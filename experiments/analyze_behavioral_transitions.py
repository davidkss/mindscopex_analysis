"""Validate and summarize fine-grained behavioral transitions using only stdlib."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = (30908, 22552)
COEFFICIENTS = (4.0, 5.0, 6.0, 7.0, 8.0)
FAMILIES = ("crt_difference", "crt_rate", "crt_growth")
TRANSITIONS = {
    ("lure", "correct"): "lure_to_correct",
    ("correct", "lure"): "correct_to_lure",
    ("lure", "lure"): "lure_to_lure",
    ("correct", "correct"): "correct_to_correct",
}
COUNTS = ("n_cases", *TRANSITIONS.values(), "net_improvement")
SUMMARY_FIELDS = ("feature_id", "coefficient", *COUNTS)
FAMILY_FIELDS = ("feature_id", "coefficient", "family", *COUNTS)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def latest_run(runs_dir: Path, feature_id: int) -> Path:
    suffix = f"_study_behavioral_fine_f{feature_id}_2b"
    candidates = []
    for path in runs_dir.glob(f"*{suffix}"):
        if path.is_dir():
            timestamp = datetime.strptime(path.name[:-len(suffix)], "%Y%m%d-%H%M%S")
            candidates.append((timestamp, path))
    require(bool(candidates), f"No run found for feature {feature_id} in {runs_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def coefficient_index(rows: list, context: str) -> dict:
    require(isinstance(rows, list), f"{context}: expected a list")
    indexed = {}
    for row in rows:
        require(isinstance(row, dict), f"{context}: expected an object")
        raw = row.get("coefficient")
        require(not isinstance(raw, bool), f"{context}: invalid coefficient {raw!r}")
        try:
            coefficient = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: invalid coefficient {raw!r}") from exc
        require(coefficient in COEFFICIENTS, f"{context}: unexpected coefficient {raw!r}")
        require(coefficient not in indexed, f"{context}: duplicate coefficient {coefficient}")
        indexed[coefficient] = row
    require(set(indexed) == set(COEFFICIENTS),
            f"{context}: missing coefficients {sorted(set(COEFFICIENTS) - set(indexed))}")
    return indexed


def case_index(rows: list, context: str) -> dict:
    require(isinstance(rows, list) and bool(rows), f"{context}: expected nonempty rows")
    indexed = {}
    for row in rows:
        require(isinstance(row, dict), f"{context}: expected a case object")
        case_id = row.get("case_id")
        require(isinstance(case_id, str) and bool(case_id.strip()),
                f"{context}: missing or invalid case_id")
        require(case_id not in indexed, f"{context}: duplicate case_id {case_id}")
        require(row.get("label") in ("correct", "lure"),
                f"{context}: case {case_id}: invalid label {row.get('label')!r}")
        require(row.get("family") in FAMILIES,
                f"{context}: case {case_id}: invalid family {row.get('family')!r}")
        indexed[case_id] = row
    return indexed


def empty_counts() -> dict:
    return dict.fromkeys(COUNTS, 0)


def finish_counts(counts: dict, context: str) -> None:
    counts["net_improvement"] = counts["lure_to_correct"] - counts["correct_to_lure"]
    require(sum(counts[key] for key in TRANSITIONS.values()) == counts["n_cases"],
            f"{context}: transition total differs from n_cases")


def validate_metrics(counts: dict, csv_row: dict, context: str) -> None:
    n = counts["n_cases"]
    lc, cl, ll, cc = (counts[key] for key in TRANSITIONS.values())
    computed = {
        "baseline_accuracy": (cl + cc) / n,
        "steered_accuracy": (lc + cc) / n,
        "accuracy_delta": counts["net_improvement"] / n,
        "baseline_lure_rate": (lc + ll) / n,
        "steered_lure_rate": (cl + ll) / n,
        "lure_rate_delta": -counts["net_improvement"] / n,
    }
    for metric, value in computed.items():
        try:
            expected = float(csv_row[metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{context}: missing or invalid CSV metric {metric}") from exc
        require(math.isfinite(expected) and math.isclose(
            value, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"{context}: {metric}: computed={value!r}, CSV={expected!r}")


def analyze_run(run_dir: Path, feature_id: int) -> tuple[list, list]:
    artifacts = run_dir / "artifacts"
    feature = json.loads((artifacts / "study_feature.json").read_text(encoding="utf-8"))
    require(isinstance(feature, dict) and isinstance(feature.get("feature"), dict),
            f"{run_dir}: invalid study_feature.json")
    require(feature["feature"].get("layer") == 5
            and feature["feature"].get("feature_id") == feature_id,
            f"{run_dir}: expected Layer 5 / Feature {feature_id}")
    blocks = coefficient_index(json.loads(
        (artifacts / "behavioral/generations.json").read_text(encoding="utf-8")),
        f"{run_dir}: generations.json")
    with (artifacts / "behavioral.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = coefficient_index(list(csv.DictReader(handle)), f"{run_dir}: behavioral.csv")
    summary, by_family = [], []
    for coefficient in COEFFICIENTS:
        context = f"feature={feature_id}, coefficient={coefficient}"
        block = blocks[coefficient]
        require(block.get("output_mode") == "binary_choice", f"{context}: expected binary_choice")
        baseline = case_index(block.get("baseline_rows"), f"{context}: baseline")
        steered = case_index(block.get("steered_rows"), f"{context}: steered")
        require(baseline.keys() == steered.keys(),
                f"{context}: case_id mismatch; missing steered={sorted(baseline.keys() - steered.keys())}; "
                f"missing baseline={sorted(steered.keys() - baseline.keys())}")
        total = empty_counts()
        families = {family: empty_counts() for family in FAMILIES}
        for case_id, before in baseline.items():
            after = steered[case_id]
            require(before["family"] == after["family"], f"{context}: {case_id}: family mismatch")
            transition = TRANSITIONS[(before["label"], after["label"])]
            for counts in (total, families[before["family"]]):
                counts["n_cases"] += 1
                counts[transition] += 1
        require(total["n_cases"] == len(baseline), f"{context}: case count mismatch")
        finish_counts(total, context)
        for family, counts in families.items():
            finish_counts(counts, f"{context}: {family}")
        for key in COUNTS:
            require(sum(counts[key] for counts in families.values()) == total[key],
                    f"{context}: family total mismatch for {key}")
        validate_metrics(total, csv_rows[coefficient], context)
        identity = {"feature_id": feature_id, "coefficient": coefficient}
        summary.append({**identity, **total})
        by_family.extend({**identity, "family": family, **families[family]} for family in FAMILIES)
    return summary, by_family


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/analysis")
    args = parser.parse_args()
    summary, by_family = [], []
    for feature_id in FEATURES:
        run_dir = latest_run(args.runs_dir, feature_id)
        print(f"Selected run: {run_dir}", flush=True)
        rows, family_rows = analyze_run(run_dir, feature_id)
        summary.extend(rows)
        by_family.extend(family_rows)
    print("Validation passed: 2 features, 10 coefficient groups, 30 family groups; "
          "case IDs, labels, families, totals and all 6 CSV metrics verified.", flush=True)
    # No output directory or file is created until every input has passed validation.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, fields, rows in (
        ("behavioral_transition_summary.csv", SUMMARY_FIELDS, summary),
        ("behavioral_transition_by_family.csv", FAMILY_FIELDS, by_family),
    ):
        path = args.output_dir / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
