#!/usr/bin/env python3
"""Join seed/case behavioral slots to case-level first-divergence margins (CPU only)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR = Path("results/analysis/fresh_v2_f30908_cplus7_vs_cplus8.csv")
MARGIN = Path("results/diagnostics/fresh_v2_f30908_matched_branch_margin.csv")
FALLBACK = Path(
    "results/runs/20261023-152801_diagnose_fresh_v2_matched_branch_margin_f30908_2b/"
    "artifacts/diagnostics/fresh_v2_f30908_matched_branch_margin.csv"
)
OUTPUT = Path("results/analysis")
STEM = "fresh_v2_growth_behavior_margin"
TRANSITIONS = ("correct->correct", "correct->lure", "lure->correct", "lure->lure")
MARGINS = ("baseline_margin", "plus7_margin", "plus8_margin")
FIELDS = (
    "seed", "case_id", "family", "plus7_label", "plus8_label", "behavior_transition",
    "margin_available", *MARGINS, "plus8_minus_plus7", "margin_direction",
    "margin_sign_transition", "margin_sign_changed", "strict_margin_sign_flip",
    "behavior_margin_direction_agrees",
)


def read_csv(path, required):
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not set(required) <= set(columns):
            raise ValueError(f"{path}: missing/duplicate columns; required={sorted(required)}, actual={columns}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    for index, row in enumerate(rows, 2):
        if None in row or any(row.get(key) is None or not row[key].strip() for key in required):
            raise ValueError(f"{path}:{index}: missing or extra field values")
    return rows, columns


def sign(value):
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def direction(value):
    return "increase" if value > 0 else "decrease" if value < 0 else "equal"


def summarize(rows):
    available = [r for r in rows if r["margin_available"]]
    unique = {r["case_id"]: r for r in available}
    comparable = [r for r in available if r["behavior_transition"] in ("correct->lure", "lure->correct")]

    def margin_counts(items):
        counts = Counter(r["margin_direction"] for r in items)
        return {
            "increase": counts["increase"], "decrease": counts["decrease"], "equal": counts["equal"],
            "sign_changed": sum(r["margin_sign_changed"] for r in items),
            "strict_sign_flip": sum(r["strict_margin_sign_flip"] for r in items),
            "sign_transitions": dict(Counter(r["margin_sign_transition"] for r in items)),
        }

    return {
        "evaluation_slots": len(rows), "unique_cases": len({r["case_id"] for r in rows}),
        "matched_margin_slots": len(available), "missing_margin_slots": len(rows) - len(available),
        "matched_margin_unique_cases": len(unique),
        "missing_margin_case_ids": sorted({r["case_id"] for r in rows if not r["margin_available"]}),
        "slot_weighted_margins": margin_counts(available),
        "unique_case_margins": margin_counts(list(unique.values())),
        "direction_comparable_slots": len(comparable),
        "direction_agreeing_slots": sum(r["behavior_margin_direction_agrees"] for r in comparable),
        "direction_agreement_rate": (
            sum(r["behavior_margin_direction_agrees"] for r in comparable) / len(comparable)
            if comparable else None
        ),
    }


def analyze(behavior_path, margin_path):
    required = {"seed", "case_id", "family", "steered_7", "steered_8", "correct_answer", "lure_answer"}
    behavior, behavior_columns = read_csv(behavior_path, required)
    raw_margins, margin_columns = read_csv(
        margin_path, {"case_id", "family", "correct_answer", "lure_answer", *MARGINS})
    seen, definitions = set(), {}
    for row in behavior:
        try:
            seed = int(row["seed"])
        except ValueError as exc:
            raise ValueError(f"{behavior_path}: invalid seed {row['seed']!r}") from exc
        key = (seed, row["case_id"])
        if key in seen:
            raise ValueError(f"{behavior_path}: duplicate seed/case slot {key}")
        seen.add(key)
        row["seed"] = seed
        if any(row[field] not in {"correct", "lure"} for field in ("steered_7", "steered_8")):
            raise ValueError(f"{behavior_path}: invalid behavior labels for {key}")
        transition = f"{row['steered_7']}->{row['steered_8']}"
        if "change_7_to_8" in row and row["change_7_to_8"] != transition:
            raise ValueError(f"{behavior_path}: inconsistent transition for {key}")
        definition = tuple(row[field].strip() for field in ("family", "correct_answer", "lure_answer"))
        if row["case_id"] in definitions and definitions[row["case_id"]] != definition:
            raise ValueError(f"{behavior_path}: inconsistent case definition {row['case_id']}")
        definitions[row["case_id"]] = definition

    margins = {}
    for row in raw_margins:
        cid = row["case_id"]
        if cid in margins:
            raise ValueError(f"{margin_path}: duplicate margin case_id {cid}")
        for field in MARGINS:
            try:
                row[field] = float(row[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{margin_path}: invalid {field} for {cid}") from exc
            if not math.isfinite(row[field]):
                raise ValueError(f"{margin_path}: nonfinite {field} for {cid}")
        delta = row["plus8_margin"] - row["plus7_margin"]
        for field, expected in (
            ("plus8_minus_plus7", delta),
            ("plus7_minus_baseline", row["plus7_margin"] - row["baseline_margin"]),
            ("plus8_minus_baseline", row["plus8_margin"] - row["baseline_margin"]),
        ):
            if field in row:
                original = row[field]
                try:
                    raw_delta = float(original)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{margin_path}: invalid {field} for {cid}") from exc
                if not math.isclose(raw_delta, expected, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"{margin_path}: inconsistent {field} for {cid}")
        row["plus8_minus_plus7"] = delta
        margins[cid] = row

    growth = [r for r in behavior if r["family"] == "crt_growth"]
    if not growth:
        raise ValueError(f"{behavior_path}: no Growth evaluation slots")
    joined = []
    for row in sorted(growth, key=lambda r: (r["seed"], r["case_id"])):
        transition = f"{row['steered_7']}->{row['steered_8']}"
        item = dict(seed=row["seed"], case_id=row["case_id"], family=row["family"],
                    plus7_label=row["steered_7"], plus8_label=row["steered_8"],
                    behavior_transition=transition, margin_available=row["case_id"] in margins)
        measured = margins.get(row["case_id"])
        if measured is None:
            if transition == "correct->lure":
                raise ValueError(f"{margin_path}: missing margin for target case {row['case_id']}")
            item.update({field: None for field in FIELDS if field not in item})
        else:
            definition = tuple(measured[field].strip() for field in ("family", "correct_answer", "lure_answer"))
            if definition != definitions[row["case_id"]]:
                raise ValueError(f"Case definition differs between inputs: {row['case_id']}")
            delta = measured["plus8_minus_plus7"]
            s7, s8 = sign(measured["plus7_margin"]), sign(measured["plus8_margin"])
            item.update({field: measured[field] for field in (*MARGINS, "plus8_minus_plus7")})
            item.update(
                margin_direction=direction(delta), margin_sign_transition=f"{s7}->{s8}",
                margin_sign_changed=s7 != s8,
                strict_margin_sign_flip={s7, s8} == {"negative", "positive"},
                behavior_margin_direction_agrees=(
                    delta > 0 if transition == "correct->lure" else
                    delta < 0 if transition == "lure->correct" else None),
            )
        joined.append(item)
    target = [r for r in joined if r["behavior_transition"] == "correct->lure"]
    checks = {
        "growth_evaluation_slots": (len(joined), 64),
        "correct_to_lure_slots": (len(target), 14),
        "correct_to_lure_unique_cases": (len({r["case_id"] for r in target}), 6),
        "target_unique_cases_with_positive_delta": (
            len({r["case_id"] for r in target if r["plus8_minus_plus7"] > 0}), 6),
    }
    summary = {
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "columns": columns}
            for name, path, columns in (
                ("behavior", behavior_path, behavior_columns), ("margin", margin_path, margin_columns))
        },
        "growth": summarize(joined), "correct_to_lure": summarize(target),
        "by_seed": {str(seed): {
            "growth": summarize([r for r in joined if r["seed"] == seed]),
            "correct_to_lure": summarize([r for r in target if r["seed"] == seed]),
        } for seed in sorted({r["seed"] for r in joined})},
        "by_behavior_transition": {
            transition: summarize([r for r in joined if r["behavior_transition"] == transition])
            for transition in TRANSITIONS
        },
        "validation": {
            key: {"actual": actual, "expected": expected, "matches": actual == expected}
            for key, (actual, expected) in checks.items()
        },
        "mismatches": [f"{key}: actual={actual}, expected={expected}"
                       for key, (actual, expected) in checks.items() if actual != expected],
        "definitions": {
            "margin": "lure_branch_logit - correct_branch_logit; positive favors lure at the measured branch",
            "sign_transition": "+7 sign -> +8 sign; zero is a separate category",
            "strict_sign_flip": "negative <-> positive only; excludes transitions involving zero",
            "direction_agreement": "correct->lure requires delta>0; lure->correct requires delta<0; stable labels not assessed",
        },
        "limitations": [
            "Margins are case-level measurements reused across seeds, not independent seed-level measurements.",
            "Only measured cases support margin comparisons; missing margins are blank, never zero or imputed.",
            "Diagnostic Growth cases were selected for correct->lure regressions; coverage is outcome-selected.",
            "Stable-transition rows on those same cases are not representative of all stable Growth cases.",
            "First-divergence conditional logits are not full-answer likelihoods or complete decoding equivalence.",
            "Directional agreement does not establish a causal mechanism or CRT reasoning improvement.",
            "CSV inputs alone do not verify model/layer/feature provenance or exact behavioral context equivalence.",
        ],
    }
    return joined, summary


def run(behavior_path, margin_path, output_dir):
    outputs = (output_dir / f"{STEM}_join.csv", output_dir / f"{STEM}_summary.json")
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        if path.resolve() in {behavior_path.resolve(), margin_path.resolve()}:
            raise ValueError(f"Output aliases input: {path}")
    rows, summary = analyze(behavior_path, margin_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with outputs[0].open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with outputs[1].open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return outputs, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", type=Path, default=ROOT / BEHAVIOR)
    parser.add_argument("--margin", type=Path, help="Explicit margin CSV; otherwise try canonical then known Colab artifact")
    parser.add_argument("--output-dir", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args()
    margin_path = args.margin
    if margin_path is None:
        margin_path = next((ROOT / p for p in (MARGIN, FALLBACK) if (ROOT / p).is_file()), None)
        if margin_path is None:
            parser.exit(1, f"ERROR: Margin CSV not found. Tried: {ROOT / MARGIN}, {ROOT / FALLBACK}\n")
    try:
        outputs, summary = run(args.behavior, margin_path, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(f"Behavior input: {args.behavior}\nMargin input: {margin_path}")
    print(json.dumps({key: summary[key] for key in ("validation", "mismatches")}, indent=2))
    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
