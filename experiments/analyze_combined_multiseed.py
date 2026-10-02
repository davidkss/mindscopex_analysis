"""Validate and compare five-seed +7 behavioral artifacts using only the stdlib.

Run with Python 3.11+. Existing outputs are never overwritten. Seeds overlap;
all pooled counts describe observations, not independent samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import statistics
import subprocess
import tempfile
import tomllib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = (30908, 22552)
SEEDS = tuple(range(5))
FAMILIES = ("crt_difference", "crt_rate", "crt_growth")
LABELS = ("lure", "correct")
CATEGORY_MAP = {
    ("lure", "correct", "correct"): "both_fix",
    ("lure", "correct", "lure"): "30908_only_fix",
    ("lure", "lure", "correct"): "22552_only_fix",
    ("lure", "lure", "lure"): "neither_fix",
    ("correct", "lure", "correct"): "30908_only_regression",
    ("correct", "correct", "lure"): "22552_only_regression",
    ("correct", "lure", "lure"): "both_regress",
    ("correct", "correct", "correct"): "both_preserve_correct",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index_cases(rows, context):
    require(isinstance(rows, list) and rows, f"{context}: empty/missing rows")
    indexed = {}
    for row in rows:
        require(isinstance(row, dict), f"{context}: invalid case row")
        cid = row.get("case_id")
        require(isinstance(cid, str) and cid.strip(), f"{context}: missing case_id")
        require(cid not in indexed, f"{context}: duplicate case_id {cid}")
        require(row.get("label") in LABELS, f"{context}: {cid}: invalid label")
        require(row.get("family") in FAMILIES, f"{context}: {cid}: invalid family")
        indexed[cid] = row
    return indexed


def compare_sets(a, b, context):
    require(
        set(a) == set(b),
        f"{context}: case set mismatch; "
        f"left_only={sorted(set(a) - set(b))}, right_only={sorted(set(b) - set(a))}",
    )


CONDITIONS = ("feature30908", "feature22552", "combined")
EFFECTS = (
    "preserved_single_success",
    "lost_single_success",
    "combined_rescue",
    "combined_regression",
    "preserved_correct",
    "still_lure",
)


def effect(baseline, first, second, combined):
    if baseline == "correct":
        return "preserved_correct" if combined == "correct" else "combined_regression"
    if "correct" in (first, second):
        return "preserved_single_success" if combined == "correct" else "lost_single_success"
    return "combined_rescue" if combined == "correct" else "still_lure"


def select_seven(rows, context):
    require(isinstance(rows, list) and rows, f"{context}: expected nonempty rows")
    indexed = {}
    for row in rows:
        value = float(row["coefficient"])
        require(
            math.isfinite(value) and value not in indexed,
            f"{context}: duplicate/nonfinite coefficient",
        )
        indexed[value] = row
    require(7.0 in indexed, f"{context}: missing +7")
    return indexed[7.0]


def load_run(root, condition, seed):
    suffix = (
        f"study_behavioral_combined_f30908_f22552_seed{seed}_2b"
        if condition == "combined"
        else (
            f"study_behavioral_fine_f{condition.removeprefix('feature')}_2b"
            if seed == 0
            else f"study_behavioral_robust_f{condition.removeprefix('feature')}_seed{seed}_2b"
        )
    )
    candidates = [
        (datetime.strptime(p.name[: -(len(suffix) + 1)], "%Y%m%d-%H%M%S"), p)
        for p in root.glob(f"*_{suffix}")
        if p.is_dir()
    ]
    require(candidates, f"No run for {condition}")
    path = max(candidates, key=lambda pair: pair[0])[1]
    a = path / "artifacts"
    for name in (
        "behavioral/generations.json",
        "behavioral.csv",
        "study_feature.json",
        "config.toml",
    ):
        require((a / name).is_file(), f"Latest run missing artifact: {a / name}")
    with (a / "config.toml").open("rb") as handle:
        config = tomllib.load(handle)
    expected = (
        {"layer": 5, "feature_ids": [30908, 22552]}
        if condition == "combined"
        else {"layer": 5, "feature_id": int(condition.removeprefix("feature"))}
    )
    block_name = "features" if condition == "combined" else "feature"
    require(config[block_name] == expected, f"{path}: wrong feature config")
    require(
        ("feature" not in config if condition == "combined" else "features" not in config),
        f"{path}: conflicting feature blocks",
    )
    require(
        config["data"]["split_seed"] == seed
        and config["behavioral"]["output_mode"] == "binary_choice",
        f"{path}: incorrect seed/output mode",
    )
    require(7.0 in config["behavioral"]["coefficients"], f"{path}: missing config coefficient")
    if condition == "combined" or seed != 0:
        require(
            config["behavioral"]["coefficients"] == [7.0],
            f"{path}: incorrect combined coefficients",
        )
    meta = json.loads((a / "study_feature.json").read_text())
    require(
        meta["feature"] == expected and meta["source"] == "pinned",
        f"{path}: wrong feature metadata",
    )
    detail = select_seven(json.loads((a / "behavioral/generations.json").read_text()), str(path))
    require(detail["output_mode"] == "binary_choice", f"{path}: wrong generation mode")
    if condition == "combined":
        require(
            detail.get("layer") == 5 and detail.get("feature_ids") == [30908, 22552],
            f"{path}: wrong combined generation metadata",
        )
    before = index_cases(detail["baseline_rows"], f"{path}: baseline")
    after = index_cases(detail["steered_rows"], f"{path}: steered")
    compare_sets(before, after, str(path))
    for cid in before:
        require(before[cid]["family"] == after[cid]["family"], f"{path}: {cid}: family mismatch")
    n = len(before)
    require(n == 40, f"{path}: expected 40 cases, got {n}")
    ba = sum(r["label"] == "correct" for r in before.values()) / n
    sa = sum(r["label"] == "correct" for r in after.values()) / n
    bl = sum(r["label"] == "lure" for r in before.values()) / n
    sl = sum(r["label"] == "lure" for r in after.values()) / n
    metrics = dict(
        baseline_accuracy=ba,
        steered_accuracy=sa,
        accuracy_delta=sa - ba,
        baseline_lure_rate=bl,
        steered_lure_rate=sl,
        lure_rate_delta=sl - bl,
    )
    with (a / "behavioral.csv").open(newline="") as handle:
        recorded = select_seven(list(csv.DictReader(handle)), str(path))
    for key, value in metrics.items():
        other = float(recorded[key])
        require(
            math.isfinite(other) and math.isclose(value, other, rel_tol=1e-12, abs_tol=1e-12),
            f"{path}: {key}: computed={value}, CSV={other}",
        )
    return path, before, after, metrics


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(root, reviewed_path=None):
    """Strict by default; reviewed policy accepts only the exact audited discordances."""
    loaded, runs, checks, discordances = {}, {}, [], []
    for seed in SEEDS:
        loaded[seed] = {c: load_run(root, c, seed) for c in CONDITIONS}
        runs[str(seed)] = {c: str(v[0].resolve()) for c, v in loaded[seed].items()}
        reference = loaded[seed][CONDITIONS[0]][1]
        for c, (path, before, _, metrics) in loaded[seed].items():
            compare_sets(reference, before, f"seed={seed}, {c}")
            require(
                all(before[cid]["family"] == reference[cid]["family"] for cid in reference),
                f"seed={seed}, {c}: family mismatch",
            )
            hashes = {
                name: sha256(path / "artifacts" / name)
                for name in (
                    "config.toml",
                    "study_feature.json",
                    "behavioral.csv",
                    "behavioral/generations.json",
                )
            }
            checks.append(
                dict(
                    seed=seed,
                    condition=c,
                    run=str(path.resolve()),
                    status="passed",
                    n_cases=len(before),
                    reconstructed_csv_metrics=metrics,
                    artifact_sha256=hashes,
                )
            )
        for cid in sorted(reference):
            labels = {c: loaded[seed][c][1][cid]["label"] for c in CONDITIONS}
            if len(set(labels.values())) != 1:
                discordances.append(
                    dict(seed=seed, case_id=cid, error="baseline label mismatch", values=labels)
                )
    review = None
    if reviewed_path is None:
        require(not discordances, f"Strict baseline validation failed: {discordances}")
    else:
        reviewed = json.loads(reviewed_path.read_text())
        require(reviewed.get("status") == "failed", "Expected an existing failed validation audit")
        require(
            reviewed.get("runs") == runs,
            "Run provenance differs from reviewed audit; review required",
        )
        expected = reviewed.get("errors")
        require(
            isinstance(expected, list)
            and all(e.get("error") == "baseline label mismatch" for e in expected),
            "Reviewed audit contains unsupported errors",
        )

        def canonical(entries):
            return sorted(json.dumps(e, sort_keys=True) for e in entries)

        require(
            canonical(discordances) == canonical(expected),
            f"Unexpected/changed baseline discordance; hard error. Observed: {discordances}",
        )
        review = dict(
            path=str(reviewed_path.resolve()),
            sha256=sha256(reviewed_path),
            accepted_discordances=expected,
        )
    return loaded, dict(
        status="passed_with_reviewed_baseline_discordance" if discordances else "passed",
        structural_and_per_run_validation="passed",
        baseline_equality="discordant" if discordances else "equal",
        baseline_policy="reviewed" if reviewed_path else "strict",
        runs=runs,
        per_run_checks=checks,
        discordances=discordances,
        reviewed_audit=review,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def diagnostic_evidence(path):
    """Independent case-level follow-up evidence, never historical-run measurements."""
    if path is None:
        return None
    data = json.loads(path.read_text())
    candidates = {c["label"]: c for c in data["candidates"]}
    ids = [candidates[c]["token_ids"] for c in ("correct", "lure")]
    divergence = next((i for i in range(min(map(len, ids))) if ids[0][i] != ids[1][i]), None)
    require(divergence is not None, "Diagnostic needs an actual two-token candidate divergence")
    require(
        divergence == data["candidate_layout"]["first_divergence_position_zero_based"],
        "Diagnostic divergence metadata mismatch",
    )
    require(len(data["repeats"]) >= 5, "Diagnostic needs at least five repeats")
    observations = []
    for repeat in data["repeats"]:
        branch = next(b for b in repeat["branch_steps"] if b["step_zero_based"] == divergence)
        require(
            branch["prefix_token_ids"] == ids[0][:divergence] == ids[1][:divergence],
            "Diagnostic prefix mismatch",
        )
        tokens = {t["token_id"]: t for t in branch["tokens"]}
        correct, lure = (float(tokens[seq[divergence]]["raw_logit"]) for seq in ids)
        require(math.isfinite(correct) and math.isfinite(lure), "Nonfinite diagnostic logit")
        margin = correct - lure
        require(
            margin == branch["correct_minus_lure_raw_logit_margin"], "Diagnostic margin mismatch"
        )
        candidate = candidates[repeat["label"]]
        generated = repeat["generated_token_ids"]
        eos = data["environment"]["generation_config"]["eos_token_id"]
        eos_ids = eos if isinstance(eos, list) else [eos]
        while generated and generated[-1] in eos_ids:
            generated = generated[:-1]
        require(
            generated == candidate["token_ids"] and repeat["answer"] == candidate["text"].strip(),
            "Diagnostic answer/label/token mismatch",
        )
        observations.append(
            dict(
                repetition=repeat["repetition"],
                answer=repeat["answer"],
                label=repeat["label"],
                correct_raw_logit=correct,
                lure_raw_logit=lure,
                margin=margin,
            )
        )
    return dict(
        path=str(path.resolve()),
        sha256=sha256(path),
        case_id=data["case_id"],
        scope="separate_followup_diagnostic_only",
        historical_run_tie_status="not_measured",
        first_divergence_position_zero_based=divergence,
        candidates=data["candidates"],
        repeats=observations,
        exact_tie_all_repeats=all(r["margin"] == 0 for r in observations),
        environment=data["environment"],
    )


def case_rows(loaded, evidence):
    rows = []
    for seed in SEEDS:
        for cid, reference in sorted(loaded[seed][CONDITIONS[0]][1].items()):
            baselines = {c: loaded[seed][c][1][cid]["label"] for c in CONDITIONS}
            labels = {c: loaded[seed][c][2][cid]["label"] for c in CONDITIONS}
            comparable = len(set(baselines.values())) == 1
            shared = baselines[CONDITIONS[0]] if comparable else None
            row = dict(
                seed=seed,
                case_id=cid,
                family=reference["family"],
                baseline_discordant=not comparable,
                shared_baseline_comparable=comparable,
                shared_baseline_label=shared,
            )
            for c in CONDITIONS:
                row.update(
                    {
                        f"{c}_baseline_label": baselines[c],
                        f"{c}_label": labels[c],
                        f"{c}_transition": f"{baselines[c]}_to_{labels[c]}",
                        f"{c}_run": str(loaded[seed][c][0].resolve()),
                    }
                )
            row["single_category"] = (
                CATEGORY_MAP[shared, labels[CONDITIONS[0]], labels[CONDITIONS[1]]]
                if comparable
                else None
            )
            row["combined_effect_category"] = (
                effect(shared, *(labels[c] for c in CONDITIONS)) if comparable else None
            )
            row["shared_category_status"] = "classified" if comparable else "not_comparable"
            row["diagnostic_case_evidence_available"] = bool(
                evidence and cid == evidence["case_id"]
            )
            row["diagnostic_evidence_path"] = (
                evidence["path"] if row["diagnostic_case_evidence_available"] else None
            )
            row["historical_run_tie_status"] = "not_measured"
            rows.append(row)
    require(
        len(rows) == len({(r["seed"], r["case_id"]) for r in rows}) == 200,
        "Expected 200 unique (seed, case_id) observations",
    )
    discordant_ids = {r["case_id"] for r in rows if r["baseline_discordant"]}
    for row in rows:
        row.update(
            primary_included=True,
            S1_included=not row["baseline_discordant"],
            S2_included=row["case_id"] not in discordant_ids,
        )
    return rows


def summarize(rows):
    n = len(rows)
    require(n > 0, "Empty analysis group")
    comparable = [r for r in rows if r["shared_baseline_comparable"]]
    result = dict(
        n_cases=n,
        shared_baseline_comparable_n=len(comparable),
        baseline_discordant_n=n - len(comparable),
    )
    for c in CONDITIONS:
        baseline = sum(r[f"{c}_baseline_label"] == "correct" for r in rows) / n
        accuracy = sum(r[f"{c}_label"] == "correct" for r in rows) / n
        fixes = sum(r[f"{c}_transition"] == "lure_to_correct" for r in rows)
        regressions = sum(r[f"{c}_transition"] == "correct_to_lure" for r in rows)
        result.update(
            {
                f"{c}_baseline_accuracy": baseline,
                f"{c}_accuracy": accuracy,
                f"{c}_accuracy_delta": accuracy - baseline,
                f"{c}_fixes": fixes,
                f"{c}_regressions": regressions,
                f"{c}_net_improvement": fixes - regressions,
            }
        )
        require(
            math.isclose(
                accuracy - baseline, (fixes - regressions) / n, rel_tol=1e-12, abs_tol=1e-12
            ),
            f"{c}: transition/accuracy mismatch",
        )
    for left, right in (
        ("combined", "feature30908"),
        ("combined", "feature22552"),
        ("feature30908", "feature22552"),
    ):
        for metric in ("accuracy", "accuracy_delta"):
            result[f"{left}_minus_{right}_{metric}"] = (
                result[f"{left}_{metric}"] - result[f"{right}_{metric}"]
            )
    singles = Counter(r["single_category"] for r in comparable)
    effects = Counter(r["combined_effect_category"] for r in comparable)
    require(
        sum(singles.values()) == sum(effects.values()) == len(comparable),
        "Category partition mismatch",
    )
    require(
        set(singles) <= set(CATEGORY_MAP.values()) and set(effects) <= set(EFFECTS),
        "Invalid category",
    )
    require(
        all(
            r["single_category"] is None and r["combined_effect_category"] is None
            for r in rows
            if r["baseline_discordant"]
        ),
        "Discordant row classified as ordinary effect",
    )
    result.update({f"single_category_{k}": singles[k] for k in CATEGORY_MAP.values()})
    result.update({f"effect_{k}": effects[k] for k in EFFECTS})
    result["shared_single_success_union"] = sum(
        r["shared_baseline_label"] == "lure"
        and "correct" in (r["feature30908_label"], r["feature22552_label"])
        for r in comparable
    )
    require(
        result["shared_single_success_union"]
        == effects["preserved_single_success"] + effects["lost_single_success"],
        "Shared union mismatch",
    )
    return result


def equal_seed_mean(stats):
    """Unweighted mean of seed-level metrics, including mean counts/N (not totals)."""
    return {k: statistics.mean(s[k] for s in stats) for k in summarize_keys(stats)}


def summarize_keys(stats):
    return [k for k in stats[0] if k not in ("seed", "family")]


def analyze_subset(name, all_rows, runs):
    rows = [r for r in all_rows if r[f"{name}_included"]]
    seeds = [dict(seed=s, **summarize([r for r in rows if r["seed"] == s])) for s in SEEDS]
    families = [
        dict(
            seed=s, family=f, **summarize([r for r in rows if r["seed"] == s and r["family"] == f])
        )
        for s in SEEDS
        for f in FAMILIES
    ]
    family_means = [
        dict(family=f, **equal_seed_mean([r for r in families if r["family"] == f]))
        for f in FAMILIES
    ]
    total = summarize(rows)
    count_keys = [k for k in total if "accuracy" not in k]
    for key in count_keys:
        require(sum(s[key] for s in seeds) == total[key], f"Seed count mismatch: {key}")
        require(sum(f[key] for f in families) == total[key], f"Family count mismatch: {key}")
    robustness = {}
    for c in CONDITIONS:
        values = [s[f"{c}_accuracy_delta"] for s in seeds]
        robustness[c] = dict(
            mean=statistics.mean(values),
            sample_stdev=statistics.stdev(values),
            min=min(values),
            max=max(values),
            positive_seed_count=sum(v > 0 for v in values),
        )
    comparisons = {}
    for c in CONDITIONS[:2]:
        differences = [s[f"combined_minus_{c}_accuracy"] for s in seeds]
        comparisons[c] = dict(
            better_seed_count=sum(v > 1e-12 for v in differences),
            equal_seed_count=sum(abs(v) <= 1e-12 for v in differences),
            lower_seed_count=sum(v < -1e-12 for v in differences),
        )
    return dict(
        analysis=name,
        n_observations=len(rows),
        n_unique_cases=len({r["case_id"] for r in rows}),
        runs=runs,
        seed_summary=seeds,
        family_seed_summary=families,
        seed_equal_weighted_mean=equal_seed_mean(seeds),
        family_seed_equal_weighted_mean=family_means,
        observation_weighted_summary=total,
        family_observation_weighted_summary=[
            dict(family=f, **summarize([r for r in rows if r["family"] == f])) for f in FAMILIES
        ],
        delta_robustness=robustness,
        combined_steered_comparison_seed_counts=comparisons,
        baseline_discordant_observations=[r for r in rows if r["baseline_discordant"]],
        source_baseline_discordant_observations=[r for r in all_rows if r["baseline_discordant"]],
        excluded_observations=[
            dict(
                seed=r["seed"],
                case_id=r["case_id"],
                family=r["family"],
                reason="baseline_discordant_observation"
                if name == "S1"
                else "case_has_baseline_discordance",
            )
            for r in all_rows
            if not r[f"{name}_included"]
        ],
    )


def unique_cases(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["case_id"]].append(row)
    fields = [f"{c}_{suffix}" for c in CONDITIONS for suffix in ("baseline_label", "label")]
    fields += ["single_category", "combined_effect_category", "baseline_discordant"]
    result = []
    for cid, observations in sorted(groups.items()):
        require(len({r["family"] for r in observations}) == 1, f"{cid}: cross-seed family mismatch")
        inconsistent = [f for f in fields if len({r[f] for r in observations}) > 1]
        result.append(
            dict(
                case_id=cid,
                family=observations[0]["family"],
                n_observations=len(observations),
                seeds=json.dumps([r["seed"] for r in observations]),
                consistency_status="single_observation"
                if len(observations) == 1
                else "inconsistent"
                if inconsistent
                else "consistent",
                inconsistent_fields=json.dumps(inconsistent),
                observations_json=json.dumps(observations),
            )
        )
    require(sum(r["n_observations"] for r in result) == len(rows), "Unique-case observations lost")
    return result


def analyze(root, reviewed_path=None, diagnostic_path=None):
    require(
        set(CATEGORY_MAP) == set(itertools.product(LABELS, repeat=3)), "Incomplete category map"
    )
    require(
        all(effect(*x) in EFFECTS for x in itertools.product(LABELS, repeat=4)),
        "Incomplete effects",
    )
    loaded, validation = validate_inputs(root, reviewed_path)
    evidence = diagnostic_evidence(diagnostic_path)
    rows = case_rows(loaded, evidence)
    analyses = {
        name: analyze_subset(name, rows, validation["runs"]) for name in ("primary", "S1", "S2")
    }
    for check in validation["per_run_checks"]:
        seed = analyses["primary"]["seed_summary"][check["seed"]]
        c = check["condition"]
        for key, target in (
            ("baseline_accuracy", "baseline_accuracy"),
            ("steered_accuracy", "accuracy"),
            ("accuracy_delta", "accuracy_delta"),
        ):
            require(
                math.isclose(
                    seed[f"{c}_{target}"],
                    check["reconstructed_csv_metrics"][key],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                f"Primary result differs from source CSV: {c}, {key}",
            )
    unique = unique_cases(rows)
    summary = dict(
        policy=dict(
            primary="C+E: retain all observations and actual condition-specific baselines",
            S1="B: exclude discordant (seed, case_id) symmetrically across all conditions",
            S2="A: exclude all observations of case IDs with discordance symmetrically",
        ),
        validation=validation,
        run_provenance=validation["runs"],
        diagnostic_evidence=evidence,
        analyses=analyses,
        unique_case_consistency=dict(Counter(r["consistency_status"] for r in unique)),
        interpretation="Descriptive behavioral comparison. "
        "Seeds overlap; observations are not independent. "
        "Condition-specific delta differences need not equal steered accuracy differences. "
        "Shared categories use baseline-concordant observations. Diagnostic ties are follow-up "
        "measurements, never imputed to historical runs. Environment confounding remains; "
        "no causal synergy/interference claim.",
        aggregation_note="seed_equal_weighted_mean averages five seed-level values; "
        "N and counts there are means, "
        "not sums. observation_weighted_summary contains pooled rates and total counts. "
        "Subset denominators change; do not compare pooled and seed-equal means interchangeably.",
    )
    return rows, unique, summary


def publish(output_dir, rows, unique, summary):
    primary = summary["analyses"]["primary"]
    sensitivity = []
    for name in ("S1", "S2"):
        analysis = summary["analyses"][name]
        for aggregation, entries in (
            ("seed", analysis["seed_summary"]),
            ("seed_family", analysis["family_seed_summary"]),
            ("seed_equal_weighted_mean", [analysis["seed_equal_weighted_mean"]]),
            ("family_seed_equal_weighted_mean", analysis["family_seed_equal_weighted_mean"]),
        ):
            for entry in entries:
                sensitivity.append(
                    dict(
                        analysis=name,
                        aggregation=aggregation,
                        seed=entry.get("seed", ""),
                        family=entry.get("family", ""),
                        **{k: v for k, v in entry.items() if k not in ("seed", "family")},
                    )
                )
    tables = dict(
        combined_multiseed_primary_by_seed=primary["seed_summary"],
        combined_multiseed_primary_by_family=primary["family_seed_summary"],
        combined_multiseed_sensitivity=sensitivity,
        combined_multiseed_case_comparison=rows,
        combined_multiseed_primary_unique_cases=unique,
    )
    names = [f"{name}.csv" for name in tables] + ["combined_multiseed_primary_summary.json"]
    require(
        all(not (output_dir / name).exists() for name in names),
        "Output exists; choose a new --output-dir",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary["output_files"] = [str((output_dir / name).resolve()) for name in names]
    published = []
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".combined-multiseed-") as tmp:
        for name, data in tables.items():
            with (Path(tmp) / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        (Path(tmp) / names[-1]).write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        try:
            for name in names:
                target = output_dir / name
                os.link(Path(tmp) / name, target)
                published.append(target)
        except BaseException:
            for path in published:
                path.unlink()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/analysis")
    parser.add_argument("--baseline-policy", choices=("strict", "reviewed"), default="strict")
    parser.add_argument("--reviewed-validation", type=Path)
    parser.add_argument("--diagnostic-json", type=Path)
    args = parser.parse_args()
    if (args.baseline_policy == "reviewed") != (args.reviewed_validation is not None):
        parser.error(
            "Reviewed policy requires --reviewed-validation; strict policy does not accept it"
        )
    rows, unique, summary = analyze(args.runs_dir, args.reviewed_validation, args.diagnostic_json)
    publish(args.output_dir, rows, unique, summary)
    print(json.dumps(summary, indent=2))
    print("GIT STATUS --SHORT", flush=True)
    subprocess.run(["git", "--no-optional-locks", "status", "--short"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
