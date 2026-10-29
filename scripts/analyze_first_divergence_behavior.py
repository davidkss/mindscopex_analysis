#!/usr/bin/env python3
"""Audit first-divergence preference against observed constrained greedy CRT labels.

The first-divergence margin is a diagnostic from two teacher-forced paths,
not a capture of actual generate() logits. Interpret agreement accordingly.
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path, required=frozenset()):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"Duplicate CSV columns in {path}")
        if not required <= set(fields):
            raise ValueError(f"Missing required columns in {path}: {sorted(required)}")
        result = list(reader)
        if any(None in row or any(row[k] is None for k in required) for row in result):
            raise ValueError(f"Malformed CSV row in {path}")
        return result


def finite(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value: {value}")
    return number


def preference(margin):
    return "tie" if margin == 0 else "lure" if margin > 0 else "correct"


def agreement(left, right):
    return None if "tie" in (left, right) else int(left == right)


def summarize(group, metric, samples):
    valid = [r for r in group if r[metric] is not None]
    bycase = defaultdict(list)
    for row in valid:
        bycase[row["case_id"]].append(row[metric])
    casevals = {case: mean(values) for case, values in bycase.items()}
    # Resample complete case clusters, retaining repeated draws of each case.
    # A case with no eligible slots contributes no denominator for this metric.
    draws = []
    for sample in samples:
        values = [casevals[c] for c in sample if c in casevals]
        if values:
            draws.append(mean(values))
    return (
        dict(count=sum(r[metric] for r in valid), n_valid=len(valid),
             n_excluded=len(group)-len(valid), n_ties_excluded=len(group)-len(valid),
             rate=mean([r[metric] for r in valid]) if valid else None),
        dict(n_valid_cases=len(casevals), n_excluded_cases=len({r["case_id"] for r in group})-len(casevals),
             mean=mean(list(casevals.values())) if casevals else None,
             ci95=ci(draws) if draws else None, bootstrap_valid_resamples=len(draws)),
    )


def mean(values):
    return sum(values) / len(values)


def ci(draws):
    ordered = sorted(draws)
    return [ordered[int(.025 * len(ordered))], ordered[min(len(ordered)-1, int(.975 * len(ordered)))]]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--slots", type=Path, default=ROOT/"results/analysis/protocol_b_behavioral_join_slots.csv")
    p.add_argument("--scores", type=Path, default=ROOT/"results/analysis/protocol_b_20261029/full_answer_scores.csv")
    p.add_argument("--output", type=Path, default=ROOT/"results/analysis/protocol_b_first_divergence_behavior_summary.json")
    p.add_argument("--details", type=Path, default=ROOT/"results/analysis/protocol_b_first_divergence_behavior_slots.csv")
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20261029)
    a = p.parse_args(argv)
    if a.bootstrap < 100:
        p.error("--bootstrap must be >= 100")
    if a.output.resolve() == a.details.resolve():
        raise ValueError("Summary and details paths must differ")
    if a.output.exists() or a.details.exists():
        raise FileExistsError("Output exists; refusing overwrite")
    slots = rows(a.slots, {"case_id", "seed", "coefficient", "actual", "full_margin"})
    scores = rows(a.scores, {"case_id", "coefficient", "input_protocol", "include_eos",
                            "first_divergence_margin", "full_margin"})
    lookup = {}
    for r in scores:
        k = (r["case_id"], finite(r["coefficient"]))
        if k in lookup:
            raise ValueError(f"Duplicate score {k}")
        if not k[0] or k[1] not in (0, 7, 8):
            raise ValueError(f"Unexpected score key {k}")
        if r["input_protocol"] != "behavior_compatible" or r["include_eos"] != "False":
            raise ValueError(f"Protocol/EOS mismatch {k}")
        finite(r["first_divergence_margin"])
        finite(r["full_margin"])
        lookup[k] = r
    if len(lookup) != 99:
        raise ValueError(f"Expected 99 score rows; found {len(lookup)}")
    if len(slots) != 128:
        raise ValueError(f"Expected 128 joined slots; found {len(slots)}")
    score_cases = {key[0] for key in lookup}
    if len(score_cases) != 33 or set(lookup) != {(c, v) for c in score_cases for v in (0, 7, 8)}:
        raise ValueError("Expected the same 33 cases at coefficients 0, 7, 8")
    results = []
    seen_slots = set()
    for r in slots:
        k = (r["case_id"], finite(r["coefficient"]))
        if k not in lookup:
            raise ValueError(f"Missing score {k}")
        slot_key = (k[0], r["seed"], k[1])
        if k[1] not in (7, 8) or r["seed"] not in {"1", "2", "3", "4", "5"} or slot_key in seen_slots:
            raise ValueError(f"Invalid or duplicate slot {slot_key}")
        seen_slots.add(slot_key)
        s = lookup[k]
        if s["input_protocol"] != "behavior_compatible" or s["include_eos"] != "False":
            raise ValueError(f"Protocol/EOS mismatch {k}")
        branch = s["first_divergence_margin"]
        if not branch:
            raise ValueError(f"Missing branch margin {k}")
        branch = finite(branch)
        full = finite(s["full_margin"])
        if not math.isclose(full, finite(r["full_margin"]), rel_tol=0, abs_tol=1e-5):
            raise ValueError(f"Full margin mismatch {k}")
        if r["actual"] not in ("correct", "lure"):
            raise ValueError(f"Unknown actual label {r['actual']}")
        bp = preference(branch)
        fp = preference(full)
        results.append(dict(case_id=r["case_id"], seed=r["seed"], coefficient=r["coefficient"],
                            actual=r["actual"], first_branch_preference=bp,
                            full_preference=fp, first_branch_margin=branch, full_margin=full,
                            branch_actual_agreement=agreement(bp, r["actual"]),
                            full_actual_agreement=agreement(fp, r["actual"]),
                            branch_full_agreement=agreement(bp, fp)))
        results[-1]["paired_branch_minus_full_agreement"] = (
            agreement(bp, r["actual"]) - agreement(fp, r["actual"])
            if "tie" not in (bp, fp) else None
        )
    if {(c, seed) for c, seed, v in seen_slots if v == 7} != {(c, seed) for c, seed, v in seen_slots if v == 8}:
        raise ValueError("Behavioral slot identities differ across coefficients")
    groups = defaultdict(list)
    for r in results:
        groups[finite(r["coefficient"])].append(r)
    if set(groups) != {7, 8}:
        raise ValueError("Expected +7 and +8")
    cases = sorted({r["case_id"] for r in results})
    if len(cases) != 33 or any(len(groups[c]) != 64 for c in (7, 8)):
        raise ValueError("Unexpected case/slot count")
    metrics = ("branch_actual_agreement", "full_actual_agreement", "branch_full_agreement", "paired_branch_minus_full_agreement")
    rng = random.Random(a.seed)
    samples = [[rng.choice(cases) for _ in cases] for _ in range(a.bootstrap)]
    output = dict(method="teacher_forced_first_divergence_proxy_vs_constrained_greedy_labels",
                  limitations=["first_divergence_margin comes from separate teacher-forced candidate paths, not actual generate() logits",
                               "EOS omitted in full-answer scoring, but generation permits EOS",
                               "case_id clustered bootstrap; seed slots not independent cases",
                               "zero first-divergence margin is a tie, excluded from branch agreement denominators and reported separately",
                               "zero full-answer margin is a tie, excluded from full and paired agreement denominators"],
                  n_unique_cases=len(cases), n_join_rows=len(results),
                  tie_policy="exact zero is tie; exclude per metric; paired difference uses common eligible slots",
                  bootstrap_estimand="equal-weight mean of eligible case means; empty case clusters omitted",
                  bootstrap_unit="case_id", bootstrap_resamples=a.bootstrap, seed=a.seed, coefficients={})
    for coefficient, group in sorted(groups.items()):
        bycase = defaultdict(list)
        for r in group:
            bycase[r["case_id"]].append(r)
        if set(bycase) != set(cases):
            raise ValueError(f"Case mismatch for coefficient {coefficient}")
        summary = dict(n_slots=len(group), slot_weighted={}, case_weighted={})
        for metric in metrics:
            slot_summary, case_summary = summarize(group, metric, samples)
            summary["slot_weighted"][metric] = slot_summary
            summary["case_weighted"][metric] = case_summary
        summary["branch_tie_slots"] = sum(r["first_branch_preference"] == "tie" for r in group)
        summary["full_tie_slots"] = sum(r["full_preference"] == "tie" for r in group)
        summary["case_slot_multiplicity"] = {case: len(bycase[case]) for case in cases}
        summary["branch_lure_slots"] = sum(r["first_branch_preference"] == "lure" for r in group)
        summary["full_lure_slots"] = sum(r["full_preference"] == "lure" for r in group)
        summary["actual_lure_slots"] = sum(r["actual"] == "lure" for r in group)
        output["coefficients"][str(int(coefficient))] = summary
    a.details.parent.mkdir(parents=True, exist_ok=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.details.exists() or a.output.exists():
        raise FileExistsError("Output exists; refusing overwrite")
    created = []
    try:
        with a.details.open("x", newline="", encoding="utf-8") as f:
            created.append(a.details)
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        with a.output.open("x", encoding="utf-8") as f:
            created.append(a.output)
            f.write(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    except OSError:
        # Remove only files created by this invocation if either write fails.
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
