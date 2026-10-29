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


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values):
    return sum(values) / len(values)


def ci(draws):
    ordered = sorted(draws)
    return [ordered[int(.025 * len(ordered))], ordered[min(len(ordered)-1, int(.975 * len(ordered)))]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slots", type=Path, default=ROOT/"results/analysis/protocol_b_behavioral_join_slots.csv")
    p.add_argument("--scores", type=Path, default=ROOT/"results/analysis/protocol_b_20261029/full_answer_scores.csv")
    p.add_argument("--output", type=Path, default=ROOT/"results/analysis/protocol_b_first_divergence_behavior_summary.json")
    p.add_argument("--details", type=Path, default=ROOT/"results/analysis/protocol_b_first_divergence_behavior_slots.csv")
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20261029)
    a = p.parse_args()
    if a.bootstrap < 100:
        p.error("--bootstrap must be >= 100")
    slots, scores = rows(a.slots), rows(a.scores)
    lookup = {}
    for r in scores:
        k = (r["case_id"], float(r["coefficient"]))
        if k in lookup:
            raise ValueError(f"Duplicate score {k}")
        lookup[k] = r
    if len(lookup) != 99:
        raise ValueError(f"Expected 99 score rows; found {len(lookup)}")
    if len(slots) != 128:
        raise ValueError(f"Expected 128 joined slots; found {len(slots)}")
    results = []
    for r in slots:
        k = (r["case_id"], float(r["coefficient"]))
        if k not in lookup:
            raise ValueError(f"Missing score {k}")
        s = lookup[k]
        if s["input_protocol"] != "behavior_compatible" or s["include_eos"] != "False":
            raise ValueError(f"Protocol/EOS mismatch {k}")
        branch = s["first_divergence_margin"]
        if not branch:
            raise ValueError(f"Missing branch margin {k}")
        branch = float(branch)
        full = float(s["full_margin"])
        if not math.isclose(full, float(r["full_margin"]), abs_tol=1e-5):
            raise ValueError(f"Full margin mismatch {k}")
        if not math.isfinite(branch):
            raise ValueError(f"Non-finite first divergence {k}")
        if r["actual"] not in ("correct", "lure"):
            raise ValueError(f"Unknown actual label {r['actual']}")
        bp = "tie" if branch == 0 else ("lure" if branch > 0 else "correct")
        fp = "lure" if full > 0 else "correct"
        results.append(dict(case_id=r["case_id"], seed=r["seed"], coefficient=r["coefficient"],
                            actual=r["actual"], first_branch_preference=bp,
                            full_preference=fp, first_branch_margin=branch, full_margin=full,
                            branch_actual_agreement=None if bp == "tie" else int(bp == r["actual"]),
                            full_actual_agreement=int(fp == r["actual"]),
                            branch_full_agreement=None if bp == "tie" else int(bp == fp)))
    groups = defaultdict(list)
    for r in results:
        groups[int(float(r["coefficient"]))].append(r)
    if set(groups) != {7, 8}:
        raise ValueError("Expected +7 and +8")
    cases = sorted({r["case_id"] for r in results})
    if len(cases) != 33 or any(len(groups[c]) != 64 for c in (7, 8)):
        raise ValueError("Unexpected case/slot count")
    metrics = ("branch_actual_agreement", "full_actual_agreement", "branch_full_agreement")
    rng = random.Random(a.seed)
    samples = [[rng.choice(cases) for _ in cases] for _ in range(a.bootstrap)]
    output = dict(method="teacher_forced_first_divergence_proxy_vs_constrained_greedy_labels",
                  limitations=["first_divergence_margin comes from separate teacher-forced candidate paths, not actual generate() logits",
                               "EOS omitted in full-answer scoring, but generation permits EOS",
                               "case_id clustered bootstrap; seed slots not independent cases",
                               "zero first-divergence margin is a tie, excluded from branch agreement denominators and reported separately"],
                  n_unique_cases=len(cases), n_join_rows=len(results),
                  bootstrap_unit="case_id", bootstrap_resamples=a.bootstrap, seed=a.seed, coefficients={})
    for coefficient, group in sorted(groups.items()):
        bycase = defaultdict(list)
        for r in group:
            bycase[r["case_id"]].append(r)
        if set(bycase) != set(cases):
            raise ValueError(f"Case mismatch for coefficient {coefficient}")
        summary = dict(n_slots=len(group), slot_weighted={}, case_weighted={})
        for metric in metrics:
            valid = [r[metric] for r in group if r[metric] is not None]
            summary["slot_weighted"][metric] = dict(count=sum(valid), n_valid=len(valid),
                                                      n_ties_excluded=len(group)-len(valid),
                                                      rate=mean(valid) if valid else None)
            casevals = {case: mean([r[metric] for r in bycase[case] if r[metric] is not None])
                        for case in cases if any(r[metric] is not None for r in bycase[case])}
            draws = []
            for sample in samples:
                valid_sample = [casevals[case] for case in sample if case in casevals]
                if valid_sample:
                    draws.append(mean(valid_sample))
            summary["case_weighted"][metric] = dict(n_valid_cases=len(casevals),
                mean=mean(list(casevals.values())) if casevals else None,
                ci95=ci(draws) if draws else None)
        summary["branch_tie_slots"] = sum(r["first_branch_preference"] == "tie" for r in group)
        summary["branch_lure_slots"] = sum(r["first_branch_preference"] == "lure" for r in group)
        summary["full_lure_slots"] = sum(r["full_preference"] == "lure" for r in group)
        summary["actual_lure_slots"] = sum(r["actual"] == "lure" for r in group)
        output["coefficients"][str(coefficient)] = summary
    a.details.parent.mkdir(parents=True, exist_ok=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.details.exists() or a.output.exists():
        raise FileExistsError("Output exists; refusing overwrite")
    with a.details.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    a.output.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
