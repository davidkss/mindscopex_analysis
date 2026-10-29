#!/usr/bin/env python3
"""Analyze first-divergence vs remaining-token contributions for Protocol A/B.

No GPU needed. Validates trajectory against full-answer score CSV, then
reports paired coefficient deltas at the case level.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

BASE = Path("results/analysis")

def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def analyze_protocol(directory):
    scores = rows(directory / "full_answer_scores.csv")
    trajectory = rows(directory / "token_trajectory.csv")
    groups = defaultdict(list)
    for r in trajectory:
        key = (r["case_id"], float(r["coefficient"]), r["candidate_type"])
        groups[key].append(r)
    if len(scores) != 99:
        raise ValueError(f"{directory}: expected 99 score rows, found {len(scores)}")
    cases = defaultdict(dict)
    for s in scores:
        case, c = s["case_id"], float(s["coefficient"])
        div = int(s["first_divergence_position"])
        candidate_sums = {}
        for kind in ("correct", "lure"):
            seq = sorted(groups[(case, c, kind)], key=lambda r: int(r["token_index"]))
            if not seq or [int(x["token_index"]) for x in seq] != list(range(1, len(seq) + 1)):
                raise ValueError(f"Missing or unordered trajectory: {case}/{c}/{kind}")
            expected = float(s[f"{kind}_score"])
            total = sum(float(x["token_logprob"]) for x in seq)
            if abs(total - expected) > 1e-4:
                raise ValueError(f"Trajectory-score mismatch: {case}/{c}/{kind}: {total} != {expected}")
            if not 1 <= div <= len(seq):
                raise ValueError(f"Invalid divergence position: {case}/{c}/{kind}")
            candidate_sums[kind] = {
                "first": float(seq[div-1]["token_logprob"]),
                "remaining": sum(float(x["token_logprob"]) for x in seq[div:]),
                "prefix": sum(float(x["token_logprob"]) for x in seq[:div-1]),
            }
        correct, lure = candidate_sums["correct"], candidate_sums["lure"]
        prefix_diff = lure["prefix"] - correct["prefix"]
        if abs(prefix_diff) > 1e-4:
            raise ValueError(f"Shared prefix did not cancel: {case}/{c}")
        first = lure["first"] - correct["first"]
        remaining = lure["remaining"] - correct["remaining"]
        full = float(s["full_margin"])
        if abs(first - float(s["first_divergence_margin"])) > 1e-4:
            raise ValueError(f"First divergence mismatch: {case}/{c}")
        if abs(first + remaining - full) > 1e-4:
            raise ValueError(f"Margin decomposition mismatch: {case}/{c}")
        if c in cases[case]:
            raise ValueError(f"Duplicate score: {case}/{c}")
        cases[case][c] = {"full": full, "first": first, "remaining": remaining}
    if len(cases) != 33 or any(set(v) != {0.0, 7.0, 8.0} for v in cases.values()):
        raise ValueError("Expected 33 complete cases with coefficients 0, 7, 8")
    return cases, len(trajectory)

def mean(values):
    return sum(values) / len(values)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=Path, default=BASE / "protocol_a_20261023")
    p.add_argument("--b", type=Path, default=BASE / "protocol_b_20261029")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    a, na = analyze_protocol(args.a)
    b, nb = analyze_protocol(args.b)
    if set(a) != set(b):
        raise ValueError("A/B case IDs differ")
    result = {"n_cases": len(a), "trajectory_rows": {"A": na, "B": nb}, "coefficient_deltas": {}}
    for c in (7.0, 8.0):
        result["coefficient_deltas"][str(int(c))] = {}
        for label, cases in (("A", a), ("B", b)):
            result["coefficient_deltas"][str(int(c))][label] = {
                part: mean([cases[case][c][part] - cases[case][0.0][part] for case in sorted(cases)])
                for part in ("full", "first", "remaining")
            }
    output = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    print(output, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")

if __name__ == "__main__":
    main()
