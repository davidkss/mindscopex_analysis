#!/usr/bin/env python3
"""Paired bootstrap comparison of MindScopeX full-answer Protocol A and B."""
import csv
import json
import argparse
from pathlib import Path

def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = {(r["case_id"], float(r["coefficient"])): r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate case/coefficient")
    return result

class RNG:
    def __init__(self):
        self.state = 20261029
    def random(self):
        self.state = (1664525 * self.state + 1013904223) & 0xffffffff
        return self.state / 4294967296

def ci(xs, rng):
    n = len(xs)
    means = sorted(sum(xs[int(rng.random() * n)] for _ in range(n)) / n for _ in range(20000))
    return [means[499], means[19499]]

def mean(xs):
    return sum(xs) / len(xs)

def analyze(a, b):
    if set(a) != set(b) or len(a) != 99:
        raise ValueError("Expected 99 aligned case/coefficient rows")
    cases = sorted({k[0] for k in a})
    if len(cases) != 33 or any((case, c) not in a for case in cases for c in (0., 7., 8.)):
        raise ValueError("Expected 33 complete cases")
    rng = RNG()
    result = {"n_cases": 33, "n_rows_each": 99, "bootstrap_resamples": 20000, "seed": 20261029, "coefficients": {}}
    for c in (7., 8.):
        da = [float(a[case,c]["full_margin"]) - float(a[case,0.]["full_margin"]) for case in cases]
        db = [float(b[case,c]["full_margin"]) - float(b[case,0.]["full_margin"]) for case in cases]
        diff = [y-x for x,y in zip(da,db)]
        result["coefficients"][str(int(c))] = {
            "A": {"mean_delta": mean(da), "ci95": ci(da,rng)},
            "B": {"mean_delta": mean(db), "ci95": ci(db,rng)},
            "B_minus_A": {"mean_delta": mean(diff), "ci95": ci(diff,rng)},
            "first_divergence_mean_delta": {
                name: mean([float(rows[case,c]["first_divergence_margin"]) - float(rows[case,0.]["first_divergence_margin"]) for case in cases])
                for name,rows in (("A",a),("B",b))
            }
        }
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="results/analysis/protocol_a_20261023/full_answer_scores.csv")
    p.add_argument("--b", default="results/analysis/protocol_b_20261029/full_answer_scores.csv")
    p.add_argument("--output")
    args = p.parse_args()
    report = json.dumps(analyze(load(args.a), load(args.b)), indent=2, ensure_ascii=False) + "\n"
    print(report, end="")
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
