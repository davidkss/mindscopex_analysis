#!/usr/bin/env python3
"""Join Growth behavioral labels with Protocol B likelihood, cluster by case.

Repeated seeds are evaluation slots, not independent cases. Bootstrap resamples
case IDs (with all their slots), not individual rows. No GPU or dependencies.
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def mean(xs):
    return sum(xs) / len(xs)

def ci(values):
    values = sorted(values)
    return [values[int(.025 * len(values))], values[int(.975 * len(values)) - 1]]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--behavior", type=Path, default=Path("results/analysis/fresh_v2_f30908_cplus7_vs_cplus8.csv"))
    p.add_argument("--scores", type=Path, default=Path("results/analysis/protocol_b_20261029/full_answer_scores.csv"))
    p.add_argument("--output", type=Path, default=Path("results/analysis/protocol_b_behavioral_join_summary.json"))
    p.add_argument("--detail", type=Path, default=Path("results/analysis/protocol_b_behavioral_join_slots.csv"))
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20261029)
    args = p.parse_args()
    if args.bootstrap < 100:
        p.error("--bootstrap must be at least 100")
    score_rows = read(args.scores)
    scores = {}
    for r in score_rows:
        key = (r["case_id"], int(float(r["coefficient"])))
        if key in scores:
            raise ValueError(f"duplicate score {key}")
        scores[key] = r
    behavior = [r for r in read(args.behavior) if r["family"] == "crt_growth"]
    if len(behavior) != 64 or len({r["case_id"] for r in behavior}) != 33:
        raise ValueError("expected 64 Growth slots across 33 unique cases")
    cases = sorted({r["case_id"] for r in behavior})
    if len(score_rows) != 99 or set(scores) != {(case, c) for case in cases for c in (0, 7, 8)}:
        raise ValueError("expected exactly 33 cases x 3 coefficients in score CSV")
    by_case = defaultdict(list)
    detail = []
    for b in behavior:
        case = b["case_id"]
        for c in (7, 8):
            r = scores[case, c]
            zero = scores[case, 0]
            if (b["correct_answer"].strip() != r["correct_answer"].strip()
                or b["lure_answer"].strip() != r["lure_answer"].strip()):
                raise ValueError(f"candidate mismatch: {case}")
            margin = float(r["full_margin"])
            delta = margin - float(zero["full_margin"])
            if not (math.isfinite(margin) and math.isfinite(delta)):
                raise ValueError(f"non-finite score: {case}")
            predicted = "lure" if margin > 0 else "correct" if margin < 0 else "tie"
            actual = b[f"steered_{c}"]
            baseline = b[f"baseline_{c}"]
            if actual not in ("correct", "lure") or baseline not in ("correct", "lure"):
                raise ValueError(f"unrecognized behavioral label: {case} seed {b['seed']}")
            item = {
                "case_id": case, "seed": b["seed"], "coefficient": c,
                "baseline": baseline, "actual": actual, "likelihood_preference": predicted,
                "full_margin": margin, "delta_margin": delta,
                "agreement": int(actual == predicted),
                "actual_lure": int(actual == "lure"),
                "likelihood_lure": int(predicted == "lure"),
                "delta_positive": int(delta > 0),
                "correct_to_lure": int(baseline == "correct" and actual == "lure"),
                "lure_to_correct": int(baseline == "lure" and actual == "correct")
            }
            by_case[case].append(item)
            detail.append(item)
    def case_avg(case, coefficient, key):
        return mean([r[key] for r in by_case[case] if r["coefficient"] == coefficient])
    result = {
        "n_unique_cases": len(cases), "n_behavioral_slots": len(behavior),
        "n_join_rows": len(detail), "bootstrap_unit": "case_id",
        "bootstrap_resamples": args.bootstrap, "seed": args.seed,
        "protocol": "B behavior_compatible", "coefficients": {}
    }
    rng = random.Random(args.seed)
    draws = [[cases[rng.randrange(len(cases))] for _ in cases] for _ in range(args.bootstrap)]
    for c in (7, 8):
        subset = [r for r in detail if r["coefficient"] == c]
        per_case = {key: [case_avg(case, c, key) for case in cases]
                    for key in ("agreement", "actual_lure", "likelihood_lure", "delta_positive")}
        result["coefficients"][str(c)] = {
            "slot_weighted": {
                "n": len(subset),
                "actual_correct": sum(r["actual"] == "correct" for r in subset),
                "actual_lure": sum(r["actual_lure"] for r in subset),
                "likelihood_correct": sum(r["likelihood_preference"] == "correct" for r in subset),
                "likelihood_lure": sum(r["likelihood_lure"] for r in subset),
                "likelihood_tie": sum(r["likelihood_preference"] == "tie" for r in subset),
                "agreement": sum(r["agreement"] for r in subset),
                "correct_to_lure": sum(r["correct_to_lure"] for r in subset),
                "lure_to_correct": sum(r["lure_to_correct"] for r in subset)
            },
            "case_weighted": {
                key: {"mean": mean(vals),
                      "ci95": ci([mean([case_avg(case, c, key) for case in draw]) for draw in draws])}
                for key, vals in per_case.items()
            }
        }
    differences = [case_avg(case, 8, "actual_lure") - case_avg(case, 7, "actual_lure") for case in cases]
    result["paired_case_comparison"] = {
        "actual_lure_rate_plus8_minus_plus7": {
            "mean": mean(differences),
            "ci95": ci([mean([case_avg(case, 8, "actual_lure") - case_avg(case, 7, "actual_lure") for case in draw]) for draw in draws])
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.detail.parent.mkdir(parents=True, exist_ok=True)
    with args.detail.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
