#!/usr/bin/env python3
"""Offline tokenizer audit for paired Fresh v2 +7/+8 comparison.

Requires existing paired CSV from compare_fresh_v2_f30908_cplus7_cplus8.py.
Uses exact Qwen/Qwen3.5-2B-Base tokenizer; may download tokenizer files, no GPU.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def load_tokenizer(model_id):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install dependencies: pip install transformers") from exc
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def token_stats(tokenizer, correct, lure):
    c = tokenizer.encode(correct.strip(), add_special_tokens=False)
    l = tokenizer.encode(lure.strip(), add_special_tokens=False)
    if not c or not l:
        raise ValueError("Empty candidate token sequence")
    shared = 0
    for a, b in zip(c, l):
        if a != b:
            break
        shared += 1
    return {
        "correct_token_count": len(c),
        "lure_token_count": len(l),
        "correct_token_ids": " ".join(map(str, c)),
        "lure_token_ids": " ".join(map(str, l)),
        "common_prefix_token_count": shared,
        "first_token_same": int(c[0] == l[0]),
        "correct_first_distinct_token": c[shared] if shared < len(c) else "",
        "lure_first_distinct_token": l[shared] if shared < len(l) else "",
        "token_length_relation": "correct_shorter" if len(c) < len(l) else "correct_longer" if len(c) > len(l) else "equal",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--model-id", default="Qwen/Qwen3.5-2B-Base")
    p.add_argument("--input", type=Path, default=Path("results/analysis/fresh_v2_f30908_cplus7_vs_cplus8.csv"))
    p.add_argument("--output", type=Path, default=Path("results/analysis/fresh_v2_f30908_token_audit.csv"))
    args = p.parse_args()
    root = args.repo_root.resolve()
    src = args.input if args.input.is_absolute() else root / args.input
    dst = args.output if args.output.is_absolute() else root / args.output
    if not src.is_file():
        raise SystemExit(f"Missing paired CSV: {src}\nRun scripts/compare_fresh_v2_f30908_cplus7_cplus8.py first.")
    tokenizer = load_tokenizer(args.model_id)
    with src.open(newline="", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))
    if len(records) != 200:
        raise ValueError(f"Expected 200 matched slots, found {len(records)}")
    cache = {}
    output = []
    for row in records:
        key = (row["correct_answer"], row["lure_answer"])
        if key not in cache:
            cache[key] = token_stats(tokenizer, *key)
        output.append({**row, **cache[key]})
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(output[0].keys()))
        w.writeheader()
        w.writerows(output)
    print(f"Model tokenizer: {args.model_id}")
    print(f"Rows: {len(output)}; unique case_ids: {len(set(r['case_id'] for r in output))}")
    print(f"CSV: {dst}")
    for family in ("crt_difference", "crt_growth", "crt_rate"):
        family_rows = [r for r in output if r["family"] == family]
        print(f"\n{family}: n={len(family_rows)}")
        for relation in ("correct_shorter", "equal", "correct_longer"):
            subset = [r for r in family_rows if r["token_length_relation"] == relation]
            if not subset:
                continue
            s7 = sum(r["steered_7"] == "correct" for r in subset)
            s8 = sum(r["steered_8"] == "correct" for r in subset)
            print(f"  {relation}: n={len(subset)}, +7 correct={s7}, +8 correct={s8}, change={s8-s7:+d}")
        print("  same first token:", sum(r["first_token_same"] for r in family_rows))
    print("\nGrowth +7 correct -> +8 lure cases:")
    changed = [r for r in output if r["family"] == "crt_growth" and r["steered_7"] == "correct" and r["steered_8"] == "lure"]
    for r in changed:
        print(f"  seed={r['seed']} {r['case_id']} {r['template_id']} tokens(correct/lure)={r['correct_token_count']}/{r['lure_token_count']} first_distinct={r['correct_first_distinct_token']}/{r['lure_first_distinct_token']}")
    print(f"  total={len(changed)}")


if __name__ == "__main__":
    main()
