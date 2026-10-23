#!/usr/bin/env python3
"""Compare Fresh v2 Feature 30908 +7 and +8 on identical seed/case slots.

Reads committed +7 generations.json and local +8 artifacts.zip; no GPU needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

TIMES7 = ("155728", "162901", "163038", "163135", "163232")
TIMES8 = ("135827", "140018", "140118", "140216", "140400")
DATE7 = "20261019"
DATE8 = "20261023"
PREFIX = "study_transfer_fresh_v2_f30908"
FIELDS = (
    "seed", "case_id", "family", "template_id", "question", "correct_answer",
    "lure_answer", "baseline_7", "baseline_8", "steered_7", "steered_8",
    "transition_7", "transition_8", "change_7_to_8",
)


def load_generation(path: Path, *, archive: bool):
    if archive:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith("behavioral/generations.json")]
            if len(names) != 1:
                raise ValueError(f"{path}: expected one behavioral/generations.json, found {names}")
            payload = json.loads(z.read(names[0]))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"{path}: expected one coefficient result")
    item = payload[0]
    if float(item["coefficient"]) != (8.0 if archive else 7.0):
        raise ValueError(f"{path}: unexpected coefficient {item['coefficient']}")
    return item


def keyed(rows, path):
    result = {}
    for row in rows:
        case_id = row["case_id"]
        if case_id in result:
            raise ValueError(f"{path}: duplicate {case_id}")
        if row["label"] not in ("correct", "lure"):
            raise ValueError(f"{path}: unexpected label {row['label']}")
        result[case_id] = row["label"]
    return result


def transition(before, after):
    return f"{before}->{after}"


def summarize(rows, name, predicate):
    chosen = [r for r in rows if predicate(r)]
    counts = Counter((r["steered_7"], r["steered_8"]) for r in chosen)
    b7 = sum(r["baseline_7"] == "correct" for r in chosen)
    b8 = sum(r["baseline_8"] == "correct" for r in chosen)
    s7 = sum(r["steered_7"] == "correct" for r in chosen)
    s8 = sum(r["steered_8"] == "correct" for r in chosen)
    print(f"{name:24} n={len(chosen):3} base7={b7:3} base8={b8:3} "
          f"+7={s7:3} +8={s8:3} delta8-7={s8-s7:+3} "
          f"7correct->8lure={counts['correct','lure']:3} "
          f"7lure->8correct={counts['lure','correct']:3}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/analysis/fresh_v2_f30908_cplus7_vs_cplus8.csv"))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    data = json.loads((root / "src/mindscopex_analysis/data/crt_fresh_v2.json").read_text(encoding="utf-8"))
    cases = {c["case_id"]: c for c in data["cases"]}
    rows = []
    for seed, (t7, t8) in enumerate(zip(TIMES7, TIMES8), start=1):
        p7 = root / "results/runs" / f"{DATE7}-{t7}_{PREFIX}_seed{seed}_2b/artifacts/behavioral/generations.json"
        p8 = root / "results/runs" / f"{DATE8}-{t8}_{PREFIX}_cplus8_seed{seed}_2b/artifacts.zip"
        if not p7.is_file() or not p8.is_file():
            raise FileNotFoundError(f"Missing seed {seed} input: +7={p7.exists()}, +8={p8.exists()}\n"
                                    f"Expected +8 at: {p8}")
        a, b = load_generation(p7, archive=False), load_generation(p8, archive=True)
        a_base, b_base = keyed(a["baseline_rows"], p7), keyed(b["baseline_rows"], p8)
        a_steer, b_steer = keyed(a["steered_rows"], p7), keyed(b["steered_rows"], p8)
        if not (a_base.keys() == b_base.keys() == a_steer.keys() == b_steer.keys()):
            raise ValueError(f"Seed {seed}: case_id sets do not match")
        for case_id in sorted(a_base):
            c = cases[case_id]
            rows.append({
                "seed": seed, "case_id": case_id, "family": c["family"],
                "template_id": c["template_id"], "question": c["question"],
                "correct_answer": c["correct_answer"], "lure_answer": c["lure_answer"],
                "baseline_7": a_base[case_id], "baseline_8": b_base[case_id],
                "steered_7": a_steer[case_id], "steered_8": b_steer[case_id],
                "transition_7": transition(a_base[case_id], a_steer[case_id]),
                "transition_8": transition(b_base[case_id], b_steer[case_id]),
                "change_7_to_8": transition(a_steer[case_id], b_steer[case_id]),
            })
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Matched slots: {len(rows)}")
    summarize(rows, "ALL", lambda r: True)
    for family in ("crt_difference", "crt_growth", "crt_rate"):
        summarize(rows, family, lambda r, f=family: r["family"] == f)
    for template in sorted({r["template_id"] for r in rows}):
        summarize(rows, template, lambda r, t=template: r["template_id"] == t)
    mismatches = [r for r in rows if r["baseline_7"] != r["baseline_8"]]
    print(f"Baseline mismatches between runs: {len(mismatches)}")
    if mismatches:
        print("CAUTION: baseline differences indicate an additional run-level difference.")
    print(f"CSV: {output}")


if __name__ == "__main__":
    main()
