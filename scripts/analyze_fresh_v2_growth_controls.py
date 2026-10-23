#!/usr/bin/env python3
"""Build Growth inventory, then combine new controls with preserved case margins."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts import analyze_fresh_v2_growth_behavior_margin as previous
from mindscopex_analysis.growth_control_inventory import CONDITIONS, load_inventory

GROUPS = ("correct_to_lure", "correct_to_correct", "lure_to_lure", "lure_to_correct")
METRICS = (*previous.MARGINS, "plus8_minus_plus7")
INVENTORY_FIELDS = (
    "case_id", "groups", "seeds", "evaluation_slots", *GROUPS,
    "margin_available", "needs_measurement",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_outputs(csv_path, json_path, rows, summary, fields, extra_json=None):
    paths = [csv_path, json_path] + ([extra_json[0]] if extra_json else [])
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Output paths must be distinct")
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for path, payload in [(json_path, summary)] + ([extra_json] if extra_json else []):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")


def build_inventory(behavior_path, margin_path):
    # Reuse validation of seed/case duplicates, labels, answers and numeric deltas.
    slots, validated = previous.analyze(behavior_path, margin_path)
    behavior, _ = previous.read_csv(behavior_path, {"case_id", "correct_answer", "lure_answer"})
    definitions = {r["case_id"]: r for r in behavior}
    measured = {r["case_id"]: r for r in slots if r["margin_available"]}
    raw_margin, _ = previous.read_csv(margin_path, {"case_id", *previous.MARGINS})
    existing = [r for r in raw_margin if r["case_id"] in measured]
    cases, inventory_rows = [], []
    for cid in sorted({row["case_id"] for row in slots}):
        selected = [r for r in slots if r["case_id"] == cid]
        counts = Counter(r["behavior_transition"].replace("->", "_to_") for r in selected)
        groups = sorted(counts)
        cases.append(dict(
            case_id=cid, groups=groups, evaluation_slots=len(selected),
            needs_measurement=cid not in measured,
            correct_answer=definitions[cid]["correct_answer"].strip(),
            lure_answer=definitions[cid]["lure_answer"].strip(),
        ))
        inventory_rows.append(dict(
            case_id=cid, groups=";".join(groups),
            seeds=";".join(str(r["seed"]) for r in selected),
            evaluation_slots=len(selected), margin_available=cid in measured,
            needs_measurement=cid not in measured,
            **{group: counts[group] for group in GROUPS},
        ))
    counts = {group: {
        "evaluation_slots": sum(r["behavior_transition"].replace("->", "_to_") == group for r in slots),
        "unique_cases": sum(group in case["groups"] for case in cases),
        "case_ids": [case["case_id"] for case in cases if group in case["groups"]],
    } for group in GROUPS}
    expected = {"correct_to_lure": 14, "correct_to_correct": 34, "lure_to_lure": 16}
    mismatches = [f"{group}: actual={counts[group]['evaluation_slots']}, expected={count}"
                  for group, count in expected.items() if counts[group]["evaluation_slots"] != count]
    if len(slots) != 64:
        mismatches.append(f"Growth slots: actual={len(slots)}, expected=64")
    data = dict(
        schema=1, conditions=CONDITIONS, inputs=validated["inputs"],
        cases=cases, slots=[{key: row[key] for key in (
            "seed", "case_id", "family", "plus7_label", "plus8_label", "behavior_transition")}
            for row in slots],
        existing_margins=existing,
        case_ids=[case["case_id"] for case in cases if case["needs_measurement"]],
    )
    summary = dict(
        inputs=validated["inputs"], evaluation_slots=len(slots), unique_cases=len(cases),
        existing_margin_unique_cases=len(measured), additional_unique_cases=len(data["case_ids"]),
        additional_case_ids=data["case_ids"], by_transition=counts,
        overlapping_group_case_ids=[c["case_id"] for c in cases if len(c["groups"]) > 1],
        expected_count_mismatches=mismatches,
        scope="Cases may belong to multiple transition groups across seeds; group unique counts need not sum to union.",
    )
    return inventory_rows, summary, data


def read_margins(path):
    rows, _ = previous.read_csv(path, {
        "case_id", "family", "correct_answer", "lure_answer", *previous.MARGINS})
    result = {}
    for row in rows:
        cid = row["case_id"]
        if cid in result:
            raise ValueError(f"Duplicate margin case_id: {cid}")
        for key in previous.MARGINS:
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError(f"Nonfinite margin: {cid}/{key}")
        for key, expected in (
            ("plus8_minus_plus7", row["plus8_margin"] - row["plus7_margin"]),
            ("plus7_minus_baseline", row["plus7_margin"] - row["baseline_margin"]),
            ("plus8_minus_baseline", row["plus8_margin"] - row["baseline_margin"]),
        ):
            if key in row and not math.isclose(float(row[key]), expected, abs_tol=1e-9, rel_tol=1e-9):
                raise ValueError(f"Inconsistent margin delta: {cid}/{key}")
            row[key] = expected
        result[cid] = row
    return result


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def describe(rows):
    result = {"n": len(rows), "metrics": {}}
    for key in METRICS:
        values = [r[key] for r in rows]
        result["metrics"][key] = (
            dict(mean=statistics.mean(values), median=statistics.median(values),
                 minimum=min(values), q25=quantile(values, .25), q75=quantile(values, .75),
                 maximum=max(values), values=values) if values else None)
    n = len(rows)
    counts = Counter(previous.direction(r["plus8_minus_plus7"]) for r in rows)
    result["delta_direction"] = {
        key: {"count": counts[key], "rate": counts[key] / n if n else None}
        for key in ("increase", "decrease", "equal")
    }
    result["sign_transitions"] = dict(Counter(r["margin_sign_transition"] for r in rows))
    for key in ("margin_sign_changed", "strict_margin_sign_flip"):
        count = sum(r[key] for r in rows)
        result[key] = {"count": count, "rate": count / n if n else None}
    return result


def cluster_bootstrap(rows, replicates, seed):
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    ids = sorted({r["case_id"] for r in rows})
    group_cases = {
        group: {r["case_id"]: r["plus8_minus_plus7"] for r in rows if r["group"] == group}
        for group in GROUPS
    }
    active = [g for g in GROUPS if group_cases[g]]
    pairs = list(itertools.combinations(active, 2))
    draws = {g: [] for g in active}
    contrasts = {f"{a} minus {b}": [] for a, b in pairs}
    rng = random.Random(seed)
    for _ in range(replicates):
        # One shared union-of-case draw preserves cross-group case overlap.
        weights = Counter(rng.choices(ids, k=len(ids)))
        means = {}
        for group in active:
            selected = group_cases[group]
            denominator = sum(weights[cid] for cid in selected)
            if denominator:
                means[group] = sum(weights[cid] * value for cid, value in selected.items()) / denominator
                draws[group].append(means[group])
        for a, b in pairs:
            if a in means and b in means:
                contrasts[f"{a} minus {b}"].append(means[a] - means[b])

    def interval(values):
        return dict(valid_replicates=len(values),
                    lower=quantile(values, .025) if values else None,
                    upper=quantile(values, .975) if values else None)

    return dict(
        replicates=replicates, seed=seed, confidence=.95,
        estimand="Unique-case group mean of plus8_minus_plus7; paired differences between group means",
        sampling_unit="case_id; shared resample of the case union, retaining group memberships",
        group_mean_delta={g: interval(values) for g, values in draws.items()},
        group_mean_delta_differences={key: interval(values) for key, values in contrasts.items()},
        limitations="Exploratory percentile intervals; small, selected case sample; not a seed-level or causal test.",
    )


def compare(inventory_path, new_margin_path, replicates=2000, seed=0):
    data = load_inventory(inventory_path)
    # Validate embedded old measurements with the same numeric reader without writing temp files.
    old = {r["case_id"]: dict(r) for r in data["existing_margins"]}
    for row in old.values():
        for key in previous.MARGINS:
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError("Nonfinite embedded existing margin")
        row["plus8_minus_plus7"] = row["plus8_margin"] - row["plus7_margin"]
    new = read_margins(new_margin_path)
    required = set(data["case_ids"])
    allowed = {r["case_id"] for r in data["cases"]}
    if not required <= new.keys() or not new.keys() <= allowed:
        raise ValueError(f"New margin coverage differs: missing={sorted(required-new.keys())}, unexpected={sorted(new.keys()-allowed)}")
    overlap = sorted(old.keys() & new.keys())
    for cid in overlap:
        for key in (*previous.MARGINS, "plus8_minus_plus7",
                    "branch_position_1based", "correct_branch_token_id", "lure_branch_token_id"):
            if key in old[cid] or key in new[cid]:
                if key not in old[cid] or key not in new[cid] or not math.isclose(
                        float(old[cid][key]), float(new[cid][key]), abs_tol=1e-9, rel_tol=1e-9):
                    raise ValueError(f"Conflicting old/new margin: {cid}/{key}")
    # Always retain original values on overlaps.
    merged = {**new, **old}
    definitions = {r["case_id"]: r for r in data["cases"]}
    for source in (old, new):
        for cid, row in source.items():
            case = definitions[cid]
            if (row["family"] != "crt_growth"
                    or any(row[key].strip() != case[key] for key in ("correct_answer", "lure_answer"))):
                raise ValueError(f"Margin answer/family mismatch: {cid}")
    rows = []
    for slot in data["slots"]:
        row = dict(slot)
        measured = merged[row["case_id"]]
        row.update({key: measured[key] for key in METRICS})
        s7, s8 = previous.sign(row["plus7_margin"]), previous.sign(row["plus8_margin"])
        row.update(group=f"{row['plus7_label']}_to_{row['plus8_label']}",
                   margin_source="existing" if row["case_id"] in old else "new",
                   margin_sign_transition=f"{s7}->{s8}", margin_sign_changed=s7 != s8,
                   strict_margin_sign_flip={s7, s8} == {"negative", "positive"})
        rows.append(row)
    groups = {}
    for group in GROUPS:
        selected = [r for r in rows if r["group"] == group]
        unique = {r["case_id"]: r for r in selected}
        groups[group] = dict(
            evaluation_slots=len(selected), unique_cases=len(unique), case_ids=sorted(unique),
            unique_case_statistics=describe(list(unique.values())),
            slot_weighted_descriptive_statistics=describe(selected))
    summary = dict(
        inventory={"path": str(inventory_path.resolve()), "sha256": sha(inventory_path)},
        new_margin={"path": str(new_margin_path.resolve()), "sha256": sha(new_margin_path)},
        evaluation_slots=len(rows), unique_cases=len(allowed), groups=groups,
        consistency=dict(existing_cases_preserved=sorted(old), repeated_cases_compared=overlap,
                         repeated_values_consistent=True if overlap else None,
                         note="No direct remeasurement check when overlap is empty; existing cases are reused unchanged."),
        overlapping_group_case_ids=[case["case_id"] for case in data["cases"] if len(case["groups"]) > 1],
        bootstrap=cluster_bootstrap(rows, replicates, seed),
        limitations=[
            "Case-level margins are reused across seeds; slot-weighted statistics are descriptive.",
            "A case can belong to multiple groups; groups are not independent or mutually exclusive case samples.",
            "Prompt/tokenization/intervention follow the existing diagnostic; conditional logits are not full-answer likelihoods.",
            "Old and new cases run separately; without repeated cases, batch-to-batch numerical consistency is not directly measured.",
            "Model/layer/feature provenance requires diagnostic manifests; CSV values alone cannot establish it.",
            "Small selected sample and exploratory bootstrap intervals do not establish a causal mechanism.",
        ])
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--behavior", type=Path, required=True)
    inventory.add_argument("--existing-margin", type=Path, required=True)
    inventory.add_argument("--output-dir", type=Path, default=ROOT / "results/analysis")
    inventory.add_argument("--selection", type=Path, required=True)
    comparison = sub.add_parser("compare")
    comparison.add_argument("--inventory", type=Path, required=True)
    comparison.add_argument("--new-margin", type=Path, required=True)
    comparison.add_argument("--output-dir", type=Path, default=ROOT / "results/analysis")
    comparison.add_argument("--bootstrap", type=int, default=2000)
    comparison.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.command == "inventory":
            rows, summary, data = build_inventory(args.behavior, args.existing_margin)
            summary["portable_inventory"] = str(args.selection)
            write_outputs(args.output_dir / "fresh_v2_growth_control_inventory.csv",
                          args.output_dir / "fresh_v2_growth_control_inventory_summary.json",
                          rows, summary, INVENTORY_FIELDS, (args.selection, data))
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            rows, summary = compare(args.inventory, args.new_margin, args.bootstrap, args.seed)
            write_outputs(args.output_dir / "fresh_v2_growth_controls_behavior_margin_join.csv",
                          args.output_dir / "fresh_v2_growth_controls_behavior_margin_summary.json",
                          rows, summary, list(rows[0]))
            print(f"Joined {len(rows)} slots / {summary['unique_cases']} unique cases")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
