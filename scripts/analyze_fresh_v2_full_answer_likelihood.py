#!/usr/bin/env python3
"""Validate full-answer scores/trajectories, join 64 slots, compare branch margins."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from mindscopex_analysis.full_answer_protocols import (
    BEHAVIOR, DIAGNOSTIC, PROTOCOLS, candidate_texts, protocol_spec, require_protocol, token_ids_sha256,
)
from mindscopex_analysis.growth_control_inventory import CONDITIONS, load_inventory
from scripts import analyze_fresh_v2_growth_controls as controls
from scripts import diagnose_fresh_v2_full_answer_likelihood as diagnostic

COEFFICIENTS = (0.0, 7.0, 8.0)
STEM = "fresh_v2_growth_full_answer"


def close(a, b, name):
    if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, abs_tol=1e-5, rel_tol=1e-6):
        raise ValueError(f"Inconsistent/nonfinite {name}: {a} versus {b}")


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError(f"Invalid CSV boolean: {value!r}")
    return value == "True"


def finite(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {name}")
    return result


def read_diagnostic(run_dir, inventory_path, *, input_protocol=None):
    data = load_inventory(inventory_path)
    if len(data["cases"]) != 33 or len(data["slots"]) != 64:
        raise ValueError("Expected exactly the existing 33 cases / 64 slots")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    declared_protocol = require_protocol(manifest.get("input_protocol"))
    if input_protocol is not None and require_protocol(input_protocol) != declared_protocol:
        raise ValueError("Requested input_protocol differs from diagnostic manifest")
    if manifest.get("schema") != 2 or manifest.get("input_protocol_spec") != protocol_spec(declared_protocol):
        raise ValueError("Missing/unsupported input protocol schema or definition")
    if manifest.get("eos_policy") != ("include" if manifest["include_eos"] else "exclude"):
        raise ValueError("Manifest EOS policy differs")
    expected = {r["case_id"] for r in data["cases"]}
    manifest_ids = manifest.get("case_ids", [])
    if (manifest.get("status") != "succeeded" or manifest.get("conditions") != CONDITIONS
            or manifest.get("coefficients") != list(COEFFICIENTS)
            or len(manifest_ids) != len(expected) or set(manifest_ids) != expected
            or manifest.get("inventory_sha256") != controls.sha(inventory_path)):
        raise ValueError("Manifest conditions, case IDs, status or inventory hash differ")
    prompt_records = manifest.get("prompts", [])
    if (len(prompt_records) != len(expected)
            or {row["case_id"] for row in prompt_records} != expected):
        raise ValueError("Missing/duplicate per-case prompt fingerprints")
    prompts = {row["case_id"]: row for row in prompt_records}
    for record in prompt_records:
        ids = record["prompt_token_ids"]
        if (record["input_protocol"] != declared_protocol or not ids
                or any(type(token) is not int or token < 0 for token in ids)
                or record["prompt_token_ids_sha256"] != token_ids_sha256(ids)):
            raise ValueError("Invalid prompt token-ID fingerprint or protocol")
    sae = manifest.get("sae_settings", {})
    model = manifest.get("model_settings", {})
    if (model.get("model_id") != CONDITIONS["model_id"]
            or sae.get("layer") != 5 or sae.get("feature_id") != 30908
            or sae.get("coefficients") != list(COEFFICIENTS)
            or sae.get("token_position") != "all" or not sae.get("repo_id")):
        raise ValueError("Manifest model/SAE settings differ from pinned conditions")
    for filename, key in (("full_answer_scores.csv", "score_csv_sha256"),
                          ("token_trajectory.csv", "trajectory_csv_sha256")):
        if controls.sha(run_dir / filename) != manifest.get(key):
            raise ValueError(f"CSV/manifest SHA-256 mismatch: {filename}")
    raw_scores, _ = controls.previous.read_csv(run_dir / "full_answer_scores.csv", {
        "case_id", "coefficient", "correct_score", "lure_score", "full_margin",
        "correct_mean_logprob", "lure_mean_logprob", "correct_token_length", "lure_token_length",
        "correct_token_ids", "lure_token_ids", "first_divergence_position", "include_eos",
        "input_protocol", "prompt_token_ids_sha256", "correct_token_ids_sha256", "lure_token_ids_sha256"})
    raw_tokens, _ = controls.previous.read_csv(run_dir / "token_trajectory.csv", {
        "case_id", "coefficient", "candidate_type", "token_index", "token_id", "token_logprob",
        "cumulative_logprob", "shared_prefix", "first_divergence_position", "is_eos",
        "input_protocol", "prompt_token_ids_sha256", "candidate_token_ids_sha256"})
    if any(row["input_protocol"] != declared_protocol for row in raw_scores + raw_tokens):
        raise ValueError("Mixed input_protocol values in diagnostic CSV files")
    scores = {}
    for row in raw_scores:
        key = (row["case_id"], float(row["coefficient"]))
        if key in scores:
            raise ValueError(f"Duplicate score case/coefficient: {key}")
        scores[key] = row
    required_keys = {(cid, c) for cid in expected for c in COEFFICIENTS}
    if set(scores) != required_keys:
        raise ValueError(f"Score coverage mismatch: missing={sorted(required_keys-scores.keys())}, extra={sorted(scores.keys()-required_keys)}")
    tokens = {}
    for row in raw_tokens:
        key = (row["case_id"], float(row["coefficient"]), row["candidate_type"], int(row["token_index"]))
        if key in tokens:
            raise ValueError(f"Duplicate trajectory token: {key}")
        tokens[key] = row
    expected_token_keys = set()
    _, reference_cases = diagnostic.select_growth_cases(inventory_path)
    definitions = {case.case_id: candidate_texts(case, declared_protocol) for case in reference_cases}
    for (cid, coefficient), row in scores.items():
        record = prompts[cid]
        if declared_protocol == BEHAVIOR and any(
                audit.get("behavior_reference_token_ids_verified") is not True
                or audit.get("reference_candidate_paths_checked") != 2
                for audit in record["boundary_audits"].values()):
            raise ValueError("Missing behavior reference token-ID compatibility verification")
        if (row["prompt_token_ids_sha256"] != record["prompt_token_ids_sha256"]
                or record["candidate_texts"] != definitions[cid]):
            raise ValueError("Score prompt fingerprint/candidate text differs from manifest")
        if row["family"] != "crt_growth":
            raise ValueError(f"Unexpected family: {cid}")
        if boolean(row["include_eos"]) != manifest["include_eos"]:
            raise ValueError("Mixed EOS scoring policies")
        recorded_eos = int(row["eos_token_id"]) if row["eos_token_id"] else None
        if recorded_eos != manifest["eos_token_id"]:
            raise ValueError("Score/manifest EOS ID differs")
        sequences = {kind: json.loads(row[f"{kind}_token_ids"]) for kind in ("correct", "lure")}
        if sequences["correct"] == sequences["lure"]:
            raise ValueError("Identical candidate tokens")
        common = diagnostic.common_prefix(sequences["correct"], sequences["lure"])
        if int(row["shared_prefix_length"]) != common or int(row["first_divergence_position"]) != common+1:
            raise ValueError("Incorrect shared prefix/divergence metadata")
        traces = {}
        for kind, ids in sequences.items():
            if (not ids or any(type(token) is not int or token < 0 for token in ids)
                    or int(row[f"{kind}_token_length"]) != len(ids)):
                raise ValueError(f"Invalid token IDs/length: {cid}/{kind}")
            if (record["candidate_token_ids"].get(kind) != ids
                    or record["candidate_token_ids_sha256"].get(kind) != token_ids_sha256(ids)
                    or row[f"{kind}_token_ids_sha256"] != token_ids_sha256(ids)):
                raise ValueError(f"Candidate token-ID fingerprint differs: {cid}/{kind}")
            answer_ids = ids[:-1] if manifest["include_eos"] else ids
            if record["answer_token_ids_sha256"].get(kind) != token_ids_sha256(answer_ids):
                raise ValueError("Answer-only token-ID fingerprint differs")
            if row[f"{kind}_answer"] != definitions[cid][kind]:
                raise ValueError(f"Candidate answer differs from inventory: {cid}/{kind}")
            if ids != json.loads(scores[(cid, 0.0)][f"{kind}_token_ids"]):
                raise ValueError("Candidate tokens differ across coefficients")
            if manifest["include_eos"] and ids[-1] != manifest["eos_token_id"]:
                raise ValueError("Missing configured EOS target")
            answer_length = len(ids) - int(manifest["include_eos"])
            if int(row[f"{kind}_answer_token_length"]) != answer_length:
                raise ValueError("Wrong non-EOS answer length")
            logprobs = []
            for index, token_id in enumerate(ids, 1):
                key = (cid, coefficient, kind, index)
                expected_token_keys.add(key)
                if key not in tokens:
                    raise ValueError(f"Missing trajectory token: {key}")
                token = tokens[key]
                if (token["prompt_token_ids_sha256"] != record["prompt_token_ids_sha256"]
                        or token["candidate_token_ids_sha256"] != row[f"{kind}_token_ids_sha256"]
                        or int(token["token_id"]) != token_id
                        or boolean(token["shared_prefix"]) != (index <= common)
                        or int(token["first_divergence_position"]) != common+1
                        or boolean(token["is_eos"]) != (manifest["include_eos"] and index == len(ids))):
                    raise ValueError(f"Trajectory metadata mismatch: {key}")
                lp = finite(token["token_logprob"], "token logprob")
                if lp > 1e-6:
                    raise ValueError("Log probability cannot be positive")
                logprobs.append(lp)
                close(math.fsum(logprobs), float(token["cumulative_logprob"]), "cumulative logprob")
            row[f"{kind}_score"] = finite(row[f"{kind}_score"], "score")
            close(math.fsum(logprobs), row[f"{kind}_score"], "trajectory sum")
            close(math.fsum(logprobs)/len(ids), float(row[f"{kind}_mean_logprob"]), "mean logprob")
            traces[kind] = logprobs
        row["full_margin"] = finite(row["full_margin"], "full margin")
        close(row["lure_score"]-row["correct_score"], row["full_margin"], "full margin")
        has_branch = common < min(map(len, sequences.values()))
        if has_branch:
            row["first_divergence_margin"] = finite(row["first_divergence_margin"], "branch margin")
            close(traces["lure"][common]-traces["correct"][common],
                  row["first_divergence_margin"], "branch margin")
            for kind in sequences:
                if int(row[f"{kind}_branch_token_id"]) != sequences[kind][common]:
                    raise ValueError("Branch token ID differs from trajectory")
        elif row["first_divergence_margin"]:
            raise ValueError("Strict-prefix candidate has no two-token branch without EOS")
    if set(tokens) != expected_token_keys:
        raise ValueError("Unexpected extra trajectory tokens")
    for (cid, coefficient), row in scores.items():
        close(row["full_margin"]-scores[(cid, 0.0)]["full_margin"],
              float(row["full_margin_minus_baseline"]), "baseline full-margin delta")
        close(row["full_margin"]-scores[(cid, 7.0)]["full_margin"],
              float(row["full_margin_minus_plus7"]), "+7 full-margin delta")
    return data, scores, manifest


def branch_inputs(data, paths):
    result = {}
    for row in data["existing_margins"]:
        copied = dict(row)
        for key in controls.previous.MARGINS:
            copied[key] = finite(copied[key], key)
        copied["plus8_minus_plus7"] = copied["plus8_margin"]-copied["plus7_margin"]
        result[row["case_id"]] = copied
    expected = {case["case_id"]: case for case in data["cases"]}
    for path in paths:
        for cid, row in controls.read_margins(path).items():
            if cid not in expected:
                continue  # Older matched CSV also includes ten Rate cases.
            if cid in result:
                for key in (*controls.previous.MARGINS, "branch_position_1based",
                            "correct_branch_token_id", "lure_branch_token_id"):
                    if key in result[cid] or key in row:
                        close(float(result[cid][key]), float(row[key]), f"repeated branch margin {cid}/{key}")
            else:
                result[cid] = row
    for cid, row in result.items():
        if row["family"] != "crt_growth" or any(
                row[key].strip() != expected[cid][key] for key in ("correct_answer", "lure_answer")):
            raise ValueError(f"Existing branch answer/family mismatch: {cid}")
    return result


def describe(rows, metrics):
    result = {}
    for metric in metrics:
        values = [row[metric] for row in rows]
        result[metric] = dict(
            n=len(values), mean=statistics.mean(values), median=statistics.median(values),
            minimum=min(values), q25=controls.quantile(values, .25),
            q75=controls.quantile(values, .75), maximum=max(values), values=values,
        ) if values else None
    return result


def analyze(run_dir, inventory_path, branch_paths=(), bootstrap=2000, seed=0, *, input_protocol=None):
    data, scores, manifest = read_diagnostic(run_dir, inventory_path, input_protocol=input_protocol)
    protocol = manifest["input_protocol"]
    branches = branch_inputs(data, branch_paths)
    case_rows = {}
    for case in data["cases"]:
        cid = case["case_id"]
        row = dict(case_id=cid, input_protocol=protocol, eos_policy=manifest["eos_policy"], include_eos=manifest["include_eos"])
        for coefficient in COEFFICIENTS:
            score = scores[(cid, coefficient)]
            suffix = str(int(coefficient))
            for key in ("correct_score", "lure_score", "correct_mean_logprob", "lure_mean_logprob",
                        "correct_token_length", "lure_token_length", "full_margin", "first_divergence_margin"):
                row[f"{key}_{suffix}"] = score[key]
        row.update(
            full_margin_delta_7_minus_0=row["full_margin_7"]-row["full_margin_0"],
            full_margin_delta_8_minus_7=row["full_margin_8"]-row["full_margin_7"],
            existing_branch_available=cid in branches,
        )
        signs = [controls.previous.sign(row[f"full_margin_{c}"]) for c in (7, 8)]
        row["full_margin_sign_transition"] = "->".join(signs)
        row["full_margin_strict_sign_flip"] = set(signs) == {"negative", "positive"}
        row["full_margin_delta_direction"] = controls.previous.direction(row["full_margin_delta_8_minus_7"])
        row["existing_branch_input_protocol"] = DIAGNOSTIC if cid in branches else None
        row["existing_branch_protocol_compatible"] = (protocol == DIAGNOSTIC) if cid in branches else None
        row["existing_branch_comparison_status"] = (
            "missing" if cid not in branches else
            "cross_protocol_not_comparable" if protocol != DIAGNOSTIC else "tokenization_mismatch")
        row["existing_branch_tokenization_compatible"] = None
        for key in ("baseline_margin", "plus7_margin", "plus8_margin", "plus8_minus_plus7"):
            row[f"existing_branch_{key}"] = branches[cid][key] if cid in branches else None
        row["existing_vs_trajectory_branch_max_abs_error"] = None
        row["existing_vs_trajectory_branch_close_at_1e_4"] = None
        if cid in branches and protocol == DIAGNOSTIC:
            source = branches[cid]
            reference = scores[(cid, 0.0)]
            compatible = bool(reference["first_divergence_margin"] != "" and all(
                int(source[old]) == int(reference[new])
                for old, new in (
                    ("branch_position_1based", "first_divergence_position"),
                    ("correct_branch_token_id", "correct_branch_token_id"),
                    ("lure_branch_token_id", "lure_branch_token_id"))))
            row["existing_branch_tokenization_compatible"] = compatible
            if compatible:
                row["existing_branch_comparison_status"] = "compatible_tokens_prompt_provenance_unverified"
                error = max(abs(source[key]-row[f"first_divergence_margin_{c}"])
                            for c, key in ((0, "baseline_margin"), (7, "plus7_margin"), (8, "plus8_margin")))
                row["existing_vs_trajectory_branch_max_abs_error"] = error
                row["existing_vs_trajectory_branch_close_at_1e_4"] = error <= 1e-4
        case_rows[cid] = row
    rows = []
    for slot in data["slots"]:
        rows.append(dict(
            **case_rows[slot["case_id"]], seed=slot["seed"], plus7_label=slot["plus7_label"],
            plus8_label=slot["plus8_label"], group=f"{slot['plus7_label']}_to_{slot['plus8_label']}"))
    metrics = ("full_margin_0", "full_margin_7", "full_margin_8",
               "full_margin_delta_7_minus_0", "full_margin_delta_8_minus_7")
    groups = {}
    for group in controls.GROUPS:
        selected = [r for r in rows if r["group"] == group]
        unique = list({r["case_id"]: r for r in selected}.values())
        groups[group] = dict(
            evaluation_slots=len(selected), unique_cases=len(unique),
            unique_case_statistics=describe(unique, metrics),
            slot_weighted_descriptive_statistics=describe(selected, metrics),
            unique_case_delta_directions=dict(controls.Counter(r["full_margin_delta_direction"] for r in unique)),
            slot_delta_directions=dict(controls.Counter(r["full_margin_delta_direction"] for r in selected)),
            existing_branch_unique_cases=sum(r["existing_branch_available"] for r in unique))
    bootstrap_rows = [dict(r, plus8_minus_plus7=r["full_margin_delta_8_minus_7"]) for r in rows]
    intervals = controls.cluster_bootstrap(bootstrap_rows, bootstrap, seed)
    intervals["estimand"] = "Unique-case mean FULL margin +8-minus-+7; paired group mean differences"
    prompts = manifest.get("prompts", [])
    audits = [audit for p in prompts for audit in p["boundary_audits"].values()]
    summary = dict(
        input_protocol=protocol, input_protocol_spec=manifest["input_protocol_spec"],
        eos_policy=manifest["eos_policy"],
        model_settings=manifest["model_settings"], sae_settings=manifest["sae_settings"],
        cross_protocol_results_are_repeated_measurements=False,
        first_divergence_reference_protocol=DIAGNOSTIC,
        first_divergence_reference_comparable=(protocol == DIAGNOSTIC),
        inventory_sha256=controls.sha(inventory_path), diagnostic_dir=str(run_dir.resolve()),
        manifest_sha256=controls.sha(run_dir / "manifest.json"),
        branch_inputs=[dict(path=str(p.resolve()), sha256=controls.sha(p)) for p in branch_paths],
        evaluation_slots=len(rows), unique_cases=len(case_rows), include_eos=manifest["include_eos"],
        eos_token_id=manifest["eos_token_id"], groups=groups,
        by_seed={str(s): dict(
            evaluation_slots=len([r for r in rows if r["seed"] == s]),
            group_counts=dict(controls.Counter(r["group"] for r in rows if r["seed"] == s)),
            slot_descriptive_statistics=describe([r for r in rows if r["seed"] == s], metrics),
            by_group={g: describe([r for r in rows if r["seed"] == s and r["group"] == g], metrics)
                      for g in controls.GROUPS})
            for s in sorted({r["seed"] for r in rows})},
        existing_branch_unique_cases=len(branches),
        missing_existing_branch_case_ids=sorted(set(case_rows)-branches.keys()),
        branch_comparison=list(case_rows.values()),
        boundary_audit=dict(
            candidate_audits=len(audits),
            joint_encoding_mismatches=sum(not a["joint_matches_explicit_sequence"] for a in audits),
            prompt_roundtrip_failures=sum(not a["prompt_roundtrip_matches"] for a in audits)),
        bootstrap=intervals,
        limitations=manifest["limitations"] + [
            "Case margins are reused across seeds; slot summaries are descriptive, not independent replicates.",
            "Different input_protocol values are distinct experimental conditions, never repeated measurements.",
            "Legacy diagnostic_compatible branch margins are reference-only for behavior_compatible runs; no numeric equivalence test is performed.",
            "Case overlap across groups is retained by the shared case bootstrap; intervals are exploratory.",
            "Previous branch CSVs may cover only six cases; missing branch data are not imputed.",
            "Branch token metadata must match before numerical comparison; CSVs alone cannot prove prompt equivalence.",
            "Length effects and missing EOS can change full-margin signs without a corresponding greedy label change.",
        ])
    return rows, summary


def run(run_dir, inventory_path, output_dir, branch_paths=(), bootstrap=2000, seed=0, *, input_protocol=None):
    csv_path, json_path = output_dir / f"{STEM}_behavior_join.csv", output_dir / f"{STEM}_summary.json"
    for path in (csv_path, json_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    rows, summary = analyze(run_dir, inventory_path, branch_paths, bootstrap, seed, input_protocol=input_protocol)
    controls.write_outputs(csv_path, json_path, rows, summary, list(rows[0]))
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Downloaded artifacts/likelihood directory")
    parser.add_argument("--input-protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--inventory", type=Path, default=ROOT / "experiments/inputs/fresh_v2_growth_controls_f30908.json")
    parser.add_argument("--branch-margin", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True, help="A new analysis directory")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        paths = run(args.run_dir, args.inventory, args.output_dir, args.branch_margin, args.bootstrap, args.seed, input_protocol=args.input_protocol)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
