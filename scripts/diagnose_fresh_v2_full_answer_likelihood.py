#!/usr/bin/env python3
"""Pinned Growth candidate teacher-forcing diagnostic; no sampling or generation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from contextlib import nullcontext, contextmanager
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch

from mindscopex_analysis.full_answer_protocols import (
    BEHAVIOR, DIAGNOSTIC, PROTOCOLS, candidate_texts, capture_behavior_inputs, protocol_spec,
    require_protocol, token_ids_sha256,
)
from mindscopex_analysis.generation import _input_device, _prepare_inputs
from mindscopex_analysis.growth_control_inventory import CONDITIONS, load_inventory
from mindscopex_analysis.lure_datasets import load_lure_dataset
from mindscopex_analysis.models import get_qwen35_analysis_profile, load_qwen_text_generation_model
from mindscopex_analysis.prompts import instruct_lure_case
from mindscopex_analysis.qwen_scope import load_qwen_scope_sae, make_feature_steering_hook
from mindscopex_analysis.research import find_decoder_block

COEFFICIENTS = (0.0, 7.0, 8.0)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_growth_cases(inventory_path):
    inventory = load_inventory(inventory_path)
    if len(inventory["cases"]) != 33 or len(inventory["slots"]) != 64:
        raise ValueError("Expected the existing 33 Growth cases / 64 slots")
    cases = {case.case_id: case for case in load_lure_dataset("crt_fresh_v2")}
    selected = []
    for definition in inventory["cases"]:
        cid = definition["case_id"]
        if cid not in cases or cases[cid].family != "crt_growth":
            raise ValueError(f"Missing/non-Growth case: {cid}")
        case = cases[cid]
        if any(getattr(case, key).strip() != definition[key] for key in ("correct_answer", "lure_answer")):
            raise ValueError(f"Inventory/dataset answer mismatch: {cid}")
        selected.append(instruct_lure_case(case))
    return inventory, selected


def eos_id(tokenizer, include_eos=False, explicit_id=None):
    available = getattr(tokenizer, "eos_token_id", None)
    ids = list(available) if isinstance(available, (list, tuple)) else (
        [available] if available is not None else [])
    if not include_eos:
        if explicit_id is not None:
            raise ValueError("eos_token_id requires include_eos=true")
        return None
    if explicit_id is not None:
        if type(explicit_id) is not int or explicit_id not in ids:
            raise ValueError("Explicit EOS ID must be a tokenizer EOS token")
        return explicit_id
    if len(ids) != 1:
        raise ValueError("EOS inclusion requires one tokenizer EOS or explicit --eos-token-id")
    return ids[0]


def common_prefix(a, b):
    count = 0
    while count < min(len(a), len(b)) and a[count] == b[count]:
        count += 1
    return count


def boundary_audit(tokenizer, prompt_ids, answer, candidate_ids):
    # Audit the actual rendered/template prompt, retaining special token text.
    rendered = tokenizer.decode(prompt_ids, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
    roundtrip = list(tokenizer.encode(rendered, add_special_tokens=False))
    combined = list(tokenizer.encode(rendered + answer, add_special_tokens=False))
    prefix_preserved = combined[:len(prompt_ids)] == prompt_ids
    return dict(
        prompt_roundtrip_matches=roundtrip == prompt_ids,
        joint_prompt_prefix_preserved=prefix_preserved,
        joint_suffix_matches_candidate=(combined[len(prompt_ids):] == candidate_ids
                                        if prefix_preserved else False),
        joint_matches_explicit_sequence=combined == prompt_ids + candidate_ids,
        joint_token_ids=combined,
        policy="Score prepared prompt IDs + protocol-specific standalone candidate IDs; joint encoding is audit only.",
    )


def prepare_case(tokenizer, case, terminal_id=None, *, input_protocol=DIAGNOSTIC):
    require_protocol(input_protocol)
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if input_protocol == DIAGNOSTIC:
        prepared = _prepare_inputs(tokenizer, case.prompt, system_prompt="",
                                   enable_thinking=False, use_chat_template=True)
    else:
        # Exact input/tokenization calls used by research._binary_choice_completion.
        prepared = dict(tokenizer(case.prompt, return_tensors="pt"))
        if tokenizer.eos_token_id is None:
            raise ValueError("Behavior-compatible reference requires an eos_token_id")
    prompt = prepared["input_ids"][0].tolist()
    if not prompt:
        raise ValueError("Empty prompt token sequence")
    if "attention_mask" in prepared and not bool(prepared["attention_mask"].all()):
        raise ValueError("Single-case diagnostic expects an unpadded prompt")
    candidates, audits = {}, {}
    for kind, answer in candidate_texts(case, input_protocol).items():
        ids = list(tokenizer.encode(answer, add_special_tokens=False))
        if not ids:
            raise ValueError(f"Empty {kind} candidate: {case.case_id}")
        audits[kind] = boundary_audit(tokenizer, prompt, answer, ids)
        if input_protocol == BEHAVIOR:
            joint = tokenizer(case.prompt + answer, return_tensors="pt")["input_ids"][0].tolist()
            audits[kind].update(raw_prompt_joint_token_ids=joint,
                               raw_prompt_joint_matches_explicit=joint == prompt + ids)
        candidates[kind] = ids + ([terminal_id] if terminal_id is not None else [])
    if input_protocol == BEHAVIOR:
        reference_prompt, reference_candidates = capture_behavior_inputs(tokenizer, case)
        answer_ids = {kind: ids[:-1] if terminal_id is not None else ids
                      for kind, ids in candidates.items()}
        if reference_prompt != prompt or reference_candidates != answer_ids:
            raise ValueError("Behavioral reference prompt/candidate token IDs differ")
        for audit in audits.values():
            audit["behavior_reference_token_ids_verified"] = True
            audit["reference_candidate_paths_checked"] = 2
    if candidates["correct"] == candidates["lure"]:
        raise ValueError("Candidate token sequences are identical")
    common = common_prefix(candidates["correct"], candidates["lure"])
    return prompt, candidates, audits, common

def teacher_force(model, tokenizer, prompt_ids, candidate_ids):
    """Align logits at prompt_length-1+i with target candidate_ids[i]."""
    if not prompt_ids or not candidate_ids:
        raise ValueError("Prompt and candidate tokens must be nonempty")
    context = torch.tensor([prompt_ids + candidate_ids[:-1]], dtype=torch.long,
                           device=_input_device(model))
    with torch.inference_mode():
        logits = model(input_ids=context, use_cache=False).logits[0]
        relevant = logits[len(prompt_ids)-1:len(prompt_ids)+len(candidate_ids)-1].float()
        targets = torch.tensor(candidate_ids, device=relevant.device, dtype=torch.long)
        values = torch.log_softmax(relevant, dim=-1).gather(1, targets[:, None])[:, 0]
    logprobs = values.cpu().tolist()
    if len(logprobs) != len(candidate_ids) or not all(math.isfinite(v) for v in logprobs):
        raise ValueError("Missing/nonfinite teacher-forced token log probabilities")
    trajectory = []
    for index, (token, logprob) in enumerate(zip(candidate_ids, logprobs, strict=True), 1):
        trajectory.append(dict(
            token_index=index, token_id=token,
            token_text=tokenizer.decode([token], skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False),
            token_logprob=logprob, cumulative_logprob=math.fsum(logprobs[:index]),
        ))
    return math.fsum(logprobs), trajectory


@contextmanager
def steering(block, sae, coefficient):
    handle = block.register_forward_hook(make_feature_steering_hook(
        sae, [30908], coefficient=coefficient, token_position="all"))
    try:
        yield
    finally:
        handle.remove()


def measure_case(model, tokenizer, block, sae, case, terminal_id=None, *, input_protocol=DIAGNOSTIC):
    prompt, candidates, audits, common = prepare_case(
        tokenizer, case, terminal_id, input_protocol=input_protocol)
    texts = candidate_texts(case, input_protocol)
    prompt_hash = token_ids_sha256(prompt)
    candidate_hashes = {kind: token_ids_sha256(ids) for kind, ids in candidates.items()}
    scores, tokens = [], []
    for coefficient in COEFFICIENTS:
        row = dict(case_id=case.case_id, family=case.family, coefficient=coefficient,
                   input_protocol=input_protocol, prompt_token_ids_sha256=prompt_hash,
                   correct_token_ids_sha256=candidate_hashes["correct"],
                   lure_token_ids_sha256=candidate_hashes["lure"],
                   include_eos=terminal_id is not None, eos_token_id=terminal_id,
                   correct_answer=texts["correct"], lure_answer=texts["lure"],
                   shared_prefix_length=common, first_divergence_position=common+1)
        traces = {}
        with (nullcontext() if coefficient == 0 else steering(block, sae, coefficient)):
            for kind, ids in candidates.items():
                score, trace = teacher_force(model, tokenizer, prompt, ids)
                traces[kind] = trace
                row.update({
                    f"{kind}_score": score, f"{kind}_mean_logprob": score / len(ids),
                    f"{kind}_token_length": len(ids),
                    f"{kind}_answer_token_length": len(ids) - int(terminal_id is not None),
                    f"{kind}_token_ids": json.dumps(ids),
                    f"{kind}_joint_encoding_matches": audits[kind]["joint_matches_explicit_sequence"],
                })
                for item in trace:
                    tokens.append(dict(
                        case_id=case.case_id, coefficient=coefficient, candidate_type=kind,
                        input_protocol=input_protocol, prompt_token_ids_sha256=prompt_hash,
                        candidate_token_ids_sha256=candidate_hashes[kind],
                        **item, shared_prefix=item["token_index"] <= common,
                        first_divergence_position=common+1,
                        is_eos=terminal_id is not None and item["token_index"] == len(ids),
                    ))
        row["full_margin"] = row["lure_score"] - row["correct_score"]
        # Difference of logprobs at a common context equals a logit difference.
        has_branch = common < min(map(len, candidates.values()))
        row["first_divergence_margin"] = (
            traces["lure"][common]["token_logprob"] - traces["correct"][common]["token_logprob"]
            if has_branch else None)
        row["correct_branch_token_id"] = candidates["correct"][common] if has_branch else None
        row["lure_branch_token_id"] = candidates["lure"][common] if has_branch else None
        scores.append(row)
    for row in scores:
        row["full_margin_minus_baseline"] = row["full_margin"] - scores[0]["full_margin"]
        row["full_margin_minus_plus7"] = row["full_margin"] - scores[1]["full_margin"]
    return scores, tokens, dict(
        case_id=case.case_id, input_protocol=input_protocol,
        prompt_token_ids=prompt, prompt_token_ids_sha256=prompt_hash,
        candidate_texts=texts, candidate_token_ids=candidates,
        candidate_token_ids_sha256=candidate_hashes,
        answer_token_ids_sha256={
            kind: token_ids_sha256(ids[:-1] if terminal_id is not None else ids)
            for kind, ids in candidates.items()},
        prompt_text_sha256=hashlib.sha256(case.prompt.encode()).hexdigest(),
        boundary_audits=audits,
    )


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(inventory_path, output_dir, include_eos=False, explicit_eos=None, *, input_protocol):
    require_protocol(input_protocol)
    inventory, cases = select_growth_cases(inventory_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    profile = get_qwen35_analysis_profile("2b")
    manifest = dict(
        schema=2, input_protocol=input_protocol, input_protocol_spec=protocol_spec(input_protocol),
        token_id_hash_format="SHA-256 of UTF-8 JSON integer array, separators=(',', ':')",
        eos_policy="include" if include_eos else "exclude",
        model_settings=dict(profile="2b", model_id=profile.analysis_model_id,
                            dtype="auto", device_map="auto", eval=True, use_cache=False),
        sae_settings=dict(repo_id=profile.sae_repo_id, layer=5, feature_id=30908,
                          coefficients=list(COEFFICIENTS), device="cpu", token_position="all",
                          block_path_template=profile.block_path_template),
        reference_source_sha256={
            "research.py": digest(ROOT / "src/mindscopex_analysis/research.py"),
            "generation.py": digest(ROOT / "src/mindscopex_analysis/generation.py")},
        conditions=CONDITIONS, coefficients=list(COEFFICIENTS),
        inventory_sha256=digest(inventory_path), case_ids=[case.case_id for case in cases],
        include_eos=include_eos, eos_token_id=explicit_eos, status="running",
        started_at=datetime.now(UTC).isoformat(),
        score_definition="Sum of natural-log probabilities of explicit candidate tokens, without length normalization",
        tokenization=protocol_spec(input_protocol),
        cross_protocol_results_are_repeated_measurements=False,
        limitations=[
            "EOS excluded by default: scores are prefix likelihoods, not probabilities of complete terminated answers.",
            "Candidate lengths differ; sums and token means answer different questions.",
            "Shared-prefix probabilities contribute to both sums; paths are scored separately after branching.",
            "Joint text tokenization may differ; audit flags do not change the explicit constrained-candidate token path.",
            "A failed prompt decode/re-encode roundtrip makes the joint-text audit inconclusive.",
            "Teacher forcing follows prescribed candidates; it is not greedy/constrained generation behavior.",
            "Logits are scored with full-vocabulary softmax, without candidate-only renormalization.",
            "Full-sequence no-cache computation may differ numerically from branch-only or cached execution.",
        ])
    path = output_dir / "manifest.json"

    def save():
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; CPU tests use stubs")
        model, processor = load_qwen_text_generation_model(
            profile.analysis_model_id, device_map="auto", dtype="auto")
        model.eval()
        manifest["model_settings"]["resolved_revision"] = getattr(
            getattr(model, "config", None), "_commit_hash", None)
        tokenizer = getattr(processor, "tokenizer", processor)
        terminal = eos_id(tokenizer, include_eos, explicit_eos)
        manifest["eos_token_id"] = terminal
        sae = load_qwen_scope_sae(profile.sae_repo_id, 5, device="cpu")
        block = find_decoder_block(model, 5, block_path_template=profile.block_path_template)
        scores, tokens, prompts = [], [], []
        for case in cases:
            case_scores, case_tokens, audit = measure_case(model, tokenizer, block, sae, case, terminal, input_protocol=input_protocol)
            scores.extend(case_scores)
            tokens.extend(case_tokens)
            prompts.append(audit)
            print(f"{case.case_id}: full margins " +
                  ", ".join(f"{r['coefficient']:+g}={r['full_margin']:+.6f}" for r in case_scores),
                  flush=True)
        write_csv(output_dir / "full_answer_scores.csv", scores)
        write_csv(output_dir / "token_trajectory.csv", tokens)
        (output_dir / "inventory.json").write_bytes(inventory_path.read_bytes())
        manifest.update(status="succeeded", score_rows=len(scores), trajectory_rows=len(tokens),
                        candidate_paths=2*len(scores), prompts=prompts,
                        torch_version=torch.__version__,
                        score_csv_sha256=digest(output_dir / "full_answer_scores.csv"),
                        trajectory_csv_sha256=digest(output_dir / "token_trajectory.csv"))
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        save()
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--include-eos", action="store_true")
    parser.add_argument("--eos-token-id", type=int)
    args = parser.parse_args()
    if args.eos_token_id is not None and not args.include_eos:
        parser.error("--eos-token-id requires --include-eos")
    run(args.inventory, args.output_dir, args.include_eos, args.eos_token_id, input_protocol=args.input_protocol)


if __name__ == "__main__":
    main()
