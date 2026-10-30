"""Opt-in Fresh v2/30908 capture using the unchanged production generate helper.

--plan reads existing CSVs only. --run-model is a separate, explicit GPU action.
No model libraries are imported by --plan. Output is exclusive-create JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.diagnose_rate027_baseline import RecordingModel, allowed_at, candidate_layout
from scripts.analyze_first_divergence_behavior import finite, preference, rows

CASE_IDS = tuple(f"crt_fresh_v2_growth_{i:03d}" for i in (7, 8, 9, 12, 13, 19, 26))
COEFFICIENTS = (0, 7, 8)
INPUTS = {
    "behavior": ROOT / "results/analysis/protocol_b_behavioral_join_slots.csv",
    "scores": ROOT / "results/analysis/protocol_b_20261029/full_answer_scores.csv",
    "audit": ROOT / "research/audits/2026-10-29-first-divergence/slots.csv",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(records, key):
    result = {}
    for row in records:
        k = key(row)
        if k in result:
            raise ValueError(f"Duplicate input key: {k}")
        result[k] = row
    return result


def load_plan(paths=None):
    """Validate all three read-only inputs, then select seven unique cases."""
    paths = INPUTS if paths is None else paths
    behavior = rows(paths["behavior"], {"case_id", "seed", "coefficient", "actual",
                                         "baseline", "full_margin"})
    scores = rows(paths["scores"], {"case_id", "coefficient", "input_protocol", "include_eos",
                                   "first_divergence_margin", "full_margin", "correct_token_ids",
                                   "lure_token_ids", "prompt_token_ids_sha256", "correct_answer",
                                   "lure_answer"})
    audit = rows(paths["audit"], {"case_id", "seed", "coefficient", "actual",
                                "first_branch_margin", "full_margin", "first_branch_preference"})
    score_key = lambda r: (r["case_id"], finite(r["coefficient"]))
    slot_key = lambda r: (*score_key(r), r["seed"])
    ss, bb, aa = unique(scores, score_key), unique(behavior, slot_key), unique(audit, slot_key)
    if len(ss) != 99 or len(bb) != 128 or bb.keys() != aa.keys():
        raise ValueError("Expected 99 scores and matching 128 behavior/audit slots")
    cases = {c for c, _ in ss}
    if len(cases) != 33 or set(ss) != {(c, a) for c in cases for a in COEFFICIENTS}:
        raise ValueError("Expected 33 cases x coefficients 0/7/8")
    for score in scores:
        if score["input_protocol"] != "behavior_compatible" or score["include_eos"] != "False":
            raise ValueError("Protocol/EOS mismatch")
        finite(score["first_divergence_margin"])
        finite(score["full_margin"])
    targets, counts = [], Counter()
    for key, b in bb.items():
        cid, coefficient, seed = key
        if coefficient not in (7, 8) or seed not in {"1", "2", "3", "4", "5"}:
            raise ValueError(f"Invalid slot {key}")
        if (cid, coefficient) not in ss:
            raise ValueError(f"Missing score for slot {key}")
        a, s = aa[key], ss[cid, coefficient]
        if b["actual"] not in ("correct", "lure") or b["baseline"] not in ("correct", "lure"):
            raise ValueError("Invalid behavior label")
        branch = finite(s["first_divergence_margin"])
        if (a["actual"] != b["actual"] or a["first_branch_preference"] != preference(branch)
                or finite(a["first_branch_margin"]) != branch
                or abs(finite(a["full_margin"]) - finite(s["full_margin"])) > 1e-5
                or abs(finite(b["full_margin"]) - finite(s["full_margin"])) > 1e-5):
            raise ValueError(f"CSV join mismatch: {key}")
        kind = "tie" if branch == 0 else "mismatch" if preference(branch) != b["actual"] else None
        if kind:
            counts[f"{int(coefficient)}_{kind}"] += 1
            targets.append(dict(case_id=cid, seed=int(seed), coefficient=int(coefficient), kind=kind))
    if dict(counts) != {"7_tie": 11, "8_tie": 6, "8_mismatch": 2}:
        raise ValueError(f"Audit target counts changed: {counts}")
    if {r["case_id"] for r in targets} != set(CASE_IDS):
        raise ValueError("Audit target cases changed")
    for cid in cases:
        b7 = {seed: b for (c, a, seed), b in bb.items() if c == cid and a == 7}
        b8 = {seed: b for (c, a, seed), b in bb.items() if c == cid and a == 8}
        if b7.keys() != b8.keys() or any(b7[k]["baseline"] != b8[k]["baseline"] for k in b7):
            raise ValueError("Seed identities/baseline differ between coefficients")
    measurements, comparisons = [], []
    for cid in CASE_IDS:
        for coefficient in COEFFICIENTS:
            measurement_id = f"{cid}:c{coefficient}"
            s = ss[cid, coefficient]
            measurements.append(dict(measurement_id=measurement_id, case_id=cid,
                                     coefficient=coefficient, score_reference=s))
            for (c, a, seed), b in sorted(bb.items()):
                if c != cid or a != (7 if coefficient == 0 else coefficient):
                    continue
                comparisons.append(dict(
                    measurement_id=measurement_id, case_id=cid, seed=int(seed),
                    coefficient=coefficient, existing_behavioral_label=(
                        b["baseline"] if coefficient == 0 else b["actual"]),
                    teacher_forced_first_divergence_margin=finite(s["first_divergence_margin"]),
                    full_answer_margin=finite(s["full_margin"]),
                    audit_reference=None if coefficient == 0 else aa[c, a, seed],
                ))
    return dict(schema=2, status="planned", model_id="Qwen/Qwen3.5-2B-Base",
                layer=5, feature_id=30908, token_position="all", case_ids=list(CASE_IDS),
                coefficients=list(COEFFICIENTS), target_slots=targets, target_counts=dict(counts),
                n_unique_measurements=len(measurements), n_slot_comparisons=len(comparisons),
                n_unique_generation_conditions=len(measurements),
                n_historical_slot_references=len(comparisons), n_exception_slots=len(targets),
                seed_references_are_independent_measurements=False,
                measurement_unit="case_id x coefficient; split seeds are read-only slot references",
                source_files={k: dict(path=str(p), sha256=digest(p)) for k, p in paths.items()},
                measurements=measurements, slot_comparisons=comparisons)


def filter_plan(plan, *, case_id=None, coefficient=None, check_equivalence=False):
    """Select existing conditions and their slot references, preserving the full audit."""
    if case_id is not None and case_id not in CASE_IDS:
        raise ValueError(f"Unknown case ID: {case_id!r}; choose one of {CASE_IDS}")
    if coefficient is not None and (type(coefficient) is not int or coefficient not in COEFFICIENTS):
        raise ValueError("Coefficient must be one of 0, 7, 8")
    measurements = [m for m in plan["measurements"]
                    if (case_id is None or m["case_id"] == case_id)
                    and (coefficient is None or m["coefficient"] == coefficient)]
    if not measurements:
        raise ValueError("No measurements match the requested condition")
    selected_ids = {m["measurement_id"] for m in measurements}
    comparisons = [r for r in plan["slot_comparisons"] if r["measurement_id"] in selected_ids]
    selected_keys = {(m["case_id"], m["coefficient"]) for m in measurements}
    return dict(plan,
                filters=dict(case_id=case_id, coefficient=coefficient),
                case_ids=[cid for cid in plan["case_ids"] if any(c == cid for c, _ in selected_keys)],
                coefficients=[a for a in plan["coefficients"] if any(c == a for _, c in selected_keys)],
                measurements=measurements, slot_comparisons=comparisons,
                n_unique_measurements=len(measurements), n_unique_generation_conditions=len(measurements),
                n_slot_comparisons=len(comparisons), n_historical_slot_references=len(comparisons),
                n_selected_slot_comparisons=len(comparisons),
                n_selected_exception_slots=sum((r["case_id"], r["coefficient"]) in selected_keys
                                               for r in plan["target_slots"]),
                audit_scope="target_slots, target_counts and n_exception_slots retain the full audit",
                equivalence_check_enabled=check_equivalence,
                expected_generate_calls=len(measurements) * (2 if check_equivalence else 1))


def scalar(value):
    value = float(value)
    # Preserve masked -inf explicitly, without nonstandard JSON NaN/Infinity.
    return value if math.isfinite(value) else str(value)


def inspect_capture(wrapper, candidate_ids, eos_ids):
    """Align only actual returned sequence steps. Never synthesize a branch."""
    output = wrapper.output
    if output is None or getattr(output, "logits", None) is None:
        raise RuntimeError("Raw logits unavailable; processed scores are not a substitute")
    prompt = wrapper.kwargs["input_ids"][0].tolist()
    sequence = output.sequences[0].tolist()
    if sequence[:len(prompt)] != prompt:
        raise RuntimeError("Returned prompt differs from generation input")
    generated = sequence[len(prompt):]
    if (len(output.logits) != len(generated) or getattr(output, "scores", None) is None
            or len(output.scores) != len(generated)):
        raise RuntimeError("Raw logits/scores cannot be aligned to generated steps")
    layout = candidate_layout(candidate_ids)
    k = layout["first_divergence_position_zero_based"]
    eos_steps = [i for i, token in enumerate(generated) if token in eos_ids]
    steps = []
    for i, selected in enumerate(generated):
        raw, processed = output.logits[i], output.scores[i]
        if raw is None:
            raise RuntimeError(f"Raw logits missing at step {i}")
        if (processed is None or raw.ndim != 2 or raw.shape[0] != 1
                or processed.shape != raw.shape):
            raise RuntimeError("Expected one sequence with matching raw/processed vocabulary shapes")
        prefix = generated[:i]
        calls = [r["allowed_token_ids"] for r in wrapper.allowed_calls
                 if r["batch_id"] == 0 and r["sequence"] == prompt + prefix]
        try:
            expected = allowed_at(prefix, candidate_ids, eos_ids)
        except ValueError:
            expected = None
        actual_allowed = calls[0] if calls else []
        valid_constraint = bool(calls) and expected is not None and all(x == expected for x in calls)
        ids = sorted(set(actual_allowed + [selected]))
        steps.append(dict(
            step_zero_based=i, prefix_token_ids=prefix, allowed_token_ids=actual_allowed,
            constraint_verified=valid_constraint, selected_token_id=selected,
            tokens=[dict(token_id=t, raw_logit=scalar(raw[0, t].item()),
                         processed_score=scalar(processed[0, t].item())) for t in ids],
        ))
    reason = None
    if eos_steps and eos_steps[0] < k:
        reason = "eos_before_divergence"
    elif len(generated) <= k:
        reason = "divergence_step_not_reached"
    elif generated[:k] != layout["common_prefix_token_ids"]:
        reason = "common_prefix_mismatch"
    elif layout["divergence_involves_candidate_end"]:
        reason = "prefix_collision_eos_vs_continuation_no_scalar_margin"
    elif not steps[k]["constraint_verified"]:
        reason = "allowed_tokens_not_verified"
    elif not all(ids[k] in steps[k]["allowed_token_ids"] for ids in candidate_ids):
        reason = "branch_tokens_not_both_allowed"
    elif steps[k]["selected_token_id"] not in steps[k]["allowed_token_ids"]:
        reason = "selected_token_not_allowed"
    raw_margin = processed_margin = None
    if reason is None:
        correct, lure = (ids[k] for ids in candidate_ids)
        values = [output.logits[k][0, t].item() for t in (correct, lure)]
        if not all(math.isfinite(v) for v in values):
            reason = "nonfinite_branch_raw_logits"
        else:
            raw_margin = values[1] - values[0]
            p = [output.scores[k][0, t].item() for t in (correct, lure)]
            processed_margin = p[1] - p[0] if all(math.isfinite(v) for v in p) else None
    # Match the helper's token acceptance rule: remove trailing EOS only.
    body = list(generated)
    while body and body[-1] in eos_ids:
        body.pop()
    label = next((name for name, ids in zip(("correct", "lure"), candidate_ids)
                  if ids == body), None)
    eos_terminated = bool(eos_steps) and all(t in eos_ids for t in generated[eos_steps[0]:])
    valid_path = bool(steps) and all(step["constraint_verified"] and
                                      step["selected_token_id"] in step["allowed_token_ids"]
                                      for step in steps)
    behavioral_valid = label is not None and eos_terminated and valid_path
    return dict(prompt_token_ids=prompt, candidate_token_ids=dict(zip(("correct", "lure"), candidate_ids)),
                common_prefix_length=k, candidate_layout=layout, generated_token_ids=generated,
                eos_token_ids=eos_ids, eos_steps=eos_steps,
                termination="eos" if eos_steps else "no_eos_observed",
                eos_observed=bool(eos_steps), eos_terminated=eos_terminated,
                complete_candidate=label is not None, candidate_label=label,
                helper_accepts_tokens=label is not None,
                behavioral_response_valid=behavioral_valid,
                completion_status=("complete_with_eos" if behavioral_valid else
                                   "candidate_complete_without_eos" if label and not eos_steps else
                                   "invalid_or_incomplete"),
                new_generation_label=label if behavioral_valid else None,
                observed=reason is None, unobserved_reason=reason, actual_raw_margin=raw_margin,
                actual_raw_preference=None if raw_margin is None else preference(raw_margin),
                actual_processed_score_margin=processed_margin, steps=steps,
                score_storage="raw and processed values for actual allowed and selected IDs at every step")


@contextmanager
def intervention(block, sae, coefficient):
    """Same production hook factory; coefficient zero is a no-hook baseline."""
    if coefficient == 0:
        yield
        return
    from mindscopex_analysis.qwen_scope import make_feature_steering_hook
    handle = block.register_forward_hook(make_feature_steering_hook(
        sae, [30908], coefficient=coefficient, token_position="all"))
    try:
        yield
    finally:
        handle.remove()


def expected_incomplete_error(exc, wrapper, case, eos_ids, helper):
    """Recognize only the production helper's final post-generate rejection.

    A message alone is insufficient: model/kernel errors must propagate even
    when their wording resembles a completion error.
    """
    if wrapper.output is None or getattr(wrapper.output, "logits", None) is None:
        return False
    tb = exc.__traceback__
    while tb and tb.tb_next:
        tb = tb.tb_next
    if tb is None or tb.tb_frame.f_code is not getattr(helper, "__code__", None):
        return False
    n = int(wrapper.kwargs["input_ids"].shape[-1])
    tokens = wrapper.output.sequences[0, n:].tolist()
    while tokens and tokens[-1] in eos_ids:
        tokens.pop()
    return str(exc) == (
        f"Constrained generation did not finish a valid answer for case {case.case_id!r}: "
        f"{tokens!r}"
    )


class SequenceRecordingModel:
    """Passive return observer: delegate generate kwargs and return value unchanged.

    No logits/scores/return flags or prefix callback are added on this path.
    """
    def __init__(self, model):
        self.model = model
        self.sequence = None
        self.prompt_length = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def generate(self, **kwargs):
        output = self.model.generate(**kwargs)
        self.prompt_length = int(kwargs["input_ids"].shape[-1])
        self.sequence = output[0, self.prompt_length:].tolist()
        return output


def compare_uninstrumented(model, tokenizer, case, block, sae, coefficient, capture):
    """One additional production call at the same condition; never a new decode loop."""
    from mindscopex_analysis.research import _binary_choice_completion
    observer = SequenceRecordingModel(model)
    with intervention(block, sae, coefficient):
        answer = _binary_choice_completion(observer, tokenizer, case, max_new_tokens=16)
    ids_equal = observer.sequence == capture["generated_token_ids"]
    answers_equal = answer == capture["answer"]
    return dict(status="compared", order="instrumented_then_uninstrumented",
                uninstrumented_generated_token_ids=observer.sequence,
                uninstrumented_answer=answer, generated_token_ids_equal=ids_equal,
                answers_equal=answers_equal, equivalent=ids_equal and answers_equal,
                note="Passive observer changes no generation arguments; equality is per observed run.")


def capture_case(model, tokenizer, case, block, sae, coefficient):
    from mindscopex_analysis.research import _binary_choice_completion
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    candidates = [list(text_tokenizer.encode(text, add_special_tokens=False))
                  for text in (case.correct_answer, case.lure_answer)]
    eos = text_tokenizer.eos_token_id
    eos = list(eos) if isinstance(eos, (tuple, list)) else [eos]
    if not eos or any(type(t) is not int for t in eos):
        raise ValueError("Tokenizer requires EOS IDs")
    wrapper = RecordingModel(model)
    answer, error = None, None
    with intervention(block, sae, coefficient):
        try:
            answer = _binary_choice_completion(wrapper, tokenizer, case, max_new_tokens=16)
        except RuntimeError as exc:
            if not expected_incomplete_error(exc, wrapper, case, eos, _binary_choice_completion):
                raise
            # Only the final helper rejection of a returned incomplete sequence is expected.
            error = str(exc)
    record = inspect_capture(wrapper, candidates, eos)
    record.update(prompt=case.prompt, candidate_texts=dict(correct=case.correct_answer,
                                                         lure=case.lure_answer),
                  answer=answer, completion_error=error,
                  helper_returned_response=answer is not None)
    if error is not None:
        record.update(behavioral_response_valid=False, new_generation_label=None)
    return record


def run_model(plan, *, check_equivalence=False):
    """The sole model-loading entry point, reachable only with --run-model."""
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    import transformers
    from experiments.diagnose_rate027_baseline import environment
    from mindscopex_analysis.lure_datasets import load_lure_dataset
    from mindscopex_analysis.models import (
        default_sae_device, dtype_from_name, get_qwen35_analysis_profile,
        load_qwen_text_generation_model, recommended_dtype_name,
    )
    from mindscopex_analysis.prompts import instruct_lure_case
    from mindscopex_analysis.qwen_scope import load_qwen_scope_sae
    from mindscopex_analysis.research import find_decoder_block
    from mindscopex_analysis.full_answer_protocols import token_ids_sha256

    plan["equivalence_check_enabled"] = check_equivalence
    plan["expected_generate_calls"] = len(plan["measurements"]) * (2 if check_equivalence else 1)
    plan["started_at"] = datetime.now(UTC).isoformat()
    if not torch.cuda.is_available():
        raise RuntimeError("This opt-in diagnostic requires GPU; CPU tests use mocks")
    profile = get_qwen35_analysis_profile("2b")
    dtype = recommended_dtype_name()
    model, processor = load_qwen_text_generation_model(
        profile.analysis_model_id, device_map="auto", dtype=dtype)
    tokenizer = getattr(processor, "tokenizer", processor)
    config = model.generation_config
    if any(getattr(config, k, 1) not in (None, 1) for k in ("num_beams", "num_return_sequences")):
        raise ValueError("Single-sequence greedy capture required; generation config is not changed")
    sae_device = default_sae_device()
    sae = load_qwen_scope_sae(profile.sae_repo_id, 5, device=sae_device, dtype=dtype_from_name(dtype))
    block = find_decoder_block(model, 5, block_path_template=profile.block_path_template)
    cases = {c.case_id: instruct_lure_case(c) for c in load_lure_dataset("crt_fresh_v2")
             if c.case_id in CASE_IDS and c.family == "crt_growth"}
    # Verify all conditions against saved B tokens before the first model.generate call.
    for item in plan["measurements"]:
        case, score = cases[item["case_id"]], item["score_reference"]
        prompt = tokenizer(case.prompt, return_tensors="pt")["input_ids"][0].tolist()
        if token_ids_sha256(prompt) != score["prompt_token_ids_sha256"]:
            raise ValueError("Prompt IDs differ from saved Protocol B")
        for label in ("correct", "lure"):
            text = getattr(case, f"{label}_answer")
            if text != score[f"{label}_answer"] or list(tokenizer.encode(
                    text, add_special_tokens=False)) != json.loads(score[f"{label}_token_ids"]):
                raise ValueError("Candidate text/IDs differ from saved Protocol B")
    plan["environment"] = environment(model, processor, tokenizer, torch, transformers, dtype)
    plan["sae"] = dict(repo_id=profile.sae_repo_id, layer=5, feature_id=30908,
                        device=sae_device, dtype=str(sae.W_dec.dtype), resolved_revision=None)
    plan["source_hashes"] = {str(p.relative_to(ROOT)): digest(p) for p in (
        Path(__file__), ROOT / "experiments/diagnose_rate027_baseline.py",
        ROOT / "src/mindscopex_analysis/research.py", ROOT / "src/mindscopex_analysis/qwen_scope.py")}
    captured = {}
    for item in plan["measurements"]:
        record = capture_case(model, processor, cases[item["case_id"]], block, sae, item["coefficient"])
        score = item["score_reference"]
        if (token_ids_sha256(record["prompt_token_ids"]) != score["prompt_token_ids_sha256"]
                or any(record["candidate_token_ids"][label] != json.loads(score[f"{label}_token_ids"])
                       for label in ("correct", "lure"))):
            raise ValueError("Actual capture inputs differ from saved Protocol B")
        record["equivalence_check"] = (
            compare_uninstrumented(model, processor, cases[item["case_id"]], block, sae,
                                   item["coefficient"], record)
            if check_equivalence else dict(status="not_run"))
        item["capture"] = record
        captured[item["measurement_id"]] = record
    for slot in plan["slot_comparisons"]:
        record = captured[slot["measurement_id"]]
        slot.update(behavioral_response_valid=record["behavioral_response_valid"],
                    eos_observed=record["eos_observed"], complete_candidate=record["complete_candidate"],
                    new_generation_label=record["new_generation_label"],
                    observed=record["observed"], unobserved_reason=record["unobserved_reason"],
                    actual_raw_margin=record["actual_raw_margin"],
                    actual_processed_score_margin=record["actual_processed_score_margin"],
                    behavioral_label_agreement=(record["new_generation_label"] == slot["existing_behavioral_label"]
                                                if record["new_generation_label"] is not None else None))
    plan.update(status="captured", finished_at=datetime.now(UTC).isoformat(), limitations=[
        "One generate per unique case/coefficient; seed slots are references, not independent measurements.",
        "Historical +8 raw response files and exact model/SAE revisions remain unverified.",
        "Only allowed/selected token logits are serialized; full-vocabulary tensors are transient.",
        "Capture equivalence is CPU-mock tested; actual GPU equivalence is not established by mocks.",
        "Observation of logit/selection association does not establish feature semantics or reasoning mechanism.",
    ])
    return plan


def write_new(path, payload):
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="Read CSVs only; no ML imports")
    mode.add_argument("--run-model", action="store_true", help="Explicit model download/loading/GPU opt-in")
    parser.add_argument("--check-equivalence", action="store_true",
                        help="With future GPU execution: add one unchanged production call per condition")
    parser.add_argument("--case-id", choices=CASE_IDS, help="Select one of the seven prepared Growth cases")
    parser.add_argument("--coefficient", type=int, choices=COEFFICIENTS,
                        help="Select coefficient 0, 7 or 8; default includes all three")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing existing output: {args.output}")
    if args.output.resolve().is_relative_to(ROOT / "results/runs"):
        raise ValueError("Use a separate new diagnostics path, outside historical results/runs")
    plan = filter_plan(load_plan(), case_id=args.case_id, coefficient=args.coefficient,
                       check_equivalence=args.check_equivalence)
    print(json.dumps({k: plan[k] for k in ("target_counts", "target_slots", "coefficients",
                                          "n_unique_measurements", "n_slot_comparisons",
                                          "n_exception_slots", "n_selected_slot_comparisons",
                                          "n_selected_exception_slots", "expected_generate_calls")}, indent=2))
    write_new(args.output, run_model(plan, check_equivalence=args.check_equivalence) if args.run_model else plan)


if __name__ == "__main__":
    main()
