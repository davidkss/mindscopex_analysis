"""Matched-input first-divergence logit diagnostic for Fresh v2 CRT.

GPU required. Uses the same chat-template input preparation and constrained-answer
candidate tokenization as generation.generate_constrained_choice_response.
This is a conditional *first branch* logit measurement, not whole-answer score.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from mindscopex_analysis.generation import _prepare_inputs, _input_device
from mindscopex_analysis.lure_datasets import load_lure_dataset
from mindscopex_analysis.models import get_qwen35_analysis_profile, load_qwen_text_generation_model
from mindscopex_analysis.prompts import instruct_lure_case
from mindscopex_analysis.qwen_scope import load_qwen_scope_sae, make_feature_steering_hook
from mindscopex_analysis.research import find_decoder_block

RATE_IDS = (
    "crt_fresh_v2_rate_002", "crt_fresh_v2_rate_013",
    "crt_fresh_v2_rate_016", "crt_fresh_v2_rate_020",
    "crt_fresh_v2_rate_022", "crt_fresh_v2_rate_023",
    "crt_fresh_v2_rate_025", "crt_fresh_v2_rate_028",
    "crt_fresh_v2_rate_030", "crt_fresh_v2_rate_036",
)
# Unique case IDs of the 14 matched +7 correct -> +8 lure evaluation slots.
GROWTH_IDS = tuple(f"crt_fresh_v2_growth_{i:03d}" for i in (7, 8, 9, 14, 19, 26))
COEFFICIENTS = (0.0, 7.0, 8.0)


def first_branch(tokenizer, correct, lure):
    a = tokenizer.encode(correct.strip(), add_special_tokens=False)
    b = tokenizer.encode(lure.strip(), add_special_tokens=False)
    common = 0
    while common < min(len(a), len(b)) and a[common] == b[common]:
        common += 1
    if common == min(len(a), len(b)):
        raise ValueError("Answer prefix relation needs EOS-aware handling")
    return a[:common], a[common], b[common], common


def margin(model, input_ids, correct_id, lure_id):
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    return float((logits[lure_id] - logits[correct_id]).item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/diagnostics/fresh_v2_f30908_matched_branch_margin.csv")
    p.add_argument("--layer", type=int, default=5)
    p.add_argument("--feature", type=int, default=30908)
    p.add_argument("--only", choices=("all", "rate", "growth"), default="all")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (Colab A100 recommended).")
    profile = get_qwen35_analysis_profile("2b")
    print("Model:", profile.analysis_model_id, flush=True)
    model, processor = load_qwen_text_generation_model(profile.analysis_model_id, device_map="auto", dtype="auto")
    tokenizer = getattr(processor, "tokenizer", processor)
    sae = load_qwen_scope_sae(profile.sae_repo_id, args.layer, device="cpu")
    block = find_decoder_block(model, args.layer, block_path_template=profile.block_path_template)
    cases = {case.case_id: case for case in load_lure_dataset("crt_fresh_v2")}
    selected = (RATE_IDS if args.only != "growth" else ()) + (GROWTH_IDS if args.only != "rate" else ())
    if any(cid not in cases for cid in selected):
        raise ValueError("Missing selected Fresh v2 case IDs")
    rows = []
    for cid in selected:
        # Behavioral config sets [data].instruction=true; no separate system prompt.
        case = instruct_lure_case(cases[cid])
        prefix, correct_id, lure_id, common = first_branch(tokenizer, case.correct_answer, case.lure_answer)
        prepared = _prepare_inputs(tokenizer, case.prompt, system_prompt="", enable_thinking=False, use_chat_template=True)
        prompt_ids = prepared["input_ids"].to(_input_device(model))
        context = torch.cat((prompt_ids, torch.tensor([prefix], device=prompt_ids.device, dtype=prompt_ids.dtype)), dim=1)
        measured = {}
        for coefficient in COEFFICIENTS:
            if coefficient == 0:
                measured[coefficient] = margin(model, context, correct_id, lure_id)
            else:
                hook = make_feature_steering_hook(sae, [args.feature], coefficient=coefficient, token_position="all")
                handle = block.register_forward_hook(hook)
                try:
                    measured[coefficient] = margin(model, context, correct_id, lure_id)
                finally:
                    handle.remove()
        row = {
            "case_id": cid, "family": case.family,
            "correct_answer": case.correct_answer, "lure_answer": case.lure_answer,
            "branch_position_1based": common + 1,
            "correct_branch_token_id": correct_id, "lure_branch_token_id": lure_id,
            "baseline_margin": measured[0.0], "plus7_margin": measured[7.0], "plus8_margin": measured[8.0],
            "plus7_minus_baseline": measured[7.0] - measured[0.0],
            "plus8_minus_baseline": measured[8.0] - measured[0.0],
            "plus8_minus_plus7": measured[8.0] - measured[7.0],
        }
        rows.append(row)
        print(f"{cid} baseline={measured[0.0]:+.4f} +7={measured[7.0]:+.4f} +8={measured[8.0]:+.4f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} unique cases to {output}")
    print("First-divergence conditional logits only; not full answer likelihood.")
    print("Compare direction with existing behavioral data; do not infer complete decoding equivalence.")


if __name__ == "__main__":
    main()
