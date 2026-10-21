"""MindScopeX Fresh CRT rate: first-divergence logit diagnostic.

Run in the project's Colab GPU environment after installing project dependencies.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from mindscopex_analysis.models import get_qwen35_analysis_profile, load_qwen_text_generation_model
from mindscopex_analysis.qwen_scope import load_qwen_scope_sae, make_feature_steering_hook
from mindscopex_analysis.research import find_decoder_block

CASE_IDS = (
    'crt_fresh_v2_rate_002', 'crt_fresh_v2_rate_013',
    'crt_fresh_v2_rate_016', 'crt_fresh_v2_rate_020',
    'crt_fresh_v2_rate_022', 'crt_fresh_v2_rate_023',
    'crt_fresh_v2_rate_025', 'crt_fresh_v2_rate_028',
    'crt_fresh_v2_rate_030', 'crt_fresh_v2_rate_036',
)


def branch_ids(tokenizer, correct, lure):
    a = tokenizer.encode(correct, add_special_tokens=False)
    b = tokenizer.encode(lure, add_special_tokens=False)
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    if i == min(len(a), len(b)):
        raise ValueError('One candidate is a token prefix of the other; EOS branch needs separate handling')
    return a[:i], a[i], b[i], i


def logit_margin(model, tokenizer, prompt, common_prefix, correct_id, lure_id):
    prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids'][0].tolist()
    ids = torch.tensor([prompt_ids + common_prefix], dtype=torch.long, device=next(model.parameters()).device)
    with torch.inference_mode():
        output = model(input_ids=ids, use_cache=False)
        logits = output.logits[0, -1].float()
        return (logits[lure_id] - logits[correct_id]).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='src/mindscopex_analysis/data/crt_fresh_v2.json')
    parser.add_argument('--output', default='results/diagnostics/fresh_rate_branch_margin.csv')
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--feature', type=int, default=30908)
    parser.add_argument('--coefficient', type=float, default=7.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU runtime required. Select Colab A100 and run there.')

    profile = get_qwen35_analysis_profile('2b')
    print('Loading model:', profile.analysis_model_id, flush=True)
    model, processor = load_qwen_text_generation_model(profile.analysis_model_id, device_map='auto', dtype='auto')
    tokenizer = getattr(processor, 'tokenizer', processor)
    print('Loading SAE:', profile.sae_repo_id, 'layer', args.layer, flush=True)
    sae = load_qwen_scope_sae(profile.sae_repo_id, args.layer, device='cpu')
    block = find_decoder_block(model, args.layer, block_path_template=profile.block_path_template)
    cases = json.loads(Path(args.data).read_text(encoding='utf-8'))['cases']
    by_id = {case['case_id']: case for case in cases}
    missing = set(CASE_IDS) - set(by_id)
    if missing:
        raise ValueError(f'Missing cases: {sorted(missing)}')
    rows = []
    for case_id in CASE_IDS:
        case = by_id[case_id]
        prefix, correct_id, lure_id, branch_index = branch_ids(tokenizer, case['correct_answer'], case['lure_answer'])
        baseline = logit_margin(model, tokenizer, case['prompt'], prefix, correct_id, lure_id)
        hook = make_feature_steering_hook(sae, [args.feature], coefficient=args.coefficient, token_position='all')
        handle = block.register_forward_hook(hook)
        try:
            intervention = logit_margin(model, tokenizer, case['prompt'], prefix, correct_id, lure_id)
        finally:
            handle.remove()
        row = {
            'case_id': case_id,
            'correct_answer': case['correct_answer'],
            'lure_answer': case['lure_answer'],
            'branch_position_1based': branch_index + 1,
            'correct_branch_token_id': correct_id,
            'lure_branch_token_id': lure_id,
            'baseline_margin_lure_minus_correct': baseline,
            'intervention_margin_lure_minus_correct': intervention,
            'margin_delta_intervention_minus_baseline': intervention - baseline,
        }
        rows.append(row)
        print(f"{case_id}: branch={branch_index + 1}, baseline={baseline:+.4f}, intervention={intervention:+.4f}, delta={intervention-baseline:+.4f}", flush=True)

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', dest)
    print('NOTE: This measures the first divergence conditional on shared prefix, not full-answer log probability.')


if __name__ == '__main__':
    main()
