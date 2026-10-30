#!/usr/bin/env python3
"""Read-only, stdlib CPU audit of the three existing Growth019 A100 runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.jobs.first_divergence_trace import validate_equivalence
from experiments.trace_fresh_v2_first_divergence import INPUTS, digest, write_new
from experiments.diagnose_rate027_baseline import allowed_at
from scripts.analyze_first_divergence_behavior import preference

RUNS = {
    0: ROOT / 'results/runs/20261030-131401_trace_fresh_v2_growth019_c0_2b',
    7: ROOT / 'results/runs/20261030-131523_trace_fresh_v2_growth019_c7_2b',
    8: ROOT / 'results/runs/20261030-104935_trace_fresh_v2_growth019_c8_2b',
}
CASE = 'crt_fresh_v2_growth_019'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def token_hash(ids):
    # Same canonical integer-array format used by the diagnostic.
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def close(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(
        float(a), float(b), abs_tol=1e-6, rel_tol=0)


def read_run(directory, coefficient):
    paths = {k: directory / p for k, p in {
        'result': 'artifacts/actual_first_divergence.json', 'plan': 'artifacts/plan.json',
        'job': 'artifacts/job_manifest.json', 'launcher': 'launcher_manifest.json',
        'log': 'artifacts/diagnostic.log', 'job_log': 'artifacts/job.log',
        'config': 'artifacts/config.toml', 'source': 'source.zip',
    }.items()}
    inputs = {k: dict(path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                     sha256=digest(p)) for k, p in paths.items()}
    docs = {k: json.loads(paths[k].read_text()) for k in ('result', 'plan', 'job', 'launcher')}
    d, plan, job = docs['result'], docs['plan'], docs['job']
    config = tomllib.loads(paths['config'].read_text())
    require(job['status'] == 'succeeded' and job['equivalence_validation'] == 'passed', 'Job not successful')
    require(d['status'] == 'captured' and d['n_unique_measurements'] == 1, 'Result not single captured measurement')
    require(d['filters'] == dict(case_id=CASE, coefficient=coefficient), 'Wrong selected condition')
    require(d['coefficients'] == [coefficient] and d['layer'] == 5 and d['feature_id'] == 30908
            and d['token_position'] == 'all', 'Intervention condition mismatch')
    require(d['expected_generate_calls'] == 2 and d['n_slot_comparisons'] == 2, 'Wrong observation unit')
    validate_equivalence(d, plan)
    require(job['source_files'] == d['source_files'], 'Manifest input hashes differ')
    require(d['model_id'] == d['environment']['model_id'] == 'Qwen/Qwen3.5-2B-Base', 'Model ID mismatch')
    require(d['sae']['repo_id'] == 'Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50'
            and d['sae']['layer'] == 5 and d['sae']['feature_id'] == 30908, 'SAE identity mismatch')
    measurement = d['measurements'][0]
    require(measurement['case_id'] == CASE and measurement['coefficient'] == coefficient, 'Wrong measurement')
    c, score = measurement['capture'], measurement['score_reference']
    prompt_hash = token_hash(c['prompt_token_ids'])
    require(prompt_hash == score['prompt_token_ids_sha256'], 'Prompt ID hash mismatch')
    require(score['input_protocol'] == 'behavior_compatible' and score['include_eos'] == 'False', 'Teacher-forced protocol mismatch')
    for label in ('correct', 'lure'):
        require(c['candidate_token_ids'][label] == json.loads(score[f'{label}_token_ids']), 'Candidate IDs differ')
        require(token_hash(c['candidate_token_ids'][label]) == score[f'{label}_token_ids_sha256'], 'Candidate hash mismatch')
    ids = list(c['candidate_token_ids'].values())
    k = 0
    while k < min(map(len, ids)) and ids[0][k] == ids[1][k]:
        k += 1
    require(k < min(map(len, ids)) and k == c['common_prefix_length'], 'Invalid divergence index')
    generated = c['generated_token_ids']
    require(c['observed'] is True and c['unobserved_reason'] is None and len(generated) > k
            and generated[:k] == ids[0][:k], 'Divergence not observed')
    require(len(c['steps']) == len(generated), 'Generated step count mismatch')
    for i, step in enumerate(c['steps']):
        require(step['step_zero_based'] == i and step['prefix_token_ids'] == generated[:i]
                and step['allowed_token_ids'] == allowed_at(generated[:i], ids, c['eos_token_ids'])
                and step['selected_token_id'] == generated[i], 'Generation step/trie mismatch')
    branch = c['steps'][k]
    require(branch['step_zero_based'] == k and branch['prefix_token_ids'] == ids[0][:k]
            and branch['constraint_verified'] is True
            and {ids[0][k], ids[1][k]} <= set(branch['allowed_token_ids']), 'Invalid branch alignment')
    tokens = {t['token_id']: t for t in branch['tokens']}
    correct, lure = tokens[ids[0][k]], tokens[ids[1][k]]
    raw = lure['raw_logit'] - correct['raw_logit']
    processed = lure['processed_score'] - correct['processed_score']
    require(close(raw, c['actual_raw_margin']) and close(processed, c['actual_processed_score_margin']), 'Stored margin differs from branch scores')
    require(c['actual_raw_preference'] == preference(raw), 'Raw preference differs')
    require(close(float(score['lure_score'])-float(score['correct_score']), score['full_margin']), 'Full score difference mismatch')
    require(branch['selected_token_id'] == generated[k], 'Selected token mismatch')
    require(c['eos_observed'] is True and c['eos_terminated'] is True and c['termination'] == 'eos'
            and c['complete_candidate'] is True and c['behavioral_response_valid'] is True
            and generated[-1] in c['eos_token_ids'], 'Incomplete/EOS-invalid response')
    label = c['candidate_label']
    require(label in ('correct', 'lure') and c['new_generation_label'] == label
            and generated[:-1] == c['candidate_token_ids'][label]
            and c['answer'] == c['candidate_texts'][label].strip(), 'Answer/label/tokens differ')
    eq = c['equivalence_check']
    require(eq['uninstrumented_generated_token_ids'] == generated
            and eq['uninstrumented_answer'] == c['answer'], 'Equivalence flags disagree with actual values')
    slots = d['slot_comparisons']
    require(sorted(s['seed'] for s in slots) == [2, 4], 'Wrong historical slot references')
    for slot in slots:
        require(slot['case_id'] == CASE and slot['coefficient'] == coefficient
                and close(slot['teacher_forced_first_divergence_margin'], score['first_divergence_margin'])
                and close(slot['full_answer_margin'], score['full_margin']), 'Historical slot join differs')
    require(config['diagnostic'] == dict(case_id=CASE, coefficient=coefficient, layer=5,
                                       feature=30908, check_equivalence=True), 'Config differs')
    with zipfile.ZipFile(paths['source']) as archive:
        for name, sha in d['source_hashes'].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == sha, f'Source hash mismatch: {name}')
        for name, path in INPUTS.items():
            require(hashlib.sha256(archive.read(path.relative_to(ROOT).as_posix())).hexdigest()
                    == d['source_files'][name]['sha256'], f'Input archive hash mismatch: {name}')
    config['run'].pop('name')
    config['diagnostic'].pop('coefficient')
    return dict(coefficient=coefficient, answer=c['answer'], label=label, actual_raw_margin=raw,
                actual_raw_preference=preference(raw),
                actual_processed_margin=processed, teacher_forced_margin=float(score['first_divergence_margin']),
                full_margin=float(score['full_margin']), actual_minus_teacher_forced=raw-float(score['first_divergence_margin']),
                prompt_sha256=prompt_hash, candidate_token_ids=c['candidate_token_ids'],
                branch=branch, eos_valid=True, equivalence='passed', inputs=inputs,
                environment=d['environment'], sae=d['sae'], config_without_coefficient=config,
                source_hashes=d['source_hashes'], input_hashes={k:v['sha256'] for k,v in d['source_files'].items()},
                launcher_git_commit=docs['launcher'].get('git_commit'),
                launcher_git_status=docs['launcher'].get('git_status'),
                kernel_warnings=[line for line in paths['log'].read_text().splitlines() if 'falling back' in line])


def field(row, path):
    value = row
    for key in path.split('.'):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def analyze(runs=None):
    rows, errors = [], []
    for coefficient, directory in (RUNS if runs is None else runs).items():
        try:
            rows.append(read_run(Path(directory), coefficient))
        except (OSError, ValueError, KeyError, TypeError, IndexError, zipfile.BadZipFile) as exc:
            errors.append(dict(coefficient=coefficient, error=str(exc)))
    rows.sort(key=lambda r:r['coefficient'])
    require_rows = [r['coefficient'] for r in rows] == [0,7,8]
    if not require_rows:
        errors.append(dict(error='Need validated coefficients 0, 7, 8'))
    paths = ['environment.model_id', 'environment.model_configs.model._commit_hash',
             'environment.model_configs.text._commit_hash', 'environment.tokenizer_name_or_path',
             'environment.tokenizer_revision_metadata.revision', 'environment.tokenizer_revision_metadata._commit_hash',
             'environment.cached_config_snapshot_hint', 'sae.repo_id', 'sae.resolved_revision',
             'sae.dtype', 'sae.device', 'environment.torch', 'environment.transformers', 'environment.packages',
             'environment.requested_dtype', 'environment.model_configs', 'environment.parameter_groups',
             'environment.cuda_runtime', 'environment.cudnn_version', 'environment.gpus',
             'environment.generation_config', 'environment.deterministic_algorithms',
             'environment.environment_variables', 'config_without_coefficient', 'source_hashes', 'input_hashes']
    comparison = []
    for path in paths:
        values = [field(r,path) for r in rows]
        status = ('not recorded' if len(values)!=3 or any(v is None for v in values)
                  else 'matched recorded values' if values[0]==values[1]==values[2] else 'different')
        comparison.append(dict(field=path, status=status, values=values))
    if require_rows:
        for key in ('prompt_sha256','candidate_token_ids'):
            if not rows[0][key] == rows[1][key] == rows[2][key]:
                errors.append(dict(error=f'Cross-condition {key} mismatch'))
        for r in rows:
            r['raw_delta_from_baseline'] = r['actual_raw_margin']-rows[0]['actual_raw_margin']
            r['answer_changed_from_baseline'] = r['answer'] != rows[0]['answer']
    transitions = [dict(from_coefficient=a['coefficient'], to_coefficient=b['coefficient'],
                        raw_margin_delta=b['actual_raw_margin']-a['actual_raw_margin'],
                        from_label=a['label'], to_label=b['label'], answer_changed=a['answer']!=b['answer'])
                   for a,b in zip(rows,rows[1:])] if require_rows else []
    return dict(status='validated' if not errors else 'failed', rows=rows, validation_errors=errors,
                analysis_script_sha256=digest(Path(__file__)),
                adjacent_transitions=transitions,
                environment_comparison=comparison,
                recorded_environment_dicts_equal=(require_rows and rows[0]['environment']==rows[1]['environment']==rows[2]['environment']),
                limitations=['Model/tokenizer/SAE revisions are not recorded; matching nulls are not identity proof.',
                             'Cache snapshot path is a hint, not proof of loaded model/tokenizer revision.',
                             'Three single-case conditions; seed 2/4 are historical references, not new repeats.',
                             'Equivalence compares instrumented/uninstrumented paths within each coefficient.',
                             '+8 zero margin is a recorded tie, not Lure logit superiority.',
                             'Teacher forcing and actual generation are separate computation paths.'])


def report(summary):
    lines = ['# Growth019 A100 0/+7/+8 Artifact comparison', '', f"Status: {summary['status']}", '',
             'All margins are Lure minus Correct. TF = teacher-forced; actual = observed generation.', '',
             '| Coefficient | Actual raw | Actual processed | TF first | Full | Actual−TF | Raw Δ vs 0 | Answer | Label | EOS / Equivalence |',
             '|---:|---:|---:|---:|---:|---:|---:|---|---|---|']
    for r in summary['rows']:
        values=[r.get(k, "not available") for k in ('coefficient','actual_raw_margin','actual_processed_margin','teacher_forced_margin',
                              'full_margin','actual_minus_teacher_forced','raw_delta_from_baseline','answer','label')]
        lines.append('| '+' | '.join(f'{v:.9f}' if isinstance(v,float) else str(v) for v in values)+' | passed / passed |')
    lines += ['', '## Adjacent condition transitions', '',
              json.dumps(summary['adjacent_transitions'], ensure_ascii=False), '']
    lines += ['', '## Environment (full values and input SHA-256 in summary.json)', '', '| Field | Comparison |', '|---|---|']
    lines += [f"| {c['field']} | {c['status']} |" for c in summary['environment_comparison']]
    lines += ['', '## Validation failures', '', json.dumps(summary['validation_errors'],ensure_ascii=False), '', '## Limits', '']
    lines += ['- '+s for s in summary['limitations']]
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args=parser.parse_args(argv)
    # Require a fresh directory, including on a failed validation; never touch original runs.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary=analyze()
    write_new(args.output_dir/'summary.json',summary)
    text=report(summary)
    with (args.output_dir/'report.md').open('x',encoding='utf-8') as stream:
        stream.write(text)
    print(text)
    if summary['validation_errors']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
