"""CPU-only pilot planning/validation; execution reuses the existing trace loop."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from mindscopex_analysis.full_answer_protocols import token_ids_sha256

MANIFEST = ROOT / 'experiments/inputs/growth019_minimal_pilot.json'


def build_plan(block_id):
    from experiments.trace_fresh_v2_first_divergence import load_plan, filter_plan, digest
    manifest = json.loads(MANIFEST.read_text())
    if block_id not in manifest['blocks']:
        raise ValueError('Pilot block must be block-1 or block-2')
    base = load_plan()
    anchor = filter_plan(base, case_id=manifest['case_id'], coefficient=0,
                         check_equivalence=True)
    inputs = anchor['measurements'][0]['score_reference']
    measurements = []
    for condition in manifest['conditions']:
        feature, coefficient = condition['feature_id'], condition['coefficient']
        comparable = feature in (None,30908) and coefficient in (0,7,8)
        reference = (filter_plan(base, case_id=manifest['case_id'], coefficient=coefficient)
                     ['measurements'][0]['score_reference'] if comparable else None)
        measurements.append(dict(
            measurement_id=f"{block_id}:{condition['condition_id']}",
            case_id=manifest['case_id'], **condition, input_reference=inputs,
            score_reference=reference,
            teacher_forced_first_margin=float(reference['first_divergence_margin']) if reference else None,
            full_answer_margin=float(reference['full_margin']) if reference else None,
            tf_reference_status=('historical no-hook protocol B' if feature is None else 'historical 30908 protocol B') if reference else 'unavailable for this feature/coefficient'))
    plan = dict(anchor, pilot=True, block_id=block_id, run_id=f'pilot_growth019_{block_id}_2b',
                feature_id=None, feature_ids=[30908,12207], coefficients=[0,7,8,8.5],
                measurements=measurements, slot_comparisons=[], n_unique_measurements=5,
                n_unique_generation_conditions=5, n_slot_comparisons=0, n_historical_slot_references=0,
                n_selected_slot_comparisons=0, n_selected_exception_slots=0, expected_generate_calls=10,
                filters=dict(case_id=manifest['case_id'], coefficient=None),
                pilot_manifest=manifest, pilot_manifest_sha256=digest(MANIFEST),
                observation_unit='one condition/block; equivalence pair is not an independent repeat',
                historical_seed_references=[2,4], seed_references_are_independent_measurements=False)
    # The manifest is a fixed preregistered scope, not a free-form model runner.
    expected=[(None,0),(30908,7),(30908,8),(30908,8.5),(12207,8)]
    actual=[(m['feature_id'],m['coefficient']) for m in measurements]
    if (actual != expected or len({m['measurement_id'] for m in measurements}) != 5
            or manifest['blocks'] != ['block-1','block-2']
            or manifest['model_id'] != 'Qwen/Qwen3.5-2B-Base'
            or manifest['sae_repo_id'] != 'Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50'
            or manifest['case_id'] != 'crt_fresh_v2_growth_019'
            or manifest['layer'] != 5 or manifest['token_position'] != 'all'
            or manifest['positive_margin_expected'] is not False
            or manifest['check_equivalence'] is not True
            or manifest['n_observations'] != 10 or manifest['expected_generate_calls'] != 20
            or any(type(m['coefficient']) not in (int,float) or not math.isfinite(m['coefficient'])
                   or (m['feature_id'] is not None and type(m['feature_id']) is not int) for m in measurements)):
        raise ValueError('Pilot manifest differs from the approved preparation scope')
    return plan


def validate_result(output, plan):
    from experiments.jobs.first_divergence_trace import validate_equivalence
    validate_equivalence(output,plan)
    if (output.get('n_unique_measurements') != 5 or output.get('n_slot_comparisons') != 0
            or output.get('model_id') != plan['model_id'] or output.get('layer') != 5
            or output.get('token_position') != 'all' or output.get('status') != 'captured' or output.get('block_id') != plan['block_id']
            or output.get('run_id') != plan['run_id'] or output.get('expected_generate_calls') != 10
            or output.get('pilot_manifest_sha256') != plan['pilot_manifest_sha256']):
        raise ValueError('Pilot output scope/block mismatch')
    for actual, expected in zip(output['measurements'],plan['measurements'],strict=True):
        if any(actual.get(k) != expected[k] for k in ('case_id','feature_id','coefficient','condition_id')):
            raise ValueError('Pilot condition mismatch')
        capture = actual['capture']
        reference = expected['input_reference']
        if (token_ids_sha256(capture['prompt_token_ids']) != reference['prompt_token_ids_sha256']
                or any(capture['candidate_token_ids'][label] != json.loads(reference[f'{label}_token_ids'])
                       for label in ('correct','lure'))):
            raise ValueError('Pilot prompt/candidate token mismatch')
        if any(not math.isfinite(capture[k]) for k in ('actual_raw_margin','actual_processed_score_margin')):
            raise ValueError('Pilot margin must be finite')
        if (not all(capture.get(k) is True for k in ('observed','eos_observed','eos_terminated',
                                                   'complete_candidate','behavioral_response_valid'))
                or capture['generated_token_ids'][-1] not in capture['eos_token_ids']):
            raise ValueError('Pilot EOS/observation incomplete')
        ids = capture['generated_token_ids'][:-1]
        label = capture['new_generation_label']
        if (ids != capture['candidate_token_ids'].get(label)
                or capture['answer'] != capture['candidate_texts'][label].strip()
                or any(not step['constraint_verified'] or step['selected_token_id'] not in step['allowed_token_ids']
                       for step in capture['steps'])):
            raise ValueError('Pilot candidate/trie mismatch')
        eq = capture['equivalence_check']
        if (eq['uninstrumented_generated_token_ids'] != capture['generated_token_ids']
                or eq['uninstrumented_answer'] != capture['answer']):
            raise ValueError('Pilot equivalence flags disagree with observations')


def compare_blocks(outputs):
    if set(outputs) != {'block-1','block-2'}:
        raise ValueError('Need both distinct pilot blocks')
    for block,output in outputs.items():
        validate_result(output,build_plan(block))
    a,b = outputs['block-1'],outputs['block-2']
    rows=[]
    for left,right in zip(a['measurements'],b['measurements'],strict=True):
        lc,rc=left['capture'],right['capture']
        rows.append(dict(condition_id=left['condition_id'],
            actual_raw_margins=[lc['actual_raw_margin'],rc['actual_raw_margin']],
            actual_processed_margins=[lc['actual_processed_score_margin'],rc['actual_processed_score_margin']],
            raw_equal=lc['actual_raw_margin']==rc['actual_raw_margin'],
            processed_equal=lc['actual_processed_score_margin']==rc['actual_processed_score_margin'],
            answers=[lc['answer'],rc['answer']], answers_equal=lc['answer']==rc['answer'],
            labels=[lc['new_generation_label'],rc['new_generation_label']],
            selected_branch_tokens=[c['steps'][c['common_prefix_length']]['selected_token_id'] for c in (lc,rc)],
            raw_ties=[lc['actual_raw_margin']==0,rc['actual_raw_margin']==0],
            teacher_forced_first_margin=left['teacher_forced_first_margin'],
            tf_reference_status=left['tf_reference_status'], eos_valid=True, equivalence='passed',
            token_ids_equal=lc['generated_token_ids']==rc['generated_token_ids']))
    environment_status=('not recorded' if not a.get('environment') or not b.get('environment')
                        else 'matched recorded values' if a['environment']==b['environment'] else 'different')
    return dict(status='compared', rows=rows, recorded_environment_comparison=environment_status,
                exact_revision_comparison='unverified; matching missing values are not evidence',
                limitation='Two execution blocks measure repeatability, not generalization or statistical significance.')


def main(argv=None):
    import argparse
    from experiments.trace_fresh_v2_first_divergence import write_new
    parser=argparse.ArgumentParser(description='CPU comparison of two completed pilot blocks')
    parser.add_argument('--block-1',type=Path,required=True)
    parser.add_argument('--block-2',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('Refusing existing pilot comparison output')
    result=compare_blocks({block:json.loads(path.read_text()) for block,path in (
        ('block-1',args.block_1),('block-2',args.block_2))})
    write_new(args.output,result)


if __name__=='__main__':
    main()
