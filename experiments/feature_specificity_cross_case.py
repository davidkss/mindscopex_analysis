"""Versioned opt-in cross-case planning and validation; no ML imports at plan time."""
from __future__ import annotations
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path:
    sys.path.insert(0,str(ROOT/'src'))
from experiments.trace_fresh_v2_first_divergence import INPUTS, digest, load_plan, rows
from mindscopex_analysis.full_answer_protocols import token_ids_sha256

MANIFEST = ROOT/'experiments/inputs/feature_specificity_cross_case.json'
PROTOCOL = ROOT/'results/analysis/protocol_b_20261029/manifest.json'
CASE_IDS = ['crt_fresh_v2_growth_019','crt_fresh_v2_growth_004','crt_fresh_v2_growth_016']
CONDITIONS = [(None,0),(30908,7),(30908,8),(30908,8.5),(12207,7),(12207,8),(12207,8.5)]


def require_ready(plan):
    from experiments.revision_pinning import revision_blockers
    blockers = revision_blockers(plan['study_manifest'])
    if blockers:
        raise ValueError('Cross-case execution blocked: revisions; ' + '; '.join(blockers))


def build_plan(block_id):
    from mindscopex_analysis.lure_datasets import load_lure_dataset
    from mindscopex_analysis.prompts import instruct_lure_case
    manifest=json.loads(MANIFEST.read_text())
    if (manifest['schema'] != 1 or manifest['result_schema'] != 3
            or manifest['case_ids'] != CASE_IDS or manifest['blocks'] != ['block-1','block-2']
            or manifest['model_id'] != 'Qwen/Qwen3.5-2B-Base'
            or manifest['sae_repo_id'] != 'Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50'
            or manifest['layer'] != 5 or manifest['token_position'] != 'all'
            or [(r['feature_id'],r['coefficient']) for r in manifest['conditions']] != CONDITIONS
            or len({r['condition_id'] for r in manifest['conditions']}) != 7
            or any(type(r['coefficient']) not in (int,float) or (r['feature_id'] is not None and type(r['feature_id']) is not int) for r in manifest['conditions'])
            or manifest['additional_control_features'] != []
            or manifest['check_equivalence'] is not True or manifest['activation_diagnostics'] is not True
            or manifest['expected_observations'] != 42 or manifest['expected_generate_calls'] != 84):
        raise ValueError('Study manifest outside preregistered CPU scope')
    from experiments.revision_pinning import validate_revision_ids
    validate_revision_ids(manifest['revisions'])
    from experiments.revision_pinning import validate_revision_evidence
    evidence_path = ROOT / 'research/audits/2026-10-30-feature-specificity-cross-case/revision_evidence.json'
    if manifest.get('revision_evidence_path') != str(evidence_path.relative_to(ROOT)):
        raise ValueError('Cross-case revision evidence path differs')
    validate_revision_evidence(manifest, json.loads(evidence_path.read_text()))
    if block_id not in manifest['blocks']:
        raise ValueError('Unknown cross-case block')
    base=load_plan()  # Reuse full legacy input/audit consistency checks.
    scores=rows(INPUTS['scores'],{'case_id','coefficient'})
    behavior=rows(INPUTS['behavior'],{'case_id','baseline'})
    protocol=json.loads(PROTOCOL.read_text())
    prompts={p['case_id']:p for p in protocol['prompts']}
    cases={c.case_id:instruct_lure_case(c) for c in load_lure_dataset('crt_fresh_v2')}
    measurements=[];inputs={}
    for cid in CASE_IDS:
        case=cases[cid];p=prompts[cid]
        score=next(r for r in scores if r['case_id']==cid and float(r['coefficient'])==0)
        if case.family!='crt_growth' or hashlib.sha256(case.prompt.encode()).hexdigest()!=p['prompt_text_sha256']:
            raise ValueError('Dataset/raw prompt mismatch')
        if token_ids_sha256(p['prompt_token_ids'])!=p['prompt_token_ids_sha256'] or p['prompt_token_ids_sha256']!=score['prompt_token_ids_sha256']:
            raise ValueError('Prompt token SHA mismatch')
        ids=p['candidate_token_ids'];eos=248044
        for label in ('correct','lure'):
            if (getattr(case,label+'_answer') != score[label+'_answer'] or p['candidate_texts'][label]!=score[label+'_answer']
                    or ids[label]!=json.loads(score[label+'_token_ids'])
                    or token_ids_sha256(ids[label])!=score[label+'_token_ids_sha256']):
                raise ValueError('Candidate text/ID/SHA mismatch')
        a,b=ids['correct'],ids['lure'];k=0
        while k<min(len(a),len(b)) and a[k]==b[k]: k+=1
        if k==min(len(a),len(b)) or any(eos in seq or not seq for seq in (a,b)):
            raise ValueError('Candidate end/prefix collision incompatible with token divergence')
        refs=[r for r in behavior if r['case_id']==cid]
        if not refs or any(r['baseline'] not in ('correct','lure') for r in refs):
            raise ValueError('Missing historical baseline evidence')
        inputs[cid]=dict(prompt=case.prompt,prompt_token_ids=p['prompt_token_ids'],
                         prompt_text_sha256=p['prompt_text_sha256'],candidate_token_ids=ids,
                         common_prefix_length=k,branch_tokens=dict(correct=a[k],lure=b[k]),
                         compatibility='saved protocol compatible; runtime pinned tokenizer verification pending',
                         historical_baseline_slots=refs)
        for condition in manifest['conditions']:
            feature,coef=condition['feature_id'],condition['coefficient']
            reference=next((r for r in scores if r['case_id']==cid and float(r['coefficient'])==coef),None) if feature in (None,30908) else None
            measurements.append(dict(case_id=cid,**condition,measurement_id=f"{block_id}:{cid}:{condition['condition_id']}",
                                     input_reference=score,score_reference=reference,
                                     teacher_forced_first_margin=float(reference['first_divergence_margin']) if reference else None,
                                     full_answer_margin=float(reference['full_margin']) if reference else None))
    plan=dict(base,schema=3,cross_case=True,study_id=manifest['study_id'],study_manifest=manifest,
              study_manifest_sha256=digest(MANIFEST),block_id=block_id,run_id=f'cross_case_specificity_{block_id}_2b',
              case_ids=CASE_IDS,feature_id=None,feature_ids=[30908,12207],coefficients=[0,7,8,8.5],
              measurements=measurements,slot_comparisons=[],n_unique_measurements=21,
              n_unique_generation_conditions=21,n_slot_comparisons=0,n_historical_slot_references=0,
              n_selected_slot_comparisons=0,n_selected_exception_slots=0,
              expected_generate_calls=42,equivalence_check_enabled=True,case_inputs=inputs,
              activation_diagnostics=True,execution_ready=False,
              audit_scope='historical 2026-10-29 audit metadata; not study selection/observation counts',
              measurement_unit='case x condition x block; equivalence pair and historical seeds are not repeats')
    plan['source_files']['protocol_manifest']=dict(path=str(PROTOCOL),sha256=digest(PROTOCOL))
    plan['source_files']['revision_evidence'] = dict(path=str(evidence_path), sha256=digest(evidence_path))
    plan['revision_validation'] = manifest.get('revision_validation', {})
    try:require_ready(plan)
    except ValueError as exc:plan['execution_blockers']=[str(exc)]
    else:plan.update(execution_ready=True,execution_blockers=[])
    return plan


def validate_result(output,plan):
    from experiments.jobs.first_divergence_trace import validate_equivalence
    from experiments.growth019_minimal_pilot import validate_capture
    validate_equivalence(output,plan)
    for key in ('schema','study_id','block_id','run_id','study_manifest_sha256','model_id','layer','token_position','n_unique_measurements','n_slot_comparisons','expected_generate_calls'):
        if output.get(key)!=plan[key]:raise ValueError(f'Cross-case result scope differs: {key}')
    if output.get('status')!='captured' or output.get('sae',{}).get('repo_id')!=plan['study_manifest']['sae_repo_id']:
        raise ValueError('Cross-case captured result/SAE missing')
    if output.get('revision_requests')!=plan['study_manifest']['revisions']:
        raise ValueError('Revision request metadata missing/different')
    for actual,expected in zip(output['measurements'],plan['measurements'],strict=True):
        if any(actual.get(k)!=expected[k] for k in ('case_id','condition_id','feature_id','coefficient')):
            raise ValueError('Condition differs')
        c=actual['capture'];validate_capture(c,expected)
        if c['prompt']!=plan['case_inputs'][expected['case_id']]['prompt']:
            raise ValueError('Actual raw prompt differs')
        ids=c['candidate_token_ids'];k=c['common_prefix_length'];gen=c['generated_token_ids']
        expected_k=plan['case_inputs'][expected['case_id']]['common_prefix_length']
        if k!=expected_k or gen[:k]!=ids['correct'][:k] or len(c['steps'])!=len(gen):
            raise ValueError('Actual prefix/step alignment mismatch')
        for i,step in enumerate(c['steps']):
            from experiments.diagnose_rate027_baseline import allowed_at
            if (step['prefix_token_ids']!=gen[:i] or step['step_zero_based']!=i
                    or step['selected_token_id']!=gen[i]
                    or step['allowed_token_ids']!=allowed_at(gen[:i],list(ids.values()),c['eos_token_ids'])):
                raise ValueError('Actual trie mismatch')
        t={t['token_id']:t for t in c['steps'][k]['tokens']}
        for field,margin in [('raw_logit','actual_raw_margin'),('processed_score','actual_processed_score_margin')]:
            if t[ids['lure'][k]][field]-t[ids['correct'][k]][field]!=c[margin]:
                raise ValueError('Margin arithmetic mismatch')
        telemetry=c.get('activation_norms',{})
        if expected['feature_id'] is None:
            if telemetry != dict(status='no_hook',records=[]):raise ValueError('Baseline must remain no-hook')
        else:
            records=telemetry.get('records',[])
            if (telemetry.get('status')!='measured_at_hook' or not records
                    or not math.isfinite(telemetry.get('unit_decoder_direction_norm',float('nan')))):
                raise ValueError('Activation diagnostics missing')
            for r in records:
                if r['feature_ids']!=str(expected['feature_id']) or r['coefficient']!=expected['coefficient'] or r['token_position']!='all':
                    raise ValueError('Activation diagnostics condition mismatch')
                for key in ('requested_vector_norm','cast_vector_norm','actual_delta_mean','residual_before_mean','residual_after_mean'):
                    if r.get(key) is None or not math.isfinite(r[key]) or r[key]<0:raise ValueError('Invalid activation norm')
                if any(r.get(k,0) for k in ('nonfinite_residual_count','nonfinite_post_count','nonfinite_delta_count')):
                    raise ValueError('Nonfinite activation diagnostics')
        if c.get('raw_tie') != (c['actual_raw_margin']==0) or c.get('selected_branch_token_id')!=gen[k]:
            raise ValueError('Derived tie/selection mismatch')
    # Derived metrics are checked against this block's shared no-hook reference per case.
    baselines={m['case_id']:m['capture']['actual_raw_margin'] for m in output['measurements'] if m['feature_id'] is None}
    processed_baselines={m['case_id']:m['capture']['actual_processed_score_margin'] for m in output['measurements'] if m['feature_id'] is None}
    for m in output['measurements']:
        if m['capture'].get('processed_margin_minus_baseline')!=m['capture']['actual_processed_score_margin']-processed_baselines[m['case_id']]:
            raise ValueError('Processed baseline delta differs')
        if m['capture'].get('raw_margin_minus_baseline')!=m['capture']['actual_raw_margin']-baselines[m['case_id']]:
            raise ValueError('Baseline delta differs')


def compare_blocks(outputs):
    """CPU-only paired report; no p-values or seed-based pseudo-replication."""
    if set(outputs)!={'block-1','block-2'}:raise ValueError('Need both distinct blocks')
    for block,result in outputs.items():validate_result(result,build_plan(block))
    rows=[]
    for left,right in zip(outputs['block-1']['measurements'],outputs['block-2']['measurements'],strict=True):
        captures=[left['capture'],right['capture']]
        rows.append(dict(case_id=left['case_id'],condition_id=left['condition_id'],
            feature_id=left['feature_id'],coefficient=left['coefficient'],
            raw_margins=[c['actual_raw_margin'] for c in captures],
            processed_margins=[c['actual_processed_score_margin'] for c in captures],
            baseline_raw_deltas=[c['raw_margin_minus_baseline'] for c in captures],
            answers=[c['answer'] for c in captures],labels=[c['new_generation_label'] for c in captures],
            branch_token_ids=[c['selected_branch_token_id'] for c in captures],raw_ties=[c['raw_tie'] for c in captures],
            tokens_equal=captures[0]['generated_token_ids']==captures[1]['generated_token_ids'],
            raw_equal=captures[0]['actual_raw_margin']==captures[1]['actual_raw_margin'],
            processed_equal=captures[0]['actual_processed_score_margin']==captures[1]['actual_processed_score_margin'],
            activation_norms=[c['activation_norms'] for c in captures],
            teacher_forced_first_margin=left['teacher_forced_first_margin']))
    return dict(schema=1,status='compared recorded observations',rows=rows,n_cases=3,n_observations=42,
                limitation='Two blocks per case; no generalization, semantic specificity or independent seed repetitions inferred')
