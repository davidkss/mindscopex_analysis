"""CPU cached-tokenizer audit. No model/SAE loading, network or generation."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.revision_pinning import inspect_tokenizer_cache, validate_tokenizer_inputs
from experiments.trace_fresh_v2_first_divergence import digest, write_new
from experiments.diagnose_rate027_baseline import candidate_layout, allowed_at
from mindscopex_analysis.full_answer_protocols import token_ids_sha256

AUDIT = ROOT/'research/audits/2026-10-30-feature-specificity-cross-case'


def compare_inputs(tokenizer, evidence, prepared_cases):
    """Actual token calls plus existing production-trie compatibility checks; no generate."""
    comparison = validate_tokenizer_inputs(tokenizer, evidence)
    records=[]
    eos=tokenizer.eos_token_id
    eos_ids=list(eos) if isinstance(eos,(list,tuple)) else [eos]
    for cid,expected in evidence.items():
        prompt=prepared_cases[cid].prompt
        default=list(tokenizer(prompt)['input_ids'])
        explicit_true=list(tokenizer(prompt,add_special_tokens=True)['input_ids'])
        no_special=list(tokenizer(prompt,add_special_tokens=False)['input_ids'])
        texts=expected['candidate_texts']
        ids={label:list(tokenizer.encode(text,add_special_tokens=False)) for label,text in texts.items()}
        layout=candidate_layout(list(ids.values()))
        expected_layout=candidate_layout(list(expected['candidate_token_ids'].values()))
        decoded={label:tokenizer.decode(tokens,skip_special_tokens=False) for label,tokens in ids.items()}
        terminal={label:allowed_at(tokens,list(ids.values()),eos_ids) for label,tokens in ids.items()}
        no_early_eos=all(not set(eos_ids).intersection(allowed_at(tokens[:i],list(ids.values()),eos_ids))
                         for tokens in ids.values() for i in range(len(tokens)))
        checks=dict(prepared_prompt_matches=prompt==expected['prompt'],prompt_token_ids_equal=default==expected['prompt_token_ids'],
                    prompt_sha_equal=token_ids_sha256(default)==token_ids_sha256(expected['prompt_token_ids']),
                    default_equals_explicit_add_special_tokens_true=default==explicit_true,
                    candidate_ids_equal=ids==expected['candidate_token_ids'],
                    candidate_texts_match_dataset=all(text==getattr(prepared_cases[cid],label+'_answer') for label,text in texts.items()),
                    leading_space_preserved=all(text.startswith(' ') and decoded[label]==text for label,text in texts.items()),
                    common_prefix_and_divergence_equal=layout==expected_layout and not layout['divergence_involves_candidate_end'],
                    eos_id_matches_historical_reference=eos_ids==[248044],
                    eos_not_in_candidates=all(not set(eos_ids).intersection(tokens) for tokens in ids.values()),
                    candidate_end_allows_only_eos=all(tokens==eos_ids for tokens in terminal.values()),
                    incomplete_candidates_do_not_allow_eos=no_early_eos)
        records.append(dict(case_id=cid,status='verified' if all(checks.values()) else 'mismatch',checks=checks,
                            prepared_prompt=prompt,reference_prepared_prompt=expected['prompt'],
                            prompt_token_ids=default,reference_prompt_token_ids=expected['prompt_token_ids'],
                            prompt_token_ids_sha256=token_ids_sha256(default),reference_prompt_token_ids_sha256=token_ids_sha256(expected['prompt_token_ids']),
                            prompt_without_special_tokens_ids=no_special,prompt_default_add_special_tokens=True,
                            candidate_add_special_tokens=False,candidate_texts=texts,candidate_token_ids=ids,
                            reference_candidate_token_ids=expected['candidate_token_ids'],decoded_candidates=decoded,
                            common_prefix_and_divergence=layout,candidate_end_allowed_ids=terminal))
    return dict(status='verified' if comparison['status']=='verified' and len(records)==3
                and all(r['status']=='verified' for r in records) else 'mismatch',cases=records,
                eos_token_ids=eos_ids,pad_token_id=tokenizer.pad_token_id,
                completion_scope='EOS metadata and production trie rules checked; actual generation termination was not run')


def run(snapshot):
    from mindscopex_analysis.lure_datasets import load_lure_dataset
    from mindscopex_analysis.prompts import instruct_lure_case
    manifest=json.loads((ROOT/'experiments/inputs/feature_specificity_cross_case.json').read_text())
    saved=json.loads((AUDIT/'inputs_verified.json').read_text())
    revision=json.loads((AUDIT/'revision_evidence.json').read_text())
    snapshot=Path(snapshot).resolve()
    if snapshot.name!=manifest['revisions']['tokenizer']:
        raise ValueError('Snapshot directory must identify the pinned tokenizer SHA')
    cache=inspect_tokenizer_cache(snapshot,revision['queries']['tokenizer']['files'])
    if not cache['all_tokenizer_files_match']:
        raise ValueError('Cached tokenizer file metadata mismatch; no loading/download fallback')
    cases={c.case_id:instruct_lure_case(c) for c in load_lure_dataset('crt_fresh_v2')}
    expected=saved['case_inputs']
    for cid,item in expected.items():
        item['candidate_texts']={label:getattr(cases[cid],label+'_answer') for label in ('correct','lure')}
    # Source files below are only read. The saved expected IDs are never changed.
    for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):
        os.environ[key]='1'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TOKENIZERS_PARALLELISM']='false'
    attempts=[]
    def no_network(*args,**kwargs):
        attempts.append('socket connection attempted')
        raise RuntimeError('Network forbidden during tokenizer runtime audit')
    with patch.object(socket.socket,'connect',side_effect=no_network),patch.object(socket.socket,'connect_ex',side_effect=no_network):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False)
        actual=compare_inputs(tokenizer,expected,cases)
    absent=[name for name in ('config.json','preprocessor_config.json','video_preprocessor_config.json') if not (snapshot/name).is_file()]
    production=dict(status='unverified',loader_branch='AutoConfig qwen3_5 -> AutoProcessor (source inspected, not executed)',
                    missing_local_files=absent,
                    reason='Production Processor was not loaded; missing config files are not downloaded',
                    cached_tokenizer_is_not_production_processor_proof=True,
                    tensor_rendering='return_tensors=pt unverified; Torch not installed')
    versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    return dict(schema=1,created_at_utc=datetime.now(timezone.utc).isoformat(),status=actual['status'],
                tokenizer_revision=manifest['revisions']['tokenizer'],model_repo_id=manifest['model_id'],snapshot=str(snapshot),
                environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=versions,
                                 torch_installed=importlib.util.find_spec('torch') is not None),
                tokenizer=dict(class_name=type(tokenizer).__name__,module=type(tokenizer).__module__,is_fast=tokenizer.is_fast,
                               local_files_only=True,trust_remote_code=False),cache_evidence=cache,
                input_source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (
                    AUDIT/'inputs_verified.json',AUDIT/'revision_evidence.json',ROOT/'src/mindscopex_analysis/data/crt_fresh_v2.json',
                    ROOT/'src/mindscopex_analysis/prompts.py',ROOT/'src/mindscopex_analysis/models.py',ROOT/'src/mindscopex_analysis/research.py',Path(__file__))},
                cached_tokenizer_runtime=actual,production_processor=production,
                validation_state=dict(cached_tokenizer_runtime_verified=actual['status'],production_processor_runtime_verified='unverified',
                                      runtime_tokenizer_validation='unverified',runtime_model_loading_validation='unverified',
                                      runtime_sae_loading_validation='unverified',execution_ready=False),
                network_connection_attempts=attempts,model_generate_calls=0,model_sae_weights_loaded=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('Refusing existing tokenizer runtime output')
    result=run(args.snapshot)
    write_new(args.output,result)  # Mismatch evidence is retained before nonzero exit.
    if result['status']!='verified':raise SystemExit('Cached tokenizer runtime validation failed')
    print(json.dumps(dict(status=result['status'],validation_state=result['validation_state'],tokenizer=result['tokenizer'])))


if __name__=='__main__':main()
