"""Opt-in staged pinned runtime validation. CPU preparation never opens a GPU session."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments import feature_specificity_cross_case as study
from experiments.trace_fresh_v2_first_divergence import digest,write_new

MANIFEST=ROOT/'experiments/inputs/growth019_runtime_smoke.json'
STAGES=('files','processor','model','sae','hook')


def build_plan(through='generate'):
    manifest=json.loads(MANIFEST.read_text())
    if (through not in ('validate','generate') or manifest['run_id']!='growth019_pinned_runtime_smoke_2b'
            or [r['condition_id'] for r in manifest['conditions']]!=['baseline','f30908-c8p5','f12207-c8'] or manifest['schema']!=1 or manifest['result_schema']!=4
            or manifest['case_id']!='crt_fresh_v2_growth_019' or manifest['layer']!=5 or manifest['token_position']!='all'
            or manifest['execution_ready'] is not False or type(manifest['allocation_authorized']) is not bool
            or [(m['feature_id'],m['coefficient']) for m in manifest['conditions']]!=[(None,0),(30908,8.5),(12207,8)]
            or manifest['expected_observations']!=3 or manifest['expected_generate_calls']!=6
            or manifest['extra_model_forwards']!=0 or manifest['synthetic_hook_module_forwards']!=2
            or manifest['automatic_generation_retries']!=0
            or manifest['revision_source']!='experiments/inputs/feature_specificity_cross_case.json'):
        raise ValueError('Runtime smoke scope differs from the approved preparation')
    plan=copy.deepcopy(study.build_plan('block-1'))
    measurements=[]
    for condition in manifest['conditions']:
        row=next(m for m in plan['measurements'] if m['case_id']==manifest['case_id'] and m['condition_id']==condition['condition_id'])
        row['measurement_id']=manifest['run_id']+':'+condition['condition_id'];measurements.append(row)
    plan.update(schema=4,runtime_smoke=True,smoke_manifest=manifest,smoke_manifest_sha256=digest(MANIFEST),
                run_id=manifest['run_id'],block_id='runtime-smoke',through=through,case_ids=[manifest['case_id']],
                measurements=measurements,n_unique_measurements=3,n_unique_generation_conditions=3,
                expected_generate_calls=6 if through=='generate' else 0,full_smoke_generate_calls=6,
                extra_model_forwards=0,synthetic_hook_module_forwards=2,automatic_generation_retries=0,
                execution_ready=False,allocation_authorized=manifest['allocation_authorized'],
                execution_blockers=['Separate GPU allocation approval required; fresh runtime stages must pass before generation'])
    plan['source_files']['smoke_manifest']=dict(path=str(MANIFEST),sha256=digest(MANIFEST))
    return plan


def require_allocation_authorized(plan):
    if plan['smoke_manifest']['allocation_authorized'] is not True:
        raise ValueError('Runtime smoke allocation is not authorized; preparation/dry-run only')


def verify_file(path,metadata):
    """Stream large payloads; API Git blob IDs for LFS pointers are not payload hashes."""
    path=Path(path);size=path.stat().st_size
    if size!=metadata['size']:raise ValueError(f'File size mismatch: {metadata["rfilename"]}')
    lfs=metadata.get('lfs')
    hasher=hashlib.sha256() if lfs else hashlib.sha1(b'blob '+str(size).encode()+b'\0')
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):hasher.update(chunk)
    expected=lfs['sha256'] if lfs else metadata['blobId']
    if hasher.hexdigest()!=expected:raise ValueError(f'File integrity mismatch: {metadata["rfilename"]}')
    return dict(path=str(path),filename=metadata['rfilename'],bytes=size,
                hash_kind='lfs_payload_sha256' if lfs else 'git_blob_sha1',hash=expected)


def validate_summary(summary,plan):
    if (summary.get('status')!='validated' or summary.get('smoke_manifest_sha256')!=plan['smoke_manifest_sha256']
            or summary.get('revisions')!=plan['study_manifest']['revisions']
            or [r.get('stage') for r in summary.get('stages',[])]!=list(STAGES)
            or any(r.get('status')!='passed' for r in summary.get('stages',[]))):
        raise ValueError('Runtime smoke validation stage evidence incomplete')


def validate_result(result,plan):
    validate_summary(result.get('runtime_validation',{}),plan)
    study.validate_result(result,plan)
    if result.get('actual_generate_attempts')!=6:raise ValueError('Smoke generate call count differs')


def validate_sae_shape(record,hidden_size):
    if (record['layer']!=5 or record['d_model']!=hidden_size or record['d_sae']!=32768
            or record['encoder_shape']!=[32768,hidden_size]
            or record['decoder_shape'] not in ([hidden_size,32768],[32768,hidden_size])
            or record['encoder_bias_shape']!=[32768] or record['decoder_bias_shape']!=[hidden_size]
            or any(feature>=record['d_sae'] for feature in (30908,12207))):
        raise ValueError('SAE layer/shape/feature index mismatch')


def execute_stages(plan,backend,run_dir):
    """Fresh in-process stages only; no resume JSON or supplied verified flags are accepted."""
    run_dir=Path(run_dir)
    if run_dir.is_symlink():raise FileExistsError('Refusing symlink runtime directory')
    run_dir.mkdir(parents=True,exist_ok=True)
    allowed={'config.toml','plan.json','job.log','diagnostic.log','job_manifest.json','.trace_job_claim'}
    if any(p.name not in allowed or p.is_symlink() for p in run_dir.iterdir()):
        raise FileExistsError('Refusing existing runtime artifacts')
    parent=run_dir/'job_manifest.json'
    if parent.exists():
        metadata=json.loads(parent.read_text())
        if metadata.get('status')!='running' or metadata.get('smoke_manifest_sha256')!=plan['smoke_manifest_sha256']:
            raise FileExistsError('Runtime parent claim does not match this fresh smoke job')
    with (run_dir/'.runtime_smoke_claim').open('x'):pass
    state=dict(schema=1,status='running',smoke_manifest_sha256=plan['smoke_manifest_sha256'],
               started_at_utc=datetime.now(timezone.utc).isoformat(),
               revisions=plan['study_manifest']['revisions'],stages=[],actual_generate_attempts=0,
               synthetic_hook_module_forwards=0,extra_model_forwards=0,execution_ready=False)
    current_stage='files'
    try:
        for i,name in enumerate(STAGES):
            current_stage=name
            evidence=getattr(backend,name)()
            # Status is produced by concrete checks in the backend, never from CLI/config.
            record=dict(stage=name,status='passed',evidence=evidence,finished_at_utc=datetime.now(timezone.utc).isoformat())
            write_new(run_dir/f'stage_{i+1}_{name}.json',record)
            state['stages'].append(record)
            if name=='hook':state['synthetic_hook_module_forwards']=2
        state['status']='validated'
        validate_summary(state,plan)
        write_new(run_dir/'runtime_validation.json',state)
        if plan['through']=='generate':
            current_stage='generate'
            result=backend.generate(run_dir)  # Shared existing capture loop; never reloads the context.
            result['runtime_validation']=copy.deepcopy(state)
            result['actual_generate_attempts']=backend.actual_generate_attempts
            write_new(run_dir/'actual_first_divergence.json',result)
            validate_result(result,plan)
        state['status']='succeeded'
        return state
    except Exception as exc:
        state.update(status='failed',failed_stage=current_stage,error=f'{type(exc).__name__}: {exc}')
        write_new(run_dir/'partial_file_integrity.json',getattr(backend,'file_records',{}))
        with (run_dir/'runtime_failure.log').open('x') as stream:stream.write(traceback.format_exc())
        raise
    finally:
        state['actual_generate_attempts']=getattr(backend,'actual_generate_attempts',0)
        state['synthetic_hook_module_forwards']=getattr(backend,'synthetic_hook_module_forward_attempts',state['synthetic_hook_module_forwards'])
        write_new(run_dir/'runtime_status.json',state)


class RuntimeBackend:
    """Real pinned loading implementation; instantiated only after explicit allocation approval."""
    def __init__(self,plan):
        require_allocation_authorized(plan)
        self.plan=copy.deepcopy(plan);self.revisions=plan['study_manifest']['revisions']
        self.metadata=json.loads((ROOT/plan['study_manifest']['revision_evidence_path']).read_text())
        self.plan['started_at']=datetime.now(timezone.utc).isoformat()
        self.actual_generate_attempts=0;self.synthetic_hook_module_forward_attempts=0;self.snapshots={};self.file_records={}

    def files(self):
        import torch
        import transformers
        from huggingface_hub import hf_hub_download
        self.torch=torch;self.transformers=transformers
        if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('This authorized runtime smoke requires an A100')
        verified={}
        for role in ('model','tokenizer','sae'):
            evidence=self.metadata['queries'][role];records=[]
            self.file_records[role]=records
            files=evidence['files'] if role!='sae' else [f for f in evidence['files'] if f['rfilename'] in ('config.json','layer5.sae.pt')]
            for metadata in files:
                key=(evidence['repo_id'],self.revisions[role],metadata['rfilename'])
                if key not in verified:
                    path=Path(hf_hub_download(repo_id=evidence['repo_id'],revision=self.revisions[role],filename=metadata['rfilename']))
                    if path.parent.name!=self.revisions[role]:raise ValueError('Downloaded path is not the requested immutable snapshot')
                    verified[key]=verify_file(path,metadata)
                records.append(verified[key])
                path=Path(verified[key]['path'])
            self.snapshots[role]=path.parent
            self.file_records[role]=records
        return dict(revisions=self.revisions,files=self.file_records,unique_files_verified=len(verified),gpu=self.torch.cuda.get_device_name(0),
                    environment=dict(python=sys.version,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda))

    def processor(self):
        from transformers import AutoConfig
        from mindscopex_analysis.models import load_qwen_text_generation_model
        from mindscopex_analysis.lure_datasets import load_lure_dataset
        from mindscopex_analysis.prompts import instruct_lure_case
        from scripts.validate_cross_case_tokenizer_runtime import compare_inputs
        self.config=AutoConfig.from_pretrained(str(self.snapshots['model']),revision=self.revisions['model'],local_files_only=True)
        if self.config.model_type!='qwen3_5':raise ValueError('Unexpected production model type')
        # Model and Tokenizer currently share one snapshot; never silently mix revisions.
        if self.snapshots['model']!=self.snapshots['tokenizer']:raise ValueError('Separate processor snapshot routing requires explicit implementation')
        _,self.processor_object=load_qwen_text_generation_model(str(self.snapshots['model']),
            revision=self.revisions['model'],tokenizer_revision=self.revisions['tokenizer'],
            processor_only=True,prepared_config=self.config,local_files_only=True)
        tok=getattr(self.processor_object,'tokenizer',self.processor_object)
        cases={c.case_id:instruct_lure_case(c) for c in load_lure_dataset('crt_fresh_v2')}
        inputs=copy.deepcopy(self.plan['case_inputs'])
        for cid,item in inputs.items():item['candidate_texts']={k:getattr(cases[cid],k+'_answer') for k in ('correct','lure')}
        comparison=compare_inputs(tok,inputs,cases)
        if comparison['status']!='verified':raise ValueError('Production processor tokenizer mismatch')
        for cid,item in inputs.items():
            ids=tok(cases[cid].prompt,return_tensors='pt')['input_ids'][0].tolist()
            if ids!=item['prompt_token_ids']:raise ValueError('Production tensor token IDs differ')
        return dict(processor_class=type(self.processor_object).__name__,tokenizer_class=type(tok).__name__,
                    requested_revision=self.revisions['tokenizer'],snapshot=str(self.snapshots['tokenizer']),input_comparison=comparison)

    def model(self):
        from mindscopex_analysis.models import load_qwen_text_generation_model,get_qwen35_analysis_profile
        if not self.torch.cuda.is_bf16_supported():raise ValueError('BF16 required for this smoke scope')
        self.model_object,processor=load_qwen_text_generation_model(str(self.snapshots['model']),
            revision=self.revisions['model'],tokenizer_revision=self.revisions['tokenizer'],
            prepared_config=self.config,prepared_processor=self.processor_object,local_files_only=True,use_safetensors=True,dtype='bfloat16',device_map='auto')
        if processor is not self.processor_object:raise ValueError('Processor changed during model loading')
        recorded=getattr(self.model_object.config,'_commit_hash',None)
        if recorded is not None and recorded!=self.revisions['model']:raise ValueError('Loaded model commit mismatch')
        if type(self.model_object).__name__ not in self.metadata['queries']['model']['config_metadata']['architectures']:
            raise ValueError('Unexpected model architecture class')
        if self.model_object.training or any(getattr(self.model_object.generation_config,k,1) not in (None,1) for k in ('num_beams','num_return_sequences')):
            raise ValueError('Eval/single-sequence greedy model required')
        parameter=next(self.model_object.parameters())
        if parameter.dtype!=self.torch.bfloat16 or parameter.device.type!='cuda':raise ValueError('Loaded model dtype/device mismatch')
        self.profile=get_qwen35_analysis_profile('2b')
        from experiments.diagnose_rate027_baseline import environment
        self.loaded_environment=environment(self.model_object,self.processor_object,getattr(self.processor_object,'tokenizer',self.processor_object),self.torch,self.transformers,'bfloat16')
        return dict(environment=self.loaded_environment,model_class=type(self.model_object).__name__,requested_revision=self.revisions['model'],
                    config_commit_metadata=recorded,snapshot=str(self.snapshots['model']),
                    integrity_binding='Verified snapshot files passed to production loader; no inference from cache hint',
                    dtype=str(next(self.model_object.parameters()).dtype),device=str(next(self.model_object.parameters()).device))

    def sae(self):
        from mindscopex_analysis.qwen_scope import load_qwen_scope_sae
        from mindscopex_analysis.research import find_decoder_block
        self.sae_object=load_qwen_scope_sae(self.plan['study_manifest']['sae_repo_id'],5,revision=self.revisions['sae'],local_files_only=True,device='cuda',dtype=self.torch.bfloat16)
        # Verify the file again after the existing loader returns, binding its immutable cache source.
        meta=next(f for f in self.metadata['queries']['sae']['files'] if f['rfilename']=='layer5.sae.pt')
        proof=verify_file(self.snapshots['sae']/'layer5.sae.pt',meta)
        sae=self.sae_object
        if sae.repo_id!=self.plan['study_manifest']['sae_repo_id']:raise ValueError('Loaded SAE repository mismatch')
        hidden=getattr(self.model_object.config,'text_config',self.model_object.config).hidden_size
        shape=dict(layer=sae.layer,d_model=sae.d_model,d_sae=sae.d_sae,encoder_shape=list(sae.W_enc.shape),
                   decoder_shape=list(sae.W_dec.shape),encoder_bias_shape=list(sae.b_enc.shape),decoder_bias_shape=list(sae.b_dec.shape))
        validate_sae_shape(shape,hidden)
        for tensor in (sae.W_enc,sae.W_dec,sae.b_enc,sae.b_dec):
            if tensor.dtype!=self.torch.bfloat16 or not self.torch.isfinite(tensor).all().item():raise ValueError('SAE dtype/nonfinite tensor')
        self.block=find_decoder_block(self.model_object,5,block_path_template=self.profile.block_path_template)
        return dict(requested_revision=self.revisions['sae'],checkpoint=proof,layer=5,d_model=hidden,d_sae=sae.d_sae,
                    encoder_shape=list(sae.W_enc.shape),decoder_shape=list(sae.W_dec.shape),dtype=str(sae.W_dec.dtype))

    def hook(self):
        from mindscopex_analysis.qwen_scope import make_feature_steering_hook,sae_decoder_direction
        records=[]
        for feature,coefficient in ((30908,8.5),(12207,8)):
            diagnostics=[];module=self.torch.nn.Identity();hidden=self.torch.ones((1,2,self.sae_object.d_model),device='cuda',dtype=self.torch.bfloat16)
            handle=module.register_forward_hook(make_feature_steering_hook(self.sae_object,[feature],coefficient=coefficient,token_position='all',diagnostics=diagnostics))
            try:
                self.synthetic_hook_module_forward_attempts+=1
                with self.torch.inference_mode():output=module(hidden)
                direction=sae_decoder_direction(self.sae_object,[feature],[float(coefficient)]).to(device=hidden.device,dtype=hidden.dtype)
                if (not self.torch.isfinite(output).all().item() or not diagnostics
                        or not self.torch.equal(output,hidden+direction)
                        or not self.torch.equal(hidden,self.torch.ones_like(hidden))):raise ValueError('Hook Tensor diagnostic failed')
                r=diagnostics[0]
                if any(r[k] for k in ('nonfinite_residual_count','nonfinite_post_count','nonfinite_delta_count')) or any(r[k] is None or not math.isfinite(r[k]) for k in ('actual_delta_mean','requested_vector_norm','cast_vector_norm')):
                    raise ValueError('Nonfinite hook activation')
                records.append(dict(feature_id=feature,coefficient=coefficient,diagnostics=diagnostics))
            finally:handle.remove()
            if module._forward_hooks:raise ValueError('Hook removal failed')
        return dict(synthetic_module_forwards=2,extra_model_forwards=0,records=records,
                    limitation='Identity dispatch tests real Tensor hook arithmetic; actual Layer5 dispatch is checked during generation')

    def generate(self,run_dir):
        from experiments import trace_fresh_v2_first_divergence as trace
        backend=self
        class CountCalls:
            def __getattr__(self,name):return getattr(backend.model_object,name)
            def generate(self,**kwargs):
                backend.actual_generate_attempts+=1
                if backend.actual_generate_attempts>6:raise RuntimeError('Smoke generate budget exceeded')
                return backend.model_object.generate(**kwargs)
        from experiments.growth019_minimal_pilot import validate_capture_tokens,validate_capture
        def raw_ready(item,record):
            write_new(Path(run_dir)/f"capture_{item['condition_id']}.json",record)
            validate_capture_tokens(record,item)
            study.validate_activation_norms(record.get('activation_norms',{}),item)
        def comparison_ready(item,record):
            write_new(Path(run_dir)/f"measurement_{item['condition_id']}.json",record)
            check=record.get('equivalence_check',{})
            if check.get('status')!='compared' or any(check.get(k) is not True for k in ('generated_token_ids_equal','answers_equal','equivalent')):
                raise ValueError('Smoke equivalence mismatch; stop before the next condition')
            validate_capture(record,item)
        result=trace.run_loaded_model(self.plan,CountCalls(),self.processor_object,self.sae_object,self.block,
            torch=self.torch,transformers=self.transformers,dtype='bfloat16',profile=self.profile,sae_device='cuda',
            after_raw_capture=raw_ready,after_capture=comparison_ready)
        result['environment']=self.loaded_environment
        result['generation_source_evidence']=trace.generation_source_evidence(self.model_object)
        result['source_hashes'].update({name:digest(ROOT/name) for name in ('experiments/runtime_smoke.py','experiments/inputs/growth019_runtime_smoke.json')})
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through',choices=('validate','generate'),default='generate')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv);plan=build_plan(args.through)
    require_allocation_authorized(plan)
    execute_stages(plan,RuntimeBackend(plan),args.output_dir)


if __name__=='__main__':main()
