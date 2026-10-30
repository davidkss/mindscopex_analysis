"""CPU mock/AST study preflight; optional real Tensor test explicitly separated."""
import ast
import copy
import importlib.util
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments import feature_specificity_cross_case as study
from experiments import trace_fresh_v2_first_divergence as trace
from experiments.jobs import first_divergence_trace as job
from experiments.runners.config import load_toml
from test_growth019_minimal_pilot import PilotModel, PilotTokenizer
from test_trace_fresh_v2_first_divergence import Tensor, _binary_choice_completion, mock_environment


def extracted(path,name,namespace):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),namespace)
    return namespace[name]


def capture_fixture(plan):
    """Real trace loop/AST production helper; fake model, tensors, loader and hook."""
    from mindscopex_analysis.lure_datasets import load_lure_dataset
    from mindscopex_analysis.prompts import instruct_lure_case
    cases=[instruct_lure_case(c) for c in load_lure_dataset('crt_fresh_v2') if c.case_id in plan['case_ids']]
    class Tokenizer(PilotTokenizer):
        def __init__(self):
            self.candidates={getattr(c,f'{label}_answer'):plan['case_inputs'][c.case_id]['candidate_token_ids'][label]
                             for c in cases for label in ('correct','lure')}
        def __call__(self,text,*,return_tensors):
            ids=next(plan['case_inputs'][c.case_id]['prompt_token_ids'] for c in cases if c.prompt==text)
            return dict(input_ids=Tensor([ids]),attention_mask=Tensor([[1]*len(ids)]))
    model=PilotModel();tokenizer=Tokenizer();calls=[];loads=[]
    def factory(sae,features,*,coefficient,token_position,diagnostics=None):
        calls.append((features,coefficient,token_position,diagnostics is not None))
        if diagnostics is not None:
            diagnostics.append(dict(feature_ids=str(features[0]),coefficient=coefficient,token_position=token_position,
                requested_vector_norm=coefficient*5,cast_vector_norm=coefficient*5,actual_delta_mean=coefficient*5,
                residual_before_mean=10.,residual_after_mean=12.))
        return lambda *_:None
    def load_model(*a,**kwargs):loads.append(('model',kwargs));return model,tokenizer
    def load_sae(*a,**kwargs):loads.append(('sae',kwargs));return SimpleNamespace(W_dec=SimpleNamespace(dtype='mock.bfloat16'))
    profile=SimpleNamespace(analysis_model_id=plan['model_id'],sae_repo_id=plan['study_manifest']['sae_repo_id'],block_path_template='mock.{layer}')
    norm=SimpleNamespace(detach=lambda:SimpleNamespace(float=lambda:SimpleNamespace(norm=lambda:SimpleNamespace(item=lambda:5.))))
    modules={
        'torch':SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True)), 'transformers':SimpleNamespace(),
        'mindscopex_analysis.models':SimpleNamespace(default_sae_device=lambda:'mock-device',dtype_from_name=lambda x:x,
            get_qwen35_analysis_profile=lambda _:profile,load_qwen_text_generation_model=load_model,recommended_dtype_name=lambda:'bfloat16'),
        'mindscopex_analysis.qwen_scope':SimpleNamespace(make_feature_steering_hook=factory,load_qwen_scope_sae=load_sae,sae_decoder_direction=lambda *a:norm),
        'mindscopex_analysis.research':SimpleNamespace(_binary_choice_completion=_binary_choice_completion,find_decoder_block=lambda *a,**k:model.block)}
    env,_=mock_environment()
    # The sole readiness bypass is local to this Mock test; never used by production.
    with mock.patch.dict(sys.modules,modules),mock.patch.object(study,'require_ready'),mock.patch('experiments.diagnose_rate027_baseline.environment',return_value=env):
        result=trace.run_model(copy.deepcopy(plan),check_equivalence=True)
    return result,model,calls,loads



class NormTensor:
    """Tiny arithmetic test double, not Torch; exercises existing hook stats."""
    def __init__(self,data):self.data=data
    def detach(self):return self
    def float(self):return self
    def to(self,**kwargs):return self
    def clone(self):return NormTensor(copy.deepcopy(self.data))
    device='mock';dtype='mock.float32'
    def _binary(self,other,fn):
        other=other.data if isinstance(other,NormTensor) else other
        def apply(a,b):
            if isinstance(a,list):
                if a and isinstance(a[0],list) and isinstance(b,list) and not isinstance(b[0],list):
                    return [apply(x,b) for x in a]
                return [apply(x,b[i] if isinstance(b,list) else b) for i,x in enumerate(a)]
            return fn(a,b)
        return NormTensor(apply(self.data,other))
    def __add__(self,x):return self._binary(x,lambda a,b:a+b)
    def __sub__(self,x):return self._binary(x,lambda a,b:a-b)
    def __truediv__(self,x):return self._binary(x,lambda a,b:a/b)
    def __gt__(self,x):return self._binary(x,lambda a,b:a>b)
    def __eq__(self,x):return self._binary(x,lambda a,b:a==b)
    def __and__(self,x):return self._binary(x,lambda a,b:a and b)
    def __invert__(self):return NormTensor([not x for x in self.data])
    def flatten(self):
        def flat(x):return [v for a in x for v in flat(a)] if isinstance(x,list) else [x]
        return NormTensor(flat(self.data))
    def norm(self,dim=None):
        if dim is None:return NormTensor(math.sqrt(sum(x*x for x in self.flatten().data)))
        return NormTensor([math.sqrt(sum(x*x for x in row)) for row in self.data])
    def __getitem__(self,index):return NormTensor([x for x,keep in zip(self.data,index.data) if keep])
    def numel(self):return len(self.flatten().data)
    def mean(self):return NormTensor(sum(self.data)/len(self.data))
    def min(self):return NormTensor(min(self.data))
    def max(self):return NormTensor(max(self.data))
    def sum(self):return NormTensor(sum(self.data))
    def item(self):return self.data

class CrossCaseTests(unittest.TestCase):
    def setUp(self):self.plan=study.build_plan('block-1')

    def test_inputs_tokens_sha_baseline_and_scope(self):
        expected={'019':([220,20,20,2756],[220,17,23,2756]),'004':([220,17,20,2756],[220,16,18,2756]),'016':([220,19,24,2756],[220,17,20,2756])}
        for cid,evidence in self.plan['case_inputs'].items():
            a,b=expected[cid[-3:]]
            self.assertEqual(evidence['candidate_token_ids'],dict(correct=a,lure=b))
            self.assertEqual(evidence['common_prefix_length'],1)
            self.assertTrue(all(r['baseline']=='correct' for r in evidence['historical_baseline_slots']))
        plans=[study.build_plan(b) for b in ('block-1','block-2')]
        self.assertEqual(sum(p['expected_generate_calls'] for p in plans),84)
        self.assertEqual(len({m['measurement_id'] for p in plans for m in p['measurements']}),42)
        for cid in study.CASE_IDS:
            self.assertEqual([(m['feature_id'],m['coefficient']) for m in self.plan['measurements'] if m['case_id']==cid],study.CONDITIONS)

    def test_manifest_mutation_prompt_and_prefix_rejected(self):
        original=Path.read_text
        manifest=json.loads(study.MANIFEST.read_text());manifest['conditions'][1]['feature_id']=7438
        with mock.patch.object(Path,'read_text',autospec=True,side_effect=lambda p,*a,**k:json.dumps(manifest) if p==study.MANIFEST else original(p,*a,**k)):
            with self.assertRaises(ValueError):study.build_plan('block-1')
        with self.assertRaises(ValueError):study.build_plan('unknown')
        protocol=json.loads(study.PROTOCOL.read_text());protocol['prompts'][0]['prompt_token_ids_sha256']='invalid'
        # Corrupt one selected case, never an original file.
        selected=next(p for p in protocol['prompts'] if p['case_id']==study.CASE_IDS[0]);selected['candidate_token_ids']['lure']=selected['candidate_token_ids']['correct']
        original=Path.read_text
        with mock.patch.object(Path,'read_text',autospec=True,side_effect=lambda p,*a,**k:json.dumps(protocol) if p==study.PROTOCOL else original(p,*a,**k)):
            with self.assertRaises(ValueError):study.build_plan('block-1')

    def test_unresolved_revisions_stop_before_ml_import(self):
        with self.assertRaisesRegex(ValueError,'revisions'):trace.run_model(self.plan,check_equivalence=True)
        ready=copy.deepcopy(self.plan);ready['study_manifest']['revisions']={k:'a'*40 for k in ('model','tokenizer','sae')}
        with self.assertRaises(ValueError):study.require_ready(ready)
        ready['study_manifest']['pinning_validation']='verified'
        with self.assertRaises(ValueError):study.require_ready(ready)
        ready['study_manifest']['revision_validation']=dict(validated_revisions=ready['study_manifest']['revisions'],
            revision_metadata_verified=True,revision_loader_forwarding_verified=True,
            runtime_tokenizer_validation='verified',runtime_model_loading_validation='verified',runtime_sae_loading_validation='verified')
        study.require_ready(ready)
        self.assertFalse(self.plan['execution_ready'])

    def test_actual_loop_feature_dose_no_hook_and_result_validation(self):
        result,model,calls,loads=capture_fixture(self.plan)
        self.assertEqual(len(model.calls),42)
        self.assertEqual(sum(n==0 for _,n in model.calls),6)
        self.assertEqual(sum(n==1 for _,n in model.calls),36)
        self.assertEqual(len(calls),36)
        self.assertIn(([12207],8.5,'all',True),calls)
        self.assertEqual(model.block._forward_hooks,{})
        self.assertIn('revision',loads[0][1]);self.assertIn('tokenizer_revision',loads[0][1]);self.assertIn('revision',loads[1][1])
        study.validate_result(result,self.plan)

    def test_eos_equivalence_sha_margin_and_telemetry_fail(self):
        result,_,_,_=capture_fixture(self.plan)
        mutations=[lambda r:r['measurements'][0]['capture'].__setitem__('eos_terminated',False),
                   lambda r:r['measurements'][0]['capture']['equivalence_check'].__setitem__('equivalent',False),
                   lambda r:r['source_files']['scores'].__setitem__('sha256','bad'),
                   lambda r:r['measurements'][0]['capture'].__setitem__('actual_raw_margin',99),
                   lambda r:r['measurements'][1]['capture'].__setitem__('activation_norms',{}),
                   lambda r:r['measurements'][0]['capture']['steps'][1].__setitem__('allowed_token_ids',[20]),
                   lambda r:r.__setitem__('schema',2)]
        for mutate in mutations:
            changed=copy.deepcopy(result);mutate(changed)
            with self.assertRaises(ValueError):study.validate_result(changed,self.plan)

    def test_job_failure_keeps_json_and_log_and_refuses_overwrite(self):
        result,_,_,_=capture_fixture(self.plan);result['measurements'][0]['capture']['equivalence_check']['equivalent']=False
        config=trace.ROOT/'experiments/configs/cross_case_specificity_block-1_2b.toml'
        with tempfile.TemporaryDirectory() as tmp:
            def fake_run(command,**kwargs):
                trace.write_new(Path(command[-1]),result)
                kwargs['stdout'].write('Mock execution result preserved\n')
            with mock.patch.object(job.subprocess,'run',side_effect=fake_run):
                with self.assertRaises(ValueError):job.run(config,Path(tmp))
            run=Path(tmp)/self.plan['run_id']
            self.assertEqual(json.loads((run/'job_manifest.json').read_text())['status'],'failed')
            self.assertTrue((run/'actual_first_divergence.json').is_file());self.assertTrue((run/'diagnostic.log').read_text())
            with self.assertRaises(FileExistsError):job.run(config,Path(tmp))

    def test_plan_subprocess_and_cli_overwrite_no_ml(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'plan.json'
            command=[sys.executable,'experiments/trace_fresh_v2_first_divergence.py','--plan','--cross-case-block','block-1','--output',str(output)]
            self.assertEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(subprocess.run(command,capture_output=True).returncode,1)
        code="from experiments.feature_specificity_cross_case import build_plan; import sys; build_plan('block-1'); assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
        self.assertEqual(subprocess.run([sys.executable,'-c',code],capture_output=True).returncode,0)

    def test_launcher_blocks_allocation_and_archive_has_protocol(self):
        import zipfile
        from experiments.runners import launch_colab as launcher
        config=trace.ROOT/'experiments/configs/cross_case_specificity_block-1_2b.toml'
        with mock.patch.object(launcher,'ensure_session') as allocate:
            with self.assertRaisesRegex(ValueError,'revisions'):
                launcher.launch_one(config,root=trace.ROOT,session='unused',keep_session=False,default_gpu=None,default_tpu=None,exec_timeout=900,dry_run=False,allocation_attempts=1,allocation_retry_delay=0)
            allocate.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            archive=Path(tmp)/'source.zip'
            with zipfile.ZipFile(archive,'w'):pass
            job.include_reference_inputs(trace.ROOT,archive,additional_paths=(study.PROTOCOL,))
            with zipfile.ZipFile(archive) as z:self.assertEqual(z.read(str(study.PROTOCOL.relative_to(trace.ROOT))),study.PROTOCOL.read_bytes())

    def test_compare_two_mock_blocks_and_no_missing_block_success(self):
        outputs={block:capture_fixture(study.build_plan(block))[0] for block in ('block-1','block-2')}
        compared=study.compare_blocks(outputs)
        self.assertEqual(len(compared['rows']),21)
        self.assertEqual(compared['n_observations'],42)
        self.assertTrue(all(r['tokens_equal'] and r['raw_equal'] for r in compared['rows']))
        with self.assertRaises(ValueError):study.compare_blocks({'block-1':outputs['block-1']})

    def test_revision_forwarding_config_processor_model_and_sae(self):
        calls=[]
        class Auto:
            @staticmethod
            def from_pretrained(model,**kwargs):calls.append(kwargs);return SimpleNamespace(model_type='qwen3_5',eval=lambda:None)
        transformers=SimpleNamespace(**{k:Auto for k in ('AutoConfig','AutoModelForCausalLM','AutoTokenizer','AutoModelForMultimodalLM','AutoProcessor')})
        fn=extracted(trace.ROOT/'src/mindscopex_analysis/models.py','load_qwen_text_generation_model',dict(dtype_from_name=lambda _:None))
        with mock.patch.dict(sys.modules,{'transformers':transformers}):fn('mock',revision='a'*40,tokenizer_revision='b'*40)
        self.assertEqual([c['revision'] for c in calls],['a'*40,'b'*40,'a'*40])
        calls.clear()
        with mock.patch.dict(sys.modules,{'transformers':transformers}):fn('mock')
        self.assertTrue(all('revision' not in c for c in calls))
        downloads=[]
        fake_sae=SimpleNamespace(to=lambda **k:'loaded')
        fn=extracted(trace.ROOT/'src/mindscopex_analysis/qwen_scope.py','load_qwen_scope_sae',dict(
            hf_hub_download=lambda **k:downloads.append(k) or 'mock',torch=SimpleNamespace(load=lambda *a,**k:dict(W_enc=1,W_dec=2,b_enc=3,b_dec=4)),
            QwenScopeSAE=lambda **k:fake_sae,infer_top_k_from_repo=lambda _:50))
        self.assertEqual(fn('mock',5,revision='c'*40),'loaded');self.assertEqual(downloads[0]['revision'],'c'*40)
        fn('mock',5);self.assertNotIn('revision',downloads[-1])

    def test_mock_norm_arithmetic_at_production_hook(self):
        rows=[]
        fake_torch=SimpleNamespace(isfinite=lambda t:NormTensor([math.isfinite(v) for v in t.data]))
        fn=extracted(trace.ROOT/'src/mindscopex_analysis/qwen_scope.py','make_feature_steering_hook',dict(
            torch=fake_torch,sae_decoder_direction=lambda sae,features,coefficients:NormTensor([3*coefficients[0],4*coefficients[0],0.])))
        hook=fn(None,[30908],coefficient=.5,token_position='all',diagnostics=rows)
        hidden=NormTensor([[6.,8.,0.],[0.,5.,0.]])
        after=hook(None,(),hidden)
        self.assertEqual(after.data,[[7.5,10.,0.],[1.5,7.,0.]])
        self.assertEqual(hidden.data,[[6.,8.,0.],[0.,5.,0.]])
        self.assertEqual(rows[0]['requested_vector_norm'],2.5)
        self.assertEqual(rows[0]['cast_vector_norm'],2.5)
        self.assertEqual(rows[0]['actual_delta_mean'],2.5)
        self.assertEqual(rows[0]['residual_before_mean'],7.5)
        self.assertEqual(rows[0]['delta_residual_ratio_mean'],.375)

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('huggingface_hub'),'Torch/HF unavailable: actual Tensor hook test unverified')
    def test_real_tensor_norms_at_existing_hook(self):
        import torch
        from mindscopex_analysis.qwen_scope import make_feature_steering_hook
        sae=SimpleNamespace(W_dec=torch.tensor([[3.],[4.],[0.]]),d_model=3)
        rows=[];hidden=torch.tensor([[[6.,8.,0.],[0.,5.,0.]]])
        hook=make_feature_steering_hook(sae,[0],coefficient=.5,token_position='all',diagnostics=rows)
        result=hook(None,(),hidden)
        self.assertEqual(rows[0]['requested_vector_norm'],2.5)
        self.assertEqual(rows[0]['actual_delta_mean'],2.5)
        self.assertEqual(rows[0]['residual_before_mean'],7.5)
        self.assertTrue(torch.equal(result,hidden+torch.tensor([1.5,2.,0.])))


if __name__=='__main__':unittest.main()
