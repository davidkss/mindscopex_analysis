"""CPU staged smoke mocks; no real model, GPU, remote calls or weights."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments import runtime_smoke as smoke
from experiments import feature_specificity_cross_case as study
from experiments import trace_fresh_v2_first_divergence as trace
from experiments.jobs import first_divergence_trace as job
from experiments.runners.config import load_toml
from test_feature_specificity_cross_case import capture_fixture,extracted


class FakeBackend:
    def __init__(self,plan,fail=None):
        self.plan=plan;self.fail=fail;self.events=[];self.actual_generate_attempts=0
    def event(self,name):
        self.events.append(name)
        if self.fail==name:raise ValueError('Mock stage failure: '+name)
        return dict(mock_boundary=True)
    def files(self):return self.event('files')
    def processor(self):return self.event('processor')
    def model(self):return self.event('model')
    def sae(self):return self.event('sae')
    def hook(self):return self.event('hook')
    def generate(self,run_dir):
        self.event('generate')
        result,model,_,_=capture_fixture(self.plan)
        self.actual_generate_attempts=len(model.calls)
        return result


class RuntimeSmokeTests(unittest.TestCase):
    def setUp(self):
        import platform
        platform.platform()  # Resolve lazy CPU metadata before subprocess boundary mocks.
        self.plan=smoke.build_plan()

    def test_scope_three_conditions_six_calls_and_extra_budget(self):
        self.assertEqual([(m['feature_id'],m['coefficient']) for m in self.plan['measurements']],[(None,0),(30908,8.5),(12207,8)])
        self.assertEqual(self.plan['expected_generate_calls'],6)
        self.assertEqual(self.plan['synthetic_hook_module_forwards'],2)
        self.assertEqual(self.plan['extra_model_forwards'],0)
        self.assertFalse(self.plan['execution_ready']);self.assertFalse(study.build_plan('block-1')['execution_ready'])
        self.assertEqual(smoke.build_plan('validate')['expected_generate_calls'],0)
        with self.assertRaises(ValueError):smoke.build_plan('unknown')

    def test_no_allocations_or_loader_imports_without_authorization(self):
        from experiments.runners import launch_colab as launch
        path=trace.ROOT/'experiments/configs/growth019_runtime_smoke_2b.toml'
        with mock.patch.object(launch,'ensure_session') as allocate,mock.patch.object(launch,'run_cmd') as remote:
            with self.assertRaisesRegex(ValueError,'not authorized'):
                launch.launch_one(path,root=trace.ROOT,session='unused',keep_session=False,default_gpu='A100',default_tpu=None,exec_timeout=900,dry_run=False,allocation_attempts=1,allocation_retry_delay=0)
            allocate.assert_not_called();remote.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(smoke,'RuntimeBackend') as backend:
            with self.assertRaisesRegex(ValueError,'not authorized'):smoke.main(['--output-dir',tmp])
            backend.assert_not_called()

    def test_verified_config_field_and_existing_artifact_rejected(self):
        config=load_toml(trace.ROOT/'experiments/configs/growth019_runtime_smoke_2b.toml')
        config['runtime_smoke']['verified']=True
        with self.assertRaises(ValueError):job.prepare_plan(config)
        with tempfile.TemporaryDirectory() as tmp:
            old=Path(tmp)/'old.json';old.write_text('preserve')
            with self.assertRaises(FileExistsError):smoke.execute_stages(self.plan,FakeBackend(self.plan),tmp)
            self.assertEqual(old.read_text(),'preserve')

    def test_each_validation_failure_stops_preserves_prior_stages(self):
        for index,name in enumerate(smoke.STAGES):
            with tempfile.TemporaryDirectory() as tmp:
                backend=FakeBackend(self.plan,fail=name)
                with self.assertRaises(ValueError):smoke.execute_stages(self.plan,backend,tmp)
                self.assertEqual(backend.events,list(smoke.STAGES[:index+1]))
                status=json.loads((Path(tmp)/'runtime_status.json').read_text())
                self.assertEqual(status['status'],'failed');self.assertEqual(status['failed_stage'],name)
                self.assertEqual(status['actual_generate_attempts'],0)
                self.assertEqual(len(status['stages']),index)
                self.assertTrue((Path(tmp)/'runtime_failure.log').is_file())
                self.assertFalse((Path(tmp)/'actual_first_divergence.json').exists())

    def test_validation_only_performs_no_generates(self):
        plan=smoke.build_plan('validate')
        with tempfile.TemporaryDirectory() as tmp:
            backend=FakeBackend(plan);state=smoke.execute_stages(plan,backend,tmp)
            self.assertEqual(backend.events,list(smoke.STAGES));self.assertEqual(state['actual_generate_attempts'],0)
            self.assertFalse((Path(tmp)/'actual_first_divergence.json').exists())
            with self.assertRaises(FileExistsError):smoke.execute_stages(plan,backend,tmp)

    def test_successful_mock_pipeline_accepts_different_historical_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend=FakeBackend(self.plan);state=smoke.execute_stages(self.plan,backend,tmp)
            self.assertEqual(backend.events,list(smoke.STAGES)+['generate'])
            self.assertEqual(state['actual_generate_attempts'],6)
            result=json.loads((Path(tmp)/'actual_first_divergence.json').read_text())
            smoke.validate_result(result,self.plan)
            self.assertNotEqual(result['measurements'][1]['capture']['actual_raw_margin'],.125)
            self.assertFalse(study.build_plan('block-1')['execution_ready'])

    def test_equivalence_failure_keeps_result_json(self):
        class BadBackend(FakeBackend):
            def generate(self,run_dir):
                result=super().generate(run_dir)
                result['measurements'][0]['capture']['equivalence_check']['equivalent']=False
                return result
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):smoke.execute_stages(self.plan,BadBackend(self.plan),tmp)
            self.assertTrue((Path(tmp)/'actual_first_divergence.json').is_file())
            self.assertEqual(json.loads((Path(tmp)/'runtime_status.json').read_text())['status'],'failed')

    def test_file_integrity_streaming_git_lfs_and_corruption(self):
        data=b'cpu fixture';sha=hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'file';p.write_bytes(data)
            meta=dict(rfilename='file',size=len(data),blobId=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest())
            smoke.verify_file(p,meta)
            meta.update(blobId='a'*40,lfs=dict(sha256=sha,size=len(data)))
            self.assertEqual(smoke.verify_file(p,meta)['hash'],sha)
            p.write_bytes(b'not the file')
            with self.assertRaises(ValueError):smoke.verify_file(p,meta)

    def test_sae_layer_shape_and_feature_range(self):
        row=dict(layer=5,d_model=2048,d_sae=32768,encoder_shape=[32768,2048],decoder_shape=[2048,32768],encoder_bias_shape=[32768],decoder_bias_shape=[2048])
        smoke.validate_sae_shape(row,2048)
        for key,value in [('layer',6),('d_sae',12207),('encoder_shape',[2048,32768]),('decoder_shape',[4,4]),('decoder_bias_shape',[1])]:
            bad=dict(row);bad[key]=value
            with self.assertRaises(ValueError):smoke.validate_sae_shape(bad,2048)

    def test_processor_only_and_prepared_loader_keep_exact_revision_local_only(self):
        calls=[]
        def auto(name):
            def load(repo,**kwargs):calls.append((name,kwargs));return SimpleNamespace(model_type='qwen3_5',eval=lambda:None)
            return SimpleNamespace(from_pretrained=load)
        names=('AutoConfig','AutoModelForCausalLM','AutoTokenizer','AutoModelForMultimodalLM','AutoProcessor')
        fake=SimpleNamespace(**{name:auto(name) for name in names})
        fn=extracted(trace.ROOT/'src/mindscopex_analysis/models.py','load_qwen_text_generation_model',dict(dtype_from_name=lambda _:None))
        rev=self.plan['study_manifest']['revisions']
        with mock.patch.dict(sys.modules,{'transformers':fake}):
            model,processor=fn('mock',processor_only=True,revision=rev['model'],tokenizer_revision=rev['tokenizer'],local_files_only=True)
            self.assertIsNone(model);self.assertEqual([c[0] for c in calls],['AutoConfig','AutoProcessor'])
            self.assertTrue(all(c[1]['local_files_only'] for c in calls))
            calls.clear()
            fn('mock',prepared_config=SimpleNamespace(model_type='qwen3_5'),prepared_processor=processor,revision=rev['model'],tokenizer_revision=rev['tokenizer'],local_files_only=True)
            self.assertEqual([c[0] for c in calls],['AutoModelForMultimodalLM'])
            self.assertEqual(calls[0][1]['revision'],rev['model'])
            self.assertNotIn('processor_only',calls[0][1])
        downloads=[]
        fn=extracted(trace.ROOT/'src/mindscopex_analysis/qwen_scope.py','load_qwen_scope_sae',dict(
            hf_hub_download=lambda **k:downloads.append(k) or 'mock',torch=SimpleNamespace(load=lambda *a,**k:dict(W_enc=1,W_dec=2,b_enc=3,b_dec=4)),
            QwenScopeSAE=lambda **k:SimpleNamespace(to=lambda **k:'loaded'),infer_top_k_from_repo=lambda _:50))
        fn('mock',5,revision=rev['sae'],local_files_only=True)
        self.assertEqual(downloads[-1]['revision'],rev['sae']);self.assertTrue(downloads[-1]['local_files_only'])
        fn('mock',5);self.assertNotIn('local_files_only',downloads[-1])

    def test_live_capture_gate_stops_at_first_equivalence_failure(self):
        # Drive the real smoke callbacks against saved fake captures; no Qwen/Torch.
        fixture,_,_,_=capture_fixture(self.plan)
        backend=smoke.RuntimeBackend.__new__(smoke.RuntimeBackend)
        backend.plan=self.plan;backend.actual_generate_attempts=0
        backend.model_object=SimpleNamespace(generate=lambda **kwargs:None)
        backend.processor_object=object();backend.sae_object=object();backend.block=object()
        backend.torch=object();backend.transformers=object();backend.profile=object();backend.loaded_environment={}
        def loaded(plan,model,*args,**kwargs):
            for item in fixture['measurements']:
                model.generate();raw=copy.deepcopy(item['capture']);kwargs['after_raw_capture'](item,raw)
                model.generate();raw['equivalence_check']['equivalent']=False
                kwargs['after_capture'](item,raw)
            raise AssertionError('Must stop at first condition')
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(trace,'run_loaded_model',side_effect=loaded):
            with self.assertRaisesRegex(ValueError,'equivalence mismatch'):backend.generate(tmp)
            self.assertEqual(backend.actual_generate_attempts,2)
            self.assertTrue((Path(tmp)/'capture_baseline.json').exists())
            self.assertTrue((Path(tmp)/'measurement_baseline.json').exists())
            self.assertFalse((Path(tmp)/'capture_f30908-c8p5.json').exists())

    def test_live_nonfinite_capture_stops_before_comparison(self):
        fixture,_,_,_=capture_fixture(self.plan)
        backend=smoke.RuntimeBackend.__new__(smoke.RuntimeBackend)
        backend.plan=self.plan;backend.actual_generate_attempts=0;backend.model_object=SimpleNamespace(generate=lambda **k:None)
        backend.processor_object=object();backend.sae_object=object();backend.block=object()
        backend.torch=object();backend.transformers=object();backend.profile=object()
        def loaded(plan,model,*args,**kwargs):
            item=fixture['measurements'][0];model.generate();raw=copy.deepcopy(item['capture'])
            raw['actual_raw_margin']=None;kwargs['after_raw_capture'](item,raw)
            raise AssertionError('Must stop before comparison')
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(trace,'run_loaded_model',side_effect=loaded):
            with self.assertRaises((ValueError,TypeError)):backend.generate(tmp)
            self.assertEqual(backend.actual_generate_attempts,1)
            self.assertTrue((Path(tmp)/'capture_baseline.json').exists())

    def test_actual_model_boundary_rejects_reported_revision_mismatch(self):
        backend=smoke.RuntimeBackend.__new__(smoke.RuntimeBackend)
        backend.torch=SimpleNamespace(cuda=SimpleNamespace(is_bf16_supported=lambda:True))
        backend.revisions=self.plan['study_manifest']['revisions']
        backend.snapshots={'model':Path('/mock/snapshot')};backend.config=object();backend.processor_object=object()
        wrong=SimpleNamespace(config=SimpleNamespace(_commit_hash='a'*40))
        modules={'mindscopex_analysis.models':SimpleNamespace(load_qwen_text_generation_model=lambda *a,**k:(wrong,backend.processor_object),get_qwen35_analysis_profile=lambda *a:None)}
        with mock.patch.dict(sys.modules,modules):
            with self.assertRaisesRegex(ValueError,'commit mismatch'):backend.model()

    def test_nonfinite_activation_gate_stops_before_comparison(self):
        fixture,_,_,_=capture_fixture(self.plan)
        backend=smoke.RuntimeBackend.__new__(smoke.RuntimeBackend)
        backend.plan=self.plan;backend.actual_generate_attempts=0;backend.model_object=SimpleNamespace(generate=lambda **k:None)
        backend.processor_object=object();backend.sae_object=object();backend.block=object()
        backend.torch=object();backend.transformers=object();backend.profile=object()
        def loaded(plan,model,*args,**kwargs):
            item=fixture['measurements'][1];model.generate();raw=copy.deepcopy(item['capture'])
            raw['activation_norms']['records'][0]['actual_delta_mean']=None
            kwargs['after_raw_capture'](item,raw)
            raise AssertionError('Must stop before uninstrumented comparison')
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(trace,'run_loaded_model',side_effect=loaded):
            with self.assertRaisesRegex(ValueError,'activation norm'):backend.generate(tmp)
            self.assertEqual(backend.actual_generate_attempts,1)
            self.assertTrue((Path(tmp)/'capture_f30908-c8p5.json').exists())

    def test_adapter_runtime_command_and_failed_manifest(self):
        config=trace.ROOT/'experiments/configs/growth019_runtime_smoke_2b.toml'
        with tempfile.TemporaryDirectory() as tmp:
            def child(command,**kwargs):
                self.assertIn('runtime_smoke.py',command[2]);self.assertIn('--through',command)
                smoke.execute_stages(self.plan,FakeBackend(self.plan),Path(command[-1]))
            with mock.patch.object(job.subprocess,'run',side_effect=child):
                run=job.run(config,Path(tmp))
            self.assertEqual(json.loads((run/'job_manifest.json').read_text())['status'],'succeeded')
            with self.assertRaises(FileExistsError):job.run(config,Path(tmp))
        with tempfile.TemporaryDirectory() as tmp:
            def child(command,**kwargs):smoke.execute_stages(self.plan,FakeBackend(self.plan,fail='processor'),Path(command[-1]))
            with mock.patch.object(job.subprocess,'run',side_effect=child):
                with self.assertRaises(ValueError):job.run(config,Path(tmp))
            run=Path(tmp)/self.plan['run_id']
            self.assertEqual(json.loads((run/'job_manifest.json').read_text())['status'],'failed')
            self.assertTrue((run/'stage_1_files.json').exists());self.assertTrue((run/'diagnostic.log').is_file())


if __name__=='__main__':unittest.main()
