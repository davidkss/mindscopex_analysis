"""One-allocation validation approval tests; no network/GPU calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments import runtime_smoke as smoke
from experiments.jobs import first_divergence_trace as job
from experiments.runners import launch_colab as launch
from experiments.runners.config import load_toml
from test_runtime_smoke import FakeBackend

AUTH='validation-only-20261030'


class ValidationOnlyApprovalTests(unittest.TestCase):
    def setUp(self):self.plan=smoke.build_plan('validate',AUTH)

    def test_config_approval_budget_and_revisions(self):
        config=load_toml(smoke.ROOT/'experiments/configs/growth019_validation_only_2b.toml')
        plan=job.prepare_plan(config);smoke.require_allocation_authorized(plan)
        self.assertEqual(plan['n_unique_measurements'],0);self.assertEqual(plan['measurements'],[])
        self.assertEqual(plan['expected_generate_calls'],0);self.assertEqual(plan['extra_model_forwards'],0)
        self.assertEqual(plan['synthetic_hook_module_forwards'],2);self.assertFalse(plan['execution_ready'])
        self.assertEqual(plan['study_manifest']['revisions']['model'],'b1485b2fa6dfa1287294f269f5fb618e03d52d7c')
        self.assertEqual(plan['study_manifest']['revisions']['sae'],'132ea3697b591df9ee46d738aa1d528e3c6082f7')

    def test_generate_mode_stays_blocked_even_global_switch(self):
        with self.assertRaises(ValueError):smoke.build_plan('generate',AUTH)
        plan=smoke.build_plan('generate');plan['smoke_manifest']['allocation_authorized']=True
        with self.assertRaises(ValueError):smoke.require_allocation_authorized(plan)
        config=load_toml(smoke.ROOT/'experiments/configs/growth019_validation_only_2b.toml')
        config['runtime_smoke']['through']='generate'
        with self.assertRaises(ValueError):job.prepare_plan(config)
        original=smoke.build_plan('generate')
        self.assertFalse(original['allocation_authorized'])

    def test_launcher_rejects_retry_wrong_gpu_or_keep_before_allocation(self):
        path=smoke.ROOT/'experiments/configs/growth019_validation_only_2b.toml'
        args=dict(root=smoke.ROOT,session='mock-absent',keep_session=False,default_gpu='A100',default_tpu=None,exec_timeout=900,allocation_attempts=1,allocation_retry_delay=0,dry_run=False)
        for key,value in [('allocation_attempts',2),('default_gpu','H100'),('keep_session',True)]:
            changed=dict(args);changed[key]=value
            with mock.patch.object(launch,'ensure_validation_session_once') as allocate:
                with self.assertRaises(ValueError):launch.launch_one(path,**changed)
                allocate.assert_not_called()

    def test_single_allocation_failure_never_retries(self):
        absent=SimpleNamespace(returncode=0,stdout='Session not found')
        with mock.patch.object(launch.subprocess,'run',return_value=absent),mock.patch.object(launch,'run_cmd',side_effect=RuntimeError('allocation failed')) as allocate:
            with self.assertRaises(RuntimeError):launch.ensure_validation_session_once('mock','A100')
            allocate.assert_called_once_with(['colab','new','-s','mock','--gpu','A100'])
        for text in ('Session exists Hardware: A100','authentication error'):
            with mock.patch.object(launch.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout=text)),mock.patch.object(launch,'run_cmd') as allocate:
                with self.assertRaises(RuntimeError):launch.ensure_validation_session_once('mock','A100')
                allocate.assert_not_called()

    def test_model_generate_and_forward_are_blocked(self):
        backend=smoke.RuntimeBackend.__new__(smoke.RuntimeBackend)
        backend.plan=self.plan;backend.blocked_generate_attempts=0;backend.blocked_model_forward_attempts=0
        backend.model_object=SimpleNamespace(generate=mock.Mock(),forward=mock.Mock())
        backend.block_model_execution()
        with self.assertRaisesRegex(RuntimeError,'generate forbidden'):backend.model_object.generate()
        with self.assertRaisesRegex(RuntimeError,'forward forbidden'):backend.model_object.forward()
        self.assertEqual(backend.blocked_generate_attempts,1);self.assertEqual(backend.blocked_model_forward_attempts,1)
        with self.assertRaises(RuntimeError):backend.generate(Path('/unused'))

    def test_validation_only_stage_timing_memory_and_zero_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend=FakeBackend(self.plan);backend.metrics=lambda:dict(status='mock',allocated_bytes=0)
            state=smoke.execute_stages(self.plan,backend,tmp)
            self.assertEqual(backend.events,list(smoke.STAGES));self.assertEqual(state['actual_generate_attempts'],0)
            self.assertEqual(state['blocked_model_forward_attempts'],0)
            for item in state['stages']:
                self.assertGreaterEqual(item['elapsed_seconds'],0);self.assertIn('gpu_memory',item)
            self.assertFalse((Path(tmp)/'actual_first_divergence.json').exists())
            self.assertTrue((Path(tmp)/'runtime_validation.json').exists())

    def test_validation_adapter_forwards_only_validate_and_authorization(self):
        config=smoke.ROOT/'experiments/configs/growth019_validation_only_2b.toml'
        with tempfile.TemporaryDirectory() as tmp:
            def child(command,**kwargs):
                self.assertEqual(command[command.index('--through')+1],'validate')
                self.assertEqual(command[command.index('--authorization')+1],AUTH)
                smoke.execute_stages(self.plan,FakeBackend(self.plan),Path(command[-1]))
            with mock.patch.object(job.subprocess,'run',side_effect=child):run=job.run(config,Path(tmp))
            manifest=json.loads((run/'job_manifest.json').read_text())
            self.assertEqual(manifest['status'],'succeeded');self.assertEqual(manifest['equivalence_validation'],'not_run')
            with self.assertRaises(FileExistsError):job.run(config,Path(tmp))

    def test_failed_stage_keeps_partial_artifacts_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend=FakeBackend(self.plan,fail='sae')
            with self.assertRaises(ValueError):smoke.execute_stages(self.plan,backend,tmp)
            self.assertEqual(backend.events,['files','processor','model','sae'])
            status=json.loads((Path(tmp)/'runtime_status.json').read_text())
            self.assertEqual(status['actual_generate_attempts'],0);self.assertEqual(status['failed_stage'],'sae')
            self.assertIn('failed_stage_elapsed_seconds',status)
            self.assertTrue((Path(tmp)/'stage_3_model.json').exists());self.assertTrue((Path(tmp)/'runtime_failure.log').exists())


if __name__=='__main__':unittest.main()
