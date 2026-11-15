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

AUTH='validation-only-20261114-attempt2'


class ValidationOnlyApprovalTests(unittest.TestCase):
    def setUp(self):
        self.plan=smoke.build_plan('validate',AUTH)

    def test_final_status_fingerprint_and_default_route(self):
        self.assertEqual(smoke.ACTIVE_AUTHORIZATION, AUTH)
        self.assertEqual(self.plan['authorization_status'], 'approved')
        self.assertTrue(self.plan['allocation_authorized'])
        self.assertFalse(self.plan['execution_ready'])
        policy_path = smoke.AUTHORIZATIONS[AUTH][0]
        self.assertNotEqual(policy_path, smoke.DRAFT_POLICY)
        self.assertEqual(self.plan['authorization_sha256'], smoke.digest(policy_path))
        self.assertNotEqual(smoke.digest(policy_path), smoke.digest(smoke.DRAFT_POLICY))
        draft = smoke.build_plan('validate', AUTH, draft=True)
        self.assertEqual(draft['authorization_status'], 'pending')
        with self.assertRaises(ValueError):
            smoke.require_allocation_authorized(draft)
        for field, value in [('authorization_status', 'pending'),
                             ('authorization_status', None), ('allocation_authorized', False)]:
            altered = copy.deepcopy(self.plan)
            altered[field] = value
            with self.assertRaises(ValueError):
                smoke.require_allocation_authorized(altered)

    def test_final_policy_approved_fields_checked_in_addition_to_sha(self):
        final = smoke.AUTHORIZATIONS[AUTH]
        for field, value in [('authorization_status', 'pending'), ('allocation_authorized', False),
                             ('session', 'other'), ('gpu', 'H100'), ('max_allocation_attempts', 2),
                             ('revisions', {}), ('expected_generate_calls', 1)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                altered = dict(self.plan['authorization_policy'], **{field: value})
                path = Path(tmp)/'policy.json'
                path.write_text(json.dumps(altered))
                with mock.patch.dict(smoke.AUTHORIZATIONS,
                                     {AUTH: (path, final[1], smoke.digest(path))}):
                    with self.assertRaises(ValueError):
                        smoke.build_plan('validate', AUTH)

    def test_both_historical_approvals_stay_blocked(self):
        for old in ('validation-only-20261030', 'validation-only-20261114'):
            with self.subTest(old=old):
                plan = smoke.build_plan('validate', old)
                with self.assertRaises(ValueError):
                    smoke.require_allocation_authorized(plan)

    def test_session_or_gpu_mismatch_cannot_claim_or_allocate(self):
        for session, gpu in [('other', 'A100'), ('mindscopex-valonly-20261114-a2', 'H100')]:
            with self.subTest(session=session, gpu=gpu), tempfile.TemporaryDirectory() as tmp:
                with mock.patch.object(launch, 'ensure_validation_session_once') as allocate:
                    with self.assertRaises(ValueError):
                        launch.launch_validation_once(
                            prepared=self.plan, root=Path(tmp), session=session, gpu=gpu,
                            local_run_dir=Path(tmp), local_config=Path(tmp)/'unused',
                            remote_config_path='unused', remote_source_archive='unused',
                            bootstrap=Path(tmp)/'unused', exec_timeout=900,
                            remote_output_root='unused', name=self.plan['run_id'])
                    allocate.assert_not_called()
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_config_approval_budget_and_revisions(self):
        config=load_toml(smoke.ROOT/'experiments/configs/growth019_validation_only_20261114_attempt2_2b.toml')
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
        config=load_toml(smoke.ROOT/'experiments/configs/growth019_validation_only_20261114_attempt2_2b.toml')
        config['runtime_smoke']['through']='generate'
        with self.assertRaises(ValueError):job.prepare_plan(config)
        original=smoke.build_plan('generate')
        self.assertFalse(original['allocation_authorized'])

    def test_launcher_rejects_retry_wrong_gpu_or_keep_before_allocation(self):
        path=smoke.ROOT/'experiments/configs/growth019_validation_only_20261114_attempt2_2b.toml'
        args=dict(root=smoke.ROOT,session='mock-absent',keep_session=False,default_gpu='A100',default_tpu=None,exec_timeout=900,allocation_attempts=1,allocation_retry_delay=0,dry_run=False)
        for key,value in [('allocation_attempts',2),('default_gpu','H100'),('keep_session',True),('allocation_retry_delay',1),('default_tpu','v5e'),('session','other')]:
            changed=dict(args);changed[key]=value
            with mock.patch.object(launch,'ensure_validation_session_once') as allocate:
                with self.assertRaises(ValueError):launch.launch_one(path,**changed)
                allocate.assert_not_called()

    def test_single_allocation_failure_never_retries(self):
        absent=SimpleNamespace(returncode=0,stdout='Session not found')
        with mock.patch.object(launch.subprocess,'run',return_value=absent),mock.patch.object(launch,'run_cmd',side_effect=RuntimeError('allocation failed')) as allocate:
            with self.assertRaises(RuntimeError):launch.ensure_validation_session_once('mock','A100')
            allocate.assert_called_once_with(['colab','new','-s','mock','--gpu','A100'])
        for text in ('Session exists Hardware: A100', 'authentication error',
                     'TLS certificate file not found', 'authentication error: Session not found'):
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
        config=smoke.ROOT/'experiments/configs/growth019_validation_only_20261114_attempt2_2b.toml'
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

    def test_retired_approval_and_unknown_policy_blocked(self):
        old = smoke.build_plan('validate', 'validation-only-20261030')
        self.assertNotEqual(old['run_id'], self.plan['run_id'])
        self.assertNotEqual(old['authorization_sha256'], self.plan['authorization_sha256'])
        with self.assertRaisesRegex(ValueError, 'historical'):
            smoke.require_allocation_authorized(old)
        with self.assertRaises(ValueError):
            smoke.build_plan('validate', '/tmp/unapproved.json')
        config = smoke.ROOT/'experiments/configs/growth019_validation_only_2b.toml'
        with mock.patch.object(launch, 'run_cmd') as remote:
            with self.assertRaises(ValueError):
                launch.launch_one(config, root=smoke.ROOT, session='unused',
                                  keep_session=False, default_gpu='A100', default_tpu=None,
                                  exec_timeout=900, allocation_attempts=1,
                                  allocation_retry_delay=0, dry_run=False)
            remote.assert_not_called()

    def test_policy_sha_and_plan_tampering_blocked(self):
        path = smoke.AUTHORIZATIONS[AUTH][0]
        self.assertEqual(self.plan['authorization_sha256'], smoke.digest(path))
        self.assertEqual(self.plan['authorization_policy']['revisions'],
                         self.plan['study_manifest']['revisions'])
        with mock.patch.object(smoke, 'digest', side_effect=lambda p: 'bad' if p == path
                               else __import__('hashlib').sha256(Path(p).read_bytes()).hexdigest()):
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                smoke.build_plan('validate', AUTH)
        for key, value in [('expected_generate_calls', 6), ('extra_model_forwards', 1),
                           ('synthetic_hook_module_forwards', 3), ('run_id', 'old'),
                           ('authorization_sha256', 'bad'), ('automatic_generation_retries', 1)]:
            with self.subTest(key=key):
                plan = copy.deepcopy(self.plan)
                plan[key] = value
                with self.assertRaises(ValueError):
                    smoke.require_allocation_authorized(plan)
        state = dict(status='validated', smoke_manifest_sha256=self.plan['smoke_manifest_sha256'],
                     revisions=self.plan['study_manifest']['revisions'],
                     stages=[dict(stage=s, status='passed') for s in smoke.STAGES],
                     actual_generate_attempts=0, extra_model_forwards=0,
                     synthetic_hook_module_forwards=2)
        with self.assertRaisesRegex(ValueError, 'authorization evidence'):
            smoke.validate_summary(state, self.plan)

    def test_dry_run_then_single_consumption_even_after_failure(self):
        config = smoke.ROOT/'experiments/configs/growth019_validation_only_20261114_attempt2_2b.toml'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = dict(root=root, session='mindscopex-valonly-20261114-a2', keep_session=False, default_gpu='A100',
                        default_tpu=None, exec_timeout=900, allocation_attempts=1,
                        allocation_retry_delay=0)
            with mock.patch.object(launch, 'make_source_archive',
                                   side_effect=lambda r, p: p.write_bytes(b'fake archive')), \
                 mock.patch.object(job, 'include_reference_inputs'), \
                 mock.patch.object(launch, 'git_value', return_value='mock'), \
                 mock.patch.object(launch, 'timestamp', side_effect=['dry', 'first', 'second']), \
                 mock.patch.object(launch, 'ensure_validation_session_once',
                                   side_effect=RuntimeError('allocation failed')) as allocate, \
                 mock.patch.object(launch, 'run_cmd') as remote:
                run = launch.launch_one(config, dry_run=True, **args)
                manifest = json.loads((run/'launcher_manifest.json').read_text())
                self.assertEqual(manifest['runtime_plan']['authorization_id'], AUTH)
                self.assertEqual(manifest['runtime_plan']['authorization_sha256'],
                                 self.plan['authorization_sha256'])
                self.assertFalse((root/'results/.authorization_claims').exists())
                allocate.assert_not_called()
                remote.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, 'allocation failed'):
                    launch.launch_one(config, dry_run=False, **args)
                with self.assertRaises(FileExistsError):
                    launch.launch_one(config, dry_run=False, **args)
                allocate.assert_called_once()
                remote.assert_not_called()
                claim = root/'results/.authorization_claims'/f'{AUTH}.json'
                self.assertEqual(json.loads(claim.read_text())['authorization_id'], AUTH)

    def test_runtime_claim_prevents_second_output_before_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(smoke, 'ROOT', Path(tmp)), \
                 mock.patch.object(smoke, 'build_plan', return_value=self.plan), \
                 mock.patch.object(smoke, 'RuntimeBackend', side_effect=RuntimeError('mock load failed')) as backend:
                for output in ('first', 'second'):
                    args = ['--through', 'validate', '--authorization', AUTH,
                            '--output-dir', str(Path(tmp)/output)]
                    if output == 'first':
                        with self.assertRaisesRegex(RuntimeError, 'mock load failed'):
                            smoke.main(args)
                    else:
                        with self.assertRaises(FileExistsError):
                            smoke.main(args)
                backend.assert_called_once()

    def test_cli_preflight_exception_does_not_stop_unowned_session(self):
        config = smoke.ROOT/'experiments/configs/growth019_validation_only_20261114_attempt2_2b.toml'
        argv = ['launch', str(config), '--session', 'mock', '--gpu', 'A100',
                '--allocation-attempts', '1', '--allocation-retry-delay', '0']
        with mock.patch('sys.argv', argv), \
             mock.patch.object(launch, 'check_colab_cli'), \
             mock.patch.object(launch, 'launch_one', side_effect=RuntimeError('mock failure')), \
             mock.patch.object(launch, 'run_cmd') as remote:
            with self.assertRaisesRegex(RuntimeError, 'mock failure'):
                launch.main()
            remote.assert_not_called()

    def test_actual_hook_count_mismatch_cannot_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = FakeBackend(self.plan)
            backend.synthetic_hook_module_forward_attempts = 3
            with self.assertRaisesRegex(ValueError, 'budget mismatch'):
                smoke.execute_stages(self.plan, backend, tmp)
            status = json.loads((Path(tmp)/'runtime_status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(status['synthetic_hook_module_forwards'], 3)

    def test_bootstrap_preserves_existing_source_and_consumes_before_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root/'source'
            repo.mkdir()
            original = repo/'preserved.txt'
            original.write_text('research evidence')
            config, archive = root/'config.toml', root/'source.zip'
            config.write_text('config')
            archive.write_bytes(b'archive')
            payload = dict(validation_authorization=AUTH, source='archive',
                           remote_repo_dir=str(repo), remote_config_path=str(config),
                           remote_source_archive=str(archive),
                           config_sha256=smoke.digest(config),
                           source_archive_sha256=smoke.digest(archive))
            code = launch.render_bootstrap(payload).replace(
                '/content/mindscopex_authorizations', str(root/'ledger'))
            namespace = {}
            exec(compile(code.rsplit('main()', 1)[0], '<bootstrap>', 'exec'), namespace)
            with mock.patch.object(namespace['shutil'], 'rmtree') as delete:
                with self.assertRaisesRegex(FileExistsError, 'no automatic cleanup'):
                    namespace['prepare_source']()
                delete.assert_not_called()
            self.assertEqual(original.read_text(), 'research evidence')
            with mock.patch.dict(namespace, prepare_source=mock.Mock(side_effect=RuntimeError('mock source failure'))):
                with self.assertRaisesRegex(RuntimeError, 'mock source failure'):
                    namespace['main']()
                with self.assertRaises(FileExistsError):
                    namespace['main']()
            self.assertTrue((root/'ledger'/f'{AUTH}.json').exists())

    def test_bootstrap_blocks_changed_upload_before_source_or_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root/'config.toml'
            config.write_text('changed')
            payload = dict(validation_authorization=AUTH, remote_config_path=str(config),
                           remote_source_archive=str(root/'unused.zip'), config_sha256='wrong')
            code = launch.render_bootstrap(payload).replace(
                '/content/mindscopex_authorizations', str(root/'ledger'))
            namespace = {}
            exec(compile(code.rsplit('main()', 1)[0], '<bootstrap>', 'exec'), namespace)
            prepare = mock.Mock()
            with mock.patch.dict(namespace, prepare_source=prepare):
                with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                    namespace['main']()
                prepare.assert_not_called()


if __name__=='__main__':unittest.main()
