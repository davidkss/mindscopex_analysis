"""CPU-only pending draft boundary checks. Never enable an actual approval."""
import copy
import json
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments import runtime_smoke as smoke
from experiments.jobs import first_divergence_trace as job
from experiments.runners import launch_colab as launch
from experiments.runners.config import load_toml

CONFIG = smoke.ROOT/'experiments/configs/growth019_validation_only_20261114_attempt2_2b.toml'
SESSION = 'mindscopex-valonly-20261114-a2'


class PendingValidationDraftTests(unittest.TestCase):
    def setUp(self):
        platform.platform()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.config = Path(tmp.name)/'draft.toml'
        self.config.write_text(CONFIG.read_text().replace(
            'authorization = "validation-only-20261114-attempt2"',
            'authorization = "validation-only-20261114-attempt2"\npolicy = "draft"'))
        self.plan = job.prepare_plan(load_toml(self.config))
        self.args = dict(root=smoke.ROOT, session=SESSION, keep_session=False,
                         default_gpu='A100', default_tpu=None, exec_timeout=900,
                         allocation_attempts=1, allocation_retry_delay=0)

    def assert_active_slot(self):
        attempt4_pinned = smoke.ATTEMPT4_APPROVED_SHA is not None
        expected = smoke.ATTEMPT4_AUTHORIZATION if attempt4_pinned else smoke.ATTEMPT3_AUTHORIZATION
        self.assertEqual(smoke.ACTIVE_AUTHORIZATION, expected)
        self.assertEqual(smoke.ATTEMPT4_AUTHORIZATION in smoke.AUTHORIZATIONS, attempt4_pinned)

    def test_pending_schema_budget_revisions_and_inactive_identity(self):
        policy = self.plan['authorization_policy']
        self.assert_active_slot()
        self.assertNotEqual(smoke.AUTHORIZATIONS[smoke.ATTEMPT2_AUTHORIZATION][0], smoke.DRAFT_POLICY)
        self.assertEqual(policy['authorization_status'], 'pending')
        self.assertFalse(policy['allocation_authorized'])
        self.assertFalse(self.plan['allocation_authorized'])
        self.assertFalse(self.plan['execution_ready'])
        self.assertEqual(self.plan['run_id'], smoke.DRAFT_RUN_ID)
        self.assertEqual(self.plan['expected_generate_calls'], 0)
        self.assertEqual(self.plan['extra_model_forwards'], 0)
        self.assertEqual(self.plan['synthetic_hook_module_forwards'], 2)
        self.assertEqual(policy['max_allocation_attempts'], 1)
        self.assertEqual(policy['automatic_execution_retries'], 0)
        self.assertTrue(policy['automatic_session_cleanup'])
        self.assertEqual(policy['execution_actor'], 'user_wsl')
        self.assertEqual(policy['revisions'], dict(
            model='b1485b2fa6dfa1287294f269f5fb618e03d52d7c',
            tokenizer='b1485b2fa6dfa1287294f269f5fb618e03d52d7c',
            sae='132ea3697b591df9ee46d738aa1d528e3c6082f7'))
        self.assertEqual(self.plan['authorization_sha256'], smoke.digest(smoke.DRAFT_POLICY))
        self.assertEqual(self.plan['study_manifest']['layer'], 5)
        self.assertEqual([(c['feature_id'], c['coefficient'])
                          for c in self.plan['smoke_manifest']['conditions']],
                         [(None, 0), (30908, 8.5), (12207, 8)])

    def test_all_real_entry_points_block_without_claim_or_remote_calls(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(launch, 'run_cmd') as command, \
             mock.patch.object(launch, 'run_cmd_retry') as transfer, \
             mock.patch.object(launch.subprocess, 'run') as remote, \
             mock.patch.object(smoke, 'RuntimeBackend') as backend:
            args = dict(self.args, root=Path(tmp))
            with self.assertRaises(ValueError):
                launch.launch_one(self.config, dry_run=False, **args)
            with self.assertRaises(ValueError):
                job.run(self.config, Path(tmp))
            with self.assertRaises(ValueError):
                smoke.require_allocation_authorized(self.plan)
            with self.assertRaises(ValueError):
                smoke.claim_authorization(self.plan, Path(tmp)/'claims')
            # Substituting Draft bytes for the pinned final policy must fail before the Backend.
            final = smoke.AUTHORIZATIONS[smoke.ATTEMPT2_AUTHORIZATION]
            with mock.patch.dict(smoke.AUTHORIZATIONS, {
                    smoke.ATTEMPT2_AUTHORIZATION: (smoke.DRAFT_POLICY, final[1], final[2])}):
                with self.assertRaises(ValueError):
                    smoke.main(['--through', 'validate', '--authorization', smoke.DRAFT_AUTHORIZATION,
                                '--output-dir', str(Path(tmp)/'runtime')])
            command.assert_not_called()
            transfer.assert_not_called()
            remote.assert_not_called()
            backend.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_old_approval_retired_even_with_new_or_absent_ledger(self):
        old = smoke.build_plan('validate', 'validation-only-20261114')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                smoke.claim_authorization(old, Path(tmp)/'fresh-ledger')
            with self.assertRaises(ValueError):
                smoke.require_allocation_authorized(old)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_all_consumed_approvals_block_every_execution_entry_without_ledger(self):
        self.assert_active_slot()
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(launch, 'run_cmd') as command, \
             mock.patch.object(launch, 'run_cmd_retry') as transfer, \
             mock.patch.object(launch.subprocess, 'run') as remote, \
             mock.patch.object(smoke, 'RuntimeBackend') as backend:
            for auth in (a for a in smoke.AUTHORIZATIONS if a != smoke.ACTIVE_AUTHORIZATION):
                plan = smoke.build_plan('validate', auth)
                with self.subTest(auth=auth), self.assertRaises(ValueError):
                    smoke.claim_authorization(plan, Path(tmp)/'claims')
                with self.subTest(auth=auth), self.assertRaises(ValueError):
                    smoke.main(['--through', 'validate', '--authorization', auth,
                                '--output-dir', str(Path(tmp)/'runtime')])
            with self.assertRaises(ValueError):
                launch.launch_one(CONFIG, dry_run=False, **dict(self.args, root=Path(tmp)))
            with self.assertRaises(ValueError):
                job.run(CONFIG, Path(tmp))
            command.assert_not_called()
            transfer.assert_not_called()
            remote.assert_not_called()
            backend.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_policy_changes_and_extra_fields_rejected(self):
        original = self.plan['authorization_policy']
        changes = [('allocation_authorized', True), ('authorization_status', 'approved'),
                   ('execution_ready', True), ('gpu', 'H100'), ('max_allocation_attempts', 2),
                   ('automatic_execution_retries', 1), ('expected_generate_calls', 6),
                   ('expected_generate_calls', False), ('extra_model_forwards', 1),
                   ('synthetic_hook_module_forwards', 3), ('run_id', 'old'),
                   ('session', 'other'), ('execution_actor', 'codex'),
                   ('revisions', {}), ('unexpected', True)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'draft.json'
            for key, value in changes:
                with self.subTest(key=key):
                    policy = dict(original, **{key: value})
                    path.write_text(json.dumps(policy))
                    with mock.patch.object(smoke, 'DRAFT_POLICY', path):
                        with self.assertRaisesRegex(ValueError, 'scope mismatch'):
                            smoke.build_plan('validate', smoke.DRAFT_AUTHORIZATION, draft=True)

    def test_wrong_launcher_options_run_id_and_generation_blocked(self):
        for key, value in [('session', 'other'), ('default_gpu', 'H100'),
                           ('allocation_attempts', 2), ('allocation_retry_delay', 1),
                           ('keep_session', True), ('default_tpu', 'v5e')]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                args = dict(self.args, root=Path(tmp), **{key: value})
                with self.assertRaises(ValueError):
                    launch.launch_one(self.config, dry_run=True, **args)
                self.assertEqual(list(Path(tmp).iterdir()), [])
        config = load_toml(self.config)
        config['run']['name'] = 'wrong-run'
        with self.assertRaises(ValueError):
            job.prepare_plan(config)
        with self.assertRaises(ValueError):
            smoke.build_plan('generate', smoke.DRAFT_AUTHORIZATION)
        forged = copy.deepcopy(self.plan)
        forged.update(allocation_authorized=True, authorization_status='approved')
        with self.assertRaises(ValueError):
            smoke.require_allocation_authorized(forged)

    def test_dry_run_has_explicit_pending_plan_and_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(launch, 'make_source_archive',
                               side_effect=lambda root, path: path.write_bytes(b'archive')), \
             mock.patch.object(job, 'include_reference_inputs'), \
             mock.patch.object(launch, 'git_value', return_value='mock'), \
             mock.patch.object(launch, 'run_cmd') as command, \
             mock.patch.object(launch, 'run_cmd_retry') as transfer, \
             mock.patch.object(launch.subprocess, 'run') as remote:
            args = dict(self.args, root=Path(tmp))
            run = launch.launch_one(self.config, dry_run=True, **args)
            manifest = json.loads((run/'launcher_manifest.json').read_text())
            plan = manifest['runtime_plan']
            self.assertEqual(plan['authorization_status'], 'pending')
            self.assertFalse(plan['allocation_authorized'])
            self.assertEqual(plan['authorization_id'], smoke.DRAFT_AUTHORIZATION)
            self.assertFalse((Path(tmp)/'results/.authorization_claims').exists())
            command.assert_not_called()
            transfer.assert_not_called()
            remote.assert_not_called()

    def test_allocation_error_classification_keeps_exact_output(self):
        for output, category in [('HTTP 503 Service Unavailable', 'http_503'),
                                 ('ColabRequestError: Service Unavailable\n', 'service_unavailable'),
                                 ('service unavailable', 'service_unavailable'),
                                 ('HTTP 503', 'http_503'),
                                 ('SSLCertVerificationError', 'tls'),
                                 ('Authentication failed', 'authentication'),
                                 ('Unknown response', 'unknown')]:
            with self.subTest(category=category), tempfile.TemporaryDirectory() as tmp, \
                 mock.patch.object(launch.subprocess, 'run', return_value=(
                     SimpleNamespace(returncode=0, stdout='Session not found'))), \
                 mock.patch.object(launch, 'run_cmd', side_effect=(
                     subprocess.CalledProcessError(1, ['colab', 'new'], output=output))) as new:
                with self.assertRaises(subprocess.CalledProcessError):
                    launch.ensure_validation_session_once(SESSION, 'A100', report_dir=Path(tmp))
                record = json.loads((Path(tmp)/'allocation_attempt.json').read_text())
                self.assertEqual(record['category'], category)
                self.assertEqual(record['server_assignment_status'], 'unverified')
                self.assertEqual((Path(tmp)/'allocation_output.log').read_text(), output)
                self.assertEqual(new.call_count, 1)
