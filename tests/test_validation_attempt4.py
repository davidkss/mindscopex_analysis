"""Attempt4 authorization rotation tested with local files and no Colab calls."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments import runtime_smoke as smoke
from experiments.jobs import first_divergence_trace as job
from experiments.runners import launch_colab as launch
from experiments.runners import validation_cli_environment as environment
from experiments.runners.config import load_toml
from test_runtime_smoke import FakeBackend


AUTH = smoke.ATTEMPT4_AUTHORIZATION
CONFIG = smoke.ROOT / 'experiments/configs/growth019_validation_only_20261121_attempt4_2b.toml'
ATTEMPT3_CONFIG = smoke.ROOT / 'experiments/configs/growth019_validation_only_20261116_attempt3_2b.toml'
SESSION = 'mindscopex-valonly-20261121-a4'


class Attempt4ValidationTests(unittest.TestCase):
    def pending(self):
        prior = {key: value for key, value in smoke.AUTHORIZATIONS.items() if key != AUTH}
        registry, active = smoke._configure_authorizations(prior, None)
        return mock.patch.multiple(smoke, AUTHORIZATIONS=registry, ACTIVE_AUTHORIZATION=active)

    def activation(self, policy_path, *, pinned_sha=None):
        # Only tests substitute the approved policy. Production requires a source-pinned final SHA.
        with mock.patch.object(smoke, 'ATTEMPT4_POLICY', policy_path):
            registry, active = smoke._configure_authorizations(
                smoke.AUTHORIZATIONS, pinned_sha if pinned_sha is not None else smoke.digest(policy_path)
            )
        return mock.patch.multiple(smoke, AUTHORIZATIONS=registry, ACTIVE_AUTHORIZATION=active)

    def test_pending_config_keeps_prior_active_slot_and_pinned_scope(self):
        self.enterContext(self.pending())
        plan = job.prepare_plan(load_toml(CONFIG))
        prior = smoke.build_plan('validate', smoke.ATTEMPT3_AUTHORIZATION)
        self.assertEqual(smoke.ACTIVE_AUTHORIZATION, smoke.ATTEMPT3_AUTHORIZATION)
        self.assertNotIn(AUTH, smoke.AUTHORIZATIONS)
        self.assertEqual(plan['authorization_id'], AUTH)
        self.assertEqual(plan['run_id'], smoke.ATTEMPT4_RUN_ID)
        self.assertNotEqual(plan['run_id'], prior['run_id'])
        self.assertEqual(plan['authorization_status'], 'pending')
        self.assertIs(plan['execution_ready'], False)
        self.assertIs(plan['allocation_authorized'], False)
        self.assertEqual(plan['measurements'], [])
        self.assertEqual(plan['authorization_sha256'], smoke.digest(smoke.ATTEMPT4_DRAFT))
        policy = plan['authorization_policy']
        self.assertEqual(policy['session'], SESSION)
        self.assertNotEqual(policy['session'], prior['authorization_policy']['session'])
        self.assertEqual(policy['revisions'], prior['authorization_policy']['revisions'])
        self.assertEqual(policy['draft_metadata']['runtime_stages'], list(smoke.STAGES))
        self.assertEqual(policy['draft_metadata']['retirement_on_activation']['usage_status'], 'UNKNOWN')
        self.assertIs(policy['draft_metadata']['gpu_execution_risk_accepted'], False)
        self.assertEqual((plan['expected_generate_calls'], plan['extra_model_forwards'],
                          plan['synthetic_hook_module_forwards']), (0, 0, 2))
        with self.assertRaises(ValueError):
            smoke.build_plan('generate', AUTH)

    def test_pending_blocks_every_execution_entry_without_claim_or_remote_calls(self):
        self.enterContext(self.pending())
        plan = smoke.build_plan('validate', AUTH)
        forged = dict(plan, authorization_status='approved', allocation_authorized=True)
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(launch, 'run_cmd') as remote, \
             mock.patch.object(launch, 'run_cmd_retry') as transfer, \
             mock.patch.object(smoke, 'RuntimeBackend') as backend:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'pending'):
                launch.launch_one(CONFIG, root=root, session=SESSION, keep_session=False,
                                  default_gpu='A100', default_tpu=None, exec_timeout=900,
                                  allocation_attempts=1, allocation_retry_delay=0, dry_run=False)
            with self.assertRaisesRegex(ValueError, 'pending'):
                job.run(CONFIG, root)
            with mock.patch('sys.stderr'):
                with self.assertRaises(SystemExit) as error:
                    smoke.main(['--through', 'validate', '--authorization', AUTH, '--output-dir', tmp])
                self.assertEqual(error.exception.code, 2)
            for candidate in (plan, forged):
                with self.assertRaisesRegex(ValueError, 'pending'):
                    smoke.claim_authorization(candidate, root / 'claims')
            self.assertEqual(list(root.iterdir()), [])
            remote.assert_not_called()
            transfer.assert_not_called()
            backend.assert_not_called()

    def test_activation_keeps_exactly_one_runnable_approval_and_a_separate_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = root / 'approved-test-fixture.json'
            final.write_text(json.dumps(smoke.expected_attempt4_policy('2026-11-21')))
            with self.activation(final):
                self.assertEqual(smoke.ACTIVE_AUTHORIZATION, AUTH)
                runnable = []
                for authorization in smoke.AUTHORIZATIONS:
                    plan = smoke.build_plan('validate', authorization)
                    if authorization == AUTH:
                        smoke.require_allocation_authorized(plan)
                        runnable.append(authorization)
                    else:
                        with self.assertRaisesRegex(ValueError, 'retired'):
                            smoke.claim_authorization(plan, root / 'fresh-ledger')
                self.assertEqual(runnable, [AUTH])
                self.assertFalse((root / 'fresh-ledger').exists())
                with mock.patch.object(launch, 'run_cmd') as remote, \
                     mock.patch.object(smoke, 'RuntimeBackend') as backend:
                    with self.assertRaisesRegex(ValueError, 'retired'):
                        launch.launch_one(ATTEMPT3_CONFIG, root=root,
                                          session='mindscopex-valonly-20261116-a3', keep_session=False,
                                          default_gpu='A100', default_tpu=None, exec_timeout=900,
                                          allocation_attempts=1, allocation_retry_delay=0, dry_run=False)
                    with self.assertRaisesRegex(ValueError, 'retired'):
                        job.run(ATTEMPT3_CONFIG, root / 'remote')
                    with self.assertRaisesRegex(ValueError, 'retired'):
                        smoke.main(['--through', 'validate', '--authorization', smoke.ATTEMPT3_AUTHORIZATION,
                                    '--output-dir', str(root / 'runtime')])
                    remote.assert_not_called()
                    backend.assert_not_called()
                plan = job.prepare_plan(load_toml(CONFIG))
                retirement = plan['authorization_policy']['approval_metadata']['retirement_on_activation']
                self.assertEqual(retirement['authorization_id'], smoke.ATTEMPT3_AUTHORIZATION)
                self.assertEqual(retirement['usage_status'], 'UNKNOWN')
                ledger = root / 'claims'
                ledger.mkdir()
                prior_claim = ledger / f'{smoke.ATTEMPT3_AUTHORIZATION}.json'
                prior_claim.write_text('historical test sentinel')
                claimed = smoke.claim_authorization(plan, ledger, session=SESSION)
                self.assertEqual(claimed, ledger / f'{AUTH}.json')
                self.assertEqual(json.loads(claimed.read_text())['authorization_id'], AUTH)
                with self.assertRaises(FileExistsError):
                    smoke.claim_authorization(plan, ledger, session=SESSION)
                self.assertEqual(prior_claim.read_text(), 'historical test sentinel')
                backend = FakeBackend(plan)
                state = smoke.execute_stages(plan, backend, root / 'validation')
                self.assertEqual(backend.events, list(smoke.STAGES))
                self.assertEqual(state['actual_generate_attempts'], 0)
                self.assertEqual(state['extra_model_forwards'], 0)
                self.assertEqual(state['synthetic_hook_module_forwards'], 2)
                self.assertEqual(state['authorization_id'], AUTH)

    def test_activation_rejects_unpinned_or_altered_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'approved-test-fixture.json'
            original = smoke.expected_attempt4_policy('2026-11-21')
            path.write_text(json.dumps(original))
            with self.activation(path, pinned_sha=smoke.ATTEMPT4_DRAFT_SHA):
                with self.assertRaisesRegex(ValueError, 'policy SHA mismatch'):
                    smoke.build_plan('validate', AUTH)
            for key, value in (('gpu_execution_risk_accepted', False), ('cli_assignment_retries', 1),
                               ('retirement_on_activation', {})):
                with self.subTest(key=key):
                    policy = copy.deepcopy(original)
                    policy['approval_metadata'][key] = value
                    path.write_text(json.dumps(policy))
                    with self.activation(path), self.assertRaisesRegex(ValueError, 'approved scope mismatch'):
                        smoke.build_plan('validate', AUTH)

    def test_activated_launcher_checks_wsl_environment_before_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = root / 'approved-test-fixture.json'
            final.write_text(json.dumps(smoke.expected_attempt4_policy('2026-11-21')))
            with self.activation(final), \
                 mock.patch.object(environment, 'verify_validation_cli_environment',
                                   side_effect=ValueError('environment drift')) as gate, \
                 mock.patch.object(launch, 'ensure_validation_session_once') as remote:
                with self.assertRaisesRegex(ValueError, 'environment drift'):
                    launch.launch_validation_once(
                        prepared=smoke.build_plan('validate', AUTH), root=root, session=SESSION,
                        gpu='A100', local_run_dir=root, local_config=None, remote_config_path=None,
                        remote_source_archive=None, bootstrap=None, exec_timeout=900,
                        remote_output_root=None, name=None)
                gate.assert_called_once()
                remote.assert_not_called()
                self.assertFalse((root / 'results').exists())
