"""Approved Option B boundaries; no CLI, GPU, or credential access."""
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
from experiments.runners import validation_cli_environment as environment
from experiments.runners.config import load_toml

AUTH = 'validation-only-20261116-attempt3'
CONFIG = smoke.ROOT/'experiments/configs/growth019_validation_only_20261116_attempt3_2b.toml'
SESSION = 'mindscopex-valonly-20261116-a3'


class Attempt3ApprovalTests(unittest.TestCase):
    def test_exact_policy_risk_acceptance_and_only_active_approval(self):
        self.assertEqual(smoke.ACTIVE_AUTHORIZATION, AUTH)
        plan = job.prepare_plan(load_toml(CONFIG))
        smoke.require_allocation_authorized(plan)
        policy = plan['authorization_policy']
        self.assertEqual(policy, smoke.expected_attempt3_policy())
        self.assertEqual(plan['authorization_sha256'], smoke.AUTHORIZATIONS[AUTH][2])
        self.assertEqual((plan['expected_generate_calls'], plan['extra_model_forwards'],
                          plan['synthetic_hook_module_forwards']), (0, 0, 2))
        self.assertFalse(plan['execution_ready'])
        for old in smoke.AUTHORIZATIONS:
            if old != AUTH:
                with self.assertRaises(ValueError):
                    smoke.require_allocation_authorized(smoke.build_plan('validate', old))

    def test_policy_tampering_rejected_even_with_substituted_registry_sha(self):
        original = smoke.expected_attempt3_policy()
        final = smoke.AUTHORIZATIONS[AUTH]
        for key, value in [('session', 'wrong'), ('authorization_id', 'wrong'),
                           ('run_id', 'wrong'), ('execution_actor', 'codex'),
                           ('expected_generate_calls', 1), ('max_allocation_attempts', True),
                           ('approval_metadata', {}), ('revisions', {}),
                           ('authorization_status', 'pending')]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                altered = copy.deepcopy(original)
                altered[key] = value
                path = Path(tmp)/'policy.json'
                path.write_text(json.dumps(altered))
                with mock.patch.dict(smoke.AUTHORIZATIONS, {
                    AUTH: (path, final[1], smoke.digest(path))}):
                    with self.assertRaises(ValueError):
                        smoke.build_plan('validate', AUTH)
        draft = json.loads(smoke.ATTEMPT3_DRAFT.read_text())
        with self.assertRaises(ValueError):
            smoke.claim_authorization(draft, Path('/unused'))

    def test_bad_config_or_budget_never_calls_remote_or_creates_claim(self):
        args = dict(session=SESSION, keep_session=False, default_gpu='A100',
                    default_tpu=None, exec_timeout=900, allocation_attempts=1,
                    allocation_retry_delay=0, dry_run=False)
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(launch.subprocess, 'run') as remote:
            for key, value in [('session', 'other'), ('keep_session', True),
                               ('default_gpu', 'H100'), ('allocation_attempts', 2),
                               ('allocation_retry_delay', 1)]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    launch.launch_one(CONFIG, root=Path(tmp), **dict(args, **{key: value}))
            config = load_toml(CONFIG)
            config['runtime_smoke']['through'] = 'generate'
            with self.assertRaises(ValueError):
                job.prepare_plan(config)
            remote.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_environment_failure_precedes_claim_and_any_cli(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(environment, 'verify_validation_cli_environment',
                               side_effect=ValueError('environment drift')), \
             mock.patch.object(launch, 'run_cmd') as remote:
            with self.assertRaisesRegex(ValueError, 'environment drift'):
                launch.launch_validation_once(
                    prepared=smoke.build_plan('validate', AUTH), root=Path(tmp),
                    session=SESSION, gpu='A100', local_run_dir=Path(tmp),
                    local_config=None, remote_config_path=None, remote_source_archive=None,
                    bootstrap=None, exec_timeout=900, remote_output_root=None, name=None)
            remote.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_environment_gate_rejects_drift_and_preserves_parent_settings(self):
        with mock.patch.object(environment.platform, 'release', return_value='microsoft-WSL2'), \
             mock.patch.dict(environment.os.environ, {'WSL_DISTRO_NAME': 'test'}, clear=True), \
             mock.patch.object(environment.shutil, 'which', return_value=str(environment.CLI)), \
             mock.patch.object(Path, 'read_text', return_value='#!'+environment.PYTHON), \
             mock.patch.object(environment.subprocess, 'run') as probe:
            probe.return_value = SimpleNamespace(stdout=json.dumps({
                'versions': environment.VERSIONS, 'files': environment.FINGERPRINTS}))
            environment.verify_validation_cli_environment()
            probe.return_value = SimpleNamespace(stdout='{}')
            with self.assertRaises(ValueError):
                environment.verify_validation_cli_environment()
            with mock.patch.dict(environment.os.environ, {'PYTHONPATH': '/unverified'}):
                with self.assertRaises(ValueError):
                    environment.verify_validation_cli_environment()
