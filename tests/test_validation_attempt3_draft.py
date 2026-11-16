"""Independent Attempt3 CPU draft validation; never register or activate a policy."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import runtime_smoke as smoke
from experiments.jobs import first_divergence_trace as job
from experiments.runners import launch_colab as launch

AUTH = 'validation-only-20261116-attempt3'
RUN = 'growth019_pinned_validation_only_20261116_attempt3_2b'
SESSION = 'mindscopex-valonly-20261116-a3'
DRAFT = (smoke.ROOT/'experiments/inputs'/
         'growth019_validation_only_authorization_20261116_attempt3_draft.json')
PROPOSED_SCOPE = ('CPU draft only: Option B logical allocation; separate explicit GPU approval '
                  'and risk acceptance required')


class Attempt3DraftTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads(DRAFT.read_text())

    def test_core_schema_and_revisions_match_preserved_attempt2_scope(self):
        expected = json.loads(smoke.DRAFT_POLICY.read_text())
        expected.update(authorization_id=AUTH, run_id=RUN, session=SESSION,
                        proposed_scope=PROPOSED_SCOPE)
        core = {k: v for k, v in self.policy.items() if k != 'draft_metadata'}
        # Canonical comparison rejects bools in place of integer budgets as well.
        self.assertEqual(json.dumps(core, sort_keys=True), json.dumps(expected, sort_keys=True))
        self.assertEqual(self.policy['authorization_status'], 'pending')
        self.assertIs(self.policy['allocation_authorized'], False)
        self.assertIs(self.policy['execution_ready'], False)
        self.assertEqual(self.policy['expected_generate_calls'], 0)
        self.assertEqual(self.policy['extra_model_forwards'], 0)
        self.assertEqual(self.policy['synthetic_hook_module_forwards'], 2)
        self.assertEqual(self.policy['revisions'], {
            'model': 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c',
            'tokenizer': 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c',
            'sae': '132ea3697b591df9ee46d738aa1d528e3c6082f7'})

    def test_logical_attempt_limits_do_not_claim_post_or_server_guarantees(self):
        expected = {
            'policy_option': 'B_logical_allocation', 'cpu_preparation_only': True,
            'logical_allocation_attempt_limit': 1, 'colab_new_process_limit': 1,
            'launcher_automatic_reallocations': 0, 'cli_assignment_retries': 0,
            'cli_retry_backoff': 0, 'strict_http_post_limit': 'not_guaranteed',
            'authentication_post_replay_possible': True, 'redirect_post_replay_possible': True,
            'server_side_single_allocation_guaranteed': False,
            'risk_acceptance_status': 'pending_explicit_user_approval',
            'gpu_execution_risk_accepted': False, 'gpu_execution_status': 'NO-GO',
            'strict_single_post_previous_verdict': 'NO-GO',
            'runtime_stages': list(smoke.STAGES),
            'model_sae_weights_download_approx_gb': 5.085,
            'auxiliary_file_download_possible': True,
        }
        metadata = {k: v for k, v in self.policy['draft_metadata'].items()
                    if k != 'failure_policy'}
        self.assertEqual(json.dumps(metadata, sort_keys=True), json.dumps(expected, sort_keys=True))

    def test_uncertain_failure_policy_requires_cleanup_without_new_allocation(self):
        expected = {
            'verify_authorization_and_environment_before_allocation': True,
            'reject_existing_or_uncertain_session_before_allocation': True,
            'consume_single_use_claim_before_allocation_attempt': True,
            'automatic_new_allocation_after_failure_or_uncertainty': False,
            'authentication_or_redirect_post_replay_may_occur': True,
            'server_processing_after_401_307_308': 'unverified',
            'query_session_status_after_uncertain_allocation': True,
            'attempt_artifact_recovery': True, 'attempt_session_stop': True,
            'stop_command_success_proves_server_termination': False,
            'user_wsl_verifies_server_sessions_and_compute_units': True,
            'claim_deletion_or_reuse_permitted': False,
        }
        self.assertEqual(self.policy['draft_metadata']['failure_policy'], expected)

    def test_attempt3_is_unknown_to_every_real_execution_entry(self):
        self.assertEqual(smoke.ACTIVE_AUTHORIZATION, 'validation-only-20261116-attempt3')
        self.assertIn(AUTH, smoke.AUTHORIZATIONS)
        final = smoke.AUTHORIZATIONS[AUTH]
        with mock.patch.dict(smoke.AUTHORIZATIONS, {AUTH: (DRAFT, final[1], final[2])}), \
             tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(launch, 'run_cmd') as command, \
             mock.patch.object(launch, 'run_cmd_retry') as transfer, \
             mock.patch.object(launch.subprocess, 'run') as remote, \
             mock.patch.object(smoke, 'RuntimeBackend') as backend:
            base = Path(tmp)
            config = base/'test-only.toml'
            old_config = (smoke.ROOT/'experiments/configs'/
                          'growth019_validation_only_20261114_attempt2_2b.toml')
            config.write_text(old_config.read_text().replace(smoke.ATTEMPT2_AUTHORIZATION, AUTH)
                              .replace(smoke.DRAFT_RUN_ID, RUN))
            root = base/'execution-root'
            args = dict(root=root, session=SESSION, keep_session=False, default_gpu='A100',
                        default_tpu=None, exec_timeout=900, allocation_attempts=1,
                        allocation_retry_delay=0, dry_run=False)
            with self.assertRaises(ValueError):
                launch.launch_one(config, **args)
            with self.assertRaises(ValueError):
                job.run(config, root)
            for through in ('validate', 'generate'):
                with self.subTest(through=through), self.assertRaises(ValueError):
                    smoke.build_plan(through, AUTH)
            # The old CPU Draft route is also intentionally not extended to Attempt3.
            with self.assertRaises(ValueError):
                smoke.build_plan('validate', AUTH, draft=True)
            plan = dict(authorization_id=AUTH, authorization_status='pending',
                        authorization_policy=self.policy, allocation_authorized=False)
            for forged in (False, True):
                candidate = copy.deepcopy(plan)
                if forged:
                    candidate.update(authorization_status='approved', allocation_authorized=True)
                with self.subTest(forged=forged), self.assertRaises(ValueError):
                    smoke.claim_authorization(candidate, root/'claims')
            self.assertFalse(root.exists())
            command.assert_not_called()
            transfer.assert_not_called()
            remote.assert_not_called()
            backend.assert_not_called()

    def test_retired_approvals_and_attempt2_draft_remain_unchanged(self):
        self.assertEqual(smoke.ACTIVE_AUTHORIZATION, 'validation-only-20261116-attempt3')
        for auth in (a for a in smoke.AUTHORIZATIONS if a != smoke.ACTIVE_AUTHORIZATION):
            with self.subTest(auth=auth), self.assertRaises(ValueError):
                smoke.require_allocation_authorized(smoke.build_plan('validate', auth))
        draft = smoke.build_plan('validate', smoke.DRAFT_AUTHORIZATION, draft=True)
        self.assertEqual(draft['authorization_id'], 'validation-only-20261114-attempt2')
        self.assertEqual(draft['authorization_status'], 'pending')
        self.assertIs(draft['allocation_authorized'], False)
        self.assertEqual(smoke.digest(smoke.DRAFT_POLICY),
                         '640d2485e0a1536a78aa5ac91d98e467ab93f7015246cf931ffea999e33cc096')
        self.assertEqual(smoke.digest(smoke.AUTHORIZATIONS[smoke.ATTEMPT2_AUTHORIZATION][0]),
                         'd0d42c5456226eb86c29dcea8b188b1c77f2a3cbb9bc37330e6837b06e3e39d1')
