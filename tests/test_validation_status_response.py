"""Regression strings from WSL / Colab CLI 0.7.4; no CLI execution."""
import unittest
from types import SimpleNamespace
from unittest import mock

from experiments.runners import launch_colab as launch

SESSION = 'mindscopex-valonly-20261114'
ABSENT = "[colab] Session 'mindscopex-valonly-20261114' not found."
UPGRADE = (
    '\n[colab] A new version of Colab CLI is available: 0.7.5 (current: 0.7.4)\n'
    "[colab] You can run 'colab update --install' to upgrade in place.\n"
    "[colab] Run 'uv tool install -U google-colab-cli' to update.\n"
    '[colab] To silence this check, set "enable_update_check": false in '
    '~/.config/colab-cli/settings.json\n\n'
)


class ValidationStatusResponseTests(unittest.TestCase):
    def check_status(self, output, allowed):
        callback = mock.Mock()
        with mock.patch.object(launch.subprocess, 'run', return_value=(
                SimpleNamespace(returncode=0, stdout=output))), \
             mock.patch.object(launch, 'run_cmd') as allocate:
            if allowed:
                launch.ensure_validation_session_once(SESSION, 'A100',
                                                       on_allocation_attempt=callback)
                callback.assert_called_once_with()
                allocate.assert_called_once_with(['colab', 'new', '-s', SESSION, '--gpu', 'A100'],
                                                 env=mock.ANY)
            else:
                with self.assertRaises(RuntimeError):
                    launch.ensure_validation_session_once(SESSION, 'A100',
                                                           on_allocation_attempt=callback)
                allocate.assert_not_called()
                callback.assert_not_called()

    def test_actual_wsl_and_legacy_absence(self):
        for output in (ABSENT, ABSENT + '\n', 'Session not found', 'Error: Session not found.'):
            with self.subTest(output=output):
                self.check_status(output, True)

    def test_actual_wsl_colab_update_banner(self):
        output = (
            '[colab] A new version of Colab CLI is available: 0.8.0 (current: 0.7.4)\n'
            "[colab] Run 'colab update' to update.\n"
            '[colab] To silence this check, set "enable_update_check": false in '
            '~/.config/colab-cli/settings.json\n\n'
            "[colab] Session 'mindscopex-valonly-20261114' not found.\n"
        )
        self.check_status(output, True)
        for blocked in (
            output.replace(SESSION, 'another-session'),
            output + 'Authentication failed\n',
            'TLS error\n' + output,
            output + '[colab] Arbitrary message\n',
            output.replace("Run 'colab update'", "Run 'colab update --unknown'"),
        ):
            with self.subTest(output=blocked):
                self.check_status(blocked, False)

    def test_known_upgrade_banner_before_or_after_absence(self):
        short_banner = (
            '[colab] A new version of Colab CLI is available: 0.7.5 (current: 0.7.4)\n'
            "[colab] Run 'pip install --upgrade google-colab-cli' to update.\n"
        )
        for banner in (UPGRADE, short_banner):
            for absent in (ABSENT, 'Session not found'):
                for output in (banner + absent, absent + '\n' + banner):
                    with self.subTest(output=output):
                        self.check_status(output, True)

    def test_wrong_name_unknown_existing_and_error_responses_blocked(self):
        responses = (
            ABSENT.replace(SESSION, 'another-session'),
            ABSENT.replace(SESSION, SESSION.upper()),
            '[colab] No active sessions.', '', None, 'Unknown response',
            "[colab] Session 'mindscopex-valonly-20261114' already exists.",
            '[mindscopex-valonly-20261114] host | Hardware: A100 | Status: IDLE',
            'authentication error: Session not found',
            'TLS certificate file not found',
            'SSLCertVerificationError: self-signed certificate in certificate chain',
            ABSENT + '\nAuthentication failed',
            'TLS error\n' + ABSENT,
            ABSENT + '\n' + ABSENT,
            ABSENT + '\n' + ABSENT.replace(SESSION, 'another-session'),
            UPGRADE, UPGRADE + ABSENT + '\nTLS error',
            UPGRADE + ABSENT.replace(SESSION, 'another-session'),
            '[colab] Update check failed: TLS error\n' + ABSENT,
            UPGRADE.replace('uv tool install -U google-colab-cli', 'unknown command') + ABSENT,
        )
        for output in responses:
            with self.subTest(output=output):
                self.check_status(output, False)

    def test_session_name_is_literal_not_regex(self):
        self.assertTrue(launch.validation_session_absent("[colab] Session 'a.b' not found.", 'a.b'))
        self.assertFalse(launch.validation_session_absent(
            "[colab] Session 'axb' not found.", 'a.b'))
