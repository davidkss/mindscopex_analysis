"""Validation session lifecycle tests; all Colab calls are mocked."""
import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments import runtime_smoke as smoke
from experiments.runners import launch_colab as launch


class ValidationSessionCleanupTests(unittest.TestCase):
    def setUp(self):
        active = mock.patch.object(smoke, 'ACTIVE_AUTHORIZATION', smoke.ATTEMPT2_AUTHORIZATION)
        active.start()
        self.addCleanup(active.stop)

    def exercise(self, root, *, phase=None, stop_code=0, missing_archive=False):
        run = root / 'run'
        run.mkdir()
        for filename in ('config.toml', 'source.zip', 'remote_bootstrap.py'):
            (run / filename).write_bytes(b'preserve')
        events = []

        def command(cmd, **kwargs):
            events.append(cmd[1])
            if cmd[1] == 'new':
                self.assertEqual(kwargs['env']['COLAB_CLI_ASSIGN_RETRIES'], '0')
                self.assertEqual(kwargs['env']['COLAB_CLI_RETRY_BACKOFF'], '0')
            if cmd[1] == 'new' and phase == 'read_timeout':
                raise subprocess.CalledProcessError(1, cmd, output='ReadTimeout: read timed out')
            if cmd[1] == 'new' and phase in ('http_503', 'service_unavailable'):
                output = ('HTTP 503 Service Unavailable' if phase == 'http_503'
                          else 'ColabRequestError: Service Unavailable\n')
                raise subprocess.CalledProcessError(1, cmd, output=output)
            if cmd[1] == phase:
                raise RuntimeError(f'mock {phase} failure')
            if cmd[1] == 'exec' and phase == 'remote_failure':
                return SimpleNamespace(returncode=7)
            return SimpleNamespace(returncode=stop_code if cmd[1] == 'stop' else 0)

        def transfer(cmd, **kwargs):
            self.assertEqual(kwargs, dict(attempts=1, delay=0))
            events.append(cmd[1])
            unavailable = missing_archive and cmd[1] == 'download' and cmd[-2].endswith('.zip')
            if cmd[1] == phase or unavailable:
                raise subprocess.CalledProcessError(1, cmd)
            if cmd[1] == 'download':
                Path(cmd[-1]).write_bytes(b'partial data')

        def unpack(archive, destination):
            if phase == 'unpack':
                raise ValueError('mock corrupt ZIP')
            destination.mkdir()
            (destination / 'job_manifest.json').write_text(json.dumps(dict(status='succeeded')))

        kwargs = dict(prepared=smoke.build_plan('validate', smoke.ACTIVE_AUTHORIZATION),
                      root=root, session='mindscopex-valonly-20261114-a2', gpu='A100',
                      local_run_dir=run, local_config=run/'config.toml',
                      remote_config_path='/content/config.toml',
                      remote_source_archive='/content/source.zip',
                      bootstrap=run/'remote_bootstrap.py', exec_timeout=900,
                      remote_output_root='/content/artifacts', name='validation')
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(launch.subprocess, 'run', return_value=(
            SimpleNamespace(returncode=0, stdout='Session not found'))))
        stack.enter_context(mock.patch.object(launch, 'run_cmd', side_effect=command))
        stack.enter_context(mock.patch.object(launch, 'run_cmd_retry', side_effect=transfer))
        stack.enter_context(mock.patch.object(launch, 'unpack_archive', side_effect=unpack))
        return stack, kwargs, events

    def assert_recovered_before_stop(self, events):
        self.assertEqual(events.count('new'), 1)
        self.assertEqual(events.count('stop'), 1)
        self.assertLess(events.index('download'), events.index('stop'))
        self.assertLess(events.index('log'), events.index('stop'))
        self.assertEqual(events[-1], 'stop')

    def test_success_recovers_then_stops_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack, kwargs, events = self.exercise(Path(tmp))
            with stack:
                result = launch.launch_validation_once(**kwargs)
            self.assert_recovered_before_stop(events)
            self.assertEqual(result, kwargs['local_run_dir'])
            claims = list((Path(tmp)/'results/.authorization_claims').glob('*.json'))
            self.assertEqual(len(claims), 1)
            self.assertEqual(json.loads(claims[0].read_text())['authorization_id'],
                             smoke.ACTIVE_AUTHORIZATION)
            self.assertEqual((result/'source.zip').read_bytes(), b'preserve')
            self.assertEqual(json.loads((result/'session_cleanup.json').read_text())['status'],
                             'stop_command_succeeded')
            self.assertTrue((result/'artifact_recovery.json').exists())

    def test_http_503_is_recorded_once_and_keeps_uncertain_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack, kwargs, events = self.exercise(Path(tmp), phase='http_503')
            with stack, self.assertRaises(subprocess.CalledProcessError):
                launch.launch_validation_once(**kwargs)
            self.assert_recovered_before_stop(events)
            self.assertNotIn('upload', events)
            self.assertNotIn('exec', events)
            run = kwargs['local_run_dir']
            record = json.loads((run/'allocation_attempt.json').read_text())
            self.assertEqual(record['category'], 'http_503')
            self.assertEqual(record['allocation_attempts'], 1)
            self.assertEqual(record['server_assignment_status'], 'unverified')
            self.assertEqual((run/'allocation_output.log').read_text(),
                             'HTTP 503 Service Unavailable')
            self.assertEqual(json.loads((run/'session_cleanup.json').read_text())['status'],
                             'stop_command_succeeded')

    def test_service_unavailable_preserves_output_recovers_and_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack, kwargs, events = self.exercise(Path(tmp), phase='service_unavailable')
            with stack, self.assertRaises(subprocess.CalledProcessError):
                launch.launch_validation_once(**kwargs)
            self.assert_recovered_before_stop(events)
            self.assertNotIn('upload', events)
            self.assertNotIn('exec', events)
            run = kwargs['local_run_dir']
            record = json.loads((run/'allocation_attempt.json').read_text())
            self.assertEqual(record['category'], 'service_unavailable')
            self.assertEqual(record['allocation_attempts'], 1)
            self.assertEqual(record['server_assignment_status'], 'unverified')
            self.assertEqual((run/'allocation_output.log').read_text(),
                             'ColabRequestError: Service Unavailable\n')
            self.assertEqual(len(list((Path(tmp)/'results/.authorization_claims').glob('*.json'))), 1)

    def test_remote_failure_and_transport_exceptions_still_recover_and_stop(self):
        for phase in ('remote_failure', 'upload', 'exec', 'new', 'read_timeout'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                stack, kwargs, events = self.exercise(Path(tmp), phase=phase)
                with stack, self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                    launch.launch_validation_once(**kwargs)
                self.assert_recovered_before_stop(events)
                self.assertTrue((kwargs['local_run_dir']/'session_cleanup.json').exists())

    def test_missing_or_corrupt_zip_keeps_partial_files_and_attempts_logs(self):
        for phase in (None, 'unpack'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                stack, kwargs, events = self.exercise(Path(tmp), phase=phase,
                                                     missing_archive=phase is None)
                with stack, self.assertRaisesRegex(RuntimeError, 'could not be recovered'):
                    launch.launch_validation_once(**kwargs)
                self.assert_recovered_before_stop(events)
                run = kwargs['local_run_dir']
                self.assertEqual((run/'partial_artifacts/runtime_status.json').read_bytes(),
                                 b'partial data')
                if phase == 'unpack':
                    self.assertTrue((run/'artifacts.zip').exists())
                    self.assertTrue((run/'artifacts_fallback.zip').exists())

    def test_log_failure_does_not_prevent_other_log_or_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack, kwargs, events = self.exercise(Path(tmp), phase='log')
            with stack:
                launch.launch_validation_once(**kwargs)
            self.assertEqual(events.count('log'), 2)
            self.assert_recovered_before_stop(events)

    def test_recovery_exception_or_interrupt_still_attempts_stop(self):
        for error in (OSError('mock disk error'), KeyboardInterrupt()):
            with self.subTest(error=type(error)), tempfile.TemporaryDirectory() as tmp:
                stack, kwargs, events = self.exercise(Path(tmp))
                with stack, mock.patch.object(launch, 'recover_validation_artifacts',
                                               side_effect=error):
                    with self.assertRaises(type(error)):
                        launch.launch_validation_once(**kwargs)
                self.assertEqual(events[-1], 'stop')
                self.assertEqual(events.count('stop'), 1)

    def test_stop_failure_prints_manual_command_and_preserves_original_error(self):
        for phase in (None, 'stop', 'remote_failure'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                stack, kwargs, events = self.exercise(Path(tmp), phase=phase, stop_code=1)
                stderr = io.StringIO()
                with stack, contextlib.redirect_stderr(stderr):
                    message = ('remote validation failed' if phase == 'remote_failure'
                               else 'cleanup failed')
                    with self.assertRaisesRegex(RuntimeError, message):
                        launch.launch_validation_once(**kwargs)
                self.assert_recovered_before_stop(events)
                self.assertIn('cleanup FAILED', stderr.getvalue())
                self.assertIn('colab stop -s mindscopex-valonly-20261114-a2', stderr.getvalue())
                record = json.loads((kwargs['local_run_dir']/'session_cleanup.json').read_text())
                self.assertEqual(record['status'], 'failed')
                self.assertTrue((kwargs['local_run_dir']/'artifacts/job_manifest.json').exists())

    def test_preallocation_rejection_never_stops_existing_or_uncertain_session(self):
        for status in ('Session exists Hardware: A100', 'TLS error'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                stack, kwargs, events = self.exercise(Path(tmp))
                with stack, mock.patch.object(launch.subprocess, 'run', return_value=(
                        SimpleNamespace(returncode=1, stdout=status))):
                    with self.assertRaisesRegex(RuntimeError, 'explicitly absent'):
                        launch.launch_validation_once(**kwargs)
                self.assertEqual(events, [])

    def test_duplicate_authorization_never_stops_or_recovers_another_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack, kwargs, events = self.exercise(Path(tmp))
            with stack:
                launch.launch_validation_once(**kwargs)
                before = events.copy()
                with self.assertRaises(FileExistsError):
                    launch.launch_validation_once(**kwargs)
                self.assertEqual(events, before)
