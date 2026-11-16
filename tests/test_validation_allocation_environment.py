"""Allocation subprocess boundary checks; never execute CLI or HTTP requests."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.runners import launch_colab as launch


class ValidationAllocationEnvironmentTests(unittest.TestCase):
    def test_child_overrides_retry_settings_without_changing_parent(self):
        for retries in (None, '3', '0'):
            for backoff in (None, '25'):
                for report in (False, True):
                    with self.subTest(retries=retries, backoff=backoff, report=report), \
                         tempfile.TemporaryDirectory() as tmp, \
                         mock.patch.dict(os.environ, {'PATH': '/mock/bin',
                                                     'REQUESTS_CA_BUNDLE': '/mock/ca.pem'},
                                         clear=True):
                        if retries is not None:
                            os.environ['COLAB_CLI_ASSIGN_RETRIES'] = retries
                        if backoff is not None:
                            os.environ['COLAB_CLI_RETRY_BACKOFF'] = backoff
                        parent = dict(os.environ)
                        callback = mock.Mock()
                        with mock.patch.object(launch.subprocess, 'run', side_effect=[
                            SimpleNamespace(returncode=0, stdout='Session not found'),
                            SimpleNamespace(returncode=0, stdout='allocated'),
                        ]) as child:
                            launch.ensure_validation_session_once(
                                'mock', 'A100', on_allocation_attempt=callback,
                                report_dir=Path(tmp) if report else None)
                        self.assertEqual(child.call_count, 2)  # status + one allocation CLI
                        status, allocation = child.call_args_list
                        self.assertNotIn('env', status.kwargs)
                        self.assertEqual(allocation.args[0],
                                         ['colab', 'new', '-s', 'mock', '--gpu', 'A100'])
                        self.assertEqual(allocation.kwargs['env'], dict(
                            parent, COLAB_CLI_ASSIGN_RETRIES='0', COLAB_CLI_RETRY_BACKOFF='0'))
                        self.assertEqual(dict(os.environ), parent)
                        callback.assert_called_once_with()

    def test_failure_output_and_environment_at_actual_subprocess_boundary(self):
        errors = [('HTTP 503 Service Unavailable\n', 'http_503'),
                  ('ColabRequestError: Service Unavailable\n', 'service_unavailable'),
                  ('TLS certificate verification failed\n', 'tls'),
                  ('Authentication failed\n', 'authentication'),
                  ('Unexpected failure\n', 'unknown'),
                  ('ReadTimeout: read timed out\n', 'unknown')]
        for output, category in errors:
            with self.subTest(category=category, output=output), \
                 tempfile.TemporaryDirectory() as tmp, \
                 mock.patch.dict(os.environ, COLAB_CLI_ASSIGN_RETRIES='3',
                                 COLAB_CLI_RETRY_BACKOFF='10'), \
                 mock.patch.object(launch.subprocess, 'run', side_effect=[
                     SimpleNamespace(returncode=0, stdout='Session not found'),
                     subprocess.CalledProcessError(1, ['colab', 'new'], output=output),
                 ]) as child:
                with self.assertRaises(subprocess.CalledProcessError):
                    launch.ensure_validation_session_once('mock', 'A100', report_dir=Path(tmp))
                self.assertEqual(child.call_count, 2)
                self.assertEqual(child.call_args.kwargs['env']['COLAB_CLI_ASSIGN_RETRIES'], '0')
                self.assertEqual(child.call_args.kwargs['env']['COLAB_CLI_RETRY_BACKOFF'], '0')
                record = json.loads((Path(tmp)/'allocation_attempt.json').read_text())
                self.assertEqual(record['category'], category)
                self.assertEqual(record['allocation_attempts'], 1)
                self.assertEqual(record['server_assignment_status'], 'unverified')
                self.assertEqual((Path(tmp)/'allocation_output.log').read_text(), output)

    def test_generic_commands_keep_inherited_environment(self):
        with mock.patch.object(launch.subprocess, 'run') as child:
            launch.run_cmd(['mock-command'])
            self.assertNotIn('env', child.call_args.kwargs)
