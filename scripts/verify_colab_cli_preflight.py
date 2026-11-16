"""Offline installed-CLI checks. Run with the uv tool's Python, never the CLI.

Only synthetic credentials/responses are used. No user credential loader is called.
This diagnostic does not authorize allocation or prove wire-level POST counts.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import socket
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

EXPECTED_CLIENT_SHA = '7081d2c9decbc9a4cafb4ab89202c7e7a644f72bd74a7e846bffce5344839a06'


def build_suite(*, replay_boundary_only=False):
    # Called only after the socket guards have been installed by main().
    import requests
    from colab_cli import client
    from google.auth import credentials
    from google.auth.transport.requests import AuthorizedSession
    from urllib3.exceptions import MaxRetryError, ReadTimeoutError

    class SyntheticCredentials(credentials.Credentials):
        def __init__(self):
            super().__init__()
            self.refreshes = 0

        def before_request(self, *args, **kwargs):
            pass

        def refresh(self, request):
            self.refreshes += 1  # No auth request or user token access.

    def response(status, payload=None, location=None):
        result = requests.Response()
        result.status_code = status
        result.reason = 'synthetic response'
        result._content = json.dumps(payload or {}).encode()
        if location:
            result.headers['Location'] = location
        return result

    pending = {'acc': 'A100', 'nbh': 'synthetic', 'token': 'synthetic', 'variant': 'GPU'}
    assigned = {'endpoint': 'synthetic', 'runtimeProxyInfo': {
        'token': 'synthetic', 'tokenExpiresInSeconds': 60, 'url': 'https://offline.invalid'}}

    class InstalledPreflightTests(unittest.TestCase):
        def setUp(self):
            self.stack = contextlib.ExitStack()
            self.addCleanup(self.stack.close)
            self.stack.enter_context(mock.patch.object(
                requests.adapters.HTTPAdapter, 'send',
                side_effect=AssertionError('real HTTP adapter blocked')))
            self.stack.enter_context(mock.patch.dict(os.environ, {
                'COLAB_CLI_ASSIGN_RETRIES': '0', 'COLAB_CLI_RETRY_BACKOFF': '0'}))
            self.sleep = self.stack.enter_context(mock.patch.object(client.time, 'sleep'))
            with self.assertRaisesRegex(AssertionError, 'socket blocked'):
                socket.create_connection(('offline.invalid', 443))
            with requests.Session() as probe:
                probe.trust_env = False
                with self.assertRaisesRegex(AssertionError, 'real HTTP adapter blocked'):
                    probe.post('https://offline.invalid/blocked')

        def test_network_guards_are_active(self):
            with self.assertRaisesRegex(AssertionError, 'socket blocked'):
                socket.create_connection(('offline.invalid', 443))
            with requests.Session() as session:
                session.trust_env = False
                with self.assertRaisesRegex(AssertionError, 'real HTTP adapter blocked'):
                    session.post('https://offline.invalid/blocked')

        def test_installed_identity_and_official_client_fingerprint(self):
            self.assertEqual(importlib.metadata.version('google-colab-cli'), '0.7.5')
            self.assertEqual(hashlib.sha256(Path(client.__file__).read_bytes()).hexdigest(),
                             EXPECTED_CLIENT_SHA)

        def exercise_assign(self, retries, failure, *, reconcile=False):
            requests_seen = []
            posts = 0

            def request(method, url, **kwargs):
                nonlocal posts
                requests_seen.append(method)
                if method == 'GET':
                    payload = assigned if reconcile and posts else pending
                    result = response(200, payload)
                else:
                    self.assertEqual(method, 'POST')
                    posts += 1
                    if failure == 'timeout':
                        raise requests.exceptions.ReadTimeout('synthetic timeout')
                    result = response(failure)
                result.request = requests.Request(method, url).prepare()
                return result

            session = mock.Mock(spec=requests.Session)
            session.request.side_effect = request
            actual = client.Client(client.Prod(), session)
            with mock.patch.dict(os.environ, COLAB_CLI_ASSIGN_RETRIES=str(retries)):
                if reconcile:
                    self.assertIsInstance(actual.assign(uuid.UUID(int=0)), client.Assignment)
                else:
                    error = (requests.exceptions.ReadTimeout if failure == 'timeout' else
                             client.TooManyAssignmentsError if failure == 412 else
                             client.ColabRequestError)
                    with self.assertRaises(error):
                        actual.assign(uuid.UUID(int=0))
            return requests_seen

        def test_retry_zero_503_and_read_timeout_have_one_post(self):
            for failure in (503, 'timeout'):
                with self.subTest(failure=failure):
                    self.assertEqual(self.exercise_assign(0, failure), ['GET', 'POST'])
            self.sleep.assert_not_called()

        def test_retry_three_transient_errors_have_four_posts(self):
            for failure in (502, 503, 504, 'timeout'):
                with self.subTest(failure=failure):
                    self.assertEqual(self.exercise_assign(3, failure), ['GET', 'POST'] * 4)

        def test_412_does_not_retry(self):
            for retries in (0, 3):
                with self.subTest(retries=retries):
                    self.assertEqual(self.exercise_assign(retries, 412), ['GET', 'POST'])
            self.sleep.assert_not_called()

        def test_reconciliation_prevents_second_post(self):
            self.assertEqual(self.exercise_assign(3, 503, reconcile=True),
                             ['GET', 'POST', 'GET'])

        def test_existing_assignment_has_no_post(self):
            session = mock.Mock(spec=requests.Session)
            result = response(200, assigned)
            result.request = requests.Request('GET', 'https://offline.invalid').prepare()
            session.request.return_value = result
            self.assertIsInstance(client.Client(client.Prod(), session).assign(uuid.UUID(int=0)),
                                  client.Assignment)
            self.assertEqual([c.args[0] for c in session.request.call_args_list], ['GET'])

        def test_default_adapter_has_no_status_or_read_retries(self):
            with AuthorizedSession(SyntheticCredentials()) as session:
                adapter = session.get_adapter('https://offline.invalid')
                retry = adapter.max_retries
                self.assertEqual(retry.total, 0)
                self.assertIs(retry.read, False)
                for status in (502, 503, 504):
                    self.assertFalse(retry.is_retry('POST', status, has_retry_after=True))
                with self.assertRaises(ReadTimeoutError):
                    retry.increment(method='POST', url='https://offline.invalid',
                                    error=ReadTimeoutError(None, None, 'synthetic timeout'))
                with self.assertRaises(MaxRetryError):
                    retry.increment(method='POST', url='https://offline.invalid',
                                    error=OSError('synthetic connection failure'))
                self.assertEqual(session._auth_request_session.get_adapter(
                    'https://offline.invalid').max_retries.total, 3)

        def transport(self, statuses):
            seen = []

            class SyntheticAdapter(requests.adapters.BaseAdapter):
                def send(self, prepared, **kwargs):
                    seen.append(prepared.method)
                    status = statuses[len(seen) - 1]
                    result = response(status, location='/redirected' if status in (307, 308)
                                      else None)
                    result.url = prepared.url
                    result.request = prepared
                    return result

                def close(self):
                    pass

            creds = SyntheticCredentials()
            with AuthorizedSession(creds) as session:
                session.trust_env = False  # Synthetic test only; no netrc/proxy credential lookup.
                session.mount('https://', SyntheticAdapter())
                result = session.post('https://offline.invalid/assign')
            return seen, creds.refreshes, result.status_code

        def test_auth_401_can_replay_post_despite_assign_retry_zero(self):
            seen, refreshes, status = self.transport([401, 401, 200])
            self.assertEqual(seen, ['POST', 'POST', 'POST'])
            self.assertEqual(refreshes, 2)
            self.assertEqual(status, 200)

        def test_redirect_307_308_preserves_post(self):
            for redirect in (307, 308):
                with self.subTest(redirect=redirect):
                    seen, refreshes, status = self.transport([redirect, 200])
                    self.assertEqual(seen, ['POST', 'POST'])
                    self.assertEqual(refreshes, 0)
                    self.assertEqual(status, 200)

        def test_installed_assign_and_transport_replay_boundary(self):
            # New integration boundary: real assign + auth + redirects, synthetic adapter only.
            self.assertEqual(importlib.metadata.version('google-colab-cli'), '0.7.5')
            self.assertEqual(hashlib.sha256(Path(client.__file__).read_bytes()).hexdigest(),
                             EXPECTED_CLIENT_SHA)
            for status, expected_posts in ((200, 1), (401, 3), (307, 2), (308, 2),
                                           (502, 1), (503, 1), (504, 1), ('timeout', 1)):
                with self.subTest(status=status):
                    seen = []

                    class AssignmentAdapter(requests.adapters.BaseAdapter):
                        def send(self, prepared, **kwargs):
                            seen.append(prepared.method)
                            if prepared.method == 'GET':
                                result = response(200, pending)
                            else:
                                posts = seen.count('POST')
                                if status == 'timeout':
                                    raise requests.exceptions.ReadTimeout('synthetic timeout')
                                # Redirect once, then return a valid assignment response.
                                code = 200 if status in (307, 308) and posts > 1 else status
                                payload = dict(assigned, accelerator='A100', variant=1)
                                result = response(code, payload,
                                                  location='/tun/m/assign'
                                                  if code in (307, 308) else None)
                            result.url = prepared.url
                            result.request = prepared
                            return result

                        def close(self):
                            pass

                    creds = SyntheticCredentials()
                    with AuthorizedSession(creds) as session:
                        session.trust_env = False  # Synthetic test, never user credentials.
                        session.mount('https://', AssignmentAdapter())
                        actual = client.Client(client.Prod(), session)
                        with mock.patch.object(actual, '_post_assignment',
                                               wraps=actual._post_assignment) as post:
                            if status in (200, 307, 308):
                                actual.assign(uuid.UUID(int=0))
                            else:
                                error = (requests.exceptions.ReadTimeout if status == 'timeout'
                                         else client.ColabRequestError)
                                with self.assertRaises(error):
                                    actual.assign(uuid.UUID(int=0))
                            self.assertEqual(post.call_count, 1)
                    self.assertEqual(seen.count('GET'), 1)
                    self.assertEqual(seen.count('POST'), expected_posts)
                    self.assertEqual(creds.refreshes, 2 if status == 401 else 0)
                    print(f'boundary status={status}: assign_POST=1 '
                          f'synthetic_adapter_POST={expected_posts}', flush=True)
            self.sleep.assert_not_called()

    if replay_boundary_only:
        return unittest.TestSuite([InstalledPreflightTests(
            'test_installed_assign_and_transport_replay_boundary')])
    return unittest.defaultTestLoader.loadTestsFromTestCase(InstalledPreflightTests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-boundary-only', action='store_true',
                        help='Run only the new installed assign/transport integration check')
    args = parser.parse_args()
    with contextlib.ExitStack() as guards:
        for target in ('socket.socket.connect', 'socket.socket.connect_ex',
                       'socket.create_connection', 'socket.getaddrinfo'):
            guards.enter_context(mock.patch(target, side_effect=AssertionError('socket blocked')))
        # Guard before importing the installed CLI or HTTP stack. Never call auth.get_credentials.
        suite = build_suite(replay_boundary_only=args.replay_boundary_only)
        from colab_cli import client
        print(json.dumps({'python': sys.executable, 'client': client.__file__,
                          'versions': {name: importlib.metadata.version(name) for name in
                                       ('google-colab-cli', 'requests', 'google-auth', 'urllib3')}},
                         indent=2), flush=True)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
