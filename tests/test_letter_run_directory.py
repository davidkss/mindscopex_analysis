"""CPU regression for bootstrap-created letter experiment directories."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.jobs import research_experiments as job

CONFIG = job.ROOT / 'experiments/configs/study_choice_letter_f30908_seed1_2b.toml'


class ReachedSetup(Exception):
    pass


class LetterRunDirectoryTests(unittest.TestCase):
    def test_precreated_empty_directory_reaches_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / job.run_name(job.load_toml(CONFIG))).mkdir()
            with mock.patch.object(job, '_resolve_env', side_effect=ReachedSetup), \
                 mock.patch.object(job, 'load_qwen_text_generation_model') as load:
                with self.assertRaises(ReachedSetup):
                    job.run(CONFIG, root)
                load.assert_not_called()

    def test_current_bootstrap_log_handoff_reaches_setup_once(self):
        import os
        from experiments.runners.letter_run_directory import TOKEN_ENV, STATUS_FILE, prepare
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / job.run_name(job.load_toml(CONFIG))
            token = prepare(run_dir, CONFIG)
            (run_dir / 'job.log').write_text('current bootstrap log\n')
            with mock.patch.dict(os.environ, {TOKEN_ENV: token}), \
                 mock.patch.object(job, '_resolve_env', side_effect=ReachedSetup):
                with self.assertRaises(ReachedSetup):
                    job.run(CONFIG, root)
                with self.assertRaises(FileExistsError):
                    job.run(CONFIG, root)
            import json
            self.assertEqual(json.loads((run_dir / STATUS_FILE).read_text())['phase'], 'running')
            self.assertEqual((run_dir / 'job.log').read_text(), 'current bootstrap log\n')

    def test_unknown_results_stale_status_and_symlinks_rejected_unchanged(self):
        import os
        import json
        from experiments.runners.letter_run_directory import TOKEN_ENV, STATUS_FILE, prepare
        for kind in ('log_only', 'results', 'manifest', 'config_only', 'unknown', 'stale',
                     'bad_token', 'changed_config', 'malformed', 'directory_symlink', 'log_symlink'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_dir = root / job.run_name(job.load_toml(CONFIG))
                token = ''
                if kind in ('stale', 'bad_token', 'changed_config', 'malformed', 'log_symlink'):
                    token = prepare(run_dir, CONFIG)
                    path = run_dir / STATUS_FILE
                    if kind == 'stale':
                        data = json.loads(path.read_text()); data['phase'] = 'failed'
                        path.write_text(json.dumps(data))
                    elif kind == 'bad_token': token = 'wrong'
                    elif kind == 'changed_config':
                        data = json.loads(path.read_text()); data['config_sha256'] = 'wrong'
                        path.write_text(json.dumps(data))
                    elif kind == 'malformed': path.write_text('not json')
                    elif kind == 'log_symlink':
                        target = root / 'other.log'; target.write_text('preserve')
                        (run_dir / 'job.log').symlink_to(target)
                elif kind == 'directory_symlink':
                    target = root / 'original'; target.mkdir()
                    run_dir.symlink_to(target, target_is_directory=True)
                else:
                    run_dir.mkdir()
                    filename = {'log_only': 'job.log', 'results': 'behavioral/generations.json',
                                'manifest': 'manifest.json', 'config_only': 'config.toml',
                                'unknown': 'unknown.txt'}[kind]
                    path = run_dir / filename; path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('preserve')
                before = {p: p.read_bytes() for p in run_dir.rglob('*') if p.is_file()}
                with mock.patch.dict(os.environ, {TOKEN_ENV: token}), \
                     mock.patch.object(job, 'load_qwen_text_generation_model') as load:
                    with self.assertRaises(FileExistsError): job.run(CONFIG, root)
                    load.assert_not_called()
                self.assertEqual(before, {p: p.read_bytes() for p in run_dir.rglob('*') if p.is_file()})

    def test_bootstrap_archives_success_and_failure_with_status(self):
        import json
        import os
        import zipfile
        from experiments.runners.remote_bootstrap import render_bootstrap
        from experiments.runners.letter_run_directory import TOKEN_ENV, STATUS_FILE, claim
        for mode in ('success', 'failure', 'spawn_failure', 'value_control', 'negative_dose'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                if mode == 'value_control':
                    config = CONFIG.with_name('study_choice_order_counterbalance_f30908_seed1_2b.toml')
                elif mode == 'negative_dose':
                    config = CONFIG.with_name('study_transfer_fresh_v2_negative_dose_f30908_seed1_2b.toml')
                else:
                    config = CONFIG
                name = job.run_name(job.load_toml(config))
                payload = dict(remote_repo_dir=str(job.ROOT), remote_output_root=str(root / 'output'),
                               remote_config_path=str(config), run_name=name, job='research_experiments',
                               job_path='experiments/jobs/research_experiments.py')
                # Execute rendered bootstrap with all setup/process operations stubbed.
                source = render_bootstrap(payload).rsplit('\nmain()', 1)[0]
                source = source.replace('Path("/content")', f'Path({str(root)!r})')
                namespace = {}
                exec(compile(source, '<test bootstrap>', 'exec'), namespace)
                namespace['prepare_source'] = lambda: job.ROOT
                namespace['ensure_dependencies'] = lambda repo: None
                run_dir = root / 'output' / name
                def fake_capture(command, log, **kwargs):
                    log.write_text('synthetic process log\n')
                    if mode in ('value_control', 'negative_dose'):
                        self.assertNotIn('env', kwargs)
                        return 0
                    with mock.patch.dict(os.environ, kwargs['env']):
                        claim(run_dir, config)
                    if mode == 'spawn_failure': raise OSError('synthetic spawn error')
                    return 0 if mode == 'success' else 7
                namespace['run_capturing'] = fake_capture
                # Both ZIP packaging and fallback copying use temporary paths.
                with mock.patch.object(os, 'chdir'), mock.patch.dict(os.environ):
                    if mode in ('failure', 'spawn_failure'):
                        with self.assertRaises(SystemExit) as exc: namespace['main']()
                        self.assertEqual(exc.exception.code, 7 if mode == 'failure' else 1)
                    else:
                        namespace['main']()
                with zipfile.ZipFile(run_dir.with_suffix('.zip')) as archive:
                    self.assertIn('job.log', archive.namelist())
                    if mode in ('value_control', 'negative_dose'):
                        self.assertNotIn(STATUS_FILE, archive.namelist())
                    else:
                        status = json.loads(archive.read(STATUS_FILE))
                        self.assertEqual(status['phase'], 'succeeded' if mode == 'success' else 'failed')
                        self.assertEqual(status['returncode'], 0 if mode == 'success' else 7 if mode == 'failure' else 1)
                        before = {p: p.read_bytes() for p in run_dir.rglob('*') if p.is_file()}
                        with mock.patch.object(os, 'chdir'), self.assertRaises(FileExistsError):
                            namespace['main']()
                        self.assertEqual(before, {p: p.read_bytes() for p in run_dir.rglob('*') if p.is_file()})
