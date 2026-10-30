"""Stdlib CPU tests for Colab routing; no Colab, ML imports or downloads."""
import ast
import copy
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from experiments.jobs import first_divergence_trace as job
from experiments.runners import launch_colab as launch
from experiments.runners.collect_artifacts import make_source_archive
from experiments.runners.config import load_toml
from experiments.runners.remote_bootstrap import render_bootstrap

CONFIG = job.ROOT / 'experiments/configs/trace_fresh_v2_growth019_c8_2b.toml'


class TraceJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def invoke_launch(self, dry_run=True):
        return launch.launch_one(CONFIG, root=self.root, session='mock-session', keep_session=True,
                                 default_gpu='A100', default_tpu=None, exec_timeout=60,
                                 allocation_attempts=1, allocation_retry_delay=0, dry_run=dry_run)

    def stage_inputs(self):
        for source in job.INPUTS.values():
            dest = self.root / source.relative_to(job.ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read_bytes())

    def successful_output(self):
        output = job.filter_plan(job.load_plan(), case_id=job.PINNED['case_id'], coefficient=8,
                                 check_equivalence=True)
        output['status'] = 'captured'
        output['measurements'][0]['capture'] = dict(equivalence_check=dict(
            status='compared', generated_token_ids_equal=True, answers_equal=True, equivalent=True))
        return output

    def run_with_output(self, output, directory=None):
        def execute(command, **kwargs):
            kwargs['stdout'].write('mock generated result\n')
            Path(command[-1]).write_text(json.dumps(output))
        with mock.patch.object(job.subprocess, 'run', side_effect=execute):
            return job.run(CONFIG, directory or self.root)

    def assert_failed_output_preserved(self, output, directory=None):
        root = directory or self.root
        with self.assertRaises(ValueError):
            self.run_with_output(output, root)
        run_dir = root / load_toml(CONFIG)['run']['name']
        self.assertEqual(json.loads((run_dir/'actual_first_divergence.json').read_text()), output)
        metadata = json.loads((run_dir/'job_manifest.json').read_text())
        self.assertEqual(metadata['status'], 'failed')
        self.assertIn(metadata['error'], (run_dir/'diagnostic.log').read_text())
        return metadata

    def test_all_equivalence_checks_true_succeeds(self):
        directory = self.run_with_output(self.successful_output())
        metadata = json.loads((directory/'job_manifest.json').read_text())
        self.assertEqual(metadata['status'], 'succeeded')
        self.assertEqual(metadata['equivalence_validation'], 'passed')

    def test_equivalent_false_preserves_result_and_failure_reason(self):
        output = self.successful_output()
        output['measurements'][0]['capture']['equivalence_check']['equivalent'] = False
        self.assertIn('equivalent', self.assert_failed_output_preserved(output)['error'])

    def test_generated_token_ids_equal_false_is_failure(self):
        output = self.successful_output()
        output['measurements'][0]['capture']['equivalence_check']['generated_token_ids_equal'] = False
        self.assertIn('generated_token_ids_equal', self.assert_failed_output_preserved(output)['error'])

    def test_answers_equal_false_is_failure(self):
        output = self.successful_output()
        output['measurements'][0]['capture']['equivalence_check']['answers_equal'] = False
        self.assertIn('answers_equal', self.assert_failed_output_preserved(output)['error'])

    def test_missing_or_nonaffirmative_equivalence_evidence_fails(self):
        mutations = [
            lambda o: o['measurements'][0]['capture'].pop('equivalence_check'),
            lambda o: o['measurements'][0]['capture']['equivalence_check'].pop('status'),
            lambda o: o['measurements'][0]['capture']['equivalence_check'].pop('equivalent'),
            lambda o: o['measurements'][0]['capture']['equivalence_check'].pop('answers_equal'),
            lambda o: o['measurements'][0]['capture']['equivalence_check'].pop('generated_token_ids_equal'),
            lambda o: o['measurements'][0]['capture']['equivalence_check'].update(status='not_run'),
            lambda o: o['measurements'][0]['capture']['equivalence_check'].update(equivalent=1),
            lambda o: o.update(measurements=[]),
        ]
        for i, mutate in enumerate(mutations):
            output = self.successful_output()
            mutate(output)
            with self.subTest(i=i):
                self.assert_failed_output_preserved(output, self.root / str(i))

    def test_input_hash_metadata_preserved_in_plan_result_manifest_and_missing_hash_rejected(self):
        output = self.successful_output()
        directory = self.run_with_output(output)
        for filename in ('plan.json', 'actual_first_divergence.json', 'job_manifest.json'):
            self.assertEqual(json.loads((directory/filename).read_text())['source_files'], output['source_files'])
        for i, mutation in enumerate(('missing', 'changed')):
            altered = copy.deepcopy(output)
            if mutation == 'missing':
                altered.pop('source_files')
            else:
                altered['source_files']['scores']['sha256'] = 'wrong'
            self.assert_failed_output_preserved(altered, self.root / str(i))

    def test_plan_is_single_condition_and_ml_free_in_subprocess(self):
        script = ("import sys; from experiments.jobs.first_divergence_trace import prepare_plan; "
                  "from experiments.runners.config import load_toml; "
                  f"p=prepare_plan(load_toml({str(CONFIG)!r})); "
                  "assert p['n_unique_measurements']==1 and p['expected_generate_calls']==2; "
                  "assert [r['seed'] for r in p['slot_comparisons']]==[2,4]; "
                  "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
        # load_toml expects a Path.
        script = script.replace('import sys;', 'import sys; from pathlib import Path;').replace(
            f'load_toml({str(CONFIG)!r})', f'load_toml(Path({str(CONFIG)!r}))')
        result = subprocess.run([sys.executable, '-c', script], cwd=job.ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_config_rejects_scope_expansion_before_subprocess(self):
        for key, value in [('case_id', 'crt_fresh_v2_growth_007'), ('coefficient', 9),
                           ('check_equivalence', False), ('layer', 6)]:
            cfg = copy.deepcopy(load_toml(CONFIG))
            cfg['diagnostic'][key] = value
            with mock.patch.object(job.subprocess, 'run') as execute, self.assertRaises(ValueError):
                job.prepare_plan(cfg)
            execute.assert_not_called()

    def test_coefficients_zero_seven_eight_prepare_distinct_single_measurements(self):
        for coefficient in (0, 7, 8):
            with self.subTest(coefficient=coefficient):
                config = load_toml(job.ROOT / f'experiments/configs/trace_fresh_v2_growth019_c{coefficient}_2b.toml')
                plan = job.prepare_plan(config)
                self.assertEqual(plan['n_unique_measurements'], 1)
                self.assertEqual(plan['coefficients'], [coefficient])
                self.assertEqual(plan['expected_generate_calls'], 2)
                self.assertEqual([s['seed'] for s in plan['slot_comparisons']], [2, 4])

    def test_coefficient_type_and_extra_diagnostic_fields_rejected(self):
        for coefficient in (True, 7.0, '7', -1):
            with self.subTest(coefficient=coefficient):
                config = copy.deepcopy(load_toml(CONFIG))
                config['diagnostic']['coefficient'] = coefficient
                with self.assertRaises(ValueError):
                    job.prepare_plan(config)
        config = copy.deepcopy(load_toml(CONFIG))
        config['diagnostic']['unexpected'] = True
        with self.assertRaises(ValueError):
            job.prepare_plan(config)

    def test_zero_and_seven_adapter_runs_validate_matching_filters(self):
        for coefficient in (0, 7):
            with self.subTest(coefficient=coefficient):
                config_path = job.ROOT / f'experiments/configs/trace_fresh_v2_growth019_c{coefficient}_2b.toml'
                output = job.filter_plan(job.load_plan(), case_id=job.PINNED['case_id'],
                                         coefficient=coefficient, check_equivalence=True)
                output['status'] = 'captured'
                output['measurements'][0]['capture'] = dict(equivalence_check=dict(
                    status='compared', generated_token_ids_equal=True,
                    answers_equal=True, equivalent=True))
                def execute(command, **kwargs):
                    self.assertEqual(command[command.index('--coefficient') + 1], str(coefficient))
                    kwargs['stdout'].write('mock diagnostic result\\n')
                    Path(command[-1]).write_text(json.dumps(output))
                with mock.patch.object(job.subprocess, 'run', side_effect=execute):
                    directory = job.run(config_path, self.root / f'coef_{coefficient}')
                self.assertEqual(json.loads((directory / 'job_manifest.json').read_text())['status'],
                                 'succeeded')

    def test_adapter_forwards_args_and_requires_captured_output(self):
        def success(command, **kwargs):
            self.assertEqual(command[command.index('--case-id') + 1], 'crt_fresh_v2_growth_019')
            self.assertEqual(command[command.index('--coefficient') + 1], '8')
            self.assertIn('--check-equivalence', command)
            self.assertIn('--run-model', command)
            kwargs['stdout'].write('mock diagnostic log\n')
            Path(command[-1]).write_text(json.dumps(self.successful_output()))
        with mock.patch.object(job.subprocess, 'run', side_effect=success):
            directory = job.run(CONFIG, self.root)
        self.assertEqual(json.loads((directory/'job_manifest.json').read_text())['status'], 'succeeded')
        self.assertEqual((directory/'diagnostic.log').read_text(), 'mock diagnostic log\n')

    def test_adapter_failure_preserves_logs_and_failed_manifest(self):
        def fail(command, **kwargs):
            kwargs['stdout'].write('mock OOM evidence\n')
            raise subprocess.CalledProcessError(17, command)
        with mock.patch.object(job.subprocess, 'run', side_effect=fail), self.assertRaises(subprocess.CalledProcessError):
            job.run(CONFIG, self.root)
        directory = self.root / load_toml(CONFIG)['run']['name']
        metadata = json.loads((directory/'job_manifest.json').read_text())
        self.assertEqual((metadata['status'], metadata['returncode']), ('failed', 17))
        self.assertIn('OOM', (directory/'diagnostic.log').read_text())
        self.assertTrue((directory/'plan.json').is_file())

    def test_adapter_zero_exit_without_result_is_failure(self):
        with mock.patch.object(job.subprocess, 'run'), self.assertRaises(FileNotFoundError):
            job.run(CONFIG, self.root)
        directory = self.root / load_toml(CONFIG)['run']['name']
        self.assertEqual(json.loads((directory/'job_manifest.json').read_text())['status'], 'failed')

    def test_adapter_refuses_old_artifacts_before_execution(self):
        directory = self.root / load_toml(CONFIG)['run']['name']
        directory.mkdir()
        old = directory/'actual_first_divergence.json'
        old.write_text('preserve')
        with mock.patch.object(job.subprocess, 'run') as execute, self.assertRaises(FileExistsError):
            job.run(CONFIG, self.root)
        execute.assert_not_called()
        self.assertEqual(old.read_text(), 'preserve')

    def test_source_archive_includes_exact_required_csvs(self):
        self.stage_inputs()
        archive = make_source_archive(self.root, self.root/'source.zip')
        with zipfile.ZipFile(archive) as z:
            self.assertNotIn('results/analysis/protocol_b_behavioral_join_slots.csv', z.namelist())
        job.include_reference_inputs(self.root, archive)
        with zipfile.ZipFile(archive) as z:
            for path in job.INPUTS.values():
                rel = path.relative_to(job.ROOT).as_posix()
                self.assertEqual(z.read(rel), path.read_bytes())
                self.assertEqual(z.namelist().count(rel), 1)

    def test_launcher_dry_run_prepares_archive_without_any_remote_calls_and_refuses_reuse(self):
        self.stage_inputs()
        with mock.patch.object(launch, 'timestamp', return_value='fixed'), \
             mock.patch.object(launch, 'git_value', return_value=''), \
             mock.patch.object(launch, 'ensure_session') as session, \
             mock.patch.object(launch, 'run_cmd') as command, \
             mock.patch.object(launch, 'run_cmd_retry') as retry:
            directory = self.invoke_launch()
            with self.assertRaises(FileExistsError):
                self.invoke_launch()
        session.assert_not_called()
        command.assert_not_called()
        retry.assert_not_called()
        self.assertTrue((directory/'remote_bootstrap.py').is_file())
        with zipfile.ZipFile(directory/'source.zip') as z:
            self.assertIn('results/analysis/protocol_b_20261029/full_answer_scores.csv', z.namelist())

    def test_launcher_downloads_failure_artifacts_and_logs_then_raises(self):
        self.stage_inputs()
        seen = []
        def command(args, **kwargs):
            seen.append(args[1])
            return subprocess.CompletedProcess(args, 9 if args[1]=='exec' else 0)
        def retry(args, **kwargs):
            if args[1]=='download':
                with zipfile.ZipFile(args[-1], 'w') as z:
                    z.writestr('job.log', 'mock remote failure')
        with mock.patch.object(launch, 'git_value', return_value=''), \
             mock.patch.object(launch, 'ensure_session'), \
             mock.patch.object(launch, 'run_cmd', side_effect=command), \
             mock.patch.object(launch, 'run_cmd_retry', side_effect=retry), \
             self.assertRaisesRegex(RuntimeError, 'returncode=9'):
            self.invoke_launch(dry_run=False)
        self.assertEqual(seen, ['exec', 'log', 'log'])
        files = list(self.root.glob('results/runs/*/artifacts/job.log'))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].read_text(), 'mock remote failure')

    def test_launcher_rejects_failed_manifest_even_if_cli_exit_is_zero(self):
        self.stage_inputs()
        def retry(args, **kwargs):
            if args[1] == 'download':
                with zipfile.ZipFile(args[-1], 'w') as z:
                    z.writestr('job_manifest.json', '{"status":"failed"}')
        with mock.patch.object(launch, 'git_value', return_value=''), \
             mock.patch.object(launch, 'ensure_session'), \
             mock.patch.object(launch, 'run_cmd', return_value=subprocess.CompletedProcess([], 0)), \
             mock.patch.object(launch, 'run_cmd_retry', side_effect=retry), \
             self.assertRaisesRegex(RuntimeError, 'do not confirm success'):
            self.invoke_launch(dry_run=False)

    def test_trace_failure_not_suppressed_by_continue_on_error(self):
        args = ['launch_colab', str(CONFIG), '--dry-run', '--continue-on-error']
        with mock.patch.object(sys, 'argv', args), \
             mock.patch.object(launch, 'launch_one', side_effect=RuntimeError('remote failed')), \
             self.assertRaisesRegex(RuntimeError, 'remote failed'):
            launch.main()

    def test_bootstrap_failure_archives_job_log_and_exits_nonzero(self):
        config = self.root/'config.toml'
        config.write_bytes(CONFIG.read_bytes())
        payload = dict(job='first_divergence_trace', job_path='unused.py', run_name='new-trace',
                       remote_config_path=str(config), remote_output_root=str(self.root/'out'))
        code = render_bootstrap(payload).replace('Path("/content")', f'Path({str(self.root)!r})')
        tree = ast.parse(code)
        tree.body.pop()  # do not auto-call main while preparing mocked bootstrap
        namespace = {}
        exec(compile(tree, '<bootstrap>', 'exec'), namespace)
        namespace['prepare_source'] = lambda: self.root
        namespace['ensure_dependencies'] = mock.Mock()
        def fail(cmd, log_path, **kwargs):
            log_path.write_text('mock failure evidence')
            return 19
        namespace['run_capturing'] = fail
        with mock.patch('os.chdir'), self.assertRaises(SystemExit) as failure:
            namespace['main']()
        self.assertEqual(failure.exception.code, 19)
        with zipfile.ZipFile(self.root/'out/new-trace.zip') as z:
            self.assertEqual(z.read('job.log'), b'mock failure evidence')
        with mock.patch('os.chdir'), self.assertRaises(FileExistsError):
            namespace['main']()


if __name__ == '__main__':
    unittest.main()
