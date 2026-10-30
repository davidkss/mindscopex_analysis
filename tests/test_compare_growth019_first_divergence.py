"""CPU audit regressions on copies of existing runs; never modify raw artifacts."""
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import compare_growth019_first_divergence as audit


@unittest.skipUnless(all(p.is_dir() for p in audit.RUNS.values()), 'Local A100 artifacts required')
class Growth019ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def copied_runs(self):
        paths = {}
        for coefficient, original in audit.RUNS.items():
            paths[coefficient] = self.root / str(coefficient)
            shutil.copytree(original, paths[coefficient])
        return paths

    def change(self, runs, coefficient, filename, mutate):
        path = runs[coefficient] / filename
        data = json.loads(path.read_text())
        mutate(data)
        path.write_text(json.dumps(data))

    def test_actual_margins_deltas_labels_and_equivalence(self):
        result = audit.analyze()
        self.assertEqual(result['status'], 'validated')
        self.assertEqual(result['validation_errors'], [])
        self.assertEqual([r['actual_raw_margin'] for r in result['rows']], [-1.375, -.375, 0])
        self.assertEqual([r['raw_delta_from_baseline'] for r in result['rows']], [0, 1, 1.375])
        self.assertEqual([r['label'] for r in result['rows']], ['correct', 'correct', 'lure'])
        self.assertEqual([r['answer_changed_from_baseline'] for r in result['rows']], [False, False, True])
        self.assertAlmostEqual(result['rows'][0]['actual_minus_teacher_forced'], 5.960464477539063e-8)
        self.assertEqual(result['rows'][2]['actual_processed_margin'], 0)

    def test_environment_null_revision_is_not_claimed_equal(self):
        result = audit.analyze()
        self.assertTrue(result['recorded_environment_dicts_equal'])
        compared = {r['field']:r for r in result['environment_comparison']}
        self.assertEqual(compared['sae.resolved_revision']['status'], 'not recorded')
        self.assertEqual(compared['environment.model_configs.model._commit_hash']['status'], 'not recorded')
        self.assertEqual(compared['environment.torch']['status'], 'matched recorded values')
        rows = copy.deepcopy(result['rows'])
        rows[1]['environment']['torch'] = 'different-version'
        with mock.patch.object(audit, 'read_run', side_effect=rows):
            changed = audit.analyze()
        self.assertEqual(next(c for c in changed['environment_comparison']
                              if c['field']=='environment.torch')['status'], 'different')

    def test_tampered_branch_margin_is_failed_and_report_still_renders(self):
        runs = self.copied_runs()
        self.change(runs, 8, 'artifacts/actual_first_divergence.json',
                    lambda d: d['measurements'][0]['capture'].update(actual_raw_margin=.5))
        result = audit.analyze(runs)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any('margin' in e['error'] for e in result['validation_errors']))
        self.assertIn('failed', audit.report(result))

    def test_false_equivalence_and_missing_result_fail(self):
        runs = self.copied_runs()
        self.change(runs, 7, 'artifacts/actual_first_divergence.json',
                    lambda d: d['measurements'][0]['capture']['equivalence_check'].update(equivalent=False))
        (runs[0]/'artifacts/actual_first_divergence.json').unlink()  # Temporary copy only.
        result = audit.analyze(runs)
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any('Equivalence' in e['error'] for e in result['validation_errors']))
        self.assertEqual(len(result['rows']), 1)

    def test_prompt_and_eos_tampering_fail(self):
        runs = self.copied_runs()
        self.change(runs, 0, 'artifacts/actual_first_divergence.json',
                    lambda d: d['measurements'][0]['capture']['prompt_token_ids'].append(1))
        self.change(runs, 8, 'artifacts/actual_first_divergence.json',
                    lambda d: d['measurements'][0]['capture'].update(eos_terminated=False))
        result = audit.analyze(runs)
        self.assertEqual(result['status'], 'failed')
        errors = str(result['validation_errors'])
        self.assertIn('Prompt ID hash', errors)
        self.assertIn('EOS', errors)

    def test_cli_cpu_only_and_refuses_existing_output(self):
        output = self.root/'new-report'
        code = (f"import sys; from scripts.compare_growth019_first_divergence import main; "
                f"main(['--output-dir',{str(output)!r}]); "
                "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
        result = subprocess.run([sys.executable, '-c', code], cwd=audit.ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        before = (output/'summary.json').read_bytes()
        with self.assertRaises(FileExistsError):
            audit.main(['--output-dir',str(output)])
        self.assertEqual((output/'summary.json').read_bytes(), before)

    def test_failed_job_and_input_hash_mismatch_fail(self):
        runs = self.copied_runs()
        self.change(runs, 0, 'artifacts/job_manifest.json', lambda d:d.update(status='failed'))
        self.change(runs, 7, 'artifacts/actual_first_divergence.json',
                    lambda d:d['source_files']['scores'].update(sha256='bad'))
        result = audit.analyze(runs)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['rows']), 1)


if __name__ == '__main__':
    unittest.main()
