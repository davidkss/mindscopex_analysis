"""CPU-only regression checks; all writes are confined to temporary directories."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from scripts import analyze_first_divergence_behavior as audit


class FirstDivergenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.slots = audit.rows(audit.ROOT / 'results/analysis/protocol_b_behavioral_join_slots.csv', set())
        self.scores = audit.rows(audit.ROOT / 'results/analysis/protocol_b_20261029/full_answer_scores.csv', set())

    def write_csv(self, name, rows):
        path = self.root / name
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def run_audit(self, bootstrap=100):
        slots = self.write_csv('slots.csv', self.slots)
        scores = self.write_csv('scores.csv', self.scores)
        with contextlib.redirect_stdout(io.StringIO()):
            audit.main(['--slots', str(slots), '--scores', str(scores),
                        '--output', str(self.root / 'out/summary.json'),
                        '--details', str(self.root / 'out/details.csv'),
                        '--bootstrap', str(bootstrap)])
        return json.loads((self.root / 'out/summary.json').read_text())

    def test_exact_zero_and_small_nonzero(self):
        for value in (0.0, -0.0):
            self.assertEqual(audit.preference(value), 'tie')
        self.assertEqual(audit.preference(1e-100), 'lure')
        self.assertEqual(audit.preference(-1e-100), 'correct')
        self.assertIsNone(audit.agreement('tie', 'correct'))
        self.assertIsNone(audit.agreement('lure', 'tie'))

    def test_cluster_multiplicity_and_empty_clusters(self):
        group = [dict(case_id='a', m=1), dict(case_id='a', m=0),
                 dict(case_id='b', m=0), dict(case_id='c', m=None)]
        samples = [['a', 'a', 'b'], ['b', 'b', 'a'], ['c', 'c', 'c']]
        slot, case = audit.summarize(group, 'm', samples)
        self.assertAlmostEqual(slot['rate'], 1/3)
        self.assertEqual(slot['n_valid'], 3)
        self.assertEqual(case['mean'], .25)
        self.assertEqual(case['n_valid_cases'], 2)
        self.assertEqual(case['bootstrap_valid_resamples'], 2)
        self.assertEqual(case['ci95'], [1/6, 1/3])
        slot, case = audit.summarize([dict(case_id='c', m=None)], 'm', [['c']])
        self.assertIsNone(slot['rate'])
        self.assertIsNone(case['mean'])
        self.assertIsNone(case['ci95'])

    def test_real_csv_counts_denominators_and_reproducibility(self):
        result = self.run_audit(20000)
        self.assertEqual(result['n_unique_cases'], 33)
        self.assertEqual(result['n_join_rows'], 128)
        for coeff, count, valid, cases, ties, full_count in [
            ('7', 53, 53, 29, 11, 28), ('8', 56, 58, 31, 6, 36)
        ]:
            group = result['coefficients'][coeff]
            self.assertEqual(Counter(group['case_slot_multiplicity'].values()),
                             {1: 14, 2: 10, 3: 6, 4: 3})
            self.assertEqual(group['branch_tie_slots'], ties)
            branch = group['slot_weighted']['branch_actual_agreement']
            self.assertEqual((branch['count'], branch['n_valid']), (count, valid))
            self.assertEqual(group['case_weighted']['branch_actual_agreement']['n_valid_cases'], cases)
            full = group['slot_weighted']['full_actual_agreement']
            self.assertEqual((full['count'], full['n_valid']), (full_count, 64))
        details = audit.rows(self.root / 'out/details.csv', set())
        self.assertEqual(len(details), 128)
        ties = [r for r in details if r['first_branch_preference'] == 'tie']
        self.assertEqual(len(ties), 17)
        self.assertTrue(all(r['branch_actual_agreement'] == '' for r in ties))
        (self.root / 'out/summary.json').unlink()
        (self.root / 'out/details.csv').unlink()
        self.assertEqual(result, self.run_audit(20000))

    def test_full_tie_excluded_from_full_and_paired_metrics(self):
        score = next(r for r in self.scores if r['coefficient'] == '7.0' and float(r['first_divergence_margin']) != 0)
        score['full_margin'] = '0'
        affected = [r for r in self.slots if r['case_id'] == score['case_id'] and r['coefficient'] == '7']
        for row in affected:
            row['full_margin'] = '0'
        result = self.run_audit()['coefficients']['7']
        self.assertEqual(result['full_tie_slots'], len(affected))
        self.assertEqual(result['slot_weighted']['full_actual_agreement']['n_valid'], 64-len(affected))
        self.assertEqual(result['slot_weighted']['paired_branch_minus_full_agreement']['n_valid'], 53-len(affected))

    def test_invalid_inputs_do_not_create_outputs(self):
        original_slots = [r.copy() for r in self.slots]
        original_scores = [r.copy() for r in self.scores]
        mutations = [
            lambda: self.slots.pop(),
            lambda: self.scores.pop(),
            lambda: self.slots.__setitem__(1, self.slots[0].copy()),
            lambda: self.scores.__setitem__(1, self.scores[0].copy()),
            lambda: self.slots[0].update(actual='unknown'),
            lambda: self.slots[0].update(seed='6'),
            lambda: self.slots[0].update(coefficient='7.9'),
            lambda: self.slots[0].update(case_id='missing'),
            lambda: self.slots[0].update(full_margin='nan'),
            lambda: self.slots[0].update(full_margin='1000000'),
            lambda: self.scores[0].update(first_divergence_margin='inf'),
            lambda: self.scores[0].update(first_divergence_margin=''),
            lambda: self.scores[0].update(full_margin='nan'),
            lambda: self.scores[0].update(input_protocol='wrong'),
            lambda: self.scores[0].update(include_eos='True'),
            lambda: self.slots[0].pop('actual'),
        ]
        for mutation in mutations:
            self.slots = [r.copy() for r in original_slots]
            self.scores = [r.copy() for r in original_scores]
            mutation()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.run_audit()
            self.assertFalse((self.root / 'out/summary.json').exists())
            self.assertFalse((self.root / 'out/details.csv').exists())

    def test_slot_identity_mismatch(self):
        occupied = {(r['case_id'], r['seed']) for r in self.slots if r['coefficient'] == '8'}
        row = next(r for r in self.slots if r['coefficient'] == '8' and
                   (r['case_id'], '5') not in occupied)
        row['seed'] = '5'
        with self.assertRaisesRegex(ValueError, 'identities differ'):
            self.run_audit()

    def test_output_overwrite_protection(self):
        self.run_audit()
        summary = self.root / 'out/summary.json'
        details = self.root / 'out/details.csv'
        before = (summary.read_bytes(), details.read_bytes())
        with self.assertRaises(FileExistsError):
            self.run_audit()
        self.assertEqual(before, (summary.read_bytes(), details.read_bytes()))

    def test_second_write_failure_rolls_back_details(self):
        original = Path.open
        def failing_open(path, *args, **kwargs):
            if path.name == 'summary.json':
                raise PermissionError('simulated write failure')
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, 'open', failing_open):
            with self.assertRaises(PermissionError):
                self.run_audit()
        self.assertFalse((self.root / 'out/details.csv').exists())

    def test_csv_schema_and_missing_file(self):
        path = self.root / 'bad.csv'
        for text in ('other\n1\n', 'a,b\n1\n', 'a,b\n1,2,3\n'):
            path.write_text(text)
            with self.assertRaises(ValueError):
                audit.rows(path, {'a', 'b'})
        with self.assertRaises(FileNotFoundError):
            audit.rows(self.root / 'missing.csv', set())

    def test_legacy_output_keys_and_paired_same_slot_difference(self):
        result = self.run_audit()
        details = audit.rows(self.root / 'out/details.csv')
        for coefficient, group in result['coefficients'].items():
            rows = [r for r in details if r['coefficient'] == coefficient]
            paired = [r for r in rows if r['paired_branch_minus_full_agreement'] != '']
            differences = [int(r['branch_actual_agreement']) - int(r['full_actual_agreement'])
                           for r in paired]
            metric = group['slot_weighted']['paired_branch_minus_full_agreement']
            self.assertEqual(metric['count'], sum(differences))
            self.assertEqual(metric['n_valid'], len(paired))
            self.assertAlmostEqual(metric['rate'], sum(differences) / len(paired))
            bycase = {}
            for row, difference in zip(paired, differences):
                bycase.setdefault(row['case_id'], []).append(difference)
            expected = sum(sum(v) / len(v) for v in bycase.values()) / len(bycase)
            self.assertAlmostEqual(
                group['case_weighted']['paired_branch_minus_full_agreement']['mean'], expected)
            for value in group['slot_weighted'].values():
                self.assertEqual(value['n_ties_excluded'], value['n_excluded'])

    def test_duplicate_csv_headers_rejected(self):
        path = self.root / 'duplicate.csv'
        path.write_text('case_id,case_id\na,b\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate CSV columns'):
            audit.rows(path, {'case_id'})

    def test_racing_output_creation_preserves_existing_file(self):
        original = Path.open
        def racing_open(path, *args, **kwargs):
            if path.name == 'summary.json' and args and args[0] == 'x':
                with original(path, 'w') as f:
                    f.write('concurrent output')
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, 'open', racing_open):
            with self.assertRaises(FileExistsError):
                self.run_audit()
        self.assertEqual((self.root / 'out/summary.json').read_text(), 'concurrent output')
        self.assertFalse((self.root / 'out/details.csv').exists())

    def test_cli_validation(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            audit.main(['--bootstrap', '99'])
        with self.assertRaisesRegex(ValueError, 'paths must differ'):
            audit.main(['--output', str(self.root / 'same'), '--details', str(self.root / 'same')])


if __name__ == '__main__':
    unittest.main()
