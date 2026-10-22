"""Synthetic fixtures never substitute for the separate real-data regression check."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'multi', Path(__file__).resolve().parents[1] / 'scripts/analyze_crt_negative_dose_multiseed.py')
multi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(multi)


def fixture():
    rows = [dict(case_id=f'crt_fresh_v2_{multi.single.FAMILIES[i % 3][4:]}_{i:03}',
                 family=multi.single.FAMILIES[i % 3],
                 label='correct' if i % 2 else 'lure', answer='correct' if i % 2 else 'lure')
            for i in range(40)]
    data = []
    for c in multi.single.COEFFICIENTS:
        after = copy.deepcopy(rows)
        if c:
            after[0].update(label='correct', answer='correct')
        data.append(dict(coefficient=c, output_mode='binary_choice',
                         baseline_rows=copy.deepcopy(rows), steered_rows=after))
    return data


class MultiSeedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'runs'
        self.paths = {}
        for seed in range(1, 6):
            path = self.root / f'test_study_transfer_fresh_v2_negative_dose_f30908_seed{seed}_2b' / 'artifacts/behavioral/generations.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(fixture()))
            self.paths[seed] = path
        # Explicitly synthetic legacy reference, rather than disabling legacy validation.
        self.reference = Path(self.tmp.name) / 'legacy'
        self.reference.mkdir()
        totals, families, items = multi.single.analyze(fixture())
        for name, rows in (('coefficients', totals), ('families', families), ('transitions', items)):
            multi.single.write_csv(self.reference / f'{name}.csv', rows)

    def summary(self):
        # Synthetic accuracies intentionally differ from published seed 1; use seeds 2-5.
        return multi.summarize({s: p for s, p in self.paths.items() if s != 1})[0]

    def test_five_seed_discovery_and_summary(self):
        self.assertEqual(multi.discover(self.root, range(1, 6)), self.paths)
        with patch.object(multi, 'check_seed1') as check:
            data, _ = multi.summarize(self.paths)
            check.assert_called_once()
        self.assertEqual(len(data['seed_summary']), 30)
        self.assertEqual(data['coefficient_summary'][1]['total'], 200)
        self.assertEqual(data['seed_summary'][1]['net_improvement_pp'], 2.5)
        self.assertEqual(data['seed_summary'][1]['steered_correct'], 21)
        self.assertEqual(len(data['case_overlap']), 40)
        self.assertTrue(all(r['seed_count'] == 5 for r in data['case_overlap']))
        unique = next(r for r in data['unique_case_summary']
                      if r['coefficient'] == -1 and r['case_id'].endswith('000'))
        self.assertEqual(unique['lure_to_correct'], 5)
        family = next(r for r in data['family_summary']
                      if r['seed'] == 1 and r['coefficient'] == -1 and r['family'] == 'crt_difference')
        self.assertEqual((family['total'], family['lure_to_correct']), (14, 1))

    def test_missing_duplicate_inputs(self):
        self.paths[5].unlink()
        with self.assertRaisesRegex(ValueError, 'Seed 5.*found 0'):
            multi.discover(self.root, range(1, 6))
        duplicate = self.root / 'other_study_transfer_fresh_v2_negative_dose_f30908_seed1_2b/artifacts/behavioral/generations.json'
        duplicate.parent.mkdir(parents=True)
        duplicate.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'Seed 1.*found 2'):
            multi.discover(self.root, [1])

    def test_invalid_data(self):
        for kind in ('coefficient_missing', 'coefficient_duplicate', 'ids', 'zero', 'baseline',
                     'count', 'field', 'label', 'answer'):
            with self.subTest(kind=kind):
                data = fixture()
                if kind == 'coefficient_missing':
                    data.pop()
                elif kind == 'coefficient_duplicate':
                    data.append(copy.deepcopy(data[-1]))
                elif kind == 'ids':
                    data[1]['steered_rows'].pop()
                elif kind == 'zero':
                    data[0]['steered_rows'][0]['answer'] = 'changed'
                elif kind == 'baseline':
                    data[1]['baseline_rows'][0]['answer'] = 'changed'
                elif kind == 'count':
                    for block in data:
                        for key in ('baseline_rows', 'steered_rows'):
                            block[key].pop()
                elif kind == 'field':
                    del data[1]['steered_rows'][0]['answer']
                elif kind == 'label':
                    data[1]['steered_rows'][0]['label'] = 'unknown'
                elif kind == 'answer':
                    data[1]['steered_rows'][0]['answer'] = 'lure'
                self.paths[2].write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    multi.summarize({2: self.paths[2]})

    def test_transition_sum_and_undefined_rates(self):
        with self.assertRaisesRegex(ValueError, 'transition counts'):
            multi.single.aggregate([dict(baseline_label='lure', transition='invalid')], 0)
        for label, field in (('correct', 'correction_rate'), ('lure', 'regression_rate')):
            row = multi.single.aggregate([dict(baseline_label=label,
                                               transition=f'{label}_to_{label}')], 0)
            self.assertIsNone(row[field])

    def test_disagreements_preserved(self):
        data = fixture()
        data[1]['steered_rows'] = copy.deepcopy(data[1]['baseline_rows'])
        self.paths[2].write_text(json.dumps(data))
        rows = self.summary()['unique_case_summary']
        row = next(r for r in rows if r['case_id'].endswith('000') and r['coefficient'] == -1)
        self.assertTrue(row['steered_label_disagreement'])
        self.assertEqual(row['lure_to_correct'], 3)
        self.assertEqual(row['lure_to_lure'], 1)
        self.assertEqual(len(json.loads(row['seed_outcomes'])), 4)

    def test_actual_seed1_legacy_and_all_seeds(self):
        root = multi.ROOT / 'results/runs'
        if not root.exists():
            self.skipTest('real research inputs unavailable')
        data, _ = multi.summarize(multi.discover(root, range(1, 6)))
        self.assertEqual(len(data['seed_summary']), 30)
        paths = multi.discover(root, [1])
        totals, families, items = multi.single.analyze(json.loads(paths[1].read_text()))
        totals[0]['overall_accuracy'] = 0
        with self.assertRaisesRegex(ValueError, 'published accuracy mismatch'):
            multi.check_seed1(totals, families, items, multi.REFERENCE)

    def test_outputs_and_no_overwrite(self):
        output = Path(self.tmp.name) / 'analysis'
        before = {s: p.read_bytes() for s, p in self.paths.items()}
        multi.run(self.root, output, (2, 3, 4, 5))
        self.assertEqual(len(list(output.iterdir())), 9)
        saved = {p.name: p.read_bytes() for p in output.iterdir()}
        with self.assertRaises(FileExistsError):
            multi.run(self.root, output, (2, 3, 4, 5))
        self.assertEqual(saved, {p.name: p.read_bytes() for p in output.iterdir()})
        self.assertEqual(before, {s: p.read_bytes() for s, p in self.paths.items()})


if __name__ == '__main__':
    unittest.main()
