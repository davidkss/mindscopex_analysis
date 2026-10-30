"""No model libraries required for the representation/reference-selection tests."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import probe_growth019_precision as probe


class PrecisionTests(unittest.TestCase):
    def test_equal_bf16_values_remain_equal_after_fp32_conversion(self):
        row = probe.scalar_probe()['rows'][0]
        self.assertEqual(row['bf16_values'], [16.375,16.375])
        self.assertEqual(row['fp32_after_bf16'], row['bf16_values'])
        self.assertEqual(row['reference_bf16_token'],17)
        self.assertEqual(row['reference_widened_token'],17)

    def test_small_fp32_advantage_can_be_lost_by_bf16_rounding(self):
        row = probe.scalar_probe()['rows'][1]
        self.assertEqual(row['reference_fp32_token'],20)
        self.assertEqual(row['reference_bf16_token'],17)
        self.assertEqual(row['reference_widened_token'],17)

    def test_spacing_and_surviving_difference(self):
        result = probe.scalar_probe()
        self.assertEqual(result['bf16_spacing_at_16_375'],.125)
        self.assertEqual(result['rows'][3]['reference_bf16_token'],20)
        self.assertEqual(probe.bf16_rne(16.4375),16.5)  # ties to even significand
        self.assertEqual(probe.bf16_rne(16.3125),16.25)

    def test_mask_order_and_gather_order_are_different_rules(self):
        rows = probe.scalar_probe()['allowed_order_probe']
        self.assertEqual([r['full_vocab_reference_token'] for r in rows],[17,17])
        self.assertEqual([r['gathered_first_max_token'] for r in rows],[17,20])

    def test_torch_unavailable_is_reported_without_install(self):
        with mock.patch.object(probe.importlib.util,'find_spec',return_value=None):
            self.assertEqual(probe.torch_probe(),dict(status='not installed',actual_tensor_cases=0))

    def test_output_exclusive_and_nan_rejected(self):
        with self.assertRaises(ValueError):
            probe.bf16_rne(float('nan'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'probe.json'
            with mock.patch.object(probe,'torch_probe',return_value=dict(status='not installed')), mock.patch('builtins.print'):
                probe.main(['--output',str(path)])
            before = path.read_bytes()
            self.assertEqual(json.loads(before)['torch']['status'],'not installed')
            with self.assertRaises(FileExistsError):
                probe.main(['--output',str(path)])
            self.assertEqual(path.read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
