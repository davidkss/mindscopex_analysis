"""CPU-only stability grid, fixed-vector edit telemetry and collection regressions."""
import copy
import importlib.util
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from experiments.jobs import research_experiments as job
from mindscopex_analysis import research
from mindscopex_analysis.open_ended_crt import STABILITY_COEFFICIENTS, evaluate
from mindscopex_analysis.qwen_scope import make_feature_steering_hook
from test_crt_open_ended import NonThinkingTokenizer, StubGenerator, cases

SPEC = importlib.util.spec_from_file_location('stability_common', job.ROOT / 'scripts/analyze_crt_open_ended.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def sae_fixture():
    decoder = torch.zeros(3, 5)
    decoder[:, 1] = torch.tensor([3., 4., 0.])
    return SimpleNamespace(W_dec=decoder, d_model=3)


def diagnostics(coefficient):
    records = []
    hook = make_feature_steering_hook(sae_fixture(), [1], coefficient=coefficient,
                                     token_position='all', diagnostics=records)
    hook(None, (), torch.ones(1, 2, 3))
    return records


def collect(items, coefficient):
    rows = research._generate_labels(StubGenerator(), NonThinkingTokenizer(), items,
                                    max_new_tokens=64, output_mode='open_ended_crt', thinking_mode='disabled')
    return dict(baseline_rows=rows, steered_rows=copy.deepcopy(rows),
                steering_diagnostics=diagnostics(coefficient))


class StabilityTests(unittest.TestCase):
    def test_configs_same_full_cases_and_smoke_is_six_case_subset(self):
        base_path = job.ROOT / 'experiments/configs/study_open_ended_crt_nothink64_f30908_seed1_2b.toml'
        base = job.load_toml(base_path)
        original = job.family_balanced_subset(job._load_splits(base)['test'], max_cases=40)
        for smoke in (False, True):
            protocol = 'stability_smoke' if smoke else 'stability_sweep'
            config = job.load_toml(base_path.with_name(f'study_steering_{protocol}_f30908_seed1_2b.toml'))
            for field in ('model', 'data', 'feature'): self.assertEqual(config[field], base[field])
            self.assertEqual(config['behavioral']['coefficients'], list(STABILITY_COEFFICIENTS))
            self.assertEqual(config['behavioral']['thinking_mode'], 'disabled')
            self.assertEqual(config['behavioral']['max_new_tokens'], 64)
            self.assertTrue(config['behavioral']['steering_diagnostics'])
            selected = job.family_balanced_subset(job._load_splits(config)['test'], max_cases=config['behavioral']['max_cases'])
            self.assertEqual(selected, original[:6] if smoke else original)
            if smoke:
                self.assertEqual({c.family: sum(r.family == c.family for r in selected) for c in selected},
                                 dict(crt_difference=2, crt_rate=2, crt_growth=2))

    def test_fixed_vector_norms_and_observer_does_not_change_edit(self):
        hidden = torch.tensor([[[6., 8., 0.], [0., 5., 0.]]])
        for position in ('all', 'last'):
            rows = []
            observed = make_feature_steering_hook(sae_fixture(), [1], coefficient=-.5,
                                                  token_position=position, diagnostics=rows)
            plain = make_feature_steering_hook(sae_fixture(), [1], coefficient=-.5, token_position=position)
            original = hidden.clone()
            result = observed(None, (), (hidden, 'cache'))
            self.assertTrue(torch.equal(result[0], plain(None, (), hidden)))
            self.assertEqual(result[1], 'cache')
            self.assertTrue(torch.equal(hidden, original))
            r = rows[0]
            self.assertEqual(r['requested_vector_norm'], 2.5)  # Decoder norm=5, not unit-normalized.
            self.assertEqual(r['actual_delta_mean'], 2.5)
            self.assertEqual(r['affected_token_count'], 2 if position == 'all' else 1)
            self.assertAlmostEqual(r['delta_residual_ratio_mean'], .375 if position == 'all' else .5)
            self.assertAlmostEqual(r['residual_before_mean'], 7.5 if position == 'all' else 5)

    def test_zero_edit_zero_residual_and_dtype_rounding(self):
        records = diagnostics(0)
        self.assertEqual(records[0]['actual_delta_mean'], 0)
        self.assertEqual(records[0]['delta_residual_ratio_mean'], 0)
        rows = []
        hook = make_feature_steering_hook(sae_fixture(), [1], coefficient=-.25, token_position='all', diagnostics=rows)
        hook(None, (), torch.zeros(1, 2, 3))
        self.assertEqual(rows[0]['zero_residual_count'], 2)
        self.assertIsNone(rows[0]['delta_residual_ratio_mean'])
        self.assertEqual(rows[0]['ratio_defined_count'], 0)
        hook(None, (), torch.full((1, 1, 3), 10000., dtype=torch.float16))
        self.assertGreater(rows[1]['cast_vector_norm'], 0)
        self.assertEqual(rows[1]['actual_delta_mean'], 0)  # Addition rounded away in float16.

    def test_six_coefficient_validation_and_summaries(self):
        blocks = evaluate(cases(), STABILITY_COEFFICIENTS, collect, 64, 'disabled', 'stability_sweep')
        rows, _ = analysis.analyze(blocks, 1, 'stability_sweep')
        data = analysis.summarize(rows, STABILITY_COEFFICIENTS)
        self.assertEqual(len(data['seed_summary']), 6)
        self.assertTrue(all(r['steered_eos_rate'] == 1 for r in data['seed_summary']))
        self.assertTrue(all(r['steered_repetition_rate'] == 0 for r in data['seed_summary']))
        self.assertEqual(blocks[0]['baseline_rows'], blocks[0]['steered_rows'])
        wrong = copy.deepcopy(blocks)
        wrong[1]['coefficient'] = -.75
        with self.assertRaises(ValueError): analysis.analyze(wrong, 1, 'stability_sweep')
        wrong = copy.deepcopy(blocks)
        wrong[0]['steered_rows'][0]['raw_text'] += ' '
        with self.assertRaisesRegex(ValueError, 'zero'): analysis.analyze(wrong, 1, 'stability_sweep')
        with self.assertRaises(ValueError): evaluate(cases(), STABILITY_COEFFICIENTS, collect, 64)
        with self.assertRaises(ValueError): evaluate(cases()[:6], STABILITY_COEFFICIENTS, collect, 16, 'disabled', 'stability_smoke')
        with self.assertRaises(ValueError): evaluate(cases()[:6], STABILITY_COEFFICIENTS, collect, 64, 'disabled', 'stability_sweep')

    def test_diagnostics_flag_routed_and_hook_cleanup(self):
        block, handle = mock.Mock(), mock.Mock()
        block.register_forward_hook.return_value = handle
        sae = sae_fixture()
        def fake_labels(*args, **kwargs):
            if kwargs['phase'].startswith('steer'):
                callback = block.register_forward_hook.call_args.args[0]
                callback(None, (), torch.ones(1, 2, 3))
            return [{'label': 'correct'}]
        with mock.patch.object(research, '_generate_labels', side_effect=fake_labels), \
             mock.patch.object(research, 'find_decoder_block', return_value=block):
            result = research.steer_generation_labels(object(), object(), [], layer=5, sae=sae,
                        feature_id=1, coefficient=-.5, token_position='all', steering_diagnostics=True)
        self.assertEqual(len(result['steering_diagnostics']), 1)
        handle.remove.assert_called_once()
        handle.reset_mock()
        with mock.patch.object(research, '_generate_labels', side_effect=[[], RuntimeError('CPU stub')]), \
             mock.patch.object(research, 'find_decoder_block', return_value=block):
            with self.assertRaises(RuntimeError):
                research.steer_generation_labels(object(), object(), [], layer=5, sae=sae,
                            feature_id=1, coefficient=-.5, steering_diagnostics=True)
        handle.remove.assert_called_once()

    def test_smoke_job_provenance_analysis_and_output_protection(self):
        config_path = job.ROOT / 'experiments/configs/study_steering_stability_smoke_f30908_seed1_2b.toml'
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(mock.patch('torch.cuda.is_available', return_value=False))
            stack.enter_context(mock.patch.object(job, '_load_sae', return_value=object()))
            stack.enter_context(mock.patch.object(job, 'load_qwen_text_generation_model', return_value=(object(), object())))
            def fake(model, tokenizer, items, **kw):
                self.assertTrue(kw['steering_diagnostics'])
                self.assertEqual(kw['thinking_mode'], 'disabled')
                self.assertEqual(kw['max_new_tokens'], 64)
                return collect(items, kw['coefficient'])
            stack.enter_context(mock.patch.object(job, 'steer_generation_labels', side_effect=fake))
            stack.enter_context(mock.patch.object(job, 'clear_device_cache'))
            root = Path(tmp)
            actual = job.run(config_path, root / 'job')
            before = {p: p.read_bytes() for p in actual.rglob('*') if p.is_file()}
            with self.assertRaises(FileExistsError): job.run(config_path, root / 'job')
            self.assertEqual(before, {p: p.read_bytes() for p in actual.rglob('*') if p.is_file()})
            artifact = root / 'runs/test_study_steering_stability_smoke_f30908_seed1_2b/artifacts'
            for path in actual.rglob('*'):
                if path.is_file():
                    destination = artifact / path.relative_to(actual)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(path.read_bytes())
            output = root / 'analysis'
            data = analysis.run(root / 'runs', output, seeds=(1,),
                                run_prefix='study_steering_stability_smoke', protocol='stability_smoke')
            self.assertEqual(len(data['steering_norm_diagnostics']), 6)
            self.assertTrue((output / 'steering_norm_diagnostics.csv').exists())
            self.assertTrue(all(r['total'] == 6 for r in data['seed_summary']))
            with self.assertRaises(FileExistsError):
                analysis.run(root / 'runs', output, seeds=(1,), protocol='stability_smoke')


if __name__ == '__main__':
    unittest.main()
