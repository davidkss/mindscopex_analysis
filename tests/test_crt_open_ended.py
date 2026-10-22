"""CPU-only unconstrained CRT scoring, collection, provenance and paired analysis."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from experiments.jobs import research_experiments as job
from experiments.runners import letter_run_directory as guard
from experiments.runners.launch_colab import resolve_configs
from experiments.runners.remote_bootstrap import render_bootstrap
from mindscopex_analysis import research
from mindscopex_analysis.cases import LureCase
from mindscopex_analysis.open_ended_crt import (
    COEFFICIENTS, FAMILIES, MODE, classify, evaluate, validate_case,
)

SPEC = importlib.util.spec_from_file_location('open_analysis', job.ROOT / 'scripts/analyze_crt_open_ended.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
CONFIG = job.ROOT / 'experiments/configs/study_open_ended_crt_f30908_seed1_2b.toml'


def cases():
    answers = {'crt_difference': (' $6', ' $12'), 'crt_rate': (' 3 minutes', ' 20 minutes'),
               'crt_growth': (' 19 days', ' 10 days')}
    return [LureCase(f'crt_fresh_v2_{FAMILIES[i % 3][4:]}_{i:03}', FAMILIES[i % 3],
                     'Original question requesting a single numeric answer.\nAnswer:',
                     *answers[FAMILIES[i % 3]]) for i in range(40)]


def response(case, text, reason='eos', budget=16):
    ids = [7, 9] if reason == 'eos' else [7] * budget if reason == 'max_new_tokens' else [7]
    label, audit = classify(text, case, reason)
    return dict(case_id=case.case_id, family=case.family, prompt=case.prompt,
                answer=text, raw_text=text + ('<eos>' if reason == 'eos' else ''),
                label=label, classification=audit, generated_token_ids=ids,
                generated_token_count=len(ids), eos_token_ids=[9], eos_reached=reason == 'eos',
                max_new_tokens=budget, termination_reason=reason)


def generate(items, c):
    baseline = [response(case, case.correct_answer if i % 2 else case.lure_answer)
                for i, case in enumerate(items)]
    after = copy.deepcopy(baseline)
    if c:
        after[0] = response(items[0], items[0].correct_answer)
        after[1] = response(items[1], items[1].lure_answer)
        after[2] = response(items[2], '999 days')
        after[3] = response(items[3], '')
        after[4] = response(items[4], items[4].correct_answer)
    return dict(baseline_rows=baseline, steered_rows=after)


class Tokenizer:
    eos_token_id = [9, 10]
    pad_token_id = None
    def __call__(self, prompt, **kwargs):
        self.prompt = prompt
        return dict(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.ones(1, 2))
    def decode(self, ids, skip_special_tokens):
        return '$6' if skip_special_tokens else ' $6<eos>'


class StubGenerator:
    def __init__(self, tokens=(7, 9)):
        self.tokens = tokens
    def parameters(self):
        return iter([torch.zeros(1)])
    def generate(self, input_ids, **kwargs):
        self.kwargs = kwargs
        return torch.cat([input_ids, torch.tensor([self.tokens])], dim=1)


class OpenEndedTests(unittest.TestCase):
    def test_numeric_units_and_four_labels(self):
        case = cases()[0]
        for text in ('$6', '6.00 USD', '600 cents', '600¢', '600c', '６ dollars', '6', 'Answer: $6.'):
            with self.subTest(text=text): self.assertEqual(classify(text, case)[0], 'correct')
        self.assertTrue(classify('6', case)[1]['parsed']['unit_inferred'])
        self.assertEqual(classify('$12', case)[0], 'lure')
        self.assertEqual(classify('$7', case)[0], 'other')
        for text in ('', 'A', '6 or 12 dollars', 'not $6', '6 bananas', '6 dollars because 12/2=6',
                     '$6 minutes', '6 dollars\n12 dollars', 'NaN', 'Infinity'):
            with self.subTest(text=text): self.assertEqual(classify(text, case)[0], 'invalid')
        self.assertEqual(classify('$6', case, 'max_new_tokens')[0], 'invalid')
        rate = cases()[1]
        for text in ('3 minutes', '3.0 minute', '180 seconds', '.05 hours', '3 min', '3'):
            with self.subTest(text=text): self.assertEqual(classify(text, rate)[0], 'correct')
        for text in ('19 days', '19th day', 'day 19', '456 hours'):
            self.assertEqual(classify(text, cases()[2])[0], 'correct')
        self.assertEqual(classify('1,000 dollars', replace(case, correct_answer='$1000'))[0], 'correct')
        self.assertEqual(classify('5 cents', replace(case, correct_answer='$0.05'))[0], 'correct')
        with self.assertRaisesRegex(ValueError, 'identical'):
            validate_case(replace(case, correct_answer='$6', lure_answer='600 cents'))
        with self.assertRaisesRegex(ValueError, 'identical'):
            validate_case(replace(case, correct_answer='$0', lure_answer='$-0.00'))
        with self.assertRaises(ValueError):
            validate_case(replace(case, prompt=case.prompt.replace('Answer:', 'A: $6\nB: $12\nAnswer:')))

    def test_unconstrained_generation_and_raw_metadata(self):
        case = cases()[0]
        model, tokenizer = StubGenerator(), Tokenizer()
        row = research._generate_labels(model, tokenizer, [case], max_new_tokens=16, output_mode=MODE)[0]
        self.assertEqual(row['label'], 'correct')
        self.assertEqual(row['raw_text'], ' $6<eos>')
        self.assertEqual(row['answer'], '$6')
        self.assertEqual(row['generated_token_ids'], [7, 9])
        self.assertTrue(row['eos_reached'])
        self.assertEqual(row['termination_reason'], 'eos')
        self.assertEqual(row['max_new_tokens'], 16)
        self.assertEqual(tokenizer.prompt, case.prompt)
        self.assertNotIn('$6', tokenizer.prompt)
        self.assertNotIn('$12', tokenizer.prompt)
        self.assertNotIn('A:', tokenizer.prompt)
        self.assertNotIn('prefix_allowed_tokens_fn', model.kwargs)
        self.assertNotIn('constraints', model.kwargs)
        self.assertFalse(model.kwargs['do_sample'])
        self.assertIsNone(model.kwargs['forced_eos_token_id'])
        self.assertEqual(model.kwargs['eos_token_id'], [9, 10])
        truncated = research._generate_labels(StubGenerator((7,) * 16), Tokenizer(), [case],
                                              max_new_tokens=16, output_mode=MODE)[0]
        self.assertEqual(truncated['termination_reason'], 'max_new_tokens')
        self.assertEqual(truncated['label'], 'invalid')
        stopped = research._generate_labels(StubGenerator((7,)), Tokenizer(), [case],
                                           max_new_tokens=16, output_mode=MODE)[0]
        self.assertEqual(stopped['termination_reason'], 'other_stop')
        self.assertEqual(stopped['label'], 'invalid')

    def test_seed_cases_original_prompts_and_settings(self):
        self.assertEqual(len(resolve_configs(job.ROOT / 'experiments/suites/crt_open_ended_control.toml')), 5)
        for seed in range(1, 6):
            config = job.load_toml(CONFIG.with_name(f'study_open_ended_crt_f30908_seed{seed}_2b.toml'))
            previous = job.load_toml(CONFIG.with_name(f'study_choice_letter_f30908_seed{seed}_2b.toml'))
            for key in ('model', 'data', 'feature'): self.assertEqual(config[key], previous[key])
            selected = job.family_balanced_subset(job._load_splits(config)['test'], max_cases=40)
            old = job.family_balanced_subset(job._load_splits(previous)['test'], max_cases=40)
            self.assertEqual(selected, old)
            self.assertEqual(len(selected), 40)
            for case in selected:
                validate_case(case)
                self.assertNotRegex(case.prompt, r'(?m)^\s*[AB]:')
            self.assertNotIn('choice_orders', config['behavioral'])
            self.assertEqual(config['behavioral']['max_new_tokens'], 16)

    def test_four_state_paired_aggregation(self):
        blocks = evaluate(cases(), COEFFICIENTS, generate, 16)
        rows, _ = analysis.analyze(blocks, 1)
        data = analysis.summarize(rows)
        r = data['seed_summary'][1]
        self.assertEqual(r['lure_to_correct'], 2)
        self.assertEqual(r['correct_to_lure'], 1)
        self.assertEqual(r['lure_to_other'], 1)
        self.assertEqual(r['correct_to_invalid'], 1)
        self.assertEqual(r['baseline_correct_rate'], .5)
        self.assertEqual(r['steered_correct_rate'], .5)
        self.assertEqual(r['steered_other_rate'], .025)
        self.assertEqual(r['steered_invalid_rate'], .025)
        self.assertEqual(r['accuracy_change_pp'], 0)
        self.assertAlmostEqual(r['lure_change_pp'], -5)
        self.assertEqual(sum(r[f'{b}_to_{a}'] for b in ('correct','lure','other','invalid')
                             for a in ('correct','lure','other','invalid')), 40)
        doubled = rows + [dict(r, seed=2) for r in rows]
        self.assertEqual(len(analysis.summarize(doubled)['case_overlap']), 40)
        self.assertTrue(all(r['seed_count'] == 2 for r in analysis.summarize(doubled)['case_overlap']))

    def test_invalid_inputs(self):
        for kind in ('missing', 'duplicate', 'zero', 'baseline', 'prompt', 'label', 'metadata',
                     'case_content', 'budget', 'coefficient', 'duplicate_coefficient'):
            with self.subTest(kind=kind):
                data = evaluate(cases(), COEFFICIENTS, generate, 16)
                block = data[1]
                if kind == 'missing': block['steered_rows'].pop()
                elif kind == 'duplicate': block['steered_rows'][-1] = copy.deepcopy(block['steered_rows'][0])
                elif kind == 'zero': data[0]['steered_rows'][0] = response(cases()[0], '$6')
                elif kind == 'baseline': block['baseline_rows'][0] = response(cases()[0], '$6')
                elif kind == 'prompt': block['steered_rows'][0]['prompt'] += '\nA: $6'
                elif kind == 'label': block['steered_rows'][0]['label'] = 'other'
                elif kind == 'metadata': block['steered_rows'][0]['generated_token_count'] = 999
                elif kind == 'case_content': block['cases'][0]['note'] = 'changed'
                elif kind == 'budget': block['max_new_tokens'] = 32
                elif kind == 'coefficient': block['coefficient'] = -4
                elif kind == 'duplicate_coefficient': block['coefficient'] = 0
                with self.assertRaises(ValueError): analysis.analyze(data, 1)
        with self.assertRaises(ValueError): evaluate(cases()[:39], COEFFICIENTS, generate, 16)

    def make_run(self, root, seed=1):
        config = CONFIG.with_name(f'study_open_ended_crt_f30908_seed{seed}_2b.toml')
        artifact = root / f'test_study_open_ended_crt_f30908_seed{seed}_2b' / 'artifacts'
        artifact.mkdir(parents=True)
        (artifact / 'config.toml').write_bytes(config.read_bytes())
        with ExitStack() as stack:
            stack.enter_context(mock.patch('torch.cuda.is_available', return_value=False))
            stack.enter_context(mock.patch.object(job, '_load_sae', return_value=object()))
            stack.enter_context(mock.patch.object(job, 'load_qwen_text_generation_model', return_value=(object(), object())))
            stack.enter_context(mock.patch.object(job, 'steer_generation_labels', side_effect=lambda model, tokenizer, items, **kw: generate(items, kw['coefficient'])))
            stack.enter_context(mock.patch.object(job, 'clear_device_cache'))
            # Exercise actual job lifecycle, metadata and output protection without inference.
            output = job.run(config, artifact.parent / 'job_output')
        for path in output.rglob('*'):
            if path.is_file():
                destination = artifact / path.relative_to(output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(path.read_bytes())
        return artifact

    def test_full_job_analysis_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'runs'
            artifact = self.make_run(root)
            before = {p: p.read_bytes() for p in artifact.rglob('*') if p.is_file()}
            output = Path(tmp) / 'analysis'
            data = analysis.run(root, output, seeds=(1,))
            self.assertEqual(len(data['seed_summary']), 3)
            self.assertEqual(len(list(output.iterdir())), 7)
            self.assertEqual(before, {p: p.read_bytes() for p in artifact.rglob('*') if p.is_file()})
            with self.assertRaises(FileExistsError): analysis.run(root, output, seeds=(1,))
            status = json.loads((artifact / 'open_ended_run_status.json').read_text())
            self.assertEqual(status['phase'], 'succeeded')
            self.assertEqual(status['returncode'], 0)
            raw = artifact / 'behavioral/generations.json'
            raw.write_bytes(raw.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                analysis.run(root, Path(tmp) / 'bad', seeds=(1,))
            self.assertFalse((Path(tmp) / 'bad').exists())

    def test_objective_case_comparison(self):
        # Canonical cases are preserved in both output modes, including instruction text.
        from mindscopex_analysis.choice_order import evaluate as evaluate_choice
        def objective_generate(items, c):
            baseline = [dict(case_id=case.case_id, family=case.family,
                             answer=(case.correct_answer if i % 2 else case.lure_answer).strip(),
                             label='correct' if i % 2 else 'lure') for i, case in enumerate(items)]
            return dict(baseline_rows=baseline, steered_rows=copy.deepcopy(baseline))
        rows, _ = analysis.analyze(evaluate(cases(), COEFFICIENTS, generate, 16), 1)
        for mode in ('binary_choice', 'binary_choice_letter'):
            data = evaluate_choice(cases(), COEFFICIENTS, objective_generate, mode)
            objective = analysis.choice_analysis.common.analyze(data, 1, mode)
            paired, summary = analysis.compare_objective(rows, objective, mode)
            self.assertEqual(len(paired), 240)
            self.assertTrue(all(r['baseline_semantic_agreement'] == 1 for r in summary))
            self.assertLess(summary[8]['steered_semantic_agreement'], 1)

    def test_five_seed_integrated_objective_analysis(self):
        from mindscopex_analysis.choice_order import evaluate as evaluate_choice
        def objective_generate(items, coefficient):
            rows = [dict(case_id=c.case_id, family=c.family,
                         answer=(c.correct_answer if i % 2 else c.lure_answer).strip(),
                         label='correct' if i % 2 else 'lure') for i, c in enumerate(items)]
            return dict(baseline_rows=rows, steered_rows=copy.deepcopy(rows))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'runs'
            for seed in range(1, 6):
                artifact = self.make_run(root, seed)
                raw = json.loads((artifact / 'behavioral/generations.json').read_text())
                original = [LureCase(**c) for c in raw[0]['cases']]
                for stem, mode in (('study_choice_order_counterbalance', 'binary_choice'),
                                   ('study_choice_letter', 'binary_choice_letter')):
                    destination = root / f'test_{stem}_f30908_seed{seed}_2b/artifacts/behavioral/generations.json'
                    destination.parent.mkdir(parents=True)
                    destination.write_text(json.dumps(evaluate_choice(original, COEFFICIENTS, objective_generate, mode)))
            output = Path(tmp) / 'analysis'
            data = analysis.run(root, output, objective_root=root)
            self.assertEqual(len(data['seed_summary']), 15)
            self.assertEqual(data['coefficient_summary'][0]['total'], 200)
            self.assertEqual(len(data['objective_case_comparison']), 2400)
            self.assertEqual(len(list(output.iterdir())), 9)
            self.assertLess(len(data['case_overlap']), 200)
            # A requested objective comparison must fail rather than drop a missing seed.
            missing = next(root.glob('*study_choice_letter_f30908_seed5_2b/artifacts/behavioral/generations.json'))
            missing.unlink()
            with self.assertRaisesRegex(ValueError, 'Seed 5'):
                analysis.run(root, Path(tmp) / 'incomplete', objective_root=root)
            self.assertFalse((Path(tmp) / 'incomplete').exists())

    def test_open_guard_and_bootstrap_failure_artifacts(self):
        for returncode in (0, 7):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = job.load_toml(CONFIG)
                name = job.run_name(config)
                payload = dict(remote_repo_dir=str(job.ROOT), remote_output_root=str(root / 'output'),
                               remote_config_path=str(CONFIG), run_name=name, job='research_experiments',
                               job_path='experiments/jobs/research_experiments.py')
                source = render_bootstrap(payload).rsplit('\nmain()', 1)[0]
                source = source.replace('Path("/content")', f'Path({str(root)!r})')
                namespace = {}; exec(compile(source, '<CPU bootstrap>', 'exec'), namespace)
                namespace['prepare_source'] = lambda: job.ROOT
                namespace['ensure_dependencies'] = lambda repo: None
                run_dir = root / 'output' / name
                def capture(command, log_path, **kwargs):
                    log_path.write_text('synthetic log')
                    with mock.patch.dict(os.environ, kwargs['env']):
                        guard.claim(run_dir, CONFIG, mode=MODE)
                    return returncode
                namespace['run_capturing'] = capture
                with mock.patch.object(os, 'chdir'), mock.patch.dict(os.environ):
                    if returncode:
                        with self.assertRaises(SystemExit): namespace['main']()
                    else: namespace['main']()
                with zipfile.ZipFile(run_dir.with_suffix('.zip')) as archive:
                    status = json.loads(archive.read('open_ended_run_status.json'))
                    self.assertEqual(status['phase'], 'failed' if returncode else 'succeeded')
                    self.assertEqual(status['returncode'], returncode)
                before = {p: p.read_bytes() for p in run_dir.rglob('*') if p.is_file()}
                with self.assertRaises(FileExistsError): guard.claim(run_dir, CONFIG, mode=MODE)
                self.assertEqual(before, {p: p.read_bytes() for p in run_dir.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
