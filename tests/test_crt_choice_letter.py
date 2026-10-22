"""CPU stubs and synthetic fixtures only; no checkpoint or model inference."""
from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.jobs import research_experiments as job
from experiments.runners.launch_colab import resolve_configs
from mindscopex_analysis import research
from mindscopex_analysis.cases import LureCase
from mindscopex_analysis.choice_order import (
    COEFFICIENTS, LETTER_MODE, ORDERS, evaluate, letter_label, ordered_case,
)

SPEC = importlib.util.spec_from_file_location('letter_analysis', job.ROOT / 'scripts/analyze_crt_choice_letter.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def cases():
    return [LureCase(f'crt_fresh_v2_{analysis.common.single.FAMILIES[i % 3][4:]}_{i:03}',
                     analysis.common.single.FAMILIES[i % 3], f'Question {i}?\nAnswer:', ' 2', ' 1')
            for i in range(40)]


def generate(items, coefficient):
    baseline = [dict(case_id=c.case_id, family=c.family,
                     answer=(c.correct_answer if i % 2 else c.lure_answer).strip(),
                     label='correct' if i % 2 else 'lure') for i, c in enumerate(items)]
    after = copy.deepcopy(baseline)
    if coefficient:
        after[0].update(answer=items[0].correct_answer.strip(), label='correct')
        after[1].update(answer=items[1].lure_answer.strip(), label='lure')
        after[2].update(answer=items[2].correct_answer.strip(), label='correct')
    return dict(baseline_rows=baseline, steered_rows=after)


class Tokenizer:
    eos_token_id = [9, 10]
    pad_token_id = None

    def __init__(self, tokens=None):
        self.tokens = tokens or {'A': [3, 5], 'B': [3, 6]}
        self.encoded = []

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        self.encoded.append(text)
        return self.tokens[text]

    def __call__(self, prompt, *, return_tensors):
        assert return_tensors == 'pt'
        return dict(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.ones(1, 2))


class StubGenerator:
    """Walk the constraint callback with predetermined CPU token IDs."""
    def __init__(self, tokens=(3, 5, 9), check=True):
        self.tokens, self.check = tokens, check

    def parameters(self):
        return iter([torch.zeros(1)])

    def generate(self, input_ids, prefix_allowed_tokens_fn, **kwargs):
        self.kwargs = kwargs
        sequence = input_ids.clone()
        self.allowed = []
        for token in self.tokens:
            allowed = prefix_allowed_tokens_fn(0, sequence[0])
            self.allowed.append(allowed)
            if self.check and token not in allowed:
                raise AssertionError(f'{token} not in {allowed}')
            sequence = torch.cat([sequence, torch.tensor([[token]])], dim=1)
        return sequence


class LetterDecodingTests(unittest.TestCase):
    def test_both_candidates_and_eos(self):
        for final, expected in ((5, 'A'), (6, 'B')):
            model, tokenizer = StubGenerator((3, final, 10)), Tokenizer()
            result = research._binary_choice_completion(
                model, tokenizer, ordered_case(cases()[0], ORDERS[0], LETTER_MODE),
                max_new_tokens=1, letter_candidates=True)
            self.assertEqual(result, expected)
            self.assertEqual(tokenizer.encoded, ['A', 'B'])
            self.assertEqual(model.allowed, [[3], [5, 6], [9, 10]])
            self.assertFalse(model.kwargs['do_sample'])
            self.assertTrue(model.kwargs['renormalize_logits'])
            self.assertEqual(model.kwargs['pad_token_id'], 9)
            self.assertEqual(model.kwargs['max_new_tokens'], 3)
            self.assertIn('attention_mask', model.kwargs)

    def test_single_token_letters_and_fixed_order(self):
        for order in ORDERS:
            tokenizer = Tokenizer({'A': [5], 'B': [6]})
            row = research._generate_labels(
                StubGenerator((5, 9)), tokenizer,
                [ordered_case(cases()[0], order, LETTER_MODE)],
                max_new_tokens=16, output_mode=LETTER_MODE)[0]
            self.assertEqual(row['answer'], 'A')
            self.assertEqual(row['label'], letter_label('A', order))
            self.assertEqual(tokenizer.encoded, ['A', 'B'])

    def test_invalid_tokenization(self):
        for tokens in ({'A': [], 'B': [6]}, {'A': [5], 'B': [5]},
                       {'A': [3], 'B': [3, 6]}, {'A': [9], 'B': [6]}):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                research._binary_choice_completion(
                    StubGenerator(), Tokenizer(tokens), cases()[0],
                    max_new_tokens=16, letter_candidates=True)
        tokenizer = Tokenizer()
        tokenizer.eos_token_id = None
        with self.assertRaisesRegex(ValueError, 'eos_token_id'):
            research._binary_choice_completion(StubGenerator(), tokenizer, cases()[0],
                                              max_new_tokens=16, letter_candidates=True)

    def test_incomplete_invalid_and_missing_eos(self):
        for tokens in ((3,), (3, 5), (7, 9), (3, 5, 6, 9)):
            with self.subTest(tokens=tokens), self.assertRaises(RuntimeError):
                research._binary_choice_completion(
                    StubGenerator(tokens, check=False), Tokenizer(), cases()[0],
                    max_new_tokens=16, letter_candidates=True)

    def test_hook_removed_on_letter_generation_failure(self):
        handle = mock.Mock()
        block = mock.Mock()
        block.register_forward_hook.return_value = handle
        with mock.patch.object(research, '_generate_labels', side_effect=[[], RuntimeError('stub failure')]), \
             mock.patch.object(research, 'find_decoder_block', return_value=block), \
             mock.patch.object(research, 'make_feature_steering_hook'):
            with self.assertRaisesRegex(RuntimeError, 'stub failure'):
                research.steer_generation_labels(
                    object(), object(), [], layer=5, sae=object(), feature_id=30908,
                    coefficient=-7, token_position='all', output_mode=LETTER_MODE)
        handle.remove.assert_called_once()


class LetterExperimentTests(unittest.TestCase):
    def blocks(self):
        return evaluate(cases(), COEFFICIENTS, generate, output_mode=LETTER_MODE)

    def test_prompts_mapping_and_original_case_storage(self):
        from mindscopex_analysis.prompts import instruct_lure_cases, CRT_FINAL_ANSWER_SYSTEM_PROMPT
        original = instruct_lure_cases(cases())[0]
        for order in ORDERS:
            transformed = ordered_case(original, order, LETTER_MODE)
            self.assertIn('exactly one uppercase letter', transformed.prompt)
            self.assertNotIn(CRT_FINAL_ANSWER_SYSTEM_PROMPT, transformed.prompt)
            self.assertIn('Question 0?', transformed.prompt)
            self.assertEqual({transformed.correct_answer, transformed.lure_answer}, {'A', 'B'})
            self.assertEqual(letter_label(transformed.correct_answer, order), 'correct')
            value_case = ordered_case(original, order)
            self.assertIn(CRT_FINAL_ANSWER_SYSTEM_PROMPT, value_case.prompt)
            self.assertEqual(value_case.correct_answer, ' 2')
        for bad in ('a', ' A', 'A ', 'C', '', 'AB', '2'):
            with self.assertRaises(ValueError): letter_label(bad, ORDERS[0])
        blocks = self.blocks()
        self.assertEqual(blocks[0]['cases'][0]['correct_answer'], ' 2')
        self.assertEqual(blocks[0]['response_encoding'], 'choice_letter')
        self.assertEqual(blocks[0]['output_mode'], LETTER_MODE)

    def test_metrics_comparison_and_variability(self):
        letters = analysis.common.analyze(self.blocks(), 1, LETTER_MODE)
        values = analysis.common.analyze(evaluate(cases(), COEFFICIENTS, generate), 1)
        data = analysis.common.summarize(letters)
        r = data['seed_order_summary'][2]
        self.assertEqual(r['baseline_accuracy'], .5)
        self.assertEqual(r['steered_accuracy'], .525)
        self.assertEqual(r['net_improvement_pp'], 2.5)
        self.assertEqual(r['lure_to_correct'], 2)
        self.assertEqual(r['correct_to_lure'], 1)
        self.assertEqual(r['correction_rate'], .1)
        self.assertEqual(r['regression_rate'], .05)
        self.assertEqual(r['baseline_A_rate'], .5)
        self.assertEqual(r['steered_A_rate'], .525)
        paired, summaries = analysis.compare(letters, values)
        self.assertEqual(len(paired), 240)
        self.assertTrue(all(r['steered_label_agrees'] for r in paired))
        self.assertTrue(all(r['steered_label_agreement'] == 1 for r in summaries))
        self.assertTrue(all(r['letter_minus_value_net_pp'] == 0 for r in summaries))
        changed = copy.deepcopy(values)
        changed[0]['steered_choice'] = 'A' if changed[0]['steered_choice'] == 'B' else 'B'
        changed[0]['steered_label'] = 'correct' if changed[0]['steered_label'] == 'lure' else 'lure'
        _, summary = analysis.compare(letters, changed)
        self.assertEqual(summary[0]['steered_label_agreement'], .975)
        self.assertEqual(analysis.variability(data)[0]['std_pp'], None)
        with self.assertRaises(ValueError): analysis.compare(letters, values[:-1])
        with self.assertRaises(ValueError): analysis.compare(letters + letters[:1], values)

    def test_bad_output_and_data(self):
        for kind in ('letter', 'mapping', 'zero', 'baseline', 'missing', 'duplicate',
                     'case_definition', 'prompt', 'coefficient', 'order', 'mode'):
            with self.subTest(kind=kind):
                blocks = self.blocks()
                block = blocks[1]
                if kind == 'letter': block['steered_rows'][0]['answer'] = 'C'
                elif kind == 'mapping': block['steered_rows'][0]['label'] = 'lure'
                elif kind == 'zero': blocks[0]['steered_rows'][0].update(answer='A', choice='A', label='correct')
                elif kind == 'baseline': block['baseline_rows'][0].update(answer='A', choice='A', label='correct')
                elif kind == 'missing': block['steered_rows'].pop()
                elif kind == 'duplicate': block['steered_rows'][-1] = copy.deepcopy(block['steered_rows'][0])
                elif kind == 'case_definition': block['cases'][0]['correct_answer'] = ' 3'
                elif kind == 'prompt': block['prompts'][cases()[0].case_id] += 'changed'
                elif kind == 'coefficient': block['coefficient'] = -4
                elif kind == 'order': block['choice_order'] = 'unknown'
                elif kind == 'mode': block['output_mode'] = 'binary_choice'
                with self.assertRaises(ValueError): analysis.common.analyze(blocks, 1, LETTER_MODE)

    def test_seed_cases_and_configs_match(self):
        suite = job.ROOT / 'experiments/suites/crt_choice_letter_control.toml'
        self.assertEqual(len(resolve_configs(suite)), 5)
        for seed in range(1, 6):
            config = job.load_toml(job.ROOT / f'experiments/configs/study_choice_letter_f30908_seed{seed}_2b.toml')
            previous = job.load_toml(job.ROOT / f'experiments/configs/study_choice_order_counterbalance_f30908_seed{seed}_2b.toml')
            for key in ('model', 'data', 'feature'):
                self.assertEqual(config[key], previous[key])
            self.assertEqual(config['behavioral']['output_mode'], LETTER_MODE)
            selected = job.family_balanced_subset(job._load_splits(config)['test'], max_cases=40)
            old = job.family_balanced_subset(job._load_splits(previous)['test'], max_cases=40)
            self.assertEqual(selected, old)
            self.assertEqual(len(selected), 40)

    def test_job_routing_and_no_overwrite(self):
        config = job.load_toml(job.ROOT / 'experiments/configs/study_choice_letter_f30908_seed1_2b.toml')
        def fake(model, tokenizer, items, **kwargs):
            return generate(items, kwargs['coefficient'])
        state = {'study_feature': {'feature': {'layer': 5, 'feature_id': 30908}, 'sae': object()}}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(job, 'steer_generation_labels', side_effect=fake) as steer:
            root = Path(tmp)
            job.run_behavioral(object(), object(), {'test': cases()}, config, {}, root, state)
            self.assertEqual(steer.call_count, 6)
            self.assertTrue(all(c.kwargs['output_mode'] == LETTER_MODE for c in steer.call_args_list))
            saved = (root / 'behavioral/generations.json').read_bytes()
            analysis.common.analyze(json.loads(saved), 1, LETTER_MODE)
            with self.assertRaises(FileExistsError):
                job.run_behavioral(object(), object(), {'test': cases()}, config, {}, root, state)
            self.assertEqual(saved, (root / 'behavioral/generations.json').read_bytes())
            existing = root / config['run']['name']
            existing.mkdir()
            (existing / 'manifest.json').write_text('{}')
            with mock.patch.object(job, 'load_qwen_text_generation_model') as load:
                with self.assertRaises(FileExistsError):
                    job.run(job.ROOT / 'experiments/configs/study_choice_letter_f30908_seed1_2b.toml', root)
                load.assert_not_called()

    def test_analysis_outputs_and_failure_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'runs'
            paths = []
            for seed in range(1, 6):
                for stem, mode in (('study_choice_letter', LETTER_MODE),
                                   ('study_choice_order_counterbalance', 'binary_choice')):
                    p = root / f'test_{stem}_f30908_seed{seed}_2b/artifacts/behavioral/generations.json'
                    p.parent.mkdir(parents=True)
                    p.write_text(json.dumps(evaluate(cases(), COEFFICIENTS, generate, mode)))
                    paths.append(p)
            before = {p: p.read_bytes() for p in paths}
            output = Path(tmp) / 'analysis'
            data = analysis.run(root, root, output)
            self.assertEqual(len(data['seed_order_summary']), 30)
            self.assertEqual(len(data['encoding_case_comparison']), 1200)
            self.assertEqual(len(list(output.iterdir())), 9)
            self.assertEqual(before, {p: p.read_bytes() for p in paths})
            with self.assertRaises(FileExistsError): analysis.run(root, root, output)
            bad = json.loads(paths[1].read_text())
            for block in bad:
                block['cases'][0]['note'] = 'different case definition'
            paths[1].write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'definitions differ'):
                analysis.run(root, root, Path(tmp) / 'mismatch')
            self.assertFalse((Path(tmp) / 'mismatch').exists())
            paths[0].unlink()
            with self.assertRaisesRegex(ValueError, 'expected one input, found 0'):
                analysis.run(root, root, Path(tmp) / 'missing')


if __name__ == '__main__':
    unittest.main()
