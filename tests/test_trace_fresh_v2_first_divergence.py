"""CPU mock capture tests; no pretrained loaders/downloads or GPU execution."""
import copy
import json
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import ast
from contextlib import nullcontext

from experiments import trace_fresh_v2_first_divergence as trace

class Tensor:
    def __init__(self, values):
        self.values = values

    @property
    def ndim(self):
        return 2 if self.values and isinstance(self.values[0], list) else 1

    @property
    def shape(self):
        return (len(self.values), len(self.values[0])) if self.ndim == 2 else (len(self.values),)

    def to(self, device):
        return self

    def clone(self):
        return copy.deepcopy(self)

    def tolist(self):
        return copy.deepcopy(self.values)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            value = self.values[row][col]
        else:
            value = self.values[key]
        return Tensor(value) if isinstance(value, list) else Number(value)

    def __setitem__(self, key, value):
        row, col = key
        self.values[row][col] = value


class Number(float):
    def item(self):
        return float(self)


# Reuse the established Rate 027 AST technique: run the unchanged production helper.
source = (trace.ROOT / 'src/mindscopex_analysis/research.py').read_text()
node = next(n for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == '_binary_choice_completion')
namespace = {'DEFAULT_CHOICE_LETTERS': ('A', 'B'), 'torch': SimpleNamespace(inference_mode=nullcontext),
             'validate_choice_letters': lambda letters: letters}
exec(compile(ast.fix_missing_locations(ast.Module(body=[
    ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node
], type_ignores=[])), '<production-helper>', 'exec'), namespace)
_binary_choice_completion = namespace['_binary_choice_completion']


class Block:
    def __init__(self):
        self._forward_hooks = {}

    def register_forward_hook(self, hook):
        key = len(self._forward_hooks)
        self._forward_hooks[key] = hook
        return SimpleNamespace(remove=lambda: self._forward_hooks.pop(key))


def hook_factory(sae, ids, *, coefficient, token_position):
    assert ids == [30908] and coefficient in (7, 8) and token_position == 'all'
    return lambda *_: None



class Tokenizer:
    eos_token_id = 99
    pad_token_id = None
    name_or_path = "mock-tokenizer"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {" 80 hours": [1, 8, 3], " 20 hours": [1, 2, 3]}[text]

    def __call__(self, text, *, return_tensors):
        assert text == "raw question\nAnswer:"
        return dict(input_ids=Tensor([[70, 71]]), attention_mask=Tensor([[1, 1]]))


class Model:
    """Mock generate: applies the production trie to scores and greedy-selects."""
    def __init__(self, *, tie=False, stop=None, missing_raw=False):
        self.block = Block()
        self.tie, self.stop, self.missing_raw = tie, stop, missing_raw
        self.hook_counts = []

    def parameters(self):
        return iter([SimpleNamespace(device="cpu")])

    def generate(self, **kwargs):
        assert kwargs["do_sample"] is False and kwargs["renormalize_logits"] is True
        seq, logits, scores = kwargs["input_ids"].clone(), [], []
        for step in range(8):
            if self.stop is not None and step == self.stop:
                break
            self.hook_counts.append(len(self.block._forward_hooks))
            allowed = kwargs["prefix_allowed_tokens_fn"](0, seq[0])
            raw = Tensor([[0.] * 128])
            raw[0, 2] = 3  # raw prefers lure; processed scores prefer correct
            processed = Tensor([[float("-inf")] * 128])
            for t in allowed:
                processed[0, t] = 0
            if 8 in allowed:
                processed[0, 8] = 0 if self.tie else 4
            token = max(range(128), key=lambda t: processed[0, t])
            logits.append(raw)
            scores.append(processed)
            seq = Tensor([seq[0].tolist() + [token]])
            if token in kwargs["eos_token_id"]:
                break
        if kwargs.get("return_dict_in_generate"):
            assert kwargs["output_logits"] and kwargs["output_scores"]
            return SimpleNamespace(sequences=seq, logits=None if self.missing_raw else logits, scores=scores)
        return seq


def case():
    return SimpleNamespace(case_id="mock", correct_answer=" 80 hours", lure_answer=" 20 hours",
                           prompt="raw question\nAnswer:")


def wrapper_capture(**kwargs):
    model = Model(**kwargs)
    wrapper = trace.RecordingModel(model)
    try:
        _binary_choice_completion(wrapper, Tokenizer(), case(), max_new_tokens=16)
    except RuntimeError:
        if wrapper.output is None or wrapper.output.logits is None:
            raise
    return wrapper


def inspect(wrapper, candidates=None, eos=None):
    return trace.inspect_capture(wrapper, candidates or [[1, 8, 3], [1, 2, 3]], eos or [99])


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.modules = mock.patch.dict(sys.modules, {
            'mindscopex_analysis.research': SimpleNamespace(_binary_choice_completion=_binary_choice_completion),
            'mindscopex_analysis.qwen_scope': SimpleNamespace(make_feature_steering_hook=hook_factory),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_uninstrumented_and_instrumented_production_choices_identical(self):
        for tie in (False, True):
            plain = _binary_choice_completion(Model(tie=tie), Tokenizer(), case(), max_new_tokens=16)
            wrapper = wrapper_capture(tie=tie)
            answer = _binary_choice_completion(wrapper, Tokenizer(), case(), max_new_tokens=16)
            self.assertEqual(plain, answer)
            self.assertEqual(answer, "20 hours" if tie else "80 hours")

    def test_step_alignment_raw_processed_and_margin_sign(self):
        record = inspect(wrapper_capture())
        self.assertTrue(record["observed"])
        self.assertEqual(record["common_prefix_length"], 1)
        self.assertEqual(record["actual_raw_margin"], 3)
        self.assertEqual(record["actual_processed_score_margin"], -4)
        self.assertEqual(record["actual_raw_preference"], "lure")
        self.assertEqual(record["new_generation_label"], "correct")
        self.assertEqual(record["steps"][1]["selected_token_id"], 8)
        self.assertEqual(record["steps"][1]["prefix_token_ids"], [1])
        self.assertEqual(record["steps"][0]["allowed_token_ids"], [1])
        self.assertEqual(record["generated_token_ids"], [1, 8, 3, 99])
        self.assertEqual(record["termination"], "eos")
        json.dumps(record, allow_nan=False)

    def test_exact_raw_tie_is_retained_without_changing_selection(self):
        wrapper = wrapper_capture(tie=True)
        wrapper.output.logits[1][0, 2] = 0
        record = inspect(wrapper)
        self.assertEqual(record["actual_raw_margin"], 0)
        self.assertEqual(record["actual_raw_preference"], "tie")
        self.assertEqual(record["actual_processed_score_margin"], 0)
        self.assertEqual(record["steps"][1]["selected_token_id"], 2)

    def test_not_reached_and_eos_before_branch(self):
        for stop in (0, 1):
            record = inspect(wrapper_capture(stop=stop))
            self.assertFalse(record["observed"])
            self.assertIsNone(record["actual_raw_margin"])
            self.assertEqual(record["unobserved_reason"], "divergence_step_not_reached")
        wrapper = wrapper_capture(stop=1)
        wrapper.output.sequences[0, -1] = 99
        record = inspect(wrapper)
        self.assertEqual(record["unobserved_reason"], "eos_before_divergence")

    def test_prefix_and_recorded_allowed_tokens_must_match(self):
        wrapper = wrapper_capture()
        wrapper.output.sequences[0, 2] = 6
        self.assertEqual(inspect(wrapper)["unobserved_reason"], "common_prefix_mismatch")
        wrapper = wrapper_capture()
        wrapper.allowed_calls[1]["allowed_token_ids"] = [8]
        record = inspect(wrapper)
        self.assertFalse(record["observed"])
        self.assertEqual(record["unobserved_reason"], "allowed_tokens_not_verified")
        wrapper = wrapper_capture()
        wrapper.allowed_calls.pop(1)
        self.assertFalse(inspect(wrapper)["observed"])

    def test_multiple_eos_and_prefix_collision_are_not_arbitrary_scalar_branches(self):
        tokenizer = Tokenizer()
        tokenizer.eos_token_id = [98, 99]
        wrapper = trace.RecordingModel(Model())
        _binary_choice_completion(wrapper, tokenizer, case(), max_new_tokens=16)
        record = inspect(wrapper, eos=[98, 99])
        self.assertTrue(record["observed"])
        self.assertEqual(record["generated_token_ids"][-1], 98)
        self.assertEqual(record["steps"][-1]["allowed_token_ids"], [98, 99])
        # End of one candidate vs continuation of the other: explicitly non-scalar.
        record = inspect(wrapper, candidates=[[1], [1, 2]], eos=[98, 99])
        self.assertFalse(record["observed"])
        self.assertIsNone(record["actual_raw_margin"])
        self.assertEqual(record["unobserved_reason"],
                         "prefix_collision_eos_vs_continuation_no_scalar_margin")

    def test_missing_raw_or_misalignment_fails_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "raw logits"):
            wrapper_capture(missing_raw=True)
        for mutation in (lambda w: w.output.logits.pop(),
                         lambda w: w.output.scores.pop(),
                         lambda w: w.output.logits.__setitem__(1, None)):
            wrapper = wrapper_capture()
            mutation(wrapper)
            with self.assertRaises(RuntimeError):
                inspect(wrapper)

    def test_production_hook_factory_contract_and_removal_success_and_failure(self):
        sae = object()
        for coefficient in (0, 7, 8):
            model = Model()
            record = trace.capture_case(model, Tokenizer(), case(), model.block, sae, coefficient)
            self.assertTrue(record["observed"])
            self.assertTrue(all(n == int(coefficient != 0) for n in model.hook_counts))
            self.assertEqual(len(model.block._forward_hooks), 0)
        model = Model(missing_raw=True)
        with self.assertRaises(RuntimeError):
            trace.capture_case(model, Tokenizer(), case(), model.block, sae, 8)
        self.assertEqual(len(model.block._forward_hooks), 0)
        with self.assertRaisesRegex(ValueError, "failure"), trace.intervention(model.block, sae, 7):
            raise ValueError("failure")
        self.assertEqual(len(model.block._forward_hooks), 0)

    def test_truncated_helper_error_is_reported_without_fabricated_label(self):
        model = Model(stop=1)
        record = trace.capture_case(model, Tokenizer(), case(), model.block, None, 0)
        self.assertIsNotNone(record["completion_error"])
        self.assertIsNone(record["new_generation_label"])
        self.assertFalse(record["observed"])

    def test_complete_candidate_without_eos_matches_helper_but_is_not_normal_completion(self):
        model = Model(stop=3)
        record = trace.capture_case(model, Tokenizer(), case(), model.block, None, 0)
        self.assertEqual(record['answer'], '80 hours')  # unchanged production return rule
        self.assertTrue(record['helper_returned_response'])
        self.assertTrue(record['helper_accepts_tokens'])
        self.assertTrue(record['complete_candidate'])
        self.assertFalse(record['eos_observed'])
        self.assertFalse(record['behavioral_response_valid'])
        self.assertIsNone(record['new_generation_label'])
        self.assertEqual(record['candidate_label'], 'correct')
        self.assertEqual(record['completion_status'], 'candidate_complete_without_eos')
        self.assertTrue(record['observed'])  # branch observation is independent of completion

    def test_eos_does_not_imply_candidate_completion_or_branch_observation(self):
        wrapper = wrapper_capture(stop=1)
        wrapper.output.sequences[0, -1] = 99
        record = inspect(wrapper)
        self.assertTrue(record['eos_observed'])
        self.assertFalse(record['complete_candidate'])
        self.assertFalse(record['behavioral_response_valid'])
        self.assertFalse(record['observed'])

    def test_partial_answer_after_branch_retains_observation_without_behavioral_label(self):
        model = Model(stop=2)
        record = trace.capture_case(model, Tokenizer(), case(), model.block, None, 0)
        self.assertTrue(record['observed'])
        self.assertFalse(record['complete_candidate'])
        self.assertFalse(record['helper_returned_response'])
        self.assertFalse(record['behavioral_response_valid'])
        self.assertIsNone(record['new_generation_label'])
        self.assertIsNotNone(record['completion_error'])

    def test_unexpected_runtime_error_after_returned_capture_propagates_and_removes_hook(self):
        def fail_after_capture(*args, **kwargs):
            _binary_choice_completion(*args, **kwargs)
            raise RuntimeError('unexpected execution failure')
        model = Model()
        module = sys.modules['mindscopex_analysis.research']
        with mock.patch.object(module, '_binary_choice_completion', fail_after_capture):
            with self.assertRaisesRegex(RuntimeError, 'unexpected execution failure'):
                trace.capture_case(model, Tokenizer(), case(), model.block, object(), 7)
        self.assertEqual(len(model.block._forward_hooks), 0)

    def test_model_error_with_completion_like_message_is_not_swallowed(self):
        class BrokenModel(Model):
            def generate(self, **kwargs):
                raise RuntimeError("Constrained generation did not finish a valid answer for case 'mock': [1]")
        model = BrokenModel()
        with self.assertRaises(RuntimeError):
            trace.capture_case(model, Tokenizer(), case(), model.block, object(), 8)
        self.assertEqual(len(model.block._forward_hooks), 0)

    def test_passive_sequence_observer_preserves_kwargs_and_both_paths_match(self):
        model = Model()
        original_generate = model.generate
        with mock.patch.object(model, 'generate', wraps=original_generate) as calls:
            record = trace.capture_case(model, Tokenizer(), case(), model.block, object(), 7)
            comparison = trace.compare_uninstrumented(
                model, Tokenizer(), case(), model.block, object(), 7, record)
        self.assertTrue(comparison['generated_token_ids_equal'])
        self.assertTrue(comparison['answers_equal'])
        self.assertTrue(comparison['equivalent'])
        instrumented, plain = [c.kwargs for c in calls.call_args_list]
        for key in ('output_logits', 'output_scores', 'return_dict_in_generate'):
            self.assertTrue(instrumented[key])
            self.assertNotIn(key, plain)
        self.assertEqual(plain['input_ids'].tolist(), instrumented['input_ids'].tolist())
        self.assertTrue(all(n == 1 for n in model.hook_counts))
        self.assertEqual(len(model.block._forward_hooks), 0)

    def test_equivalence_compares_tokens_and_answer_separately_and_cleans_up(self):
        model = Model()
        record = trace.capture_case(model, Tokenizer(), case(), model.block, None, 0)
        for field, changed, token_equal, answer_equal in (
                ('generated_token_ids', [1, 8, 3], False, True),
                ('answer', '20 hours', True, False)):
            modified = dict(record, **{field: changed})
            comparison = trace.compare_uninstrumented(
                model, Tokenizer(), case(), model.block, None, 0, modified)
            self.assertEqual(comparison['generated_token_ids_equal'], token_equal)
            self.assertEqual(comparison['answers_equal'], answer_equal)
            self.assertFalse(comparison['equivalent'])
        with mock.patch.object(model, 'generate', side_effect=RuntimeError('kernel failure')):
            with self.assertRaisesRegex(RuntimeError, 'kernel failure'):
                trace.compare_uninstrumented(model, Tokenizer(), case(), model.block, object(), 8, record)
        self.assertEqual(len(model.block._forward_hooks), 0)

    def test_nonfinite_raw_is_unobserved_and_masked_score_serialization(self):
        wrapper = wrapper_capture()
        wrapper.output.logits[1][0, 2] = float("nan")
        record = inspect(wrapper)
        self.assertEqual(record["unobserved_reason"], "nonfinite_branch_raw_logits")
        self.assertIsNone(record["actual_raw_margin"])
        json.dumps(record, allow_nan=False)


class PlanTests(unittest.TestCase):
    def test_existing_csv_join_target_counts_and_unique_measurements(self):
        plan = trace.load_plan()
        self.assertEqual(plan["target_counts"], {"7_tie": 11, "8_tie": 6, "8_mismatch": 2})
        self.assertEqual(plan["n_unique_measurements"], 21)
        self.assertEqual(plan["n_slot_comparisons"], 57)
        self.assertEqual(len(plan["target_slots"]), 19)
        self.assertEqual(plan['n_unique_generation_conditions'], 21)
        self.assertEqual(plan['n_historical_slot_references'], 57)
        self.assertEqual(plan['n_exception_slots'], 19)
        self.assertFalse(plan['seed_references_are_independent_measurements'])
        slots = [r for r in plan["slot_comparisons"] if r["case_id"].endswith("019") and r["coefficient"] == 8]
        self.assertEqual([r["seed"] for r in slots], [2, 4])
        self.assertEqual({r["teacher_forced_first_divergence_margin"] for r in slots}, {-.125})
        self.assertEqual({r["existing_behavioral_label"] for r in slots}, {"lure"})
        self.assertEqual(len({r["measurement_id"] for r in slots}), 1)

    def test_mock_run_connects_all_21_conditions_and_57_slot_references(self):
        plan = trace.load_plan()
        sha = lambda ids: hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        mapping, cases = {}, {}
        for item in plan['measurements']:
            score = item['score_reference']
            score['prompt_token_ids_sha256'] = sha([70, 71])
            cid = item['case_id']
            cases[cid] = SimpleNamespace(case_id=cid, family='crt_growth',
                                        correct_answer=score['correct_answer'],
                                        lure_answer=score['lure_answer'], prompt=case().prompt)
            for label in ('correct', 'lure'):
                mapping[score[label + '_answer']] = json.loads(score[label + '_token_ids'])
        # Keep vocabulary small in the fake model, retaining common-prefix structure.
        for item in plan['measurements']:
            score = item['score_reference']
            for label in ('correct', 'lure'):
                text = score[label + '_answer']
                mapping[text] = [1] + [int(c) + 10 for c in text if c.isdigit()] + [3]
                score[label + '_token_ids'] = json.dumps(mapping[text])

        class SavedTokenizer(Tokenizer):
            def encode(self, text, *, add_special_tokens):
                assert not add_special_tokens
                return mapping[text]

        model, tokenizer = Model(), SavedTokenizer()
        model.generation_config = SimpleNamespace(num_beams=1, num_return_sequences=1)
        profile = SimpleNamespace(analysis_model_id=plan['model_id'], sae_repo_id='mock-sae',
                                  block_path_template='mock.layers.{layer}')
        load_model, load_sae = mock.Mock(return_value=(model, tokenizer)), mock.Mock(
            return_value=SimpleNamespace(W_dec=SimpleNamespace(dtype='mock-float')))
        modules = {
            'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
            'transformers': SimpleNamespace(),
            'mindscopex_analysis.models': SimpleNamespace(
                default_sae_device=lambda: 'mock-device', dtype_from_name=lambda dtype: dtype,
                get_qwen35_analysis_profile=lambda _: profile,
                load_qwen_text_generation_model=load_model, recommended_dtype_name=lambda: 'float32'),
            'mindscopex_analysis.lure_datasets': SimpleNamespace(load_lure_dataset=lambda _: list(cases.values())),
            'mindscopex_analysis.prompts': SimpleNamespace(instruct_lure_case=lambda c: c),
            'mindscopex_analysis.full_answer_protocols': SimpleNamespace(token_ids_sha256=sha),
            'mindscopex_analysis.qwen_scope': SimpleNamespace(
                load_qwen_scope_sae=load_sae, make_feature_steering_hook=hook_factory),
            'mindscopex_analysis.research': SimpleNamespace(
                _binary_choice_completion=_binary_choice_completion, find_decoder_block=lambda *a, **k: model.block),
        }
        with mock.patch.dict(sys.modules, modules), mock.patch(
                'experiments.diagnose_rate027_baseline.environment', return_value={'mock': True}):
            result = trace.run_model(plan, check_equivalence=True)
        load_model.assert_called_once()
        load_sae.assert_called_once()
        self.assertEqual(result['status'], 'captured')
        self.assertEqual(result['expected_generate_calls'], 42)
        self.assertTrue(all(r['capture']['equivalence_check']['equivalent']
                            for r in result['measurements']))
        self.assertEqual(len(result['measurements']), 21)
        self.assertEqual(len(result['slot_comparisons']), 57)
        self.assertTrue(all(r['capture']['observed'] for r in result['measurements']))
        self.assertTrue(all('new_generation_label' in r for r in result['slot_comparisons']))
        self.assertEqual(len(model.block._forward_hooks), 0)
        json.dumps(result, allow_nan=False)

    def test_join_rejects_duplicate_or_inconsistent_input(self):
        original = trace.rows
        def bad(path, fields):
            result = original(path, fields)
            if path == trace.INPUTS["audit"]:
                result[0]["actual"] = "lure" if result[0]["actual"] == "correct" else "correct"
            return result
        with mock.patch.object(trace, "rows", side_effect=bad), self.assertRaises(ValueError):
            trace.load_plan()
        with self.assertRaises(ValueError):
            trace.unique([{"a": 1}, {"a": 1}], lambda x: x["a"])

    def test_exclusive_output_and_preflight_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "existing.json"
            output.write_text("preserve")
            with self.assertRaises(FileExistsError):
                trace.write_new(output, {})
            with mock.patch.object(trace, "run_model") as run:
                with self.assertRaises(FileExistsError):
                    trace.main(["--run-model", "--output", str(output)])
                run.assert_not_called()
            self.assertEqual(output.read_text(), "preserve")
            link = Path(tmp) / "link.json"
            link.symlink_to(Path(tmp) / "missing")
            with self.assertRaises(FileExistsError):
                trace.main(["--plan", "--output", str(link)])

    def test_plan_cli_does_not_import_ml_libraries(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "plan.json"
            script = ("import sys; from experiments.trace_fresh_v2_first_divergence import main; "
                      f"main(['--plan','--output',{str(output)!r}]); "
                      "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
            result = subprocess.run([sys.executable, "-c", script], cwd=trace.ROOT,
                                    env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text())["status"], "planned")


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.plan = trace.load_plan()
        self.cid = 'crt_fresh_v2_growth_019'

    def test_case_only_selects_three_conditions_and_six_references(self):
        selected = trace.filter_plan(self.plan, case_id=self.cid)
        self.assertEqual(selected['case_ids'], [self.cid])
        self.assertEqual(selected['coefficients'], [0, 7, 8])
        self.assertEqual(selected['n_unique_measurements'], 3)
        self.assertEqual(selected['n_slot_comparisons'], 6)
        self.assertEqual(selected['expected_generate_calls'], 3)

    def test_coefficient_only_selects_seven_conditions_and_nineteen_references(self):
        for coefficient in (0, 7, 8):
            with self.subTest(coefficient=coefficient):
                selected = trace.filter_plan(self.plan, coefficient=coefficient)
                self.assertEqual(selected['coefficients'], [coefficient])
                self.assertEqual(selected['n_unique_measurements'], 7)
                self.assertEqual(selected['n_slot_comparisons'], 19)
                self.assertTrue(all(m['coefficient'] == coefficient for m in selected['measurements']))

    def test_combined_selection_retains_saved_growth019_margin_and_labels(self):
        selected = trace.filter_plan(self.plan, case_id=self.cid, coefficient=8,
                                     check_equivalence=True)
        self.assertEqual(selected['n_unique_measurements'], 1)
        self.assertEqual(selected['n_slot_comparisons'], 2)
        self.assertEqual(selected['expected_generate_calls'], 2)
        self.assertEqual([r['seed'] for r in selected['slot_comparisons']], [2, 4])
        for slot in selected['slot_comparisons']:
            self.assertEqual(slot['teacher_forced_first_divergence_margin'], -.125)
            self.assertEqual(slot['existing_behavioral_label'], 'lure')

    def test_selected_measurements_and_slots_are_consistent_and_full_audit_preserved(self):
        original = copy.deepcopy(self.plan)
        for cid in trace.CASE_IDS:
            for coefficient in trace.COEFFICIENTS:
                selected = trace.filter_plan(self.plan, case_id=cid, coefficient=coefficient)
                mids = {m['measurement_id'] for m in selected['measurements']}
                self.assertEqual({r['measurement_id'] for r in selected['slot_comparisons']}, mids)
                self.assertEqual(selected['target_slots'], self.plan['target_slots'])
                self.assertEqual(selected['target_counts'], self.plan['target_counts'])
                self.assertEqual(selected['n_exception_slots'], 19)
                self.assertEqual(selected['n_selected_slot_comparisons'], len(selected['slot_comparisons']))
                self.assertEqual(selected['n_selected_exception_slots'], sum(
                    r['case_id'] == cid and r['coefficient'] == coefficient
                    for r in self.plan['target_slots']))
        self.assertEqual(self.plan, original)

    def test_default_scope_and_equivalence_call_budget(self):
        for equivalence, calls in ((False, 21), (True, 42)):
            selected = trace.filter_plan(self.plan, check_equivalence=equivalence)
            self.assertEqual(selected['measurements'], self.plan['measurements'])
            self.assertEqual(selected['slot_comparisons'], self.plan['slot_comparisons'])
            self.assertEqual(selected['n_unique_measurements'], 21)
            self.assertEqual(selected['n_slot_comparisons'], 57)
            self.assertEqual(selected['expected_generate_calls'], calls)
            self.assertFalse(selected['seed_references_are_independent_measurements'])

    def test_invalid_filter_values_are_rejected(self):
        for kwargs in ({'case_id': 'growth_019'}, {'case_id': 'crt_fresh_v2_growth_001'},
                       {'coefficient': 9}, {'coefficient': 7.9}, {'coefficient': '8'},
                       {'coefficient': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                trace.filter_plan(self.plan, **kwargs)

    def test_cli_filters_before_run_model_entry(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(trace, 'run_model') as run:
            run.side_effect = lambda plan, **kwargs: plan
            with mock.patch('builtins.print'):
                trace.main(['--run-model', '--case-id', self.cid, '--coefficient', '8',
                            '--check-equivalence', '--output', str(Path(tmp) / 'mock.json')])
            selected = run.call_args.args[0]
            self.assertEqual(len(selected['measurements']), 1)
            self.assertEqual(len(selected['slot_comparisons']), 2)
            self.assertEqual(selected['expected_generate_calls'], 2)
            self.assertEqual(selected['measurements'][0]['case_id'], self.cid)
            self.assertEqual(selected['measurements'][0]['coefficient'], 8)
            self.assertTrue(run.call_args.kwargs['check_equivalence'])

    def test_invalid_cli_never_enters_model_loading_or_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(trace, 'run_model') as run:
            for args in (['--case-id', 'missing'], ['--coefficient', '9'], ['--coefficient', '7.9']):
                path = Path(tmp) / 'absent.json'
                with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
                    trace.main(['--run-model', *args, '--output', str(path)])
                self.assertFalse(path.exists())
            run.assert_not_called()

    def test_filtered_plan_cli_has_no_model_library_imports_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'filtered.json'
            args = ['--plan', '--case-id', self.cid, '--coefficient', '8', '--check-equivalence',
                    '--output', str(output)]
            script = ("import sys; from experiments.trace_fresh_v2_first_divergence import main; "
                      f"main({args!r}); "
                      "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
            result = subprocess.run([sys.executable, '-c', script], cwd=trace.ROOT,
                                    env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(output.read_text())
            self.assertEqual((data['n_unique_measurements'], data['n_slot_comparisons'],
                              data['expected_generate_calls']), (1, 2, 2))
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                trace.main(args)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
