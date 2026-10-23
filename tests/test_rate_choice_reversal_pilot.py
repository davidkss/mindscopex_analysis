"""CPU-only Pilot 4A/4B regressions; no checkpoint loading or GPU runs."""
import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.jobs import research_experiments as job
from experiments.runners.launch_colab import resolve_configs
from mindscopex_analysis import research
from mindscopex_analysis.choice_order import (
    LETTER_MODE, ORDERS, choice_letter, evaluate, letter_label, ordered_case,
)
from test_crt_choice_letter import StubGenerator, Tokenizer
from test_rate_choice_xy_pilot import CONFIG as PILOT_3, generate, load_cases

CONFIGS = [
    job.ROOT / f"experiments/configs/study_rate_choice_{suffix}_pilot_f22552_seed1_2b.toml"
    for suffix in ("ba", "yx")
]
SUITE = job.ROOT / "experiments/suites/study_rate_choice_reversal_pilot_f22552_2b.toml"
REVERSALS = (("B", "A"), ("Y", "X"))


class ChoiceReversalPilotTests(unittest.TestCase):
    def test_configs_and_suite_preserve_pilot_3_conditions(self):
        original = job.load_toml(PILOT_3)
        self.assertEqual(resolve_configs(SUITE), [path.resolve() for path in CONFIGS])
        for path, letters in zip(CONFIGS, REVERSALS, strict=True):
            config = job.load_toml(path)
            self.assertEqual(config["run"]["name"], path.stem)
            self.assertEqual(config["behavioral"]["choice_letters"], list(letters))
            self.assertEqual(config["behavioral"]["coefficients"], [-7.0, 4.0, 7.0])
            self.assertEqual(config["behavioral"]["choice_orders"], list(ORDERS))
            self.assertEqual(config["behavioral"]["expected_cases"], 20)
            self.assertEqual(job._resolve_env(config)["model_id"], "Qwen/Qwen3.5-2B-Base")
            self.assertEqual(load_cases(config), load_cases(original))
            normalized = copy.deepcopy(config)
            normalized["run"]["name"] = original["run"]["name"]
            normalized["behavioral"]["choice_letters"] = original["behavioral"]["choice_letters"]
            self.assertEqual(normalized, original)

    def test_prompts_and_mapping_all_letter_orders(self):
        case = load_cases(job.load_toml(PILOT_3))[0]
        for letters in (("A", "B"), ("X", "Y"), *REVERSALS):
            for order in ORDERS:
                with self.subTest(letters=letters, order=order):
                    correct, lure = letters if order == "correct_first" else letters[::-1]
                    first_value, second_value = (
                        (case.correct_answer, case.lure_answer) if order == "correct_first"
                        else (case.lure_answer, case.correct_answer)
                    )
                    transformed = ordered_case(case, order, LETTER_MODE, letters)
                    self.assertTrue(transformed.prompt.startswith(
                        f"Answer with exactly one uppercase letter, {letters[0]} or {letters[1]}."))
                    self.assertTrue(transformed.prompt.endswith(
                        f"\n{letters[0]}: {first_value.strip()}"
                        f"\n{letters[1]}: {second_value.strip()}\nAnswer:"))
                    self.assertEqual((transformed.correct_answer, transformed.lure_answer),
                                     (correct, lure))
                    self.assertEqual(letter_label(correct, order, letters), "correct")
                    self.assertEqual(letter_label(lure, order, letters), "lure")
                    self.assertEqual(choice_letter(case.correct_answer, case, order, letters), correct)
                    self.assertEqual(choice_letter(case.lure_answer, case, order, letters), lure)
                    if letters == ("A", "B"):
                        self.assertEqual(transformed, ordered_case(case, order, LETTER_MODE))

    def test_constrained_decoding_reversed_candidate_order(self):
        case = load_cases(job.load_toml(PILOT_3))[0]
        for letters in REVERSALS:
            for order in ORDERS:
                for index in (0, 1):
                    for tokens, emitted in (
                        ({letters[0]: [5], letters[1]: [6]}, (5 + index, 9)),
                        ({letters[0]: [3, 5], letters[1]: [3, 6]}, (3, 5 + index, 9)),
                    ):
                        with self.subTest(letters=letters, order=order, emitted=emitted):
                            model, tokenizer = StubGenerator(emitted), Tokenizer(tokens)
                            rows = research._generate_labels(
                                model, tokenizer, [ordered_case(case, order, LETTER_MODE, letters)],
                                max_new_tokens=1, output_mode=LETTER_MODE, choice_letters=letters)
                            self.assertEqual(tokenizer.encoded, list(letters))
                            self.assertEqual(rows[0]["answer"], letters[index])
                            self.assertEqual(rows[0]["label"],
                                             letter_label(letters[index], order, letters))
                            self.assertEqual(model.allowed[-1], [9, 10])
                            self.assertFalse(model.kwargs["do_sample"])

    def test_baseline_and_steering_receive_identical_letter_order(self):
        case = load_cases(job.load_toml(PILOT_3))[0]
        for letters in REVERSALS:
            for order in ORDERS:
                block = mock.Mock()
                tokenizer = Tokenizer({letters[0]: [5], letters[1]: [6]})
                ordered = [ordered_case(case, order, LETTER_MODE, letters)]
                with mock.patch.object(research, "_generate_labels",
                                       wraps=research._generate_labels) as labels, \
                     mock.patch.object(research, "find_decoder_block", return_value=block), \
                     mock.patch.object(research, "make_feature_steering_hook"):
                    result = research.steer_generation_labels(
                        StubGenerator((5, 9)), tokenizer, ordered, layer=5, sae=object(),
                        feature_id=22552, coefficient=4.0, output_mode=LETTER_MODE,
                        choice_letters=letters)
                self.assertEqual(labels.call_count, 2)
                for call in labels.call_args_list:
                    self.assertEqual(call.kwargs["choice_letters"], letters)
                    self.assertEqual(call.args[2], ordered)
                self.assertEqual(tokenizer.encoded, list(letters) * 2)
                self.assertEqual(result["baseline_rows"], result["steered_rows"])
                self.assertEqual(result["baseline_rows"][0]["answer"], letters[0])
                block.register_forward_hook.return_value.remove.assert_called_once()

    def test_evaluation_blocks_and_case_counts(self):
        for path, letters in zip(CONFIGS, REVERSALS, strict=True):
            config = job.load_toml(path)
            cases = load_cases(config)
            self.assertEqual(len(cases), 20)
            bcfg = config["behavioral"]
            blocks = evaluate(
                cases, bcfg["coefficients"], generate, output_mode=LETTER_MODE,
                choice_letters=letters, expected_cases=bcfg["expected_cases"],
                expected_coefficients=bcfg["coefficients"])
            self.assertEqual(len(blocks), 6)
            self.assertEqual(
                {(b["choice_order"], b["coefficient"]) for b in blocks},
                {(order, coefficient) for order in ORDERS for coefficient in (-7.0, 4.0, 7.0)})
            for block in blocks:
                self.assertEqual(block["choice_letters"], list(letters))
                self.assertEqual(len(block["cases"]), 20)
                self.assertEqual(len(block["prompts"]), 20)
                for field in ("baseline_rows", "steered_rows"):
                    self.assertEqual(len(block[field]), 20)
                    self.assertEqual({row["case_id"] for row in block[field]},
                                     {case.case_id for case in cases})
                    for row in block[field]:
                        self.assertEqual(row["label"],
                                         letter_label(row["answer"], block["choice_order"], letters))

    def test_job_routes_both_reversals_and_saves_metadata(self):
        total_blocks = 0
        for path, letters in zip(CONFIGS, REVERSALS, strict=True):
            config = job.load_toml(path)
            cases = load_cases(config)
            state = {"study_feature": {"feature": config["feature"], "sae": object()}}

            def fake(model, tokenizer, items, **kwargs):
                self.assertEqual(kwargs["choice_letters"], letters)
                self.assertEqual(kwargs["layer"], 5)
                self.assertEqual(kwargs["feature_id"], 22552)
                self.assertEqual(kwargs["token_position"], "all")
                self.assertEqual(kwargs["output_mode"], LETTER_MODE)
                return generate(items, kwargs["coefficient"])

            with TemporaryDirectory() as tmp, \
                 mock.patch.object(job, "steer_generation_labels", side_effect=fake) as steering:
                result = job.run_behavioral(
                    object(), object(), {"all": cases}, config, {}, Path(tmp), state)
                self.assertEqual(steering.call_count, 6)
                self.assertEqual(result["n_cases"], 20)
                self.assertEqual(result["choice_letters"], list(letters))
                blocks = json.loads((Path(tmp) / "behavioral/generations.json").read_text())
                self.assertEqual(len(blocks), 6)
                total_blocks += len(blocks)
                self.assertTrue(all(block["choice_letters"] == list(letters) for block in blocks))
                self.assertTrue(all(len(block[field]) == 20 for block in blocks
                                    for field in ("baseline_rows", "steered_rows")))
        self.assertEqual(total_blocks, 12)


if __name__ == "__main__":
    unittest.main()
