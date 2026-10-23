"""CPU regressions for Pilot 3 and the Pilot 2B configuration it copies."""
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
    LETTER_MODE, ORDERS, choice_letter, evaluate, letter_label,
    ordered_case, validate_choice_letters,
)
from test_crt_choice_letter import StubGenerator, Tokenizer

CONFIG = job.ROOT / "experiments/configs/study_rate_choice_xy_pilot_f22552_seed1_2b.toml"
PREVIOUS = CONFIG.with_name("study_rate_choice_order_dose_f22552_seed1_2b.toml")
SUITE = job.ROOT / "experiments/suites/study_rate_choice_xy_pilot_f22552_2b.toml"


def load_cases(config):
    return job.family_balanced_subset(job._load_splits(config)["all"], max_cases=0)


def generate(items, coefficient):
    rows = [dict(case_id=c.case_id, family=c.family, answer=c.correct_answer,
                 label="correct") for c in items]
    after = copy.deepcopy(rows)
    if coefficient:
        after[0].update(answer=items[0].lure_answer, label="lure")
    return dict(baseline_rows=rows, steered_rows=after)


class ChoiceXYPilotTests(unittest.TestCase):
    def test_config_copies_pilot_2b(self):
        config, old = job.load_toml(CONFIG), job.load_toml(PREVIOUS)
        self.assertEqual(resolve_configs(SUITE), [CONFIG.resolve()])
        self.assertEqual(config["behavioral"]["coefficients"], [-7.0, 4.0, 7.0])
        self.assertEqual(config["behavioral"]["choice_letters"], ["X", "Y"])
        self.assertEqual(config["behavioral"]["choice_orders"], list(ORDERS))
        self.assertEqual(config["behavioral"]["expected_cases"], 20)
        self.assertEqual(config["feature"], {"layer": 5, "feature_id": 22552})
        self.assertEqual(config["data"]["dataset"], "crt_fresh_v2_rate_order_pilot")
        self.assertEqual(job._resolve_env(config)["model_id"], "Qwen/Qwen3.5-2B-Base")
        config["run"]["name"] = old["run"]["name"]
        config["behavioral"]["coefficients"] = old["behavioral"]["coefficients"]
        del config["behavioral"]["choice_letters"]
        self.assertEqual(config, old)

    def test_prompts_and_labels_both_orders_and_defaults(self):
        case = load_cases(job.load_toml(CONFIG))[0]
        for letters in (("A", "B"), ("X", "Y")):
            for order in ORDERS:
                with self.subTest(letters=letters, order=order):
                    result = ordered_case(case, order, LETTER_MODE, letters)
                    correct, lure = letters if order == ORDERS[0] else letters[::-1]
                    self.assertIn(f"{letters[0]} or {letters[1]}", result.prompt)
                    self.assertIn(f"\n{correct}: {case.correct_answer.strip()}\n", result.prompt)
                    self.assertIn(f"\n{lure}: {case.lure_answer.strip()}\n", result.prompt)
                    self.assertEqual(result.correct_answer, correct)
                    self.assertEqual(result.lure_answer, lure)
                    self.assertEqual(letter_label(correct, order, letters), "correct")
                    self.assertEqual(letter_label(lure, order, letters), "lure")
                    self.assertEqual(choice_letter(case.correct_answer, case, order, letters), correct)
                    self.assertEqual(choice_letter(case.lure_answer, case, order, letters), lure)
                    if letters == ("A", "B"):
                        self.assertEqual(result, ordered_case(case, order, LETTER_MODE))

    def test_xy_token_candidates_and_label_decoding(self):
        case = load_cases(job.load_toml(CONFIG))[0]
        for order in ORDERS:
            for final, letter in ((5, "X"), (6, "Y")):
                for tokens, emitted in (
                    ({"X": [5], "Y": [6]}, (final, 9)),
                    ({"X": [3, 5], "Y": [3, 6]}, (3, final, 9)),
                ):
                    tokenizer = Tokenizer(tokens)
                    model = StubGenerator(emitted)
                    rows = research._generate_labels(
                        model, tokenizer, [ordered_case(case, order, LETTER_MODE, ("X", "Y"))],
                        max_new_tokens=1, output_mode=LETTER_MODE, choice_letters=("X", "Y"))
                    self.assertEqual(tokenizer.encoded, ["X", "Y"])
                    self.assertEqual(rows[0]["answer"], letter)
                    self.assertEqual(rows[0]["label"], letter_label(letter, order, ("X", "Y")))
                    self.assertFalse(model.kwargs["do_sample"])
                    self.assertEqual(model.allowed[-1], [9, 10])

    def test_steering_passes_candidates_to_both_generations(self):
        block = mock.Mock()
        with mock.patch.object(research, "_generate_labels", return_value=[]) as generate_labels, \
             mock.patch.object(research, "find_decoder_block", return_value=block), \
             mock.patch.object(research, "make_feature_steering_hook"):
            research.steer_generation_labels(
                object(), object(), [], layer=5, sae=object(), feature_id=22552,
                coefficient=4.0, output_mode=LETTER_MODE, choice_letters=("X", "Y"))
        self.assertEqual(generate_labels.call_count, 2)
        for call in generate_labels.call_args_list:
            self.assertEqual(call.kwargs["choice_letters"], ("X", "Y"))
        block.register_forward_hook.return_value.remove.assert_called_once()

    def test_invalid_configurations(self):
        case = load_cases(job.load_toml(CONFIG))[0]
        for letters in (None, "XY", [], ["X"], ["X", "Y", "Z"], ["X", "X"],
                        ["x", "Y"], ["XX", "Y"], [" X", "Y"], ["1", "Y"],
                        ["É", "Y"], [1, "Y"], {"X", "Y"}):
            with self.subTest(letters=letters):
                for invoke in (
                    lambda: validate_choice_letters(letters),
                    lambda: ordered_case(case, ORDERS[0], LETTER_MODE, letters),
                    lambda: research._generate_labels(
                        object(), object(), [], max_new_tokens=1,
                        output_mode=LETTER_MODE, choice_letters=letters),
                ):
                    with self.assertRaises(ValueError):
                        invoke()
        with self.assertRaises(ValueError):
            letter_label("A", ORDERS[0], ("X", "Y"))
        with self.assertRaises(ValueError):
            research._generate_labels(
                StubGenerator((5, 9)), Tokenizer({"X": [5], "Y": [6]}),
                [ordered_case(case, ORDERS[0], LETTER_MODE)], max_new_tokens=1,
                output_mode=LETTER_MODE, choice_letters=("X", "Y"))

    def test_multi_coefficient_evaluation_pilots_2b_and_3(self):
        for path in (PREVIOUS, CONFIG):
            config = job.load_toml(path)
            bcfg = config["behavioral"]
            cases = load_cases(config)
            letters = tuple(bcfg.get("choice_letters", ("A", "B")))
            blocks = evaluate(
                cases, bcfg["coefficients"], generate, output_mode=LETTER_MODE,
                choice_letters=letters, expected_cases=20,
                expected_coefficients=bcfg["coefficients"])
            self.assertEqual(len(blocks), 2 * len(bcfg["coefficients"]))
            for block in blocks:
                self.assertEqual(block["choice_letters"], list(letters))
                self.assertEqual(len(block["baseline_rows"]), 20)
                self.assertEqual(len(block["steered_rows"]), 20)
                self.assertEqual(block["cases"][0]["correct_answer"], cases[0].correct_answer)
                for row in block["steered_rows"]:
                    self.assertEqual(letter_label(row["choice"], block["choice_order"], letters),
                                     row["label"])

    def test_job_propagates_letters_and_metadata(self):
        config = job.load_toml(CONFIG)
        cases = load_cases(config)
        state = {"study_feature": {"feature": config["feature"], "sae": object()}}

        def fake(model, tokenizer, items, **kwargs):
            self.assertEqual(kwargs["choice_letters"], ("X", "Y"))
            return generate(items, kwargs["coefficient"])

        with TemporaryDirectory() as tmp, \
             mock.patch.object(job, "steer_generation_labels", side_effect=fake) as steering:
            result = job.run_behavioral(
                object(), object(), {"all": cases}, config, {}, Path(tmp), state)
            self.assertEqual(steering.call_count, 6)
            self.assertEqual(result["choice_letters"], ["X", "Y"])
            blocks = json.loads((Path(tmp) / "behavioral/generations.json").read_text())
            self.assertEqual(len(blocks), 6)
            self.assertTrue(all(b["choice_letters"] == ["X", "Y"] for b in blocks))
            for letters, mode in ((["X", "X"], LETTER_MODE), (["X", "Y"], "binary_choice")):
                config["behavioral"].update(choice_letters=letters, output_mode=mode)
                with self.assertRaises(ValueError):
                    job.run_behavioral(
                        object(), object(), {"all": cases}, config, {}, Path(tmp), state)


if __name__ == "__main__":
    unittest.main()
