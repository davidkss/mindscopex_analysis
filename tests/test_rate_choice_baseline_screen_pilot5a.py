"""Pilot 5A CPU-only configuration and counterbalancing regression tests."""
import unittest
from pathlib import Path

from experiments.jobs import research_experiments as job
from experiments.runners.launch_colab import resolve_configs
from mindscopex_analysis.choice_order import LETTER_MODE, ORDERS, evaluate, ordered_case
from test_rate_choice_xy_pilot import generate, load_cases

ROOT = job.ROOT
PAIRS = (("cd", ("C", "D")), ("dc", ("D", "C")),
         ("mn", ("M", "N")), ("nm", ("N", "M")))
SUITE = ROOT / "experiments/suites/study_rate_choice_baseline_screen_pilot5a_f22552_2b.toml"


class Pilot5ABaselineScreenTests(unittest.TestCase):
    def test_suite_and_configs(self):
        expected = [ROOT / f"experiments/configs/study_rate_choice_{slug}_baseline_screen_f22552_seed1_2b.toml"
                    for slug, _ in PAIRS]
        self.assertEqual(resolve_configs(SUITE), [p.resolve() for p in expected])
        for path, (_, letters) in zip(expected, PAIRS, strict=True):
            with self.subTest(path=path.name):
                config = job.load_toml(path)
                self.assertEqual(config["run"]["name"], path.stem)
                self.assertEqual(config["experiment"]["kind"], "behavioral")
                self.assertEqual(config["feature"], {"layer": 5, "feature_id": 22552})
                self.assertEqual(config["behavioral"]["coefficients"], [0.0])
                self.assertEqual(config["behavioral"]["choice_letters"], list(letters))
                self.assertEqual(config["behavioral"]["choice_orders"], list(ORDERS))
                self.assertEqual(config["behavioral"]["output_mode"], LETTER_MODE)
                self.assertEqual(config["behavioral"]["expected_cases"], 20)
                self.assertEqual(job._resolve_env(config)["model_id"], "Qwen/Qwen3.5-2B-Base")
                self.assertEqual(len(load_cases(config)), 20)

    def test_zero_coefficient_and_correctness_mapping(self):
        for slug, letters in PAIRS:
            config = job.load_toml(ROOT / f"experiments/configs/study_rate_choice_{slug}_baseline_screen_f22552_seed1_2b.toml")
            cases = load_cases(config)
            for order in ORDERS:
                transformed = ordered_case(cases[0], order, LETTER_MODE, letters)
                self.assertEqual(transformed.correct_answer, letters[0] if order == "correct_first" else letters[1])
            blocks = evaluate(cases, [0.0], generate, output_mode=LETTER_MODE,
                              choice_letters=letters, expected_cases=20,
                              expected_coefficients=[0.0])
            self.assertEqual(len(blocks), 2)
            self.assertEqual({b["choice_order"] for b in blocks}, set(ORDERS))
            for block in blocks:
                self.assertEqual(block["baseline_rows"], block["steered_rows"])
                self.assertEqual(len(block["baseline_rows"]), 20)
                self.assertEqual(block["choice_letters"], list(letters))


if __name__ == "__main__":
    unittest.main()
