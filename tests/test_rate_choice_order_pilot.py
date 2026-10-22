"""CPU-only regression tests for rate choice-order Pilot 2A."""

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.jobs import research_experiments as job
from mindscopex_analysis.choice_order import (
    COEFFICIENTS,
    LETTER_MODE,
    ORDERS,
    evaluate,
)

CONFIG = (
    job.ROOT
    / "experiments/configs/"
    / "study_rate_choice_order_pilot_f22552_seed1_2b.toml"
)


def load_pilot():
    config = job.load_toml(CONFIG)
    bcfg = config["behavioral"]
    splits = job._load_splits(config)
    cases = job.family_balanced_subset(
        splits[bcfg["case_pool"]],
        max_cases=bcfg["max_cases"],
    )
    return config, cases


def fake_generate(items, coefficient):
    baseline = [
        {
            "case_id": case.case_id,
            "family": case.family,
            "answer": case.correct_answer,
            "label": "correct",
        }
        for case in items
    ]
    return {
        "baseline_rows": baseline,
        "steered_rows": copy.deepcopy(baseline),
    }


class RateChoiceOrderPilotTests(unittest.TestCase):

    def test_config_and_case_count(self):
        config, cases = load_pilot()
        bcfg = config["behavioral"]

        self.assertEqual(len(cases), 20)
        self.assertEqual(len({c.case_id for c in cases}), 20)
        self.assertEqual(bcfg["coefficients"], [7.0])
        self.assertEqual(bcfg["expected_cases"], 20)
        self.assertEqual(bcfg["choice_orders"], list(ORDERS))
        self.assertEqual(bcfg["output_mode"], LETTER_MODE)

    def test_evaluate_twenty_cases(self):
        _, cases = load_pilot()

        blocks = evaluate(
            cases,
            [7.0],
            fake_generate,
            output_mode=LETTER_MODE,
            expected_cases=20,
            expected_coefficients=(7.0,),
        )

        self.assertEqual(len(blocks), 2)
        self.assertEqual(
            {b["choice_order"] for b in blocks},
            set(ORDERS),
        )

        for block in blocks:
            self.assertEqual(len(block["baseline_rows"]), 20)
            self.assertEqual(len(block["steered_rows"]), 20)
            self.assertEqual(block["coefficient"], 7.0)
            self.assertEqual(block["response_encoding"], "choice_letter")

    def test_legacy_defaults_preserved(self):
        _, cases = load_pilot()

        with self.assertRaises(ValueError):
            evaluate(
                cases,
                [7.0],
                fake_generate,
                output_mode=LETTER_MODE,
            )

        self.assertEqual(COEFFICIENTS, (0.0, -7.0, -8.0))

    def test_job_routes_pilot_configuration(self):
        config, cases = load_pilot()

        def fake_steering(model, tokenizer, items, **kwargs):
            return fake_generate(items, kwargs["coefficient"])

        state = {
            "study_feature": {
                "feature": {"layer": 5, "feature_id": 22552},
                "sae": object(),
            }
        }

        with TemporaryDirectory() as tmp:
            with mock.patch.object(
                job,
                "steer_generation_labels",
                side_effect=fake_steering,
            ) as steering:
                result = job.run_behavioral(
                    object(),
                    object(),
                    {"all": cases, "test": cases},
                    config,
                    {},
                    Path(tmp),
                    state,
                )

            self.assertEqual(steering.call_count, 2)
            self.assertEqual(result["n_cases"], 20)
            self.assertEqual(result["output_mode"], LETTER_MODE)
            self.assertTrue(
                (Path(tmp) / "behavioral/generations.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
