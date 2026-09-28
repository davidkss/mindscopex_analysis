"""CPU-only regression for pinned behavioral runs through the job entry point."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from experiments.jobs import research_experiments as job


class PinnedBehavioralTests(unittest.TestCase):
    def test_pinned_run_initializes_feature_without_discovery(self) -> None:
        config_path = job.ROOT / "experiments/configs/study_behavioral_2b.toml"
        sae = object()
        model, tokenizer = object(), object()
        summary = {"accuracy": 0.5, "lure_rate": 0.5}
        result = {
            "baseline_summary": summary,
            "steered_summary": summary,
            "accuracy_delta": 0.0,
            "lure_rate_delta": 0.0,
            "baseline_rows": [],
            "steered_rows": [],
        }
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(mock.patch("torch.cuda.is_available", return_value=False))
            load_sae = stack.enter_context(mock.patch.object(job, "_load_sae", return_value=sae))
            discovery = stack.enter_context(mock.patch.object(job, "_discover_study_feature"))
            margin_model = stack.enter_context(mock.patch.object(job, "load_qwen_language_model"))
            stack.enter_context(
                mock.patch.object(
                    job, "load_qwen_text_generation_model", return_value=(model, tokenizer)
                )
            )
            steering = stack.enter_context(
                mock.patch.object(job, "steer_generation_labels", return_value=result)
            )
            stack.enter_context(mock.patch.object(job, "clear_device_cache"))
            stack.enter_context(mock.patch.object(job, "_safe_plot", return_value=None))

            run_dir = job.run(config_path, Path(directory))

            discovery.assert_not_called()
            margin_model.assert_not_called()
            load_sae.assert_called_once()
            self.assertEqual(load_sae.call_args.args[1], 5)
            self.assertEqual(
                [call.kwargs["coefficient"] for call in steering.call_args_list],
                [-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0],
            )
            for call in steering.call_args_list:
                self.assertIs(call.args[0], model)
                self.assertIs(call.args[1], tokenizer)
                self.assertEqual(call.kwargs["layer"], 5)
                self.assertEqual(call.kwargs["feature_id"], 30475)
                self.assertIs(call.kwargs["sae"], sae)
                self.assertEqual(call.kwargs["token_position"], "all")
                self.assertEqual(call.kwargs["output_mode"], "binary_choice")
            feature = json.loads((run_dir / "study_feature.json").read_text())
            self.assertEqual(
                feature, {"feature": {"layer": 5, "feature_id": 30475}, "source": "pinned"}
            )
            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["margin_kinds"], [])
            self.assertEqual(manifest["feature_source"], "pinned")
            self.assertEqual(manifest["study_feature"], feature["feature"])
