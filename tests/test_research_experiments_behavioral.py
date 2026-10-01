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


class CombinedPinnedTests(unittest.TestCase):
    def test_combined_job_metadata_and_shared_sae(self):
        result = {
            'baseline_summary': {'accuracy': .5, 'lure_rate': .5},
            'steered_summary': {'accuracy': .75, 'lure_rate': .25},
            'accuracy_delta': .25, 'lure_rate_delta': -.25,
            'baseline_rows': [{'case_id': 'x', 'family': 'crt_rate', 'answer': 'a', 'label': 'lure'}],
            'steered_rows': [{'case_id': 'x', 'family': 'crt_rate', 'answer': 'b', 'label': 'correct'}],
        }
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            config = Path(directory) / 'combined.toml'
            config.write_text('[run]\nname="combined"\n[experiment]\nkind="behavioral"\n'
                              '[model]\nprofile="2b"\n[features]\nlayer=5\n'
                              'feature_ids=[30908,22552]\n[behavioral]\ncoefficients=[7.0]\n')
            discovery = stack.enter_context(mock.patch.object(job, '_discover_study_feature'))
            margin = stack.enter_context(mock.patch.object(job, 'load_qwen_language_model'))
            sae = stack.enter_context(mock.patch.object(job, '_load_sae', return_value=object()))
            stack.enter_context(mock.patch.object(job, 'load_qwen_text_generation_model', return_value=(None, None)))
            steering = stack.enter_context(mock.patch.object(job, 'steer_generation_labels', return_value=result))
            stack.enter_context(mock.patch.object(job, '_safe_plot'))
            stack.enter_context(mock.patch.object(job, 'clear_device_cache'))
            stack.enter_context(mock.patch('torch.cuda.is_available', return_value=False))
            output = job.run(config, Path(directory))
            discovery.assert_not_called()
            margin.assert_not_called()
            sae.assert_called_once()
            self.assertEqual(sae.call_args.args[1], 5)
            self.assertEqual(steering.call_args.kwargs['feature_ids'], [30908,22552])
            self.assertEqual(steering.call_args.kwargs['coefficient'], 7.)
            self.assertNotIn('feature_id', steering.call_args.kwargs)
            detail = json.loads((output/'behavioral/generations.json').read_text())[0]
            self.assertEqual(detail['layer'], 5)
            self.assertEqual(detail['feature_ids'], [30908,22552])
            self.assertEqual(detail['baseline_rows'], result['baseline_rows'])
            self.assertEqual(detail['steered_rows'], result['steered_rows'])
            metadata = json.loads((output/'study_feature.json').read_text())
            self.assertEqual(metadata, {'feature': {'layer':5, 'feature_ids':[30908,22552]}, 'source':'pinned'})
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['study_feature'], metadata['feature'])
            self.assertEqual((output/'behavioral.csv').read_text().splitlines()[0],
                             'coefficient,baseline_accuracy,steered_accuracy,accuracy_delta,baseline_lure_rate,steered_lure_rate,lure_rate_delta')

    def test_invalid_combined_config(self):
        for kind, features in (
            ('behavioral', '[feature]\nlayer=5\nfeature_id=30908\n[features]\nlayer=5\nfeature_ids=[22552]'),
            ('behavioral', '[features]\nlayer=5\nfeature_ids=[]'),
            ('behavioral', '[features]\nlayer=5\nfeature_ids=[30908,30908]'),
            *[(k, '[features]\nlayer=5\nfeature_ids=[30908,22552]')
              for k in ('study', 'causal_heldout', 'control_specificity')],
        ):
            with tempfile.TemporaryDirectory() as directory:
                config = Path(directory)/'invalid.toml'
                config.write_text(f'[run]\nname="invalid"\n[experiment]\nkind="{kind}"\n{features}\n')
                with self.assertRaises(ValueError):
                    job.run(config, Path(directory))
                self.assertFalse((Path(directory)/'invalid').exists())
