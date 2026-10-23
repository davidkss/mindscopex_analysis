"""CPU token-ID compatibility proofs for both explicit input protocols."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.jobs import full_answer_likelihood as job
from mindscopex_analysis import research
from mindscopex_analysis.full_answer_protocols import (
    BEHAVIOR, DIAGNOSTIC, candidate_texts, token_ids_sha256,
)
from mindscopex_analysis.generation import _prepare_inputs
from scripts import analyze_fresh_v2_full_answer_likelihood as analysis
from scripts import diagnose_fresh_v2_full_answer_likelihood as diagnostic
from scripts import diagnose_fresh_v2_matched_branch_margin as legacy
from test_full_answer_likelihood import (
    CONFIG, INVENTORY, Model, Tokenizer, fake_hook, read_rows, refresh_manifest, replace_rows,
)

BEHAVIOR_CONFIG = CONFIG.with_name("diagnose_fresh_v2_growth_full_answer_behavior_f30908_2b.toml")


class SpaceTokenizer(Tokenizer):
    """Whitespace-sensitive tokens; raw prompt IDs differ from chat-template IDs."""
    pad_token_id = None

    def __init__(self):
        super().__init__()
        self.encoded = []

    def encode(self, text, *, add_special_tokens):
        ids = ([29] if text.startswith(" ") else []) + super().encode(
            text, add_special_tokens=add_special_tokens)
        self.encoded.append((text, list(ids)))
        return ids

    def __call__(self, text, *, return_tensors):
        assert return_tensors == "pt"
        ids = [6, 7] + self.encode(text, add_special_tokens=False)
        return dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones((1, len(ids))))

    def decode(self, ids, **kwargs):
        if ids == [29]:
            return " "
        return super().decode(ids, **kwargs)


class ReferenceCaptureModel(Model):
    """Exercise the real behavioral trie using CPU tensors, without forward()."""
    def __init__(self, tokenizer, candidate_index):
        super().__init__()
        self.tokenizer = tokenizer
        self.candidate_index = candidate_index

    def generate(self, **kwargs):
        self.prompt_ids = kwargs["input_ids"][0].tolist()
        self.candidate_ids = [ids for _, ids in self.tokenizer.encoded[:2]]
        self.generation_kwargs = kwargs
        sequence = kwargs["input_ids"].clone()
        for token in self.candidate_ids[self.candidate_index] + [kwargs["eos_token_id"][0]]:
            allowed = kwargs["prefix_allowed_tokens_fn"](0, sequence[0])
            assert token in allowed
            sequence = torch.cat((sequence, torch.tensor([[token]])), dim=1)
        return sequence


class FullAnswerProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.runs = {}
        for protocol in (DIAGNOSTIC, BEHAVIOR):
            model = Model()
            output = cls.root / protocol
            with mock.patch.object(diagnostic.torch.cuda, "is_available", return_value=True), \
                 mock.patch.object(diagnostic, "load_qwen_text_generation_model",
                                   return_value=(model, SpaceTokenizer())), \
                 mock.patch.object(diagnostic, "load_qwen_scope_sae", return_value=object()), \
                 mock.patch.object(diagnostic, "find_decoder_block", return_value=model.block), \
                 mock.patch.object(diagnostic, "make_feature_steering_hook", side_effect=fake_hook), \
                 mock.patch("builtins.print"):
                diagnostic.run(INVENTORY, output, input_protocol=protocol)
            cls.runs[protocol] = output

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        torch.set_num_threads(cls.old_threads)

    def test_a_matches_existing_first_divergence_inputs_for_all_33_cases(self):
        _, cases = diagnostic.select_growth_cases(INVENTORY)
        for case in cases:
            tokenizer = SpaceTokenizer()
            prompt, candidates, _, common = diagnostic.prepare_case(
                tokenizer, case, input_protocol=DIAGNOSTIC)
            reference = _prepare_inputs(tokenizer, case.prompt, system_prompt="",
                                        enable_thinking=False, use_chat_template=True)
            self.assertEqual(prompt, reference["input_ids"][0].tolist())
            self.assertEqual(candidates["correct"],
                             tokenizer.encode(case.correct_answer.strip(), add_special_tokens=False))
            self.assertEqual(candidates["lure"],
                             tokenizer.encode(case.lure_answer.strip(), add_special_tokens=False))
            prefix, correct, lure, branch = legacy.first_branch(
                tokenizer, case.correct_answer, case.lure_answer)
            self.assertEqual(common, branch)
            self.assertEqual(candidates["correct"][:common], prefix)
            self.assertEqual(candidates["correct"][common], correct)
            self.assertEqual(candidates["lure"][common], lure)

    def test_b_matches_actual_behavior_function_prompt_and_both_candidate_paths_all_33(self):
        _, cases = diagnostic.select_growth_cases(INVENTORY)
        for case in cases:
            for candidate_index in (0, 1):
                tokenizer = SpaceTokenizer()
                reference = ReferenceCaptureModel(tokenizer, candidate_index)
                answer = research._binary_choice_completion(
                    reference, tokenizer, case, max_new_tokens=16, letter_candidates=False)
                prompt, candidates, audits, _ = diagnostic.prepare_case(
                    SpaceTokenizer(), case, input_protocol=BEHAVIOR)
                self.assertEqual(prompt, reference.prompt_ids)
                self.assertEqual(list(candidates.values()), reference.candidate_ids)
                self.assertEqual(answer, (case.correct_answer if candidate_index == 0
                                          else case.lure_answer).strip())
                self.assertFalse(reference.generation_kwargs["do_sample"])
                self.assertTrue(reference.generation_kwargs["renormalize_logits"])
                self.assertEqual(reference.contexts, [])  # No model.forward or GPU execution.
                self.assertTrue(all(a["behavior_reference_token_ids_verified"] for a in audits.values()))

    def test_spaces_and_joint_boundary_audits_are_protocol_specific(self):
        _, cases = diagnostic.select_growth_cases(INVENTORY)
        case = cases[0]
        a_prompt, a_candidates, _, _ = diagnostic.prepare_case(
            SpaceTokenizer(), case, input_protocol=DIAGNOSTIC)
        b_prompt, b_candidates, audits, _ = diagnostic.prepare_case(
            SpaceTokenizer(), case, input_protocol=BEHAVIOR)
        self.assertNotEqual(a_prompt, b_prompt)
        for kind in ("correct", "lure"):
            self.assertEqual(b_candidates[kind], [29] + a_candidates[kind])
            self.assertTrue(candidate_texts(case, BEHAVIOR)[kind].startswith(" "))
            self.assertFalse(audits[kind]["raw_prompt_joint_matches_explicit"])
        self.assertNotEqual(token_ids_sha256(a_candidates["correct"]),
                            token_ids_sha256(b_candidates["correct"]))

    def test_manifest_and_csv_token_fingerprints_for_both_protocols(self):
        for protocol, directory in self.runs.items():
            data, scores, manifest = analysis.read_diagnostic(
                directory, INVENTORY, input_protocol=protocol)
            self.assertEqual(manifest["input_protocol"], protocol)
            self.assertEqual(manifest["eos_policy"], "exclude")
            self.assertEqual(manifest["sae_settings"]["feature_id"], 30908)
            self.assertTrue(manifest["sae_settings"]["repo_id"])
            self.assertFalse(manifest["cross_protocol_results_are_repeated_measurements"])
            self.assertEqual(len(scores), 99)
            self.assertEqual(len(manifest["prompts"]), 33)
            for record in manifest["prompts"]:
                self.assertEqual(record["prompt_token_ids_sha256"], token_ids_sha256(record["prompt_token_ids"]))
                for kind, ids in record["candidate_token_ids"].items():
                    self.assertEqual(record["candidate_token_ids_sha256"][kind], token_ids_sha256(ids))

    def test_behavior_branch_reference_is_not_same_condition_comparison(self):
        rows, summary = analysis.analyze(self.runs[BEHAVIOR], INVENTORY, bootstrap=30)
        self.assertEqual(summary["input_protocol"], BEHAVIOR)
        self.assertFalse(summary["first_divergence_reference_comparable"])
        self.assertEqual(len(rows), 64)
        selected = [r for r in rows if r["existing_branch_available"]]
        self.assertEqual(len(selected), 14)
        for row in selected:
            self.assertFalse(row["existing_branch_protocol_compatible"])
            self.assertEqual(row["existing_branch_comparison_status"], "cross_protocol_not_comparable")
            self.assertIsNone(row["existing_vs_trajectory_branch_max_abs_error"])

    def test_mixed_score_or_token_protocol_and_wrong_expected_protocol_rejected(self):
        for filename in ("full_answer_scores.csv", "token_trajectory.csv"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / "mixed"
                shutil.copytree(self.runs[DIAGNOSTIC], directory)
                rows = read_rows(directory / filename)
                rows[0]["input_protocol"] = BEHAVIOR
                replace_rows(directory / filename, rows)
                refresh_manifest(directory)
                with self.assertRaisesRegex(ValueError, "Mixed input_protocol"):
                    analysis.read_diagnostic(directory, INVENTORY)
        with self.assertRaisesRegex(ValueError, "Requested input_protocol"):
            analysis.read_diagnostic(self.runs[BEHAVIOR], INVENTORY, input_protocol=DIAGNOSTIC)

    def test_missing_protocol_or_tampered_token_hash_is_rejected(self):
        for kind in ("untagged", "prompt_hash", "candidate_hash"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / "run"
                shutil.copytree(self.runs[DIAGNOSTIC], directory)
                path = directory / "manifest.json"
                manifest = json.loads(path.read_text())
                if kind == "untagged":
                    del manifest["input_protocol"]
                elif kind == "prompt_hash":
                    manifest["prompts"][0]["prompt_token_ids_sha256"] = "wrong"
                else:
                    manifest["prompts"][0]["candidate_token_ids_sha256"]["correct"] = "wrong"
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    analysis.read_diagnostic(directory, INVENTORY)

    def test_required_protocol_configs_and_distinct_run_directories(self):
        a, b = job.load_toml(CONFIG), job.load_toml(BEHAVIOR_CONFIG)
        self.assertEqual(a["diagnostic"]["input_protocol"], DIAGNOSTIC)
        self.assertEqual(b["diagnostic"]["input_protocol"], BEHAVIOR)
        self.assertNotEqual(a["run"]["name"], b["run"]["name"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path, protocol in ((CONFIG, DIAGNOSTIC), (BEHAVIOR_CONFIG, BEHAVIOR)):
                with mock.patch.object(job.subprocess, "run") as execute:
                    run_dir = job.run(path, root)
                    command = execute.call_args.args[0]
                    self.assertEqual(command[command.index("--input-protocol")+1], protocol)
                    self.assertEqual(json.loads((run_dir / "job_manifest.json").read_text())["input_protocol"], protocol)
            missing = root / "missing.toml"
            missing.write_text(CONFIG.read_text().replace('input_protocol = "diagnostic_compatible"\n', ""))
            with mock.patch.object(job.subprocess, "run") as execute:
                with self.assertRaisesRegex(ValueError, "input_protocol is required"):
                    job.run(missing, root / "missing_output")
                execute.assert_not_called()

    def test_behavior_eos_policy_and_teacher_forcing_hook_regression(self):
        _, cases = diagnostic.select_growth_cases(INVENTORY)
        model = Model()
        with mock.patch.object(diagnostic, "make_feature_steering_hook", side_effect=fake_hook) as hooks:
            scores, tokens, record = diagnostic.measure_case(
                model, SpaceTokenizer(), model.block, object(), cases[0], 30,
                input_protocol=BEHAVIOR)
        self.assertEqual(model.hook_counts, [0, 0, 1, 1, 1, 1])
        self.assertEqual([c.kwargs["coefficient"] for c in hooks.call_args_list], [7.0, 8.0])
        for row in scores:
            for kind in ("correct", "lure"):
                trace = [r for r in tokens if r["coefficient"] == row["coefficient"]
                         and r["candidate_type"] == kind]
                self.assertAlmostEqual(sum(r["token_logprob"] for r in trace), row[f"{kind}_score"])
                self.assertTrue(trace[-1]["is_eos"])
                self.assertEqual(record["candidate_token_ids"][kind][-1], 30)

    def test_analysis_outputs_cannot_be_overwritten_by_other_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            paths = analysis.run(self.runs[DIAGNOSTIC], INVENTORY, output, bootstrap=20,
                                 input_protocol=DIAGNOSTIC)
            before = [p.read_bytes() for p in paths]
            with self.assertRaises(FileExistsError):
                analysis.run(self.runs[BEHAVIOR], INVENTORY, output, bootstrap=20,
                             input_protocol=BEHAVIOR)
            self.assertEqual(before, [p.read_bytes() for p in paths])


if __name__ == "__main__":
    unittest.main()
