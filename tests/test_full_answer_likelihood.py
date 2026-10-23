"""CPU-only full-answer calculation, trajectory, inventory and Colab regressions."""
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from experiments.jobs import full_answer_likelihood as job
from experiments.runners.collect_artifacts import make_source_archive
from experiments.runners.remote_bootstrap import render_bootstrap
from mindscopex_analysis.cases import LureCase
from scripts import analyze_fresh_v2_full_answer_likelihood as analysis
from scripts import diagnose_fresh_v2_full_answer_likelihood as diagnostic

INVENTORY = job.ROOT / "experiments/inputs/fresh_v2_growth_controls_f30908.json"
CONFIG = job.ROOT / "experiments/configs/diagnose_fresh_v2_growth_full_answer_f30908_2b.toml"


class Tokenizer:
    eos_token_id = 30

    def __init__(self, merged=False):
        self.merged = merged

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        prefix = [1, 2] if text.startswith("P:") else []
        text = text[2:] if prefix else text
        ids = [3+int(char) for char in text if char.isdigit()]
        if text.endswith(" days"):
            ids.append(25)
        if self.merged and prefix and ids:
            return [1, 29] + ids[1:]
        return prefix + ids

    def decode(self, ids, **kwargs):
        if ids == [1, 2]:
            return "P:"
        return "".join(str(i-3) if 3 <= i <= 12 else " days" if i == 25
                       else "<eos>" if i == 30 else "?" for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return dict(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.ones((1, 2)))


class Model(torch.nn.Module):
    """Known causal next-token scores: token+1 receives logit 2, others 0."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.block = torch.nn.Identity()
        self.contexts = []
        self.hook_counts = []

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self.anchor)

    def forward(self, input_ids, use_cache):
        assert use_cache is False
        self.contexts.append(input_ids.tolist())
        self.hook_counts.append(len(self.block._forward_hooks))
        logits = torch.zeros((*input_ids.shape, 32))
        logits.scatter_(2, ((input_ids+1) % 32)[:, :, None], 2)
        return SimpleNamespace(logits=self.block(logits))


def fake_hook(sae, ids, *, coefficient, token_position):
    assert ids == [30908] and token_position == "all"
    return lambda module, inputs, output: output + torch.arange(32) * coefficient * .01


def refresh_manifest(directory):
    path = directory / "manifest.json"
    data = json.loads(path.read_text())
    for filename, key in (("full_answer_scores.csv", "score_csv_sha256"),
                          ("token_trajectory.csv", "trajectory_csv_sha256")):
        data[key] = diagnostic.digest(directory / filename)
    path.write_text(json.dumps(data))


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def replace_rows(path, rows):
    path.unlink()
    diagnostic.write_csv(path, rows)


class FullAnswerLikelihoodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_count = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.run_dir = cls.root / "synthetic_cpu_run"
        cls.model = Model()
        with mock.patch.object(diagnostic.torch.cuda, "is_available", return_value=True), \
             mock.patch.object(diagnostic, "load_qwen_text_generation_model",
                               return_value=(cls.model, Tokenizer())), \
             mock.patch.object(diagnostic, "load_qwen_scope_sae", return_value=object()), \
             mock.patch.object(diagnostic, "find_decoder_block", return_value=cls.model.block), \
             mock.patch.object(diagnostic, "make_feature_steering_hook", side_effect=fake_hook), \
             mock.patch("builtins.print"):
            diagnostic.run(INVENTORY, cls.run_dir, input_protocol="diagnostic_compatible")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        torch.set_num_threads(cls.thread_count)

    def test_teacher_forcing_alignment_sum_and_cumulative(self):
        model, tokenizer = Model(), Tokenizer()
        score, trace = diagnostic.teacher_force(model, tokenizer, [1, 2], [3, 4])
        expected_lp = 2-math.log(math.exp(2)+31)
        self.assertAlmostEqual(score, 2*expected_lp, places=6)
        self.assertAlmostEqual(sum(r["token_logprob"] for r in trace), score, places=7)
        self.assertAlmostEqual(trace[-1]["cumulative_logprob"], score, places=7)
        self.assertEqual(model.contexts, [[[1, 2, 3]]])
        self.assertEqual([r["token_index"] for r in trace], [1, 2])

    def test_shared_prefix_and_separate_candidate_paths(self):
        model, tokenizer = Model(), Tokenizer()
        case = LureCase("case", "crt_growth", "Question?", "12 days", "13 days")
        with mock.patch.object(diagnostic, "make_feature_steering_hook", side_effect=fake_hook) as hooks:
            scores, tokens, _ = diagnostic.measure_case(model, tokenizer, model.block, object(), case)
        self.assertEqual(scores[0]["shared_prefix_length"], 1)
        self.assertEqual(scores[0]["first_divergence_position"], 2)
        baseline = [r for r in tokens if r["coefficient"] == 0]
        for kind in ("correct", "lure"):
            selected = [r for r in baseline if r["candidate_type"] == kind]
            self.assertEqual([r["shared_prefix"] for r in selected], [True, False, False])
        self.assertEqual(baseline[0]["token_logprob"], baseline[3]["token_logprob"])
        self.assertNotEqual(model.contexts[0][0], model.contexts[1][0])
        self.assertEqual(len(model.block._forward_hooks), 0)
        self.assertEqual(model.hook_counts, [0, 0, 1, 1, 1, 1])
        self.assertEqual([call.kwargs["coefficient"] for call in hooks.call_args_list], [7.0, 8.0])
        self.assertTrue(all(call.args[1] == [30908] and call.kwargs["token_position"] == "all"
                            for call in hooks.call_args_list))

    def test_behavioral_and_legacy_diagnostic_protocols_are_distinct(self):
        from mindscopex_analysis import research
        from mindscopex_analysis.lure_datasets import load_lure_dataset
        from mindscopex_analysis.prompts import instruct_lure_case

        class BehavioralTokenizer(Tokenizer):
            pad_token_id = None
            def encode(self, text, *, add_special_tokens):
                tokens = super().encode(text, add_special_tokens=add_special_tokens)
                return ([29] if text.startswith(" ") else []) + tokens

            def __call__(self, prompt, *, return_tensors):
                return dict(input_ids=torch.tensor([[6, 7]]), attention_mask=torch.ones((1, 2)))

        class BehavioralModel(Model):
            def generate(self, **kwargs):
                self.generated_prompt = kwargs["input_ids"][0].tolist()
                continuation = tokenizer.encode(case.correct_answer, add_special_tokens=False) + [30]
                return torch.cat((kwargs["input_ids"], torch.tensor([continuation])), dim=1)

        case = instruct_lure_case(next(c for c in load_lure_dataset("crt_fresh_v2")
                                      if c.case_id == "crt_fresh_v2_growth_007"))
        tokenizer, model = BehavioralTokenizer(), BehavioralModel()
        research._binary_choice_completion(model, tokenizer, case, max_new_tokens=16)
        diagnostic_prompt, candidates, _, _ = diagnostic.prepare_case(tokenizer, case)
        self.assertTrue(case.correct_answer.startswith(" "))
        self.assertNotEqual(model.generated_prompt, diagnostic_prompt)
        self.assertNotEqual(tokenizer.encode(case.correct_answer, add_special_tokens=False),
                            candidates["correct"])

    def test_prompt_answer_boundary_audit_preserves_scored_path(self):
        case = LureCase("case", "crt_growth", "Question?", "12 days", "13 days")
        prompt, candidates, audits, common = diagnostic.prepare_case(Tokenizer(merged=True), case)
        self.assertEqual(prompt, [1, 2])
        self.assertEqual(candidates["correct"], [4, 5, 25])
        self.assertFalse(audits["correct"]["joint_matches_explicit_sequence"])
        self.assertFalse(audits["correct"]["joint_prompt_prefix_preserved"])
        self.assertTrue(audits["correct"]["prompt_roundtrip_matches"])
        self.assertEqual(common, 1)

    def test_eos_inclusion_exclusion_and_explicit_policy(self):
        tokenizer = Tokenizer()
        self.assertIsNone(diagnostic.eos_id(tokenizer))
        self.assertEqual(diagnostic.eos_id(tokenizer, True), 30)
        tokenizer.eos_token_id = [30, 31]
        with self.assertRaises(ValueError):
            diagnostic.eos_id(tokenizer, True)
        self.assertEqual(diagnostic.eos_id(tokenizer, True, 31), 31)
        with self.assertRaises(ValueError):
            diagnostic.eos_id(tokenizer, False, 30)
        case = LureCase("case", "crt_growth", "Question?", "12 days", "13 days")
        model = Model()
        with mock.patch.object(diagnostic, "make_feature_steering_hook", side_effect=fake_hook):
            scores, tokens, _ = diagnostic.measure_case(model, Tokenizer(), model.block, object(), case, 30)
        self.assertTrue(scores[0]["include_eos"])
        self.assertEqual(scores[0]["correct_token_length"], 4)
        self.assertEqual(scores[0]["correct_answer_token_length"], 3)
        self.assertTrue(all(r["is_eos"] == (r["token_index"] == 4) for r in tokens))
        for row in scores:
            selected = [r for r in tokens if r["coefficient"] == row["coefficient"]
                        and r["candidate_type"] == "correct"]
            self.assertAlmostEqual(sum(r["token_logprob"] for r in selected), row["correct_score"])

    def test_strict_prefix_answer_has_no_branch_without_eos(self):
        class PrefixTokenizer(Tokenizer):
            def encode(self, text, *, add_special_tokens):
                if text == "1":
                    return [4]
                if text == "12":
                    return [4, 5]
                return super().encode(text, add_special_tokens=add_special_tokens)
        case = LureCase("case", "crt_growth", "Question?", "1", "12")
        model = Model()
        with mock.patch.object(diagnostic, "make_feature_steering_hook", side_effect=fake_hook):
            scores, _, _ = diagnostic.measure_case(model, PrefixTokenizer(), model.block, object(), case)
        self.assertIsNone(scores[0]["first_divergence_margin"])
        self.assertTrue(math.isfinite(scores[0]["full_margin"]))

    def test_exact_inventory_selection_and_output_sizes(self):
        data, cases = diagnostic.select_growth_cases(INVENTORY)
        self.assertEqual({c.case_id for c in cases}, {r["case_id"] for r in data["cases"]})
        self.assertEqual(len(cases), 33)
        self.assertEqual(len(data["slots"]), 64)
        scores = read_rows(self.run_dir / "full_answer_scores.csv")
        self.assertEqual(len(scores), 99)
        paths = {(r["case_id"], r["coefficient"], r["candidate_type"])
                 for r in read_rows(self.run_dir / "token_trajectory.csv")}
        self.assertEqual(len(paths), 198)

    def test_analysis_group_and_seed_counts_and_existing_branch_coverage(self):
        rows, summary = analysis.analyze(self.run_dir, INVENTORY, bootstrap=50)
        self.assertEqual(len(rows), 64)
        self.assertEqual(summary["unique_cases"], 33)
        for group, unique, slots in (("correct_to_lure", 6, 14), ("correct_to_correct", 20, 34),
                                      ("lure_to_lure", 7, 16)):
            self.assertEqual(summary["groups"][group]["unique_cases"], unique)
            self.assertEqual(summary["groups"][group]["evaluation_slots"], slots)
        self.assertEqual(summary["existing_branch_unique_cases"], 6)
        self.assertEqual(len(summary["missing_existing_branch_case_ids"]), 27)
        self.assertEqual(sum(r["evaluation_slots"] for r in summary["by_seed"].values()), 64)
        self.assertEqual(summary["boundary_audit"]["candidate_audits"], 66)
        self.assertEqual(summary["boundary_audit"]["joint_encoding_mismatches"], 0)

    def test_branch_margin_join_uses_case_ids_and_checks_tokens(self):
        data, scores, _ = analysis.read_diagnostic(self.run_dir, INVENTORY)
        definitions = {r["case_id"]: r for r in data["cases"]}
        branch_rows = []
        for cid in reversed(data["case_ids"]):
            score = scores[(cid, 0.0)]
            values = [scores[(cid, c)]["first_divergence_margin"] for c in diagnostic.COEFFICIENTS]
            branch_rows.append(dict(
                case_id=cid, family="crt_growth", correct_answer=definitions[cid]["correct_answer"],
                lure_answer=definitions[cid]["lure_answer"], baseline_margin=values[0],
                plus7_margin=values[1], plus8_margin=values[2], plus8_minus_plus7=values[2]-values[1],
                branch_position_1based=score["first_divergence_position"],
                correct_branch_token_id=score["correct_branch_token_id"],
                lure_branch_token_id=score["lure_branch_token_id"]))
        with tempfile.TemporaryDirectory() as tmp:
            branch = Path(tmp) / "synthetic_branch.csv"
            diagnostic.write_csv(branch, branch_rows)
            rows, summary = analysis.analyze(self.run_dir, INVENTORY, [branch], bootstrap=20)
        self.assertEqual(summary["existing_branch_unique_cases"], 33)
        for row in rows:
            if row["case_id"] in data["case_ids"]:
                self.assertTrue(row["existing_branch_tokenization_compatible"])
                self.assertEqual(row["existing_vs_trajectory_branch_max_abs_error"], 0.0)

    def test_duplicate_missing_and_bad_trajectory_data_rejected(self):
        for kind in ("duplicate_score", "missing_score", "duplicate_token", "missing_token", "bad_cumulative"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp) / "run"
                shutil.copytree(self.run_dir, directory)
                scores_path = directory / "full_answer_scores.csv"
                token_path = directory / "token_trajectory.csv"
                if kind in ("duplicate_score", "missing_score"):
                    rows = read_rows(scores_path)
                    rows.append(rows[0]) if kind == "duplicate_score" else rows.pop()
                    replace_rows(scores_path, rows)
                else:
                    rows = read_rows(token_path)
                    if kind == "duplicate_token":
                        rows.append(rows[0])
                    elif kind == "missing_token":
                        rows.pop()
                    else:
                        rows[0]["cumulative_logprob"] = "999"
                    replace_rows(token_path, rows)
                refresh_manifest(directory)
                with self.assertRaises(ValueError):
                    analysis.read_diagnostic(directory, INVENTORY)

    def test_eos_inclusive_complete_run_and_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "eos_run"
            model = Model()
            with mock.patch.object(diagnostic.torch.cuda, "is_available", return_value=True), \
                 mock.patch.object(diagnostic, "load_qwen_text_generation_model",
                                   return_value=(model, Tokenizer())), \
                 mock.patch.object(diagnostic, "load_qwen_scope_sae", return_value=object()), \
                 mock.patch.object(diagnostic, "find_decoder_block", return_value=model.block), \
                 mock.patch.object(diagnostic, "make_feature_steering_hook", side_effect=fake_hook), \
                 mock.patch("builtins.print"):
                diagnostic.run(INVENTORY, output, include_eos=True, input_protocol="diagnostic_compatible")
            data, scores, manifest = analysis.read_diagnostic(output, INVENTORY)
            self.assertTrue(manifest["include_eos"])
            self.assertEqual(manifest["eos_token_id"], 30)
            rows, summary = analysis.analyze(output, INVENTORY, bootstrap=20)
            self.assertEqual(len(rows), 64)
            self.assertTrue(summary["include_eos"])

    def test_invalid_config_rejected_before_subprocess(self):
        text = CONFIG.read_text().replace("feature = 30908", "feature = 22552")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.toml"
            path.write_text(text)
            with mock.patch.object(job.subprocess, "run") as execute:
                with self.assertRaises(ValueError):
                    job.run(path, Path(tmp) / "output")
                execute.assert_not_called()

    def test_duplicate_inventory_case_rejected(self):
        data = json.loads(INVENTORY.read_text())
        data["cases"].append(data["cases"][0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inventory.json"
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                diagnostic.select_growth_cases(path)

    def test_diagnostic_analysis_and_job_refuse_overwrite(self):
        with self.assertRaises(FileExistsError):
            diagnostic.run(INVENTORY, self.run_dir, input_protocol="diagnostic_compatible")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "analysis"
            outputs = analysis.run(self.run_dir, INVENTORY, directory, bootstrap=20)
            before = [p.read_bytes() for p in outputs]
            with self.assertRaises(FileExistsError):
                analysis.run(self.run_dir, INVENTORY, directory, bootstrap=20)
            self.assertEqual(before, [p.read_bytes() for p in outputs])
            with mock.patch.object(job.subprocess, "run") as execute:
                run_dir = job.run(CONFIG, Path(tmp) / "outputs")
                with self.assertRaises(FileExistsError):
                    job.run(CONFIG, Path(tmp) / "outputs")
                execute.assert_called_once()
            self.assertTrue((run_dir / "config.toml").exists())

    def test_job_refuses_preexisting_manifest_without_changing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / job.run_name(job.load_toml(CONFIG))
            run_dir.mkdir()
            existing = run_dir / "job_manifest.json"
            existing.write_text('{"status": "original research result"}')
            before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
            with mock.patch.object(job.subprocess, "run") as execute:
                with self.assertRaises(FileExistsError):
                    job.run(CONFIG, root)
                execute.assert_not_called()
            self.assertEqual(before, {p.name: p.read_bytes() for p in run_dir.iterdir()})

    def test_job_accepts_bootstrap_log_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / job.run_name(job.load_toml(CONFIG))
            run_dir.mkdir()
            log = run_dir / "job.log"
            log.write_text("current bootstrap log")
            with mock.patch.object(job.subprocess, "run") as execute:
                job.run(CONFIG, root)
                execute.assert_called_once()
            self.assertEqual(log.read_text(), "current bootstrap log")

    def test_hook_removed_when_teacher_forcing_fails(self):
        block = torch.nn.Identity()
        case = LureCase("case", "crt_growth", "Question?", "12 days", "13 days")
        calls = [0]
        def fail_after_baseline(*args, **kwargs):
            calls[0] += 1
            if calls[0] > 2:
                raise RuntimeError("synthetic failure")
            return -1.0, [dict(token_index=i+1, token_id=token, token_text="x",
                               token_logprob=-1/len(args[3]), cumulative_logprob=-(i+1)/len(args[3]))
                          for i, token in enumerate(args[3])]
        with mock.patch.object(diagnostic, "teacher_force", side_effect=fail_after_baseline), \
             mock.patch.object(diagnostic, "make_feature_steering_hook", side_effect=fake_hook):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                diagnostic.measure_case(Model(), Tokenizer(), block, object(), case)
        self.assertEqual(len(block._forward_hooks), 0)

    def test_source_zip_includes_33_case_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection = root / "experiments/inputs/fresh_v2_growth_controls_f30908.json"
            selection.parent.mkdir(parents=True)
            selection.write_bytes(INVENTORY.read_bytes())
            archive = make_source_archive(root, root / "results/source.zip")
            with zipfile.ZipFile(archive) as source:
                raw = source.read(str(selection.relative_to(root)))
                self.assertEqual(hashlib.sha256(raw).hexdigest(), diagnostic.digest(INVENTORY))
                self.assertEqual(len(json.loads(raw)["cases"]), 33)
                self.assertEqual(len(json.loads(raw)["slots"]), 64)

    def test_bootstrap_guards_full_answer_job_and_archives_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = dict(remote_repo_dir=str(job.ROOT), remote_output_root=str(root / "output"),
                           remote_config_path=str(CONFIG), run_name="full_answer",
                           job="full_answer_likelihood", job_path="experiments/jobs/full_answer_likelihood.py")
            source = render_bootstrap(payload).rsplit("\nmain()", 1)[0]
            source = source.replace('Path("/content")', f'Path({str(root)!r})')
            namespace = {}
            exec(compile(source, "<bootstrap>", "exec"), namespace)
            namespace["prepare_source"] = lambda: job.ROOT
            namespace["ensure_dependencies"] = lambda repo: None
            def capture(command, log):
                self.assertTrue(str(command[1]).endswith("full_answer_likelihood.py"))
                log.write_text("synthetic full-answer log")
                (log.parent / "likelihood").mkdir()
                (log.parent / "likelihood/manifest.json").write_text("{}")
                return 0
            namespace["run_capturing"] = capture
            with mock.patch.dict(os.environ), mock.patch("os.chdir"):
                namespace["main"]()
                with self.assertRaises(FileExistsError):
                    namespace["main"]()
            with zipfile.ZipFile(root / "output/full_answer.zip") as archive:
                self.assertIn("job.log", archive.namelist())
                self.assertIn("likelihood/manifest.json", archive.namelist())


if __name__ == "__main__":
    unittest.main()
