"""CPU regressions for Growth inventory, selection, archive transport and comparison."""
import copy
import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from experiments.jobs import matched_branch_margin as job
from experiments.runners.collect_artifacts import make_source_archive
from mindscopex_analysis.growth_control_inventory import load_inventory
from scripts import analyze_fresh_v2_growth_controls as analysis

CONFIG = job.ROOT / "experiments/configs/diagnose_fresh_v2_growth_controls_f30908_2b.toml"
PORTABLE = job.ROOT / "experiments/inputs/fresh_v2_growth_controls_f30908.json"


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def behavior(seed, number, first="correct", second="correct"):
    return dict(seed=seed, case_id=f"crt_fresh_v2_growth_{number:03}", family="crt_growth",
                steered_7=first, steered_8=second, correct_answer="31 days", lure_answer="16 days")


def measurement(number, plus7=-1, plus8=1):
    return dict(case_id=f"crt_fresh_v2_growth_{number:03}", family="crt_growth",
                correct_answer="31 days", lure_answer="16 days", baseline_margin=-.5,
                plus7_margin=plus7, plus8_margin=plus8, plus8_minus_plus7=plus8-plus7)


class GrowthControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.behavior, self.old, self.new = [self.root / name for name in ("behavior.csv", "old.csv", "new.csv")]
        write_csv(self.behavior, [
            behavior(1, 7, "correct", "lure"), behavior(2, 7, "correct", "lure"),
            behavior(1, 1), behavior(2, 1, "lure", "lure"), behavior(1, 2)])
        write_csv(self.old, [measurement(7)])
        rows, self.summary, self.data = analysis.build_inventory(self.behavior, self.old)
        self.rows = rows
        self.portable = self.root / "inventory.json"
        self.portable.write_text(json.dumps(self.data))
        write_csv(self.new, [measurement(1, 0, 1), measurement(2, 1, 1)])

    def test_inventory_preserves_slots_overlap_and_only_missing_cases(self):
        self.assertEqual(self.summary["evaluation_slots"], 5)
        self.assertEqual(self.summary["unique_cases"], 3)
        self.assertEqual(self.summary["additional_unique_cases"], 2)
        self.assertEqual(self.summary["by_transition"]["correct_to_lure"]["evaluation_slots"], 2)
        self.assertEqual(self.summary["by_transition"]["correct_to_lure"]["unique_cases"], 1)
        self.assertEqual(self.summary["overlapping_group_case_ids"], ["crt_fresh_v2_growth_001"])
        self.assertEqual(self.data["case_ids"], ["crt_fresh_v2_growth_001", "crt_fresh_v2_growth_002"])
        self.assertTrue(self.summary["expected_count_mismatches"])
        load_inventory(self.portable)

    def test_comparison_separates_unique_cases_slots_and_overlap(self):
        rows, summary = analysis.compare(self.portable, self.new, replicates=100, seed=1)
        self.assertEqual(len(rows), 5)
        groups = summary["groups"]
        self.assertEqual(groups["correct_to_lure"]["unique_case_statistics"]["n"], 1)
        self.assertEqual(groups["correct_to_lure"]["slot_weighted_descriptive_statistics"]["n"], 2)
        stable = groups["correct_to_correct"]["unique_case_statistics"]
        self.assertEqual(stable["metrics"]["plus8_minus_plus7"]["mean"], .5)
        self.assertEqual(stable["delta_direction"]["equal"]["rate"], .5)
        self.assertEqual(summary["consistency"]["existing_cases_preserved"], ["crt_fresh_v2_growth_007"])
        self.assertIsNone(summary["consistency"]["repeated_values_consistent"])
        self.assertEqual(sum(r["margin_source"] == "existing" for r in rows), 2)
        self.assertEqual(summary["bootstrap"], analysis.compare(self.portable, self.new, 100, 1)[1]["bootstrap"])

    def test_repeated_case_changes_slot_mean_but_not_unique_case_mean(self):
        data = copy.deepcopy(self.data)
        repeated = next(dict(row) for row in data["slots"]
                        if row["case_id"] == "crt_fresh_v2_growth_001" and row["plus7_label"] == "correct")
        repeated["seed"] = 3
        data["slots"].append(repeated)
        next(row for row in data["cases"] if row["case_id"] == repeated["case_id"])["evaluation_slots"] += 1
        self.portable.write_text(json.dumps(data))
        _, summary = analysis.compare(self.portable, self.new, 30)
        group = summary["groups"]["correct_to_correct"]
        self.assertEqual(group["unique_case_statistics"]["metrics"]["plus8_minus_plus7"]["mean"], .5)
        self.assertAlmostEqual(
            group["slot_weighted_descriptive_statistics"]["metrics"]["plus8_minus_plus7"]["mean"], 2/3)

    def test_full_real_inventory_with_synthetic_controls_only_in_temporary_directory(self):
        data = load_inventory(PORTABLE)
        definitions = {row["case_id"]: row for row in data["cases"]}
        synthetic = [
            dict(case_id=cid, family="crt_growth", correct_answer=definitions[cid]["correct_answer"],
                 lure_answer=definitions[cid]["lure_answer"], baseline_margin=0,
                 plus7_margin=-1, plus8_margin=-.5, plus8_minus_plus7=.5)
            for cid in data["case_ids"]
        ]
        write_csv(self.new, synthetic)
        rows, summary = analysis.compare(PORTABLE, self.new, 100)
        self.assertEqual(len(rows), 64)
        self.assertEqual(summary["unique_cases"], 33)
        self.assertEqual(summary["groups"]["correct_to_correct"]["unique_cases"], 20)
        self.assertEqual(summary["groups"]["lure_to_lure"]["evaluation_slots"], 16)
        old_rows = {r["case_id"]: r for r in rows if r["margin_source"] == "existing"}
        self.assertEqual(len(old_rows), 6)
        for old in data["existing_margins"]:
            for key in analysis.previous.MARGINS:
                self.assertEqual(old_rows[old["case_id"]][key], float(old[key]))
        analysis.write_outputs(self.root / "full_join.csv", self.root / "full_summary.json",
                               rows, summary, list(rows[0]))
        saved = json.loads((self.root / "full_summary.json").read_text())
        self.assertEqual(saved["evaluation_slots"], 64)

    def test_shared_case_bootstrap_preserves_group_dependence(self):
        rows = [
            dict(case_id=cid, group=group, plus8_minus_plus7=value)
            for cid, value in (("a", 1), ("b", 3))
            for group in ("correct_to_correct", "lure_to_lure")
        ]
        result = analysis.cluster_bootstrap(rows, 200, 0)
        interval = result["group_mean_delta_differences"]["correct_to_correct minus lure_to_lure"]
        self.assertEqual(interval["lower"], 0)
        self.assertEqual(interval["upper"], 0)
        with self.assertRaises(ValueError):
            analysis.cluster_bootstrap(rows, 0, 0)

    def test_missing_duplicate_conflicting_or_nonfinite_new_measurements_rejected(self):
        for rows, error in (
            ([measurement(1)], "coverage"),
            ([measurement(1), measurement(1), measurement(2)], "Duplicate"),
            ([measurement(1), measurement(2), measurement(7, 1, -1)], "Conflicting"),
            ([measurement(1, float("nan")), measurement(2)], "Nonfinite"),
            ([measurement(1), measurement(2), measurement(99)], "coverage"),
        ):
            with self.subTest(error=error):
                write_csv(self.new, rows)
                with self.assertRaisesRegex(ValueError, error):
                    analysis.compare(self.portable, self.new, 20)
        write_csv(self.new, [measurement(1), measurement(2), measurement(7)])
        _, summary = analysis.compare(self.portable, self.new, 20)
        self.assertTrue(summary["consistency"]["repeated_values_consistent"])

    def test_invalid_inventory_and_hash_rejected(self):
        for change in ("duplicate", "already_measured", "slot_duplicate", "wrong_conditions"):
            data = copy.deepcopy(self.data)
            if change == "duplicate":
                data["case_ids"].append(data["case_ids"][0])
            elif change == "already_measured":
                data["case_ids"].append("crt_fresh_v2_growth_007")
            elif change == "slot_duplicate":
                data["slots"].append(data["slots"][0])
            else:
                data["conditions"]["feature"] = 22552
            self.portable.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_inventory(self.portable)
        self.portable.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_inventory(self.portable, "wrong")

    def test_outputs_refuse_to_overwrite_existing_results(self):
        csv_path, summary_path = self.root / "out.csv", self.root / "out.json"
        analysis.write_outputs(csv_path, summary_path, self.rows, self.summary, analysis.INVENTORY_FIELDS)
        before = csv_path.read_bytes(), summary_path.read_bytes()
        with self.assertRaises(FileExistsError):
            analysis.write_outputs(csv_path, summary_path, self.rows, self.summary, analysis.INVENTORY_FIELDS)
        self.assertEqual(before, (csv_path.read_bytes(), summary_path.read_bytes()))

    def test_actual_portable_inventory_and_dataset_selection(self):
        data = load_inventory(PORTABLE)
        self.assertEqual(len(data["slots"]), 64)
        self.assertEqual(len(data["cases"]), 33)
        self.assertEqual(len(data["existing_margins"]), 6)
        self.assertEqual(len(data["case_ids"]), 27)
        spec = importlib.util.spec_from_file_location("growth_diag", job.SCRIPT)
        diagnostic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diagnostic)
        cases, selected = diagnostic.select_cases("growth", PORTABLE)
        self.assertEqual(list(selected), data["case_ids"])
        self.assertTrue(all(cases[cid].family == "crt_growth" for cid in selected))
        self.assertFalse(set(selected) & {r["case_id"] for r in data["existing_margins"]})
        _, legacy = diagnostic.select_cases()
        self.assertEqual(legacy, diagnostic.RATE_IDS + diagnostic.GROWTH_IDS)
        with self.assertRaises(ValueError):
            diagnostic.select_cases("all", PORTABLE)

    def test_source_archive_includes_inventory_despite_results_exclusion(self):
        portable = self.root / "experiments/inputs/inventory.json"
        portable.parent.mkdir(parents=True)
        portable.write_bytes(PORTABLE.read_bytes())
        (self.root / "results").mkdir()
        (self.root / "results/ignored.csv").write_text("not uploaded")
        archive_path = make_source_archive(self.root, self.root / "results/source.zip")
        with zipfile.ZipFile(archive_path) as archive:
            self.assertIn("experiments/inputs/inventory.json", archive.namelist())
            self.assertNotIn("results/ignored.csv", archive.namelist())
            data = json.loads(archive.read("experiments/inputs/inventory.json"))
            self.assertEqual(len(data["case_ids"]), 27)
            self.assertEqual(len(data["slots"]), 64)

    def test_growth_job_pins_measurements_and_copies_inventory(self):
        def fake(command, **kwargs):
            self.assertEqual(command[5:11], ["--layer", "5", "--feature", "30908", "--only", "growth"])
            self.assertEqual(command[11:], ["--case-ids", str(PORTABLE)])
            Path(command[4]).write_text("test stub")
        with mock.patch.object(job.subprocess, "run", side_effect=fake) as execute:
            run_dir = job.run(CONFIG, self.root / "outputs")
        execute.assert_called_once()
        self.assertEqual((run_dir / "diagnostics/inventory.json").read_bytes(), PORTABLE.read_bytes())
        manifest = json.loads((run_dir / "diagnostics/manifest.json").read_text())
        self.assertEqual(manifest["expected_rate_cases"], 0)
        self.assertEqual(manifest["expected_growth_cases"], 27)
        self.assertEqual(manifest["inventory_sha256"], hashlib.sha256(PORTABLE.read_bytes()).hexdigest())
        self.assertTrue((run_dir / "diagnostics/fresh_v2_growth_controls_f30908_branch_margin.csv").exists())

    def test_missing_inventory_or_bad_hash_prevents_job_execution(self):
        text = CONFIG.read_text().replace(
            "experiments/inputs/fresh_v2_growth_controls_f30908.json", "experiments/inputs/missing.json")
        config = self.root / "missing.toml"
        config.write_text(text)
        with mock.patch.object(job.subprocess, "run") as execute, self.assertRaises(FileNotFoundError):
            job.run(config, self.root / "outputs")
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
