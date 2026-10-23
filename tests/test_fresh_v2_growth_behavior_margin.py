"""CPU-only input validation, left-join coverage and slot/case weighting tests."""
import copy
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_fresh_v2_growth_behavior_margin.py"
SPEC = importlib.util.spec_from_file_location("growth_margin", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def behavior(seed=1, cid="growth1", a="correct", b="lure"):
    return dict(seed=seed, case_id=cid, family="crt_growth", steered_7=a, steered_8=b,
                correct_answer=" 31 days", lure_answer=" 16 days")


def margin(cid="growth1", a=-0.25, b=0.125):
    return dict(case_id=cid, family="crt_growth", correct_answer="31 days",
                lure_answer="16 days", baseline_margin=-0.5, plus7_margin=a, plus8_margin=b,
                plus8_minus_plus7=b-a)


class GrowthMarginTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bp, self.mp = self.root / "behavior.csv", self.root / "margin.csv"

    def write(self, behavior_rows=None, margin_rows=None):
        for path, rows in ((self.bp, behavior_rows if behavior_rows is not None else [behavior()]),
                           (self.mp, margin_rows if margin_rows is not None else [margin()])):
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def test_left_join_slots_unique_cases_missing_and_stable_labels(self):
        self.write([
            behavior(), behavior(2), behavior(3, a="lure", b="lure"),
            behavior(1, "unmeasured", "correct", "correct"),
        ])
        rows, summary = analysis.analyze(self.bp, self.mp)
        self.assertEqual(len(rows), 4)
        self.assertEqual(summary["growth"]["unique_cases"], 2)
        self.assertEqual(summary["growth"]["matched_margin_slots"], 3)
        self.assertEqual(summary["growth"]["missing_margin_slots"], 1)
        self.assertEqual(summary["correct_to_lure"]["evaluation_slots"], 2)
        self.assertEqual(summary["correct_to_lure"]["unique_cases"], 1)
        self.assertEqual(summary["correct_to_lure"]["slot_weighted_margins"]["increase"], 2)
        self.assertEqual(summary["correct_to_lure"]["unique_case_margins"]["increase"], 1)
        self.assertEqual(summary["growth"]["direction_agreement_rate"], 1.0)
        unmeasured = next(r for r in rows if not r["margin_available"])
        self.assertIsNone(unmeasured["plus8_minus_plus7"])
        stable = next(r for r in rows if r["behavior_transition"] == "lure->lure")
        self.assertIsNone(stable["behavior_margin_direction_agrees"])
        self.assertTrue(summary["mismatches"])
        self.assertEqual(summary["by_seed"]["2"]["growth"]["evaluation_slots"], 1)

    def test_sign_changes_zero_and_direction_agreement(self):
        for a, b, before, after, changed, flip, agrees in (
            (-1, 1, "correct", "lure", True, True, True),
            (-1, 0, "correct", "lure", True, False, True),
            (0, 1, "correct", "lure", True, False, True),
            (1, 2, "correct", "lure", False, False, True),
            (1, -1, "correct", "lure", True, True, False),
            (0, 0, "correct", "lure", False, False, False),
            (1, -1, "lure", "correct", True, True, True),
        ):
            with self.subTest(a=a, b=b, before=before):
                self.write([behavior(a=before, b=after)], [margin(a=a, b=b)])
                rows, _ = analysis.analyze(self.bp, self.mp)
                row = rows[0]
                self.assertEqual(row["margin_sign_changed"], changed)
                self.assertEqual(row["strict_margin_sign_flip"], flip)
                self.assertEqual(row["behavior_margin_direction_agrees"], agrees)

    def test_duplicates_missing_target_and_inconsistent_definitions(self):
        for kind in ("duplicate_slot", "duplicate_margin", "missing_target", "definition",
                     "bad_label", "bad_seed", "missing_column", "nonfinite", "bad_delta"):
            with self.subTest(kind=kind):
                bs, ms = [behavior()], [margin()]
                if kind == "duplicate_slot":
                    bs.append(copy.deepcopy(bs[0]))
                elif kind == "duplicate_margin":
                    ms.append(copy.deepcopy(ms[0]))
                elif kind == "missing_target":
                    ms[0]["case_id"] = "other"
                elif kind == "definition":
                    ms[0]["correct_answer"] = "999 days"
                elif kind == "bad_label":
                    bs[0]["steered_8"] = "other"
                elif kind == "bad_seed":
                    bs[0]["seed"] = "oops"
                elif kind == "missing_column":
                    del bs[0]["steered_8"]
                elif kind == "nonfinite":
                    ms[0]["plus7_margin"] = "nan"
                elif kind == "bad_delta":
                    ms[0]["plus8_minus_plus7"] = 100
                self.write(bs, ms)
                with self.assertRaises(ValueError):
                    analysis.analyze(self.bp, self.mp)

    def test_missing_input_and_empty_csv(self):
        with self.assertRaisesRegex(FileNotFoundError, "Input CSV not found"):
            analysis.analyze(self.bp, self.mp)
        self.write()
        self.bp.write_text("seed,case_id\n")
        with self.assertRaises(ValueError):
            analysis.analyze(self.bp, self.mp)

    def test_writes_only_new_outputs_and_preserves_inputs(self):
        self.write()
        before = (self.bp.read_bytes(), self.mp.read_bytes())
        output = self.root / "output"
        outputs, _ = analysis.run(self.bp, self.mp, output)
        self.assertEqual(json.loads(outputs[1].read_text())["correct_to_lure"]["evaluation_slots"], 1)
        saved = [p.read_bytes() for p in outputs]
        with self.assertRaises(FileExistsError):
            analysis.run(self.bp, self.mp, output)
        self.assertEqual(saved, [p.read_bytes() for p in outputs])
        self.assertEqual(before, (self.bp.read_bytes(), self.mp.read_bytes()))

    def test_actual_local_inputs_match_reported_counts_if_available(self):
        bp = analysis.ROOT / analysis.BEHAVIOR
        mp = next((analysis.ROOT / path for path in (analysis.MARGIN, analysis.FALLBACK)
                   if (analysis.ROOT / path).is_file()), None)
        if not bp.is_file() or mp is None:
            self.skipTest("Local research result CSVs are not included in every checkout")
        rows, summary = analysis.analyze(bp, mp)
        self.assertEqual(len(rows), 64)
        self.assertEqual(summary["mismatches"], [])
        target = summary["correct_to_lure"]
        self.assertEqual(target["evaluation_slots"], 14)
        self.assertEqual(target["unique_cases"], 6)
        self.assertEqual(target["unique_case_margins"]["increase"], 6)
        self.assertEqual(target["slot_weighted_margins"]["increase"], 14)
        self.assertEqual(target["unique_case_margins"]["strict_sign_flip"], 1)
        self.assertEqual(summary["growth"]["missing_margin_slots"], 50)

    def test_cli_reports_missing_input_without_writing_outputs(self):
        self.write()
        output = self.root / "output"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--behavior", str(self.root / "absent.csv"),
             "--margin", str(self.mp), "--output-dir", str(output)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("ERROR: Input CSV not found", result.stderr)
        self.assertFalse(output.exists())

    def test_existing_summary_blocks_csv_creation_and_invalid_data_writes_nothing(self):
        self.write()
        output = self.root / "output"
        output.mkdir()
        summary = output / f"{analysis.STEM}_summary.json"
        summary.write_text("preserve")
        with self.assertRaises(FileExistsError):
            analysis.run(self.bp, self.mp, output)
        self.assertFalse((output / f"{analysis.STEM}_join.csv").exists())
        summary.unlink()
        self.write([behavior()], [margin("other")])
        with self.assertRaises(ValueError):
            analysis.run(self.bp, self.mp, output)
        self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
