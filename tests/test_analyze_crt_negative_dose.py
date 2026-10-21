from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "negative_dose", Path(__file__).resolve().parents[1]
    / "scripts" / "analyze_crt_negative_dose.py",
)
analyze = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyze)


def fixture():
    rows = []
    for family in analyze.FAMILIES:
        for i, label in enumerate(("lure", "correct", "lure", "correct")):
            rows.append({"case_id": f"crt_fresh_v2_{family[4:]}_{i:03}",
                         "family": family, "answer": label, "label": label})
    blocks = []
    for coefficient in analyze.COEFFICIENTS:
        steered = copy.deepcopy(rows)
        if coefficient:
            for row in steered:
                if row["case_id"].endswith(("000", "001")):
                    row["label"] = "correct" if row["label"] == "lure" else "lure"
                    row["answer"] = row["label"]
        blocks.append({"coefficient": coefficient, "output_mode": "binary_choice",
                       "baseline_rows": copy.deepcopy(rows), "steered_rows": steered[::-1]})
    return blocks


class NegativeDoseTests(unittest.TestCase):
    def test_matching_metrics_and_independent_coefficients(self):
        data = fixture()
        data[2]["steered_rows"] = copy.deepcopy(data[2]["baseline_rows"])
        totals, families, items = analyze.analyze(data[::-1])
        self.assertEqual(len(items), 72)
        self.assertEqual(len(families), 18)
        row = totals[1]
        self.assertEqual([row[t] for t in analyze.TRANSITIONS], [3, 3, 3, 3])
        self.assertEqual(row["correction_rate"], 0.5)
        self.assertEqual(row["regression_rate"], 0.5)
        self.assertEqual(row["overall_accuracy"], 0.5)
        self.assertEqual(row["net_improvement"], 0)
        self.assertEqual(totals[2]["lure_to_correct"], 0)

    def test_invalid_inputs(self):
        for kind in ("duplicate_id", "missing_id", "baseline_drift", "zero_drift",
                     "duplicate_coefficient", "missing_coefficient", "invalid_label",
                     "family_mismatch", "answer_mismatch", "mode", "nan", "empty"):
            with self.subTest(kind=kind):
                data = fixture()
                block = data[1]
                if kind == "duplicate_id":
                    block["steered_rows"].append(copy.deepcopy(block["steered_rows"][0]))
                elif kind == "missing_id":
                    block["steered_rows"].pop()
                elif kind == "baseline_drift":
                    block["baseline_rows"][0]["answer"] = "changed"
                elif kind == "zero_drift":
                    data[0]["steered_rows"][0]["answer"] = "changed"
                elif kind == "duplicate_coefficient":
                    data.append(copy.deepcopy(block))
                elif kind == "missing_coefficient":
                    data.pop()
                elif kind == "invalid_label":
                    block["steered_rows"][0]["label"] = "other"
                elif kind == "family_mismatch":
                    block["steered_rows"][0]["family"] = "crt_rate"
                elif kind == "answer_mismatch":
                    block["steered_rows"][0]["answer"] = "changed"
                elif kind == "mode":
                    block["output_mode"] = "free"
                elif kind == "nan":
                    block["coefficient"] = float("nan")
                elif kind == "empty":
                    data = []
                with self.assertRaises(ValueError):
                    analyze.analyze(data)

    def test_undefined_rates(self):
        rows = [{"baseline_label": "correct", "transition": "correct_to_correct"}]
        result = analyze.aggregate(rows, 0)
        self.assertIsNone(result["correction_rate"])
        self.assertEqual(result["overall_accuracy"], 1)

    def test_outputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "generations.json"
            source.write_text(json.dumps(fixture()))
            original = source.read_bytes()
            output = Path(tmp) / "analysis"
            analyze.run(source, output)
            self.assertEqual(len(list(output.iterdir())), 6)
            self.assertEqual(source.read_bytes(), original)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                analyze.run(source, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
