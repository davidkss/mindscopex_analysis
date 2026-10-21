"""Analyze saved binary-choice CRT generations without loading models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

COEFFICIENTS = (0, -1, -2, -4, -7, -8)
FAMILIES = ("crt_difference", "crt_rate", "crt_growth")
TRANSITIONS = ("lure_to_correct", "correct_to_lure", "lure_to_lure", "correct_to_correct")


def index_rows(rows: object, context: str) -> dict:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{context}: expected nonempty row list")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{context}: row must be an object")
        for key in ("case_id", "family", "answer", "label"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"{context}: missing/invalid {key}: {row}")
        case_id = row["case_id"]
        if case_id in indexed:
            raise ValueError(f"{context}: duplicate case_id {case_id}")
        if row["family"] not in FAMILIES or row["label"] not in ("lure", "correct"):
            raise ValueError(f"{context}: unknown family/label: {row}")
        if not case_id.startswith(f"crt_fresh_v2_{row['family'][4:]}_"):
            raise ValueError(f"{context}: case_id/family mismatch: {case_id}")
        indexed[case_id] = {key: row[key] for key in ("case_id", "family", "answer", "label")}
    return indexed


def aggregate(rows: list[dict], coefficient: float, family: str = "all") -> dict:
    counts = Counter(row["transition"] for row in rows)
    n = len(rows)
    lure = sum(row["baseline_label"] == "lure" for row in rows)
    correct = n - lure
    lc, cl = counts["lure_to_correct"], counts["correct_to_lure"]
    result = {"coefficient": coefficient, "family": family, "total": n,
              "baseline_lure": lure, "baseline_correct": correct}
    result.update({name: counts[name] for name in TRANSITIONS})
    if sum(result[name] for name in TRANSITIONS) != n:
        raise ValueError("transition counts do not sum to total")
    result.update(correction_rate=lc / lure if lure else None,
                  regression_rate=cl / correct if correct else None,
                  baseline_accuracy=correct / n,
                  overall_accuracy=(correct + lc - cl) / n,
                  net_improvement=(lc - cl) / n)
    return result


def analyze(data: object) -> tuple[list[dict], list[dict], list[dict]]:
    if not isinstance(data, list) or not data:
        raise ValueError("expected nonempty coefficient object array")
    blocks = {}
    reference = None
    items = []
    for block in data:
        if not isinstance(block, dict):
            raise ValueError("coefficient entry must be an object")
        coefficient = block.get("coefficient")
        if (isinstance(coefficient, bool) or not isinstance(coefficient, (int, float))
                or not math.isfinite(coefficient)):
            raise ValueError(f"invalid coefficient: {coefficient}")
        if coefficient in blocks:
            raise ValueError(f"duplicate coefficient: {coefficient}")
        if block.get("output_mode") != "binary_choice":
            raise ValueError(f"{coefficient}: expected binary_choice")
        baseline = index_rows(block.get("baseline_rows"), f"{coefficient} baseline")
        steered = index_rows(block.get("steered_rows"), f"{coefficient} steered")
        if baseline.keys() != steered.keys():
            raise ValueError(f"{coefficient}: missing/mismatched case IDs; "
                             f"baseline-only={baseline.keys() - steered.keys()}, "
                             f"steered-only={steered.keys() - baseline.keys()}")
        if reference is None:
            reference = baseline
        if baseline != reference:
            raise ValueError(f"{coefficient}: baseline differs across coefficients")
        if coefficient == 0 and baseline != steered:
            raise ValueError("coefficient 0: baseline and steered differ")
        for case_id, before in sorted(baseline.items()):
            after = steered[case_id]
            if before["family"] != after["family"]:
                raise ValueError(f"{coefficient}/{case_id}: family mismatch")
            if (before["label"] == after["label"]) != (before["answer"] == after["answer"]):
                raise ValueError(f"{coefficient}/{case_id}: answer/label inconsistency")
            items.append({"coefficient": coefficient, "case_id": case_id,
                          "family": before["family"], "baseline_answer": before["answer"],
                          "steered_answer": after["answer"], "baseline_label": before["label"],
                          "steered_label": after["label"],
                          "transition": f"{before['label']}_to_{after['label']}"})
        blocks[coefficient] = True
    if set(blocks) != set(COEFFICIENTS):
        raise ValueError(f"expected coefficients {COEFFICIENTS}, got {list(blocks)}")
    if {row["family"] for row in items} != set(FAMILIES):
        raise ValueError("missing expected CRT family")
    totals, families = [], []
    for coefficient in COEFFICIENTS:
        selected = [row for row in items if row["coefficient"] == coefficient]
        totals.append(aggregate(selected, coefficient))
        for family in FAMILIES:
            families.append(aggregate([r for r in selected if r["family"] == family],
                                      coefficient, family))
    return totals, families, items


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(source: Path, output: Path) -> list[dict]:
    raw = source.read_bytes()
    totals, families, items = analyze(json.loads(raw))
    # Reserve a fresh directory; never overwrite earlier analyses or experiment artifacts.
    if "runs" in output.resolve().parts:
        raise ValueError("output must be outside results/runs")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "coefficients.csv", totals)
    write_csv(output / "families.csv", families)
    write_csv(output / "transitions.csv", items)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, title in (("correction_rate", "Correction rate"),
                          ("overall_accuracy", "Overall accuracy")):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(COEFFICIENTS, [r[metric] for r in totals], "o-", label="Steered")
        if metric == "overall_accuracy":
            ax.axhline(totals[0]["baseline_accuracy"], linestyle="--", color="gray",
                       label="Baseline")
            ax.legend()
        ax.set(xlabel="Coefficient", ylabel=title, title=f"CRT negative dose: {title}",
               ylim=(-0.02, 1.02), xticks=sorted(COEFFICIENTS))
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=160)
        plt.close(fig)
    lines = ["# CRT Negative Dose Behavioral Analysis", "",
             f"Input: `{source.resolve()}`", "",
             f"SHA-256: `{hashlib.sha256(raw).hexdigest()}`", "",
             "Scope: Qwen/Qwen3.5-2B-Base, SAE layer 5, feature 30908, crt_fresh_v2, "
             "seed 1, binary_choice (experiment configuration; JSON contains no model metadata).",
             "", "## Actual input schema", "",
             "Top-level array of six objects: coefficient (number), output_mode (string), "
             "baseline_rows and steered_rows (arrays). Each row contains case_id, family, "
             "answer and label (strings). Families: crt_difference, crt_rate, crt_growth; "
             "labels: lure, correct. Rows are joined by case_id within each coefficient.", "",
             "## Metrics and validation", "",
             "Correction Rate = Lure→Correct / baseline Lure. Regression Rate = "
             "Correct→Lure / baseline Correct. Overall Accuracy = steered Correct / N. "
             "Net Improvement = (Lure→Correct − Correct→Lure) / N, an accuracy difference. "
             "CSV rates are fractions; an undefined conditional rate is an empty cell.", "",
             "Validated: exact coefficient set; unique IDs; identical baseline rows across "
             "coefficients; identical ID sets and families; binary labels and consistent "
             "answers; coefficient 0 exact baseline/steered equality; transition sums equal N. "
             "Missing or inconsistent records raise errors before output creation.", "",
             "## Results", "",
             "| Coefficient | N | L→C | C→L | L→L | C→C | Correction | Regression | Accuracy "
             "| Net (pp) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for row in totals:
        rates = [f"{row[k]:.2%}" if row[k] is not None else "N/A"
                 for k in ("correction_rate", "regression_rate", "overall_accuracy")]
        values = [row[k] for k in ("coefficient", "total", *TRANSITIONS)]
        lines.append("| " + " | ".join(map(str, values + rates +
                     [f"{100 * row['net_improvement']:+.2f}"])) + " |")
    lines += ["", "## Results by problem type", "",
              "| Coefficient | Family | N | L→C | C→L | L→L | C→C | Accuracy |",
              "|---|---|---|---|---|---|---|---|"]
    for row in families:
        values = [row[k] for k in ("coefficient", "family", "total", *TRANSITIONS)]
        lines.append("| " + " | ".join(map(str, values)) +
                     f" | {row['overall_accuracy']:.2%} |")
    best = max(totals, key=lambda row: row["overall_accuracy"])
    lines += ["", f"Highest observed accuracy: coefficient {best['coefficient']}, "
              f"{best['overall_accuracy']:.2%}; correction {best['lure_to_correct']} cases, "
              f"regression {best['correct_to_lure']} cases, net "
              f"{100 * best['net_improvement']:+.2f} percentage points."]
    lines += ["", f"Baseline: {totals[0]['baseline_correct']}/{totals[0]['total']} correct "
              f"({totals[0]['baseline_accuracy']:.2%}).", "",
              "These are descriptive results for one seed and held-out subset; they do not "
              "establish an effect across seeds or unrestricted generation.", "",
              "Files: coefficients.csv (6 rows), families.csv (18 rows), transitions.csv "
              f"({len(items)} rows), correction_rate.png, overall_accuracy.png.", "",
              "## Reproduce", "", "```bash",
              f".venv/bin/python scripts/analyze_crt_negative_dose.py --input {source} "
              "--output results/analysis/crt_negative_dose_seed1_new", "```", "",
              "The output directory must not exist. No models or GPU libraries are loaded."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.input, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")


if __name__ == "__main__":
    main()
