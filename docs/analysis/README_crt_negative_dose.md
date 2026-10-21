# CRT Negative Dose Behavioral Analysis

Input: `/home/davidkss/ai/mindscopex_analysis/results/runs/20261021-162417_study_transfer_fresh_v2_negative_dose_f30908_seed1_2b/artifacts/behavioral/generations.json`

SHA-256: `d877969dd9a10ef1730d12d18a2be014502c00f6026a1bd90103cfcb0ada1dc9`

Scope: Qwen/Qwen3.5-2B-Base, SAE layer 5, feature 30908, crt_fresh_v2, seed 1, binary_choice (experiment configuration; JSON contains no model metadata).

## Actual input schema

Top-level array of six objects: coefficient (number), output_mode (string), baseline_rows and steered_rows (arrays). Each row contains case_id, family, answer and label (strings). Families: crt_difference, crt_rate, crt_growth; labels: lure, correct. Rows are joined by case_id within each coefficient.

## Metrics and validation

Correction Rate = Lure→Correct / baseline Lure. Regression Rate = Correct→Lure / baseline Correct. Overall Accuracy = steered Correct / N. Net Improvement = (Lure→Correct − Correct→Lure) / N, an accuracy difference. CSV rates are fractions; an undefined conditional rate is an empty cell.

Validated: exact coefficient set; unique IDs; identical baseline rows across coefficients; identical ID sets and families; binary labels and consistent answers; coefficient 0 exact baseline/steered equality; transition sums equal N. Missing or inconsistent records raise errors before output creation.

## Results

| Coefficient | N | L→C | C→L | L→L | C→C | Correction | Regression | Accuracy | Net (pp) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 40 | 0 | 0 | 20 | 20 | 0.00% | 0.00% | 50.00% | +0.00 |
| -1 | 40 | 0 | 2 | 20 | 18 | 0.00% | 10.00% | 45.00% | -5.00 |
| -2 | 40 | 0 | 9 | 20 | 11 | 0.00% | 45.00% | 27.50% | -22.50 |
| -4 | 40 | 1 | 18 | 19 | 2 | 5.00% | 90.00% | 7.50% | -42.50 |
| -7 | 40 | 9 | 6 | 11 | 14 | 45.00% | 30.00% | 57.50% | +7.50 |
| -8 | 40 | 10 | 6 | 10 | 14 | 50.00% | 30.00% | 60.00% | +10.00 |

## Results by problem type

| Coefficient | Family | N | L→C | C→L | L→L | C→C | Accuracy |
|---|---|---|---|---|---|---|---|
| 0 | crt_difference | 14 | 0 | 0 | 9 | 5 | 35.71% |
| 0 | crt_rate | 13 | 0 | 0 | 3 | 10 | 76.92% |
| 0 | crt_growth | 13 | 0 | 0 | 8 | 5 | 38.46% |
| -1 | crt_difference | 14 | 0 | 1 | 9 | 4 | 28.57% |
| -1 | crt_rate | 13 | 0 | 1 | 3 | 9 | 69.23% |
| -1 | crt_growth | 13 | 0 | 0 | 8 | 5 | 38.46% |
| -2 | crt_difference | 14 | 0 | 3 | 9 | 2 | 14.29% |
| -2 | crt_rate | 13 | 0 | 4 | 3 | 6 | 46.15% |
| -2 | crt_growth | 13 | 0 | 2 | 8 | 3 | 23.08% |
| -4 | crt_difference | 14 | 0 | 4 | 9 | 1 | 7.14% |
| -4 | crt_rate | 13 | 1 | 10 | 2 | 0 | 7.69% |
| -4 | crt_growth | 13 | 0 | 4 | 8 | 1 | 7.69% |
| -7 | crt_difference | 14 | 7 | 0 | 2 | 5 | 85.71% |
| -7 | crt_rate | 13 | 1 | 2 | 2 | 8 | 69.23% |
| -7 | crt_growth | 13 | 1 | 4 | 7 | 1 | 15.38% |
| -8 | crt_difference | 14 | 7 | 0 | 2 | 5 | 85.71% |
| -8 | crt_rate | 13 | 1 | 2 | 2 | 8 | 69.23% |
| -8 | crt_growth | 13 | 2 | 4 | 6 | 1 | 23.08% |

Highest observed accuracy: coefficient -8, 60.00%; correction 10 cases, regression 6 cases, net +10.00 percentage points.

Baseline: 20/40 correct (50.00%).

These are descriptive results for one seed and held-out subset; they do not establish an effect across seeds or unrestricted generation.

Files: coefficients.csv (6 rows), families.csv (18 rows), transitions.csv (240 rows), correction_rate.png, overall_accuracy.png.

## Reproduce

```bash
.venv/bin/python scripts/analyze_crt_negative_dose.py --input results/runs/20261021-162417_study_transfer_fresh_v2_negative_dose_f30908_seed1_2b/artifacts/behavioral/generations.json --output results/analysis/crt_negative_dose_seed1_new
```

The output directory must not exist. No models or GPU libraries are loaded.
