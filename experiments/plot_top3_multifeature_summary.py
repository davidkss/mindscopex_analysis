"""Validated two-panel descriptive summary; stdlib and matplotlib only."""
from __future__ import annotations

import csv
import math
import os
import statistics
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT/'results/analysis'
INPUTS = (
    ANALYSIS/'top3_multifeature_condition_summary.csv',
    ANALYSIS/'triple_strength_followup_condition_summary.csv',
    ANALYSIS/'triple_strength_followup_family_summary.csv',
)
OUTPUTS = tuple(ANALYSIS/f'top3_multifeature_summary_figure.{ext}' for ext in ('png', 'pdf'))
CONDITIONS = (
    'f30908', 'f22552', 'f14817', 'f30908_f22552',
    'f30908_f14817', 'f22552_f14817', 'f30908_f22552_f14817',
)
LABELS = ('Single 30908', 'Single 22552', 'Single 14817', '30908 + 22552',
          '30908 + 14817', '22552 + 14817', 'Triple')
COEFFICIENTS = (2, 4, 7)
FAMILIES = ('crt_difference', 'crt_rate', 'crt_growth')
SEEDS = set(range(5))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(row, key):
    value = float(row[key])
    require(math.isfinite(value), f'Nonfinite {key}: {row}')
    return value


def close(actual, expected, context):
    require(math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12),
            f'{context}: recorded={actual}, recomputed={expected}')


def validate_table(path, key_fields, expected_keys, overall=False):
    """Require one aggregate and five unique seeds for every expected group."""
    seeds, aggregates = {}, {}
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        required = set(key_fields) | {
            'seed', 'aggregation', 'n_cases', 'baseline_accuracy', 'steered_accuracy',
            'delta_pp', 'fixes', 'regressions', 'n_seeds', 'n_seed_case_observations',
            'mean_baseline_accuracy', 'mean_steered_accuracy', 'mean_delta_pp',
            'sd_delta_pp', 'sd_ddof', 'min_delta_pp', 'max_delta_pp',
            'total_fixes', 'total_regressions',
        }
        require(required <= set(reader.fieldnames or ()), f'{path}: missing columns')
        for line, row in enumerate(reader, 2):
            context = f'{path.name}:{line}'
            key = tuple(number(row, field) if field == 'coefficient' else row[field]
                        for field in key_fields)
            require(key in expected_keys, f'{context}: unexpected group {key}')
            if row['aggregation'] == 'per_seed':
                seed = int(row['seed'])
                require(seed in SEEDS and (key, seed) not in seeds,
                        f'{context}: unexpected/duplicate seed {seed}')
                n = number(row, 'n_cases')
                require(n > 0 and n.is_integer() and (not overall or n == 40),
                        f'{context}: invalid case count')
                for field in ('baseline_accuracy', 'steered_accuracy'):
                    require(0 <= number(row, field) <= 1, f'{context}: invalid {field}')
                close(number(row, 'delta_pp'), 100*(number(row, 'steered_accuracy')-
                      number(row, 'baseline_accuracy')), f'{context}: delta_pp')
                seeds[key, seed] = row
            elif row['aggregation'] == 'equal_seed_weight':
                require(row['seed'] == 'all' and key not in aggregates,
                        f'{context}: invalid/duplicate aggregate')
                require(number(row, 'n_seeds') == 5 and number(row, 'sd_ddof') == 1,
                        f'{context}: expected five seeds and sample SD')
                aggregates[key] = row
            else:
                raise ValueError(f'{context}: unexpected aggregation {row["aggregation"]}')
    expected_seeds = {(key, seed) for key in expected_keys for seed in SEEDS}
    require(set(seeds) == expected_seeds, f'{path}: missing seed rows {expected_seeds-set(seeds)}')
    require(set(aggregates) == expected_keys, f'{path}: missing aggregate groups')
    result = {}
    for key in sorted(expected_keys):
        rows = [seeds[key, seed] for seed in sorted(SEEDS)]
        aggregate = aggregates[key]
        delta = [number(r, 'delta_pp') for r in rows]
        recomputed = {
            'n_seed_case_observations': sum(number(r, 'n_cases') for r in rows),
            'mean_baseline_accuracy': statistics.mean(number(r, 'baseline_accuracy') for r in rows),
            'mean_steered_accuracy': statistics.mean(number(r, 'steered_accuracy') for r in rows),
            'mean_delta_pp': statistics.mean(delta), 'sd_delta_pp': statistics.stdev(delta),
            'min_delta_pp': min(delta), 'max_delta_pp': max(delta),
            'total_fixes': sum(number(r, 'fixes') for r in rows),
            'total_regressions': sum(number(r, 'regressions') for r in rows),
        }
        for field, value in recomputed.items():
            close(number(aggregate, field), value, f'{path.name}/{key}/{field}')
        result[key] = (number(aggregate, 'mean_delta_pp'), number(aggregate, 'sd_delta_pp'))
    return result


def no_overwrite():
    require(not any(p.exists() or p.is_symlink() for p in OUTPUTS),
            'Figure output already exists; refusing to overwrite PNG or PDF')


def plot(panel_a, overall, families):
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42}):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5),
                                       gridspec_kw={'width_ratios': [1.45, 1]})
        try:
            positions = (0, 1, 2, 3.7, 4.7, 5.7, 7.4)
            colors = ('#0072B2',)*3 + ('#009E73',)*3 + ('#D55E00',)
            for x, condition, color in zip(positions, CONDITIONS, colors):
                mean, sd = panel_a[condition,]
                ax_a.errorbar(x, mean, yerr=sd, fmt='o', color=color,
                              markersize=6, capsize=4, elinewidth=1.2)
            ax_a.set_xticks(positions, LABELS, rotation=40, ha='right')
            ax_a.set_xlim(-0.6, 8)
            ax_a.set_title('A  Single vs Pair vs Triple at +7', loc='left', pad=30)
            for x, label in ((1, 'Singles'), (4.7, 'Pairs'), (7.4, 'Triple')):
                ax_a.text(x, 1.025, label, transform=ax_a.get_xaxis_transform(),
                          ha='center', va='bottom', fontsize=10)
            for x in (2.85, 6.55):
                ax_a.axvline(x, color='0.85', linewidth=0.8)

            series = [('Overall', overall, None, '#222222', 'o'),
                      ('Difference', families, 'crt_difference', '#0072B2', 's'),
                      ('Rate', families, 'crt_rate', '#D55E00', '^'),
                      ('Growth', families, 'crt_growth', '#009E73', 'D')]
            for label, table, family, color, marker in series:
                values = [table[(c,) if family is None else (c, family)] for c in COEFFICIENTS]
                # Straight segments connect observed coefficients only; no fitting/resampling.
                ax_b.errorbar(COEFFICIENTS, [v[0] for v in values], yerr=[v[1] for v in values],
                              label=label, color=color, marker=marker, linestyle='-',
                              linewidth=1.5, markersize=5, capsize=3, elinewidth=1)
            ax_b.set_title('B  Triple intervention strength', loc='left', pad=30)
            ax_b.set_xlabel('Coefficient per feature')
            ax_b.set_xticks(COEFFICIENTS, ('+2', '+4', '+7'))
            ax_b.set_xlim(1.6, 7.4)
            ax_b.legend(frameon=False, fontsize=9)
            for ax in (ax_a, ax_b):
                ax.set_ylabel('Accuracy change (pp)')
                ax.axhline(0, color='0.45', linestyle='--', linewidth=0.9, zorder=0)
                ax.set_axisbelow(True)
            fig.text(0.5, 0.025, 'Means: equal weight across five split seeds. Error bars: sample SD.',
                     ha='center', fontsize=9)
            fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=3)
            save(fig)
        finally:
            plt.close(fig)


def save(fig):
    no_overwrite()
    with tempfile.TemporaryDirectory(dir=ANALYSIS, prefix='.multifeature-figure-') as temp:
        staged = [Path(temp)/p.name for p in OUTPUTS]
        fig.savefig(staged[0], format='png', dpi=300, metadata={'Software': 'matplotlib'})
        fig.savefig(staged[1], format='pdf', metadata={'CreationDate': None, 'ModDate': None})
        published = []
        try:
            for source, target in zip(staged, OUTPUTS):
                os.link(source, target)  # Atomic refusal if a concurrent writer created the target.
                published.append(target)
        except BaseException:
            for path in published:
                path.unlink()
            raise


def main():
    no_overwrite()
    panel_a = validate_table(INPUTS[0], ('condition',), {(c,) for c in CONDITIONS}, overall=True)
    overall = validate_table(INPUTS[1], ('coefficient',), {(c,) for c in COEFFICIENTS}, overall=True)
    families = validate_table(INPUTS[2], ('coefficient', 'family'),
                              {(c, f) for c in COEFFICIENTS for f in FAMILIES})
    plot(panel_a, overall, families)
    print('validation_status: passed')
    print('png:', OUTPUTS[0])
    print('pdf:', OUTPUTS[1])


if __name__ == '__main__':
    main()
