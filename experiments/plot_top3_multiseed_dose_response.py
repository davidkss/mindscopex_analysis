"""Plot validated Top-3 dose summaries; error bars are SD across split seeds."""
from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FEATURES = (30908, 22552, 14817)
DOSES = (-8, -4, -2, 0, 2, 4, 7, 8)
PREFIX = 'top3_multiseed_dose_response_'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_summary(path):
    rows = {}
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        required = {'feature_id', 'dose', 'n_seeds', 'n_seed_case_observations',
                    'mean_delta_pp', 'sd_delta_pp'}
        require(required <= set(reader.fieldnames or ()), 'Missing required CSV columns')
        for line, row in enumerate(reader, start=2):
            feature = int(row['feature_id'])
            if feature not in FEATURES:
                continue
            dose = float(row['dose'])
            key = (feature, dose)
            require(math.isfinite(dose) and dose in DOSES, f'Line {line}: unexpected dose')
            require(key not in rows, f'Line {line}: duplicate feature-dose combination')
            require(int(row['n_seeds']) == 5, f'Line {line}: n_seeds must be 5')
            require(int(row['n_seed_case_observations']) == 200,
                    f'Line {line}: n_seed_case_observations must be 200')
            mean, sd = float(row['mean_delta_pp']), float(row['sd_delta_pp'])
            require(math.isfinite(mean) and math.isfinite(sd) and sd >= 0,
                    f'Line {line}: mean/SD must be finite and SD nonnegative')
            rows[key] = (mean, sd)
    expected = {(feature, dose) for feature in FEATURES for dose in DOSES}
    require(len(rows) == 24 and set(rows) == expected,
            f'Expected 24 feature-dose rows; missing={sorted(expected-set(rows))}, '
            f'unexpected={sorted(set(rows)-expected)}')
    return rows


def plot(rows, paths, overwrite):
    # Explicit styles and observed-point segments only; no fitted/smoothed curve.
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        try:
            ax.axhline(0, color='0.4', linewidth=0.9, linestyle='--', zorder=1)
            for feature, color, marker in zip(
                    FEATURES, ('#0072B2', '#D55E00', '#009E73'), ('o', 's', '^')):
                ax.errorbar(DOSES, [rows[feature, d][0] for d in DOSES],
                            yerr=[rows[feature, d][1] for d in DOSES],
                            label=f'Feature {feature}', color=color, marker=marker,
                            linestyle='-', linewidth=1.5, markersize=5,
                            elinewidth=1, capsize=3, zorder=2)
            ax.set_xlabel('Steering coefficient')
            ax.set_ylabel('Accuracy change from baseline (pp)')
            ax.set_xticks(DOSES)
            ax.set_xlim(-8.7, 8.7)
            ax.grid(axis='y', color='0.9', linewidth=0.6)
            ax.set_axisbelow(True)
            ax.legend(frameon=False)
            fig.text(0.5, 0.025, 'Error bars: SD across five split seeds.',
                     ha='center', fontsize=9)
            fig.tight_layout(rect=(0, 0.055, 1, 1))
            paths[0].parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=paths[0].parent, prefix='.top3-plot-') as temp:
                staged = [Path(temp)/p.name for p in paths]
                fig.savefig(staged[0], format='png', dpi=300,
                            metadata={'Software': 'matplotlib'})
                fig.savefig(staged[1], format='pdf',
                            metadata={'CreationDate': None, 'ModDate': None})
                published = []
                try:
                    for source, target in zip(staged, paths):
                        if overwrite:
                            os.replace(source, target)
                        else:
                            # Hard links fail atomically if a target already exists.
                            os.link(source, target)
                        published.append(target)
                except BaseException:
                    if not overwrite:
                        for target in published:
                            target.unlink()
                    raise
        finally:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=ROOT/'results/analysis'/f'{PREFIX}dose_summary.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/analysis')
    parser.add_argument('--overwrite', action='store_true',
                        help='Explicitly allow replacement of existing figure files')
    args = parser.parse_args()
    paths = [args.output_dir/f'{PREFIX}overall.{ext}' for ext in ('png', 'pdf')]
    if not args.overwrite:
        require(not any(p.exists() or p.is_symlink() for p in paths),
                'Figure file exists; use --overwrite to replace it')
    rows = read_summary(args.input)
    plot(rows, paths, args.overwrite)
    print('validation_status: passed')
    print(f'rows: {len(rows)}')
    print(f'features: {list(FEATURES)}')
    print(f'doses: {list(DOSES)}')
    print(f'png: {paths[0]}')
    print(f'pdf: {paths[1]}')


if __name__ == '__main__':
    main()
