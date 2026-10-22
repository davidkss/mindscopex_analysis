"""Read-only, descriptive multi-seed analysis of saved CRT negative-dose runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_crt_negative_dose as single

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'results/analysis/crt_negative_dose_seed1_final'


def discover(root, seeds):
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 1 for s in seeds):
        raise ValueError('seeds must be distinct positive integers')
    paths = {}
    for seed in seeds:
        name = f'study_transfer_fresh_v2_negative_dose_f30908_seed{seed}_2b'
        matches = sorted(p for p in root.rglob('generations.json')
                         if p.parts[-3:-1] == ('artifacts', 'behavioral')
                         and (p.parents[2].name == name or p.parents[2].name.endswith('_' + name)))
        if len(matches) != 1:
            raise ValueError(f'Seed {seed}: expected exactly one input, found {len(matches)}: {matches}')
        paths[seed] = matches[0]
    return paths


def enrich(row):
    r = dict(row)
    r.update(steered_correct=r['baseline_correct'] + r['lure_to_correct'] - r['correct_to_lure'],
             steered_lure=r['baseline_lure'] - r['lure_to_correct'] + r['correct_to_lure'],
             baseline_accuracy_pct=100 * r['baseline_accuracy'],
             steered_accuracy=r['overall_accuracy'],
             steered_accuracy_pct=100 * r['overall_accuracy'],
             net_improvement_pp=100 * r['net_improvement'],
             net_correct_change=r['lure_to_correct'] - r['correct_to_lure'])
    return r


def check_seed1(totals, families, items, reference):
    expected = (0.5, 0.45, 0.275, 0.075, 0.575, 0.6)
    if any(r['baseline_accuracy'] != .5 or r['overall_accuracy'] != accuracy
           for r, accuracy in zip(totals, expected, strict=True)):
        raise ValueError('Seed 1: published accuracy mismatch')
    for filename, actual, keys in (
        ('coefficients.csv', totals, ('coefficient',)),
        ('families.csv', families, ('coefficient', 'family')),
        ('transitions.csv', items, ('coefficient', 'case_id')),
    ):
        with (reference / filename).open(newline='') as stream:
            saved = list(csv.DictReader(stream))
        def key(row):
            return tuple(float(row[k]) if k == 'coefficient' else row[k] for k in keys)
        indexed = {key(r): r for r in actual}
        if len(saved) != len(actual) or {key(r) for r in saved} != set(indexed):
            raise ValueError(f'Seed 1: {filename} key/count mismatch')
        for row in saved:
            current = indexed[key(row)]
            for field, value in row.items():
                v = current[field]
                equal = (value == '' if v is None else
                         abs(float(value) - v) < 1e-12 if isinstance(v, (int, float))
                         else value == v)
                if not equal:
                    raise ValueError(f'Seed 1: {filename} mismatch at {key(row)}, {field}')


def summarize(paths, reference=REFERENCE):
    seeds, families, items, provenance = [], [], [], []
    for seed, path in paths.items():
        raw = path.read_bytes()
        try:
            totals, types, cases = single.analyze(json.loads(raw))
            if any(r['total'] != 40 for r in totals):
                raise ValueError('expected 40 cases per coefficient')
            if seed == 1:
                check_seed1(totals, types, cases, reference)
        except (ValueError, KeyError) as exc:
            raise ValueError(f'Seed {seed} ({path}): {exc}') from exc
        seeds.extend(dict(seed=seed, **enrich(r)) for r in totals)
        families.extend(dict(seed=seed, **enrich(r)) for r in types)
        items.extend(dict(seed=seed, **r) for r in cases)
        provenance.append((seed, str(path.resolve()), hashlib.sha256(raw).hexdigest()))
        print(f'Validated Seed {seed}: 40 cases, six coefficients; SHA-256 {provenance[-1][2]}')
    coefficients = []
    for c in single.COEFFICIENTS:
        row = enrich(single.aggregate([r for r in items if r['coefficient'] == c], c))
        selected = [r for r in seeds if r['coefficient'] == c]
        for metric in ('baseline_accuracy', 'steered_accuracy', 'net_improvement_pp',
                       'correction_rate', 'regression_rate'):
            values = [r[metric] for r in selected if r[metric] is not None]
            for stat, value in (
                ('defined_seeds', len(values)),
                ('mean', statistics.mean(values) if values else None),
                ('std', statistics.stdev(values) if len(values) > 1 else None),
                ('min', min(values) if values else None),
                ('max', max(values) if values else None),
            ):
                row[f'{metric}_seed_{stat}'] = value
        coefficients.append(row)
    groups = defaultdict(list)
    for r in items:
        groups[r['case_id'], r['coefficient']].append(r)
    unique, overlap = [], []
    for (case_id, c), rows in sorted(groups.items()):
        if len({r['family'] for r in rows}) != 1:
            raise ValueError(f'cross-seed family mismatch: {case_id}')
        if c == 0:
            overlap.append(dict(case_id=case_id, family=rows[0]['family'],
                                seed_count=len(rows), seeds=';'.join(str(r['seed']) for r in rows)))
        row = enrich(single.aggregate(rows, c, rows[0]['family']))
        row.update(case_id=case_id, seed_count=len(rows),
                   baseline_label_disagreement=len({r['baseline_label'] for r in rows}) > 1,
                   steered_label_disagreement=len({r['steered_label'] for r in rows}) > 1,
                   baseline_answer_disagreement=len({r['baseline_answer'] for r in rows}) > 1,
                   steered_answer_disagreement=len({r['steered_answer'] for r in rows}) > 1,
                   seed_outcomes=json.dumps(rows, sort_keys=True))
        unique.append(row)
    reproducibility = []
    for c in (-7, -8):
        for family in ('all', *single.FAMILIES):
            selected = [r for r in (seeds if family == 'all' else families)
                        if r['coefficient'] == c and r['family'] == family]
            values = [r['net_improvement_pp'] for r in selected]
            row = dict(coefficient=c, family=family, improved=sum(v > 0 for v in values),
                       worsened=sum(v < 0 for v in values), unchanged=values.count(0),
                       mean_pp=statistics.mean(values),
                       std_pp=statistics.stdev(values) if len(values) > 1 else None,
                       min_pp=min(values), max_pp=max(values))
            row.update({f'seed_{r["seed"]}_net_pp': r['net_improvement_pp'] for r in selected})
            reproducibility.append(row)
    return dict(seed_summary=seeds, family_summary=families, coefficient_summary=coefficients,
                case_overlap=overlap, seed_reproducibility=reproducibility,
                unique_case_summary=unique), provenance


def table(rows, fields):
    def fmt(v):
        return 'N/A' if v is None else f'{v:.4f}' if isinstance(v, float) else str(v)
    return ['| ' + ' | '.join(fields) + ' |', '|' + '|'.join('---' for _ in fields) + '|'] + [
        '| ' + ' | '.join(fmt(r[f]) for f in fields) + ' |' for r in rows]


def report(output, data, provenance, root, seeds, reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for filename, metrics in (
        ('multiseed_accuracy', ('steered_accuracy',)),
        ('multiseed_correction_rate', ('correction_rate', 'regression_rate')),
    ):
        fig, axes = plt.subplots(1, len(metrics), figsize=(8 * len(metrics), 5), squeeze=False)
        for ax, metric in zip(axes[0], metrics):
            for seed in seeds:
                rows = [r for r in data['seed_summary'] if r['seed'] == seed]
                ax.plot(range(6), [r[metric] for r in rows], 'o-', alpha=.55, label=f'Seed {seed}')
            ax.plot(range(6), [r[metric] for r in data['coefficient_summary']],
                    'kD--', linewidth=2.5, label='Pooled seed-case observations')
            if metric == 'steered_accuracy':
                ax.axhline(data['coefficient_summary'][0]['baseline_accuracy'],
                           color='gray', linestyle=':', label='Pooled baseline')
            ax.set(xticks=range(6), xticklabels=single.COEFFICIENTS, xlabel='Coefficient (dose order)',
                   ylabel=f'{metric} (fraction)', ylim=(-.02, 1.02))
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output / f'{filename}.png', dpi=160)
        plt.close(fig)
    overlap = data['case_overlap']
    lines = ['# CRT Negative Dose Multi-Seed Integrated Analysis', '',
             'Purpose: quantify correction, regression, dose response and reproducibility for '
             'Qwen/Qwen3.5-2B-Base, crt_fresh_v2, SAE layer 5, feature 30908, token position all. '
             'Experiment identity comes from run names/configuration; generations JSON has no model metadata.', '',
             '## Inputs and validation', '',
             *[f'- Seed {s}: `{p}`; SHA-256 `{h}`' for s, p, h in provenance], '',
             'Passed: exact six coefficients, 40 cases per seed, valid required fields/families/labels, '
             'unique case IDs, matched baseline/steered IDs, stable baseline across coefficients, '
             'coefficient zero equality, answer/label consistency, transition totals. '
             + ('Seed 1 published accuracies and every field of existing coefficient, family and '
                f'transition CSVs match `{reference}`.' if 1 in seeds else 'Seed 1 not selected; legacy check not applicable.'), '',
             '## Metrics', '',
             'Accuracy and conditional rates are fractions [0,1]; *_pct columns are percentages. '
             'Net improvement pp = 100 × (steered accuracy − baseline accuracy). '
             'Correction rate = L→C / baseline lure; regression rate = C→L / baseline correct. '
             'Undefined rates are empty CSV cells, never zero. Net correct change = L→C − C→L. '
             'Seed statistics use arithmetic means and sample standard deviations (ddof=1); '
             'defined-seed counts record omitted undefined rates. Standard deviations are descriptive, not confidence intervals.', '',
             '## Coefficient results: seed-case observations', '',
             *table(data['coefficient_summary'], ('coefficient', 'total', 'baseline_accuracy_pct',
                    'steered_accuracy_pct', 'net_improvement_pp', 'lure_to_correct', 'correct_to_lure',
                    'correction_rate', 'regression_rate', 'net_improvement_pp_seed_std')), '',
             '## Seed results and dose shape', '',
             *table(data['seed_summary'], ('seed', 'coefficient', 'steered_accuracy_pct', 'net_improvement_pp')), '']
    for seed in seeds:
        rows = [r for r in data['seed_summary'] if r['seed'] == seed]
        values = [r['steered_accuracy'] for r in rows]
        shape = ('nondecreasing' if all(a <= b for a, b in zip(values, values[1:])) else
                 'nonincreasing' if all(a >= b for a, b in zip(values, values[1:])) else 'nonmonotonic')
        valley = values[3] < values[2] and values[4] > values[3] and values[5] > values[3]
        lines.append(f'- Seed {seed}: {shape} in dose order 0, -1, -2, -4, -7, -8; '
                     f'-4 drop followed by -7/-8 rebound: {valley}.')
    lines += ['', '## Family results', '',
              *table(data['family_summary'], ('seed', 'family', 'coefficient', 'baseline_accuracy_pct',
                     'steered_accuracy_pct', 'net_improvement_pp', 'lure_to_correct', 'correct_to_lure')),
              '', '## Reproducibility at -7 and -8', '',
              *table(data['seed_reproducibility'], tuple(data['seed_reproducibility'][0])), '',
              'The family rows show whether gains concentrate in Difference while Rate/Growth regress; '
              'all coefficients and both improvement and harm are retained.', '', '## Repeated cases', '',
              f'{40 * len(seeds)} seed-case observations; {len(overlap)} unique case IDs; '
              f'{sum(r["seed_count"] > 1 for r in overlap)} IDs occur in multiple seeds. '
              f'Frequency distribution (seeds per ID): '
              f'{dict(sorted(__import__("collections").Counter(r["seed_count"] for r in overlap).items()))}.', '',
              'case_overlap.csv records membership. unique_case_summary.csv preserves per-coefficient '
              'correction/regression frequencies, observation counts, label/answer disagreements and '
              'explicit seed outcomes; conflicting results are never collapsed to one binary outcome. '
              'ID equality is the operational definition of overlap; distinct IDs may still share templates.', '',
              '## Interpretation and limits', '',
              'These are descriptive binary_choice results. Negative coefficient steering is not complete '
              'feature ablation. Accuracy changes do not establish improved reasoning; answer selection '
              'bias and output format effects cannot be excluded. Repeated cases make observations '
              'potentially dependent: the pooled total is not a sample of independent unique questions. '
              'Only five seeds were requested, insufficient to establish a general causal effect; '
              'overlapping seeds must not be described as independent replications. No inferential '
              'p-values or independent-observation confidence intervals are reported. '
              'Gains and regressions at every dose are reported without selecting only favorable doses.', '',
              'Follow-up: evaluate disjoint held-out cases and answer-order counterbalancing; compare '
              'free generation and explicit ablation controls in separately authorized experiments.', '',
              '## Reproduce', '', '```bash',
              f'.venv/bin/python scripts/analyze_crt_negative_dose_multiseed.py --input-root {root} '
              f'--output {output} --seeds {" ".join(map(str, seeds))} --seed1-reference {reference}',
              '```', '', 'Use a fresh output directory. Inputs and previous analyses are read-only. '
              'No model execution, Colab, GPU, commit or push is involved. Synthetic tests are separate '
              'from these research inputs.']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')


def run(root, output, seeds=(1, 2, 3, 4, 5), reference=REFERENCE):
    if output.exists():
        raise FileExistsError(f'output directory already exists: {output}')
    if 'runs' in output.resolve().parts or output.resolve().is_relative_to(root.resolve()):
        raise ValueError('output must be outside input search root and results/runs')
    paths = discover(root, seeds)
    data, provenance = summarize(paths, reference)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in data.items():
        single.write_csv(output / f'{name}.csv', rows)
    report(output, data, provenance, root, seeds, reference)
    print(f'Wrote six CSVs, two PNGs and README to {output}')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT / 'results/runs')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/analysis/crt_negative_dose_multiseed_seed1to5')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--seed1-reference', type=Path, default=REFERENCE)
    args = parser.parse_args()
    try:
        run(args.input_root, args.output, args.seeds, args.seed1_reference)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Analysis failed: {exc}\n')


if __name__ == '__main__':
    main()
