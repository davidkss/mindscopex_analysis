"""Analyze actual letter runs and paired answer-value controls without inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_crt_choice_order as common

SEEDS = (1, 2, 3, 4, 5)


def load_runs(root, stem, mode, seeds):
    rows, definitions, provenance = [], {}, []
    for seed in seeds:
        name = f'{stem}_f30908_seed{seed}_2b'
        paths = sorted(p for p in root.rglob('generations.json')
                       if p.parts[-3:-1] == ('artifacts', 'behavioral')
                       and (p.parents[2].name == name or p.parents[2].name.endswith('_' + name)))
        if len(paths) != 1:
            raise ValueError(f'{stem}, Seed {seed}: expected one input, found {len(paths)}: {paths}')
        raw = paths[0].read_bytes()
        data = json.loads(raw)
        rows.extend(common.analyze(data, seed, output_mode=mode))
        definitions[seed] = {c['case_id']: c for c in data[0]['cases']}
        provenance.append(dict(seed=seed, mode=mode, path=str(paths[0].resolve()),
                               sha256=hashlib.sha256(raw).hexdigest()))
        print(f'Validated {mode}, Seed {seed}: 40 paired cases, two orders, three coefficients')
    return rows, definitions, provenance


def compare(letter_rows, value_rows):
    fields = ('seed', 'coefficient', 'choice_order', 'case_id')
    def index(rows):
        indexed = {tuple(r[k] for k in fields): r for r in rows}
        if len(indexed) != len(rows):
            raise ValueError('duplicate paired observation')
        return indexed
    letters, values = index(letter_rows), index(value_rows)
    if letters.keys() != values.keys():
        raise ValueError('letter/value case or condition sets differ')
    paired = []
    for key, row in letters.items():
        other = values[key]
        if row['family'] != other['family']:
            raise ValueError(f'family differs across encodings: {key}')
        record = {k: row[k] for k in (*fields, 'family')}
        for encoding, item in (('letter', row), ('value', other)):
            for field in ('baseline_choice', 'steered_choice', 'baseline_label', 'steered_label', 'transition'):
                record[f'{encoding}_{field}'] = item[field]
        for phase in ('baseline', 'steered'):
            for field in ('choice', 'label'):
                record[f'{phase}_{field}_agrees'] = row[f'{phase}_{field}'] == other[f'{phase}_{field}']
        paired.append(record)
    summaries = []
    for seed in sorted({r['seed'] for r in paired}):
        for c in common.COEFFICIENTS:
            for order in common.ORDERS:
                for family in ('all', *common.single.FAMILIES):
                    selected = [r for r in paired if r['seed'] == seed and r['coefficient'] == c
                                and r['choice_order'] == order
                                and (family == 'all' or r['family'] == family)]
                    if not selected:
                        raise ValueError('missing comparison group')
                    n = len(selected)
                    record = dict(seed=seed, coefficient=c, choice_order=order, family=family, total=n)
                    for phase in ('baseline', 'steered'):
                        for field in ('choice', 'label'):
                            record[f'{phase}_{field}_agreement'] = sum(r[f'{phase}_{field}_agrees'] for r in selected) / n
                        for encoding in ('letter', 'value'):
                            record[f'{encoding}_{phase}_accuracy'] = sum(r[f'{encoding}_{phase}_label'] == 'correct' for r in selected) / n
                        record[f'{phase}_letter_minus_value_accuracy_pp'] = 100 * (
                            record[f'letter_{phase}_accuracy'] - record[f'value_{phase}_accuracy'])
                    for encoding in ('letter', 'value'):
                        record[f'{encoding}_net_pp'] = 100 * (
                            record[f'{encoding}_steered_accuracy'] - record[f'{encoding}_baseline_accuracy'])
                    record['letter_minus_value_net_pp'] = record['letter_net_pp'] - record['value_net_pp']
                    summaries.append(record)
    return paired, summaries


def variability(data):
    rows = data['seed_order_summary'] + data['family_order_summary']
    result = []
    for c in common.COEFFICIENTS:
        for order in common.ORDERS:
            for family in ('all', *common.single.FAMILIES):
                selected = [r for r in rows if r['coefficient'] == c and r['choice_order'] == order
                            and r['family'] == family]
                values = [r['net_improvement_pp'] for r in selected]
                result.append(dict(coefficient=c, choice_order=order, family=family,
                                   seed_count=len(values), improved=sum(v > 0 for v in values),
                                   worsened=sum(v < 0 for v in values), unchanged=values.count(0),
                                   mean_pp=statistics.mean(values),
                                   std_pp=statistics.stdev(values) if len(values) > 1 else None,
                                   min_pp=min(values), max_pp=max(values)))
    return result


def run(root, value_root, output, seeds=SEEDS):
    if output.exists():
        raise FileExistsError(f'output exists: {output}')
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or s < 1 for s in seeds)
            or 'runs' in output.resolve().parts
            or any(output.resolve().is_relative_to(p.resolve()) for p in (root, value_root))):
        raise ValueError('invalid seeds or output inside input directory')
    letters, letter_cases, provenance = load_runs(root, 'study_choice_letter', common.LETTER_MODE, seeds)
    values, value_cases, value_provenance = load_runs(
        value_root, 'study_choice_order_counterbalance', 'binary_choice', seeds)
    if letter_cases != value_cases:
        raise ValueError('original case definitions differ across output encodings')
    data = common.summarize(letters)
    paired, comparison = compare(letters, values)
    data.update(encoding_case_comparison=paired, encoding_summary=comparison,
                seed_effect_variability=variability(data))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    for name, records in data.items():
        common.single.write_csv(output / f'{name}.csv', records)
    fig, ax = plt.subplots(figsize=(10, 5))
    for mode, rows in (('letter', data['seed_order_summary']),
                       ('value', common.summarize(values)['seed_order_summary'])):
        for order in common.ORDERS:
            selected = [r for r in rows if r['choice_order'] == order]
            means = [statistics.mean(r['net_improvement_pp'] for r in selected
                                     if r['coefficient'] == c) for c in common.COEFFICIENTS]
            ax.plot(range(3), means, 'o-' if mode == 'letter' else 's--', label=f'{mode}: {order}')
    ax.set(xticks=range(3), xticklabels=common.COEFFICIENTS, xlabel='Coefficient',
           ylabel='Mean seed net improvement (pp)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'encoding_net_improvement.png', dpi=160)
    plt.close(fig)
    lines = ['# CRT Explicit A/B Choice Decoding Control', '', '## Inputs', '']
    lines += [f"- Seed {r['seed']} ({r['mode']}): `{r['path']}` SHA-256 `{r['sha256']}`"
              for r in provenance + value_provenance]
    lines += ['', '## Validation and metrics', '',
              'Passed: exact condition grid; 40 unique cases per seed/order; matched case IDs and '
              'original definitions within and across encodings; exact prompts; valid letter/label mapping; '
              'coefficient zero equality; stable baselines; transition totals. Letter answers are actual '
              'generated A/B characters; value-control choices are derived from generated answer values.', '',
              'All accuracies and conditional rates are fractions [0,1]. Net improvement and encoding '
              'differences are percentage points. Correction = L→C / baseline lure; regression = C→L / '
              'baseline correct. Undefined rates are empty CSV cells. Seed-effect standard deviation '
              'is sample SD (ddof=1). Encoding agreement is paired by seed/coefficient/order/case_id; '
              'position and semantic agreement are reported separately for baseline and steered responses.', '',
              '## Letter results', '',
              '| Seed | Order | Coefficient | Baseline | Steered | Net pp |', '|---|---|---|---|---|---|']
    lines += [f"| {r['seed']} | {r['choice_order']} | {r['coefficient']} | "
              f"{r['baseline_accuracy']:.4f} | {r['steered_accuracy']:.4f} | {r['net_improvement_pp']:.2f} |"
              for r in data['seed_order_summary']]
    lines += ['', '## Files', '', *[f'- `{name}.csv`' for name in data],
              '- `encoding_net_improvement.png`', '', '## Limits', '',
              'Output instructions and decoding candidates both change between experiments. Differences '
              'cannot be attributed solely to token output format. Constrained A/B generation does not '
              'establish reasoning improvement. Steering is not ablation. Cases can repeat across seeds; '
              'five seeds and dependent observations limit generalization. All doses and family-specific '
              'gains and regressions are retained. Compare order effects, seed variability and paired '
              'semantic agreement before drawing conclusions.', '', '## Reproduce', '',
              f'`python scripts/analyze_crt_choice_letter.py --input-root {root} '
              f'--value-input-root {value_root} --output {output} --seeds {" ".join(map(str, seeds))}`', '',
              'Requires both real experiment sets. Use a fresh output directory. Inputs are read-only.']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=Path('results/runs'))
    parser.add_argument('--value-input-root', type=Path, default=Path('results/runs'))
    parser.add_argument('--output', type=Path, default=Path('results/analysis/crt_choice_letter_control'))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    args = parser.parse_args()
    try:
        run(args.input_root, args.value_input_root, args.output, tuple(args.seeds))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f'Analysis failed: {exc}\n')


if __name__ == '__main__':
    main()
