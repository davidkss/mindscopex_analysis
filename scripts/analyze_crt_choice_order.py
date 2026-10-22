"""Validate and analyze real Colab counterbalancing outputs; never run models."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import analyze_crt_negative_dose as single
from mindscopex_analysis.cases import LureCase
from mindscopex_analysis.choice_order import COEFFICIENTS, ORDERS, choice_letter, ordered_case


def analyze(data, seed):
    if not isinstance(data, list) or len(data) != 6:
        raise ValueError(f'Seed {seed}: expected six order/coefficient blocks')
    indexed, cases_ref, baselines, rows = {}, None, {}, []
    for block in data:
        order, c = block['choice_order'], block['coefficient']
        if order not in ORDERS or isinstance(c, bool) or c not in COEFFICIENTS or (order, c) in indexed:
            raise ValueError('invalid/duplicate order or coefficient')
        if block.get('output_mode') != 'binary_choice' or block.get('response_encoding') != 'answer_value':
            raise ValueError('wrong output encoding')
        cases = {r['case_id']: LureCase(**r) for r in block['cases']}
        if len(cases) != 40 or len(block['cases']) != 40:
            raise ValueError('expected 40 unique case definitions')
        if cases_ref is None:
            cases_ref = cases
        if cases != cases_ref:
            raise ValueError('case contents differ across orders/coefficients')
        if block['prompts'] != {k: ordered_case(v, order).prompt for k, v in cases.items()}:
            raise ValueError('prompt/order mismatch')
        b = single.index_rows(block['baseline_rows'], 'baseline')
        a = single.index_rows(block['steered_rows'], 'steered')
        if b.keys() != cases.keys() or a.keys() != cases.keys():
            raise ValueError('case sets differ')
        if order in baselines and baselines[order] != b:
            raise ValueError('baseline drift across coefficients')
        baselines[order] = b
        if c == 0 and b != a:
            raise ValueError('coefficient zero differs')
        for field in ('baseline_rows', 'steered_rows'):
            for r in block[field]:
                case = cases[r['case_id']]
                letter = choice_letter(r['answer'], case, order)
                label = 'correct' if r['answer'].strip() == case.correct_answer.strip() else 'lure'
                if r.get('choice') != letter or r['label'] != label or r['family'] != case.family:
                    raise ValueError('wrong choice/label/family mapping')
        for key in sorted(cases):
            rows.append(dict(seed=seed, choice_order=order, coefficient=c, case_id=key,
                             family=b[key]['family'], baseline_label=b[key]['label'],
                             steered_label=a[key]['label'], baseline_choice=choice_letter(b[key]['answer'], cases[key], order),
                             steered_choice=choice_letter(a[key]['answer'], cases[key], order),
                             transition=f"{b[key]['label']}_to_{a[key]['label']}"))
        indexed[order, c] = True
    if set(indexed) != {(o,c) for o in ORDERS for c in COEFFICIENTS}:
        raise ValueError('missing order/coefficient')
    return rows


def summarize(rows):
    summaries, families, bias, paired = [], [], [], []
    for seed in sorted({r['seed'] for r in rows}):
        for c in COEFFICIENTS:
            for family in ('all', *single.FAMILIES):
                selected = [r for r in rows if r['seed'] == seed and r['coefficient'] == c
                            and (family == 'all' or r['family'] == family)]
                by_order = {o: {r['case_id']: r for r in selected if r['choice_order'] == o} for o in ORDERS}
                first, second = (by_order[o] for o in ORDERS)
                if not first or first.keys() != second.keys():
                    raise ValueError('paired case/family mismatch')
                for o in ORDERS:
                    group = list(by_order[o].values())
                    row = dict(seed=seed, choice_order=o, **single.aggregate(group,c,family))
                    row['steered_accuracy'] = row['overall_accuracy']
                    row['net_improvement_pp'] = 100 * row['net_improvement']
                    for phase in ('baseline', 'steered'):
                        for letter in ('A', 'B'):
                            row[f'{phase}_{letter}_rate'] = sum(r[f'{phase}_choice'] == letter for r in group) / len(group)
                    (summaries if family == 'all' else families).append(row)
                record = dict(seed=seed, coefficient=c, family=family, total=len(first))
                for phase in ('baseline', 'steered'):
                    for field in ('choice', 'label'):
                        record[f'{phase}_{field}_agreement'] = sum(first[k][f'{phase}_{field}'] == second[k][f'{phase}_{field}'] for k in first) / len(first)
                for o in ORDERS:
                    record[f'{o}_corrections'] = sum(r['transition'] == 'lure_to_correct' for r in by_order[o].values())
                both = sum(first[k]['transition'] == second[k]['transition'] == 'lure_to_correct' for k in first)
                record['corrections_both_orders'] = both
                denominator = record['correct_first_corrections']
                record['correct_first_correction_retention'] = both / denominator if denominator else None
                effects = [sum((r['steered_label']=='correct')-(r['baseline_label']=='correct') for r in by_order[o].values()) / len(first) * 100 for o in ORDERS]
                record.update(correct_first_net_pp=effects[0], lure_first_net_pp=effects[1], effect_sign_reversed=effects[0]*effects[1] < 0,
                              order_effect_gap_pp=effects[0]-effects[1])
                bias.append(record)
                if family == 'all':
                    for k in first:
                        record = dict(seed=seed, coefficient=c, case_id=k, family=first[k]['family'])
                        for o in ORDERS:
                            for field in ('baseline_label','steered_label','baseline_choice','steered_choice','transition'):
                                record[f'{o}_{field}'] = by_order[o][k][field]
                        for phase in ('baseline','steered'):
                            for field in ('choice','label'):
                                record[f'{phase}_{field}_agrees'] = first[k][f'{phase}_{field}'] == second[k][f'{phase}_{field}']
                        paired.append(record)
    return dict(seed_order_summary=summaries, family_order_summary=families,
                choice_position_bias=bias, paired_case_transitions=paired)


def run(root, output, seeds=(1,2,3,4,5)):
    if output.exists():
        raise FileExistsError(f'output exists: {output}')
    if not seeds or len(set(seeds)) != len(seeds) or output.resolve().is_relative_to(root.resolve()) or 'runs' in output.resolve().parts:
        raise ValueError('invalid seeds or output inside input root/runs')
    rows, provenance = [], []
    for seed in seeds:
        name = f'study_choice_order_counterbalance_f30908_seed{seed}_2b'
        paths = [p for p in root.rglob('generations.json') if p.parts[-3:-1] == ('artifacts','behavioral') and
                 (p.parents[2].name == name or p.parents[2].name.endswith('_'+name))]
        if len(paths) != 1:
            raise ValueError(f'Seed {seed}: expected one input, found {len(paths)}')
        raw = paths[0].read_bytes()
        rows.extend(analyze(json.loads(raw), seed))
        provenance.append(f'- Seed {seed}: `{paths[0].resolve()}` SHA-256 `{hashlib.sha256(raw).hexdigest()}`')
        print(f'Validated Seed {seed}: both orders, 40 cases, three coefficients')
    data = summarize(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=False)
    for name, records in data.items():
        single.write_csv(output / f'{name}.csv', records)
    for filename, metric in (('order_accuracy_comparison','steered_accuracy'), ('order_net_improvement','net_improvement_pp')):
        fig, ax = plt.subplots(figsize=(9,5))
        for order in ORDERS:
            for seed in seeds:
                selected = [r for r in data['seed_order_summary'] if r['seed']==seed and r['choice_order']==order]
                ax.plot(range(3), [r[metric] for r in selected], 'o-' if order==ORDERS[0] else 's--', label=f'Seed {seed} {order}')
        ax.set(xticks=range(3), xticklabels=COEFFICIENTS, xlabel='Coefficient', ylabel=metric)
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(output/f'{filename}.png',dpi=160); plt.close(fig)
    lines = ['# CRT Choice Order Counterbalancing', '', *provenance, '',
             'Validated: both orders, exact coefficient grid, 40 unique cases, identical case definitions and IDs, exact rendered prompts, valid answer/label/position mappings, baseline stability and coefficient zero equality.', '',
             'Fractions are [0,1]; net_improvement_pp is percentage points. Correction = L→C / baseline lure; regression = C→L / baseline correct; undefined rates are empty cells. choice agreement measures derived A/B position; label agreement measures semantic Correct/Lure. Choices are inferred from generated answer values, not generated letters.', '',
             'choice_position_bias.csv includes family-specific agreement, correction retention, effect sign reversal and between-order effect gap. Compare gaps and seed variation at -7/-8 to assess descriptive stability; no single selected coefficient establishes general stability.', '',
             '## Results', '', '| Seed | Order | Coefficient | Baseline | Steered | Net pp |', '|---|---|---|---|---|---|']
    lines.extend(f"| {r['seed']} | {r['choice_order']} | {r['coefficient']} | {r['baseline_accuracy']:.4f} | {r['steered_accuracy']:.4f} | {r['net_improvement_pp']:.2f} |" for r in data['seed_order_summary'])
    lines += ['', '## Limits', '', 'Binary-choice constrained answer-value decoding cannot establish reasoning improvement. Steering is not ablation. Cases overlap across seeds; do not treat repeated observations as independent. Five seeds limit generalization. Original negative-dose prompts had no displayed choices, so comparisons to that study also include adding a choice display. Within this experiment only choice order changes.', '', '## Reproduce', '', f'`python scripts/analyze_crt_choice_order.py --input-root {root} --output {output}`', '', 'Use a fresh output directory. No model execution is performed by analysis.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,default=Path('results/runs'))
    p.add_argument('--output',type=Path,default=Path('results/analysis/crt_choice_order_counterbalance'))
    args=p.parse_args()
    try:
        run(args.input_root,args.output)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        p.exit(1,f'Analysis failed: {exc}\n')


if __name__=='__main__':
    main()
