"""Compare existing seed-0 single and combined artifacts; never run models."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
import tomllib

from analyze_cross_feature_behavior import CATEGORY_MAP, CATEGORIES, FAMILIES, index_cases, compare_sets, require

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('feature30908', 'feature22552', 'combined')
EFFECTS = ('preserved_single_success', 'lost_single_success', 'combined_rescue',
           'combined_regression', 'preserved_correct', 'still_lure')


def effect(baseline, first, second, combined):
    if baseline == 'correct':
        return 'preserved_correct' if combined == 'correct' else 'combined_regression'
    if 'correct' in (first, second):
        return 'preserved_single_success' if combined == 'correct' else 'lost_single_success'
    return 'combined_rescue' if combined == 'correct' else 'still_lure'


def select_seven(rows, context):
    require(isinstance(rows, list) and rows, f'{context}: expected nonempty rows')
    indexed = {}
    for row in rows:
        value = float(row['coefficient'])
        require(math.isfinite(value) and value not in indexed, f'{context}: duplicate/nonfinite coefficient')
        indexed[value] = row
    require(7.0 in indexed, f'{context}: missing +7')
    return indexed[7.0]


def load_run(root, condition):
    suffix = ('study_behavioral_combined_f30908_f22552_seed0_2b' if condition == 'combined'
              else f'study_behavioral_fine_f{condition.removeprefix("feature")}_2b')
    candidates = [(datetime.strptime(p.name[:-(len(suffix)+1)], '%Y%m%d-%H%M%S'), p)
                  for p in root.glob(f'*_{suffix}') if p.is_dir()]
    require(candidates, f'No run for {condition}')
    path = max(candidates, key=lambda pair: pair[0])[1]
    a = path / 'artifacts'
    for name in ('behavioral/generations.json', 'behavioral.csv', 'study_feature.json', 'config.toml'):
        require((a/name).is_file(), f'Latest run missing artifact: {a/name}')
    with (a/'config.toml').open('rb') as handle: config = tomllib.load(handle)
    expected = ({'layer': 5, 'feature_ids': [30908, 22552]} if condition == 'combined' else
                {'layer': 5, 'feature_id': int(condition.removeprefix('feature'))})
    block_name = 'features' if condition == 'combined' else 'feature'
    require(config[block_name] == expected, f'{path}: wrong feature config')
    require(('feature' not in config if condition == 'combined' else 'features' not in config),
            f'{path}: conflicting feature blocks')
    require(config['data']['split_seed'] == 0 and config['behavioral']['output_mode'] == 'binary_choice',
            f'{path}: incorrect seed/output mode')
    require(7.0 in config['behavioral']['coefficients'], f'{path}: missing config coefficient')
    if condition == 'combined':
        require(config['behavioral']['coefficients'] == [7.0], f'{path}: incorrect combined coefficients')
    meta = json.loads((a/'study_feature.json').read_text())
    require(meta['feature'] == expected and meta['source'] == 'pinned', f'{path}: wrong feature metadata')
    detail = select_seven(json.loads((a/'behavioral/generations.json').read_text()), str(path))
    require(detail['output_mode'] == 'binary_choice', f'{path}: wrong generation mode')
    if condition == 'combined':
        require(detail.get('layer') == 5 and detail.get('feature_ids') == [30908,22552],
                f'{path}: wrong combined generation metadata')
    before = index_cases(detail['baseline_rows'], f'{path}: baseline')
    after = index_cases(detail['steered_rows'], f'{path}: steered')
    compare_sets(before, after, str(path))
    for cid in before:
        require(before[cid]['family'] == after[cid]['family'], f'{path}: {cid}: family mismatch')
    n = len(before)
    ba = sum(r['label']=='correct' for r in before.values())/n
    sa = sum(r['label']=='correct' for r in after.values())/n
    bl = sum(r['label']=='lure' for r in before.values())/n
    sl = sum(r['label']=='lure' for r in after.values())/n
    metrics = dict(baseline_accuracy=ba, steered_accuracy=sa, accuracy_delta=sa-ba,
                   baseline_lure_rate=bl, steered_lure_rate=sl, lure_rate_delta=sl-bl)
    with (a/'behavioral.csv').open(newline='') as handle:
        recorded = select_seven(list(csv.DictReader(handle)), str(path))
    for key, value in metrics.items():
        other = float(recorded[key])
        require(math.isfinite(other) and math.isclose(value, other, rel_tol=1e-12, abs_tol=1e-12),
                f'{path}: {key}: computed={value}, CSV={other}')
    return path, before, after, metrics


def category_rows(rows):
    return [dict(single_category=category, n_cases=sum(r['single_category']==category for r in rows),
                 combined_correct=sum(r['single_category']==category and r['combined_label']=='correct' for r in rows),
                 combined_lure=sum(r['single_category']==category and r['combined_label']=='lure' for r in rows))
            for category in CATEGORIES]


def summarize(rows):
    n = len(rows)
    result = {'n_cases': n, 'baseline_accuracy': sum(r['baseline_label']=='correct' for r in rows)/n if n else ''}
    for condition in CONDITIONS:
        result[f'{condition}_accuracy'] = sum(r[f'{condition}_label']=='correct' for r in rows)/n if n else ''
        result[f'{condition}_fixes'] = sum(r[f'{condition}_transition']=='lure_to_correct' for r in rows)
        result[f'{condition}_regressions'] = sum(r[f'{condition}_transition']=='correct_to_lure' for r in rows)
    for category in category_rows(rows):
        for label in ('correct','lure'):
            result[f'{category["single_category"]}_combined_{label}'] = category[f'combined_{label}']
    return result


def analyze(root):
    require(set(CATEGORY_MAP) == set(itertools.product(('correct','lure'), repeat=3)), 'Incomplete single categories')
    require(all(effect(*labels) in EFFECTS for labels in itertools.product(('correct','lure'), repeat=4)),
            'Incomplete effect categories')
    loaded = {condition: load_run(root, condition) for condition in CONDITIONS}
    reference = loaded['feature30908'][1]
    rows = []
    for condition, (_, before, _, _) in loaded.items():
        compare_sets(reference, before, condition)
        for cid in reference:
            require(before[cid]['family']==reference[cid]['family'], f'{condition}: {cid}: family mismatch')
            require(before[cid]['label']==reference[cid]['label'], f'{condition}: {cid}: baseline mismatch')
    for cid, baseline in sorted(reference.items()):
        labels = {c: loaded[c][2][cid]['label'] for c in CONDITIONS}
        b = baseline['label']
        row = dict(seed=0, case_id=cid, family=baseline['family'], baseline_label=b)
        row.update({f'{c}_label': label for c,label in labels.items()})
        row.update({f'{c}_transition': f'{b}_to_{label}' for c,label in labels.items()})
        row['single_category'] = CATEGORY_MAP[b, labels['feature30908'], labels['feature22552']]
        row['combined_effect_category'] = effect(b, *[labels[c] for c in CONDITIONS])
        rows.append(row)
    categories = category_rows(rows)
    effects = {key: sum(r['combined_effect_category']==key for r in rows) for key in EFFECTS}
    families = [dict(family=family, **summarize([r for r in rows if r['family']==family])) for family in FAMILIES]
    require(sum(r['n_cases'] for r in categories)==len(rows)==sum(effects.values()), 'Category sum mismatch')
    require(sum(r['n_cases'] for r in families)==len(rows), 'Family sum mismatch')
    for cat in categories:
        require(cat['n_cases']==cat['combined_correct']+cat['combined_lure'], 'Category partition mismatch')
    total = summarize(rows)
    for key,value in total.items():
        if not key.endswith('_accuracy'):
            require(sum(f[key] for f in families)==value, f'Family metric mismatch: {key}')
    paired = {}
    for c in CONDITIONS[:2]:
        losses = [r['case_id'] for r in rows if r[f'{c}_label']=='correct' and r['combined_label']=='lure']
        gains = [r['case_id'] for r in rows if r[f'{c}_label']=='lure' and r['combined_label']=='correct']
        require(math.isclose((len(gains)-len(losses))/len(rows), loaded['combined'][3]['steered_accuracy']-loaded[c][3]['steered_accuracy'], abs_tol=1e-12), 'Paired delta mismatch')
        paired[c] = dict(losses=losses, gains=gains, net_correct_change=len(gains)-len(losses))
    summary = dict(runs={c:str(v[0]) for c,v in loaded.items()}, validation='passed',
                   n_cases=len(rows), metrics={c:v[3] for c,v in loaded.items()},
                   categories=categories, combined_effects=effects, by_family=families,
                   paired_comparisons=paired,
                   effect_definitions={
                       'preserved_single_success':'baseline lure; at least one single correct; combined correct',
                       'lost_single_success':'baseline lure; at least one single correct; combined lure',
                       'combined_rescue':'baseline lure; both singles lure; combined correct',
                       'still_lure':'baseline lure; both singles lure; combined lure',
                       'combined_regression':'baseline correct; combined lure (regardless of singles)',
                       'preserved_correct':'baseline correct; combined correct (regardless of singles)'})
    return rows,categories,families,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir',type=Path,default=ROOT/'results/runs')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/analysis')
    args=parser.parse_args()
    rows,categories,families,summary=analyze(args.runs_dir)
    tables={'combined_seed0_case_comparison.csv':rows,'combined_seed0_category_summary.csv':categories,
            'combined_seed0_by_family.csv':families}
    names=[*tables,'combined_seed0_summary.json']
    require(all(not (args.output_dir/name).exists() for name in names), 'Output exists; choose a new --output-dir')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    published=[]
    with tempfile.TemporaryDirectory(dir=args.output_dir,prefix='.combined-seed0-') as tmp:
        for name,data in tables.items():
            with (Path(tmp)/name).open('w',newline='',encoding='utf-8') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        (Path(tmp)/names[-1]).write_text(json.dumps(summary,indent=2)+'\n')
        try:
            for name in names:
                target=args.output_dir/name
                os.link(Path(tmp)/name,target);published.append(target)
        except BaseException:
            for path in published:path.unlink()
            raise
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
