"""Paired +7 behavioral comparison; no third-party dependencies or model execution."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
FEATURES = (30908, 22552)
SEEDS = tuple(range(5))
FAMILIES = ('crt_difference', 'crt_rate', 'crt_growth')
LABELS = ('lure', 'correct')
CATEGORY_MAP = {
    ('lure', 'correct', 'correct'): 'both_fix',
    ('lure', 'correct', 'lure'): '30908_only_fix',
    ('lure', 'lure', 'correct'): '22552_only_fix',
    ('lure', 'lure', 'lure'): 'neither_fix',
    ('correct', 'lure', 'correct'): '30908_only_regression',
    ('correct', 'correct', 'lure'): '22552_only_regression',
    ('correct', 'lure', 'lure'): 'both_regress',
    ('correct', 'correct', 'correct'): 'both_preserve_correct',
}
CATEGORIES = tuple(CATEGORY_MAP.values())
LABEL_FIELDS = ('baseline_label', 'feature30908_label', 'feature22552_label')
CASE_FIELDS = ('seed', 'coefficient', 'case_id', 'family', *LABEL_FIELDS,
               'feature30908_transition', 'feature22552_transition', 'category',
               'feature30908_run', 'feature22552_run')
UNIQUE_FIELDS = ('case_id', 'family', 'n_observations', 'seeds', 'consistency_status',
                 'inconsistent_fields', 'baseline_consistent', 'feature30908_consistent',
                 'feature22552_consistent', 'category_consistent', 'consensus_category',
                 'observations_json')
METRICS = ('n_cases', 'single_observation', 'consistent', 'inconsistent', 'n_classified',
           'baseline_lure', 'baseline_correct', *CATEGORIES,
           '30908_total_fixes', '22552_total_fixes', 'union_fixes', 'intersection_fixes',
           'fix_set_jaccard', '30908_regressions', '22552_regressions', 'both_regressions',
           '30908_fix_rate', '22552_fix_rate', '30908_regression_rate', '22552_regression_rate')
SUMMARY_FIELDS = ('analysis_unit', 'scope', 'seed', *METRICS)
FAMILY_FIELDS = ('analysis_unit', 'scope', 'seed', 'family', *METRICS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def latest_run(root, feature, seed):
    suffix = (f'_study_behavioral_fine_f{feature}_2b' if seed == 0 else
              f'_study_behavioral_robust_f{feature}_seed{seed}_2b')
    candidates = [(datetime.strptime(p.name[:-len(suffix)], '%Y%m%d-%H%M%S'), p)
                  for p in root.glob(f'*{suffix}') if p.is_dir()]
    require(candidates, f'No run: feature={feature}, seed={seed}')
    return max(candidates, key=lambda item: item[0])[1]


def coefficient_rows(rows, context, seed):
    require(isinstance(rows, list) and rows, f'{context}: expected nonempty list')
    indexed = {}
    for row in rows:
        require(isinstance(row, dict), f'{context}: invalid row')
        raw = row.get('coefficient')
        require(not isinstance(raw, bool), f'{context}: invalid coefficient')
        try:
            value = float(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{context}: invalid coefficient {raw!r}') from exc
        require(math.isfinite(value) and value not in indexed,
                f'{context}: nonfinite or duplicate coefficient {value}')
        indexed[value] = row
    expected = {4., 5., 6., 7., 8.} if seed == 0 else {7.}
    require(set(indexed) == expected, f'{context}: expected coefficients {sorted(expected)}')
    return indexed


def index_cases(rows, context):
    require(isinstance(rows, list) and rows, f'{context}: empty/missing rows')
    indexed = {}
    for row in rows:
        require(isinstance(row, dict), f'{context}: invalid case row')
        cid = row.get('case_id')
        require(isinstance(cid, str) and cid.strip(), f'{context}: missing case_id')
        require(cid not in indexed, f'{context}: duplicate case_id {cid}')
        require(row.get('label') in LABELS, f'{context}: {cid}: invalid label')
        require(row.get('family') in FAMILIES, f'{context}: {cid}: invalid family')
        indexed[cid] = row
    return indexed


def compare_sets(a, b, context):
    require(set(a) == set(b), f'{context}: case set mismatch; '
            f'left_only={sorted(set(a)-set(b))}, right_only={sorted(set(b)-set(a))}')


def load_run(path, feature, seed):
    context = f'{path} (feature={feature}, seed={seed})'
    artifacts = path / 'artifacts'
    with (artifacts / 'config.toml').open('rb') as handle:
        config = tomllib.load(handle)
    require(config['data']['split_seed'] == seed, f'{context}: wrong split_seed')
    require(config['feature']['layer'] == 5 and config['feature']['feature_id'] == feature,
            f'{context}: wrong config feature')
    expected = [4., 5., 6., 7., 8.] if seed == 0 else [7.]
    require(config['behavioral']['coefficients'] == expected, f'{context}: wrong config coefficients')
    meta = json.loads((artifacts / 'study_feature.json').read_text())
    require(meta.get('source') == 'pinned' and meta.get('feature') == {
        'layer': 5, 'feature_id': feature}, f'{context}: invalid pinned feature')
    blocks = coefficient_rows(json.loads((artifacts / 'behavioral/generations.json').read_text()),
                              context + ' JSON', seed)
    with (artifacts / 'behavioral.csv').open(newline='') as handle:
        metrics = coefficient_rows(list(csv.DictReader(handle)), context + ' CSV', seed)
    require(set(blocks) == set(metrics), f'{context}: coefficient mismatch')
    block = blocks[7.]
    require(block.get('output_mode') == 'binary_choice', f'{context}: wrong output_mode')
    before = index_cases(block.get('baseline_rows'), context + ' baseline')
    after = index_cases(block.get('steered_rows'), context + ' steered')
    compare_sets(before, after, context)
    for cid in before:
        require(before[cid]['family'] == after[cid]['family'], f'{context}: {cid}: family mismatch')
    n = len(before)
    counts = Counter((before[c]['label'], after[c]['label']) for c in before)
    require(sum(counts.values()) == n, f'{context}: transition count mismatch')
    lc, cl = counts['lure', 'correct'], counts['correct', 'lure']
    ba = sum(r['label'] == 'correct' for r in before.values()) / n
    sa = sum(r['label'] == 'correct' for r in after.values()) / n
    reconstructed = dict(baseline_accuracy=ba, steered_accuracy=sa,
                         accuracy_delta=(lc-cl)/n, baseline_lure_rate=1-ba,
                         steered_lure_rate=1-sa, lure_rate_delta=(cl-lc)/n)
    for key, value in reconstructed.items():
        try:
            recorded = float(metrics[7.][key])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f'{context}: invalid CSV {key}') from exc
        require(math.isfinite(recorded) and math.isclose(value, recorded, rel_tol=1e-12, abs_tol=1e-12),
                f'{context}: {key}: computed={value}, CSV={recorded}')
    return before, after


def pair_cases(seed, runs):
    b1, a1, p1 = runs[30908]
    b2, a2, p2 = runs[22552]
    compare_sets(b1, b2, f'seed={seed}: cross-feature')
    rows = []
    for cid in sorted(b1):
        require(b1[cid]['label'] == b2[cid]['label'], f'seed={seed}, case={cid}: baseline mismatch')
        require(b1[cid]['family'] == b2[cid]['family'], f'seed={seed}, case={cid}: family mismatch')
        labels = (b1[cid]['label'], a1[cid]['label'], a2[cid]['label'])
        matches = [category for key, category in CATEGORY_MAP.items() if key == labels]
        require(len(matches) == 1, f'seed={seed}, case={cid}: category not unique/exhaustive')
        rows.append(dict(seed=seed, coefficient=7., case_id=cid, family=b1[cid]['family'],
                         **dict(zip(LABEL_FIELDS, labels)),
                         feature30908_transition=f'{labels[0]}_to_{labels[1]}',
                         feature22552_transition=f'{labels[0]}_to_{labels[2]}', category=matches[0],
                         feature30908_run=str(p1), feature22552_run=str(p2)))
    return rows


def unique_cases(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['case_id']].append(row)
    unique = []
    for cid, observations in sorted(grouped.items()):
        observations = sorted(observations, key=lambda r: r['seed'])
        require(len({r['seed'] for r in observations}) == len(observations), f'{cid}: repeated seed-case')
        require(len({r['family'] for r in observations}) == 1, f'{cid}: family changed across seeds')
        consistent = [len({r[key] for r in observations}) == 1 for key in LABEL_FIELDS]
        category_consistent = len({r['category'] for r in observations}) == 1
        status = ('single_observation' if len(observations) == 1 else
                  'consistent' if all(consistent) else 'inconsistent')
        unique.append(dict(case_id=cid, family=observations[0]['family'],
                           n_observations=len(observations), seeds=json.dumps([r['seed'] for r in observations]),
                           consistency_status=status,
                           inconsistent_fields=json.dumps([key for key, ok in zip(LABEL_FIELDS, consistent) if not ok]),
                           baseline_consistent=consistent[0], feature30908_consistent=consistent[1],
                           feature22552_consistent=consistent[2], category_consistent=category_consistent,
                           consensus_category=observations[0]['category'] if all(consistent) else '',
                           observations_json=json.dumps(observations, ensure_ascii=False, separators=(',', ':'))))
    require(sum(r['n_observations'] for r in unique) == len(rows), 'Unique observation count mismatch')
    return unique


def summarize(rows, unit, scope, seed=''):
    unique = unit == 'unique_case'
    counts = Counter(r['consensus_category'] if unique else r['category'] for r in rows)
    statuses = Counter(r['consistency_status'] for r in rows) if unique else Counter()
    result = dict(analysis_unit=unit, scope=scope, seed=seed, n_cases=len(rows),
                  single_observation=statuses['single_observation'], consistent=statuses['consistent'],
                  inconsistent=statuses['inconsistent'])
    result.update({category: counts[category] for category in CATEGORIES})
    result['n_classified'] = sum(result[c] for c in CATEGORIES)
    require(result['n_classified'] + result['inconsistent'] == len(rows), 'Category total mismatch')
    result['baseline_lure'] = sum(result[c] for c in CATEGORIES[:4])
    result['baseline_correct'] = sum(result[c] for c in CATEGORIES[4:])
    # Compute improvement sets independently of category counts.
    usable = [r for r in rows if not unique or r['consistency_status'] != 'inconsistent']
    sets = []
    for fid in FEATURES:
        fixes = set()
        for r in usable:
            observation = json.loads(r['observations_json'])[0] if unique else r
            key = r['case_id'] if unique else (r['seed'], r['case_id'])
            if observation['baseline_label'] == 'lure' and observation[f'feature{fid}_label'] == 'correct':
                fixes.add(key)
        sets.append(fixes)
    union, intersection = len(sets[0] | sets[1]), len(sets[0] & sets[1])
    require(intersection == result['both_fix'] and union == sum(result[c] for c in CATEGORIES[:3]),
            'Fix set/category mismatch')
    result.update({'30908_total_fixes': len(sets[0]), '22552_total_fixes': len(sets[1]),
                   'union_fixes': union, 'intersection_fixes': intersection,
                   'fix_set_jaccard': intersection / union if union else '',
                   '30908_regressions': result['30908_only_regression'] + result['both_regress'],
                   '22552_regressions': result['22552_only_regression'] + result['both_regress'],
                   'both_regressions': result['both_regress']})
    for fid in FEATURES:
        for outcome, numerator, denominator in (
            ('fix', f'{fid}_total_fixes', 'baseline_lure'),
            ('regression', f'{fid}_regressions', 'baseline_correct')):
            result[f'{fid}_{outcome}_rate'] = result[numerator] / result[denominator] if result[denominator] else ''
    return result


def analyze(runs_dir):
    require(set(CATEGORY_MAP) == set(itertools.product(LABELS, repeat=3)) and len(set(CATEGORIES)) == 8,
            'Category mapping is not mutually exclusive/exhaustive')
    rows, paths = [], []
    for seed in SEEDS:
        runs = {}
        for fid in FEATURES:
            path = latest_run(runs_dir, fid, seed)
            before, after = load_run(path, fid, seed)
            runs[fid] = (before, after, path)
            paths.append(str(path))
        rows.extend(pair_cases(seed, runs))
    require(len({(r['seed'], r['case_id']) for r in rows}) == len(rows), 'Duplicate observation')
    unique = unique_cases(rows)
    groups = [('seed_case', 'seed', s, [r for r in rows if r['seed'] == s]) for s in SEEDS]
    groups += [('seed_case', 'all', '', rows), ('unique_case', 'all', '', unique)]
    summary, families = [], []
    for unit, scope, seed, group in groups:
        total = summarize(group, unit, scope, seed)
        summary.append(total)
        children = []
        for family in FAMILIES:
            child = summarize([r for r in group if r['family'] == family], unit, scope, seed)
            child['family'] = family
            children.append(child)
        for key in METRICS:
            if key.endswith('_rate') or key == 'fix_set_jaccard':
                continue
            require(sum(child[key] for child in children) == total[key], f'Family sum mismatch: {key}')
        families.extend(children)
    for key in METRICS:
        if not key.endswith('_rate') and key != 'fix_set_jaccard':
            require(sum(r[key] for r in summary[:len(SEEDS)]) == summary[len(SEEDS)][key],
                    f'Seed sum mismatch: {key}')
    ids = {seed: {r['case_id'] for r in rows if r['seed'] == seed} for seed in SEEDS}
    overlaps = [{'seed_a': a, 'seed_b': b, 'intersection': len(ids[a] & ids[b]),
                 'union': len(ids[a] | ids[b]), 'jaccard': len(ids[a] & ids[b]) / len(ids[a] | ids[b])}
                for a, b in itertools.combinations(SEEDS, 2)]
    report = dict(runs=paths, cases_per_seed={s: len(v) for s, v in ids.items()}, n_seeds=len(ids),
                  observations=len(rows), unique_cases=len(unique),
                  all_seed_common=len(set.intersection(*ids.values())), overlaps=overlaps,
                  jaccard_min=min(r['jaccard'] for r in overlaps), jaccard_max=max(r['jaccard'] for r in overlaps),
                  unique_statuses=dict(Counter(r['consistency_status'] for r in unique)),
                  paired_case_sets_equal=True, validated_run_metrics=len(paths)*6)
    return rows, summary, families, unique, report


def write_outputs(output_dir, tables):
    # Validate everything before calling this function. Never overwrite existing files.
    output_dir.mkdir(parents=True, exist_ok=True)
    require(all(not (output_dir / name).exists() for name, _, _ in tables),
            'Output already exists; use a new --output-dir to preserve existing files')
    committed = []
    with tempfile.TemporaryDirectory(prefix='.cross-feature-', dir=output_dir) as tmp:
        for name, fields, rows in tables:
            with (Path(tmp) / name).open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        try:
            for name, _, _ in tables:
                target = output_dir / name
                os.link(Path(tmp) / name, target)  # Exclusive publication; rollback on failure.
                committed.append(target)
        except BaseException:
            for path in committed:
                path.unlink()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-dir', type=Path, default=ROOT / 'results/runs')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/analysis')
    args = parser.parse_args()
    rows, summary, families, unique, report = analyze(args.runs_dir)
    tables = [('cross_feature_case_comparison.csv', CASE_FIELDS, rows),
              ('cross_feature_summary.csv', SUMMARY_FIELDS, summary),
              ('cross_feature_by_family.csv', FAMILY_FIELDS, families),
              ('cross_feature_unique_cases.csv', UNIQUE_FIELDS, unique)]
    write_outputs(args.output_dir, tables)
    print('VALIDATION PASSED; four CSV files written.')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
