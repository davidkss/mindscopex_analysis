"""Offline fixed Top-3 dose response from ZIPs; stdlib only, no overwrites.

Accuracy is a proportion; delta_pp is percentage points. Seeds reuse cases,
so 200 seed-case observations are not 200 independent unique cases.
No model execution, label reclassification, or post-hoc dose/feature selection.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import itertools
import json
import math
import os
import statistics
import tempfile
import tomllib
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = (30908, 22552, 14817)
SEEDS = tuple(range(5))
DOSES = (-8, -4, -2, 0, 2, 4, 7, 8)
COARSE = tuple(d for d in DOSES if d != 7)
FAMILIES = ('crt_difference', 'crt_rate', 'crt_growth')
PREFIX = 'top3_multiseed_dose_response_'
MEMBERS = ('behavioral.csv', 'behavioral/generations.json', 'config.toml',
           'manifest.json', 'study_feature.json')
SOURCE_COUNTS = dict(dose=84, reproduction_topk=21, reproduction_fine=2,
                     reproduction_robust=13)
SUFFIXES = ('summary.json', 'logical_runs.csv', 'dose_summary.csv',
            'family_summary.csv', 'case_transitions.csv', 'zero_control_check.csv',
            'baseline_consistency.csv')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def close(actual, expected, context):
    require(math.isfinite(float(actual)) and math.isclose(
        float(actual), expected, abs_tol=1e-12, rel_tol=1e-12), context)


def coefficient_index(rows, context):
    result = {}
    for row in rows:
        dose = float(row['coefficient'])
        require(math.isfinite(dose) and dose not in result,
                f'{context}: nonfinite/duplicate coefficient {dose}')
        result[dose] = row
    require(bool(result), f'{context}: empty coefficients')
    return result


def stats(pairs):
    n = len(pairs)
    require(n > 0, 'Empty metric group')
    b = sum(x['label'] == 'correct' for x, _ in pairs)
    s = sum(y['label'] == 'correct' for _, y in pairs)
    fixes = sum(x['label'] == 'lure' and y['label'] == 'correct' for x, y in pairs)
    regressions = sum(x['label'] == 'correct' and y['label'] == 'lure' for x, y in pairs)
    require(s - b == fixes - regressions, 'Transition identity failed')
    return dict(n_cases=n, baseline_correct=b, steered_correct=s,
                baseline_accuracy=b/n, steered_accuracy=s/n, delta_pp=100*(s-b)/n,
                fixes=fixes, regressions=regressions)


def aggregate(rows):
    require(len(rows) == 5 and {r['seed'] for r in rows} == set(SEEDS),
            'Aggregation requires exactly five distinct seeds')
    delta = [r['delta_pp'] for r in rows]
    return dict(n_seeds=5, n_seed_case_observations=sum(r['n_cases'] for r in rows),
                mean_baseline_accuracy=statistics.mean(r['baseline_accuracy'] for r in rows),
                mean_steered_accuracy=statistics.mean(r['steered_accuracy'] for r in rows),
                mean_delta_pp=statistics.mean(delta), sd_delta_pp=statistics.stdev(delta),
                sd_ddof=1, min_delta_pp=min(delta), max_delta_pp=max(delta),
                total_fixes=sum(r['fixes'] for r in rows),
                total_regressions=sum(r['regressions'] for r in rows))


def selection(kind, feature, seed):
    if feature not in FEATURES:
        return ()
    if kind == 'dose' and seed in SEEDS[1:]:
        return COARSE
    if kind == 'topk' and seed == 0:
        return COARSE
    if kind == 'fine' and feature in FEATURES[:2] and seed == 0:
        return (7,)
    if kind == 'robust' and (seed in SEEDS[1:] or (feature == 14817 and seed == 0)):
        return (7,)
    return ()


def canonical(config):
    result = copy.deepcopy(config)
    for section, key in (('run', 'name'), ('feature', 'feature_id'),
                         ('data', 'split_seed'), ('behavioral', 'coefficients')):
        result[section].pop(key)
    return result


def validate_config(config, manifest, metadata, feature, seed):
    require(config['feature'] == dict(layer=5, feature_id=feature)
            and 'features' not in config, 'Feature/layer mismatch')
    require(config['run']['job'] == 'research_experiments'
            and config['experiment']['kind'] == 'behavioral', 'Wrong job/kind')
    require(config['model']['profile'] == '2b', 'Wrong model profile')
    require(config['data'] == dict(dataset='hagendorff_crt', limit_per_family=0,
            train_frac=0.6, split_seed=seed, instruction=True), 'Wrong data protocol')
    for key, value in dict(max_cases=40, max_new_tokens=16, token_position='all',
                           output_mode='binary_choice').items():
        require(config['behavioral'].get(key) == value, f'Behavioral protocol: {key}')
    for key, value in dict(model_id='Qwen/Qwen3.5-2B-Base', profile='2b',
            dataset='hagendorff_crt', split_seed=seed, train_frac=0.6,
            job='research_experiments', requested_kind='behavioral',
            run_name=config['run']['name'], study_feature=config['feature']).items():
        require(manifest.get(key) == value, f'Manifest mismatch: {key}')
    require(bool(manifest.get('finished_at')), 'Unfinished manifest')
    require(any(r.get('kind') == 'behavioral' and r.get('n_cases') == 40
                and r.get('output_mode') == 'binary_choice'
                for r in manifest['results']), 'Manifest result mismatch')
    require(metadata == dict(feature=config['feature'], source='pinned'),
            'Feature metadata mismatch')


def validate_cases(rows, dataset, context):
    indexed = {r['case_id']: r for r in rows}
    require(len(rows) == len(indexed) == 40, f'{context}: duplicate/missing cases')
    for cid, row in indexed.items():
        require(cid in dataset, f'{context}: unknown case {cid}')
        original = dataset[cid]
        require(row['family'] in FAMILIES and row['family'] == original['family'],
                f'{context}/{cid}: family mismatch')
        require(row['label'] in ('correct', 'lure'), f'{context}/{cid}: invalid label')
        # Validate the stored label, never infer or replace it from an answer.
        require(row['answer'].strip() == original[row['label'] + '_answer'].strip(),
                f'{context}/{cid}: candidate answer/label conflict')
    return indexed


def load_zip(path, dataset):
    """Return selected logical entries; never extract or read loose artifacts."""
    raw_zip = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        names = archive.namelist()
        for name in MEMBERS:
            require(names.count(name) == 1, f'{path}: missing/duplicate member {name}')
        raw = {name: archive.read(name) for name in MEMBERS}
    config = tomllib.loads(raw['config.toml'].decode('utf-8'))
    name = config['run']['name']
    kinds = [k for k in ('dose', 'topk', 'fine', 'robust')
             if name.startswith(f'study_behavioral_{k}_')]
    require(len(kinds) == 1 and path.parent.name.endswith('_' + name),
            f'{path}: run/config name mismatch')
    kind = kinds[0]
    feature, seed = config['feature']['feature_id'], config['data']['split_seed']
    allowed = selection(kind, feature, seed)
    if not allowed:
        return [], None
    manifest = json.loads(raw['manifest.json'])
    validate_config(config, manifest, json.loads(raw['study_feature.json']), feature, seed)
    recorded = coefficient_index(list(csv.DictReader(io.StringIO(
        raw['behavioral.csv'].decode('utf-8')))), f'{path}/behavioral.csv')
    details = coefficient_index(json.loads(raw['behavioral/generations.json']),
                                f'{path}/generations.json')
    grid = config['behavioral']['coefficients']
    require(len(grid) == len(set(grid)) and all(math.isfinite(d) for d in grid),
            f'{path}: invalid config coefficients')
    require(set(grid) == set(recorded) == set(details), f'{path}: coefficient mismatch')
    expected_grid = {'topk': COARSE, 'fine': (4, 5, 6, 7, 8), 'robust': (7,)}
    if kind in expected_grid:
        require(set(grid) == set(expected_grid[kind]), f'{path}: wrong protocol grid')
    else:
        require(bool(set(grid) & set(allowed)) and set(grid) <= set(COARSE),
                f'{path}: wrong dose grid')
    chosen = sorted(set(grid) & set(allowed))
    source = 'dose' if kind == 'dose' else 'reproduction_' + kind
    provenance = dict(source_type=source, source_zip=str(path.resolve()),
                      artifact_sha256=digest(raw_zip),
                      member_sha256={n: digest(b) for n, b in raw.items()},
                      config=config, manifest=manifest, feature_id=feature, seed=seed,
                      selected_doses=chosen, ignored_doses=sorted(set(grid)-set(chosen)))
    entries = []
    for dose in chosen:
        detail = details[dose]
        require(detail['output_mode'] == 'binary_choice', f'{path}/{dose}: wrong mode')
        before = validate_cases(detail['baseline_rows'], dataset, f'{path}/{dose}/baseline')
        after = validate_cases(detail['steered_rows'], dataset, f'{path}/{dose}/steered')
        require(set(before) == set(after), f'{path}/{dose}: baseline/steered case set mismatch')
        metric = stats([(before[cid], after[cid]) for cid in sorted(before)])
        ba, sa = metric['baseline_accuracy'], metric['steered_accuracy']
        expected = dict(baseline_accuracy=ba, steered_accuracy=sa, accuracy_delta=sa-ba,
                        baseline_lure_rate=1-ba, steered_lure_rate=1-sa, lure_rate_delta=ba-sa)
        for key, value in expected.items():
            close(recorded[dose][key], value, f'{path}/{dose}: CSV mismatch {key}')
        # Some artifact versions also store counts; validate them when present.
        for key, value in metric.items():
            if key in recorded[dose]:
                close(recorded[dose][key], value, f'{path}/{dose}: CSV mismatch {key}')
        row = dict(feature_id=feature, seed=seed, dose=dose, **metric,
                   source_type=source, source_zip=provenance['source_zip'],
                   artifact_sha256=provenance['artifact_sha256'])
        entries.append((row, before, after))
    return entries, provenance


def consistency_rows(loaded):
    """All pairwise comparisons retain raw answers, labels, and discordances."""
    rows = []

    def compare(scope, left, right):
        b1, b2 = loaded[left][1], loaded[right][1]
        require(set(b1) == set(b2), f'{scope}: case sets differ: {left}, {right}')
        for cid in sorted(b1):
            x, y = b1[cid], b2[cid]
            rows.append(dict(scope=scope, seed=left[1], case_id=cid, family=x['family'],
                left_feature=left[0], left_dose=left[2], right_feature=right[0], right_dose=right[2],
                left_answer=x['answer'], right_answer=y['answer'],
                left_label=x['label'], right_label=y['label'],
                answer_discordant=x['answer'] != y['answer'],
                label_discordant=x['label'] != y['label'],
                left_source_zip=loaded[left][0]['source_zip'],
                right_source_zip=loaded[right][0]['source_zip']))

    for feature, seed in itertools.product(FEATURES, SEEDS):
        for d1, d2 in itertools.combinations(DOSES, 2):
            compare('within_feature_across_doses', (feature, seed, d1), (feature, seed, d2))
    for seed in SEEDS:
        for f1, f2 in itertools.combinations(FEATURES, 2):
            # All dose pairs avoid choosing a privileged feature baseline.
            for d1, d2 in itertools.product(DOSES, repeat=2):
                compare('across_features_same_seed', (f1, seed, d1), (f2, seed, d2))
    return rows


def direction(value):
    return 'positive' if value > 1e-12 else 'negative' if value < -1e-12 else 'zero'


def descriptive(logical, dose_summary):
    by_key = {(r['feature_id'], r['seed'], r['dose']): r for r in logical}
    means = {(r['feature_id'], r['dose']): r['mean_delta_pp'] for r in dose_summary}
    comparisons = []
    for feature in FEATURES:
        per_seed = []
        for seed in SEEDS:
            delta = {d: by_key[feature, seed, d]['delta_pp'] for d in DOSES}
            negative = statistics.mean(delta[d] for d in (-8, -4, -2))
            positive = statistics.mean(delta[d] for d in (2, 4, 7, 8))
            per_seed.append(dict(seed=seed, plus7_minus_plus8_delta_pp=delta[7]-delta[8],
                plus7_minus_plus8_steered_accuracy_pp=100*(
                    by_key[feature, seed, 7]['steered_accuracy']-
                    by_key[feature, seed, 8]['steered_accuracy']),
                negative_mean_delta_pp=negative, positive_mean_delta_pp=positive,
                positive_minus_negative_delta_pp=positive-negative,
                matched_magnitudes_positive_minus_negative_delta_pp={
                    str(d): delta[d]-delta[-d] for d in (2, 4, 8)}))
        slopes = [(means[feature, b]-means[feature, a])/(b-a)
                  for a, b in zip(DOSES, DOSES[1:])]
        signs = [direction(s) for s in slopes if direction(s) != 'zero']
        comparisons.append(dict(feature_id=feature, per_seed=per_seed,
            mean_plus7_minus_plus8_delta_pp=statistics.mean(
                r['plus7_minus_plus8_delta_pp'] for r in per_seed),
            mean_positive_minus_negative_delta_pp=statistics.mean(
                r['positive_minus_negative_delta_pp'] for r in per_seed),
            direction_by_dose=[dict(dose=d, mean_delta_pp=means[feature, d],
                direction=direction(means[feature, d]), seed_direction_counts=dict(Counter(
                    direction(by_key[feature, s, d]['delta_pp']) for s in SEEDS))) for d in DOSES],
            mean_curve_nondecreasing=all(s >= -1e-12 for s in slopes),
            mean_curve_nonincreasing=all(s <= 1e-12 for s in slopes),
            adjacent_slopes_pp_per_dose=[dict(left_dose=a, right_dose=b, slope=s)
                for a, b, s in zip(DOSES, DOSES[1:], slopes)],
            slope_direction_reversals=sum(a != b for a, b in zip(signs, signs[1:])),
            slope_range_pp_per_dose=max(slopes)-min(slopes),
            nonconstant_adjacent_slopes=not all(math.isclose(s, slopes[0],
                abs_tol=1e-12, rel_tol=1e-12) for s in slopes)))
    return comparisons


def analyze(root):
    dataset_path = root/'src/mindscopex_analysis/data/hagendorff_crt.json'
    dataset_raw = dataset_path.read_bytes()
    cases = json.loads(dataset_raw)['cases']
    dataset = {r['case_id']: r for r in cases}
    require(len(dataset) == len(cases), 'Duplicate dataset case_id')
    entries, provenance, errors = [], [], []
    paths = sorted((root/'results/runs').glob('*/artifacts.zip'))
    for path in paths:
        if not any('_study_behavioral_' + k + '_' in path.parent.name
                   for k in ('dose', 'topk', 'fine', 'robust')):
            continue
        try:
            selected, prov = load_zip(path, dataset)
            entries.extend(selected)
            if prov is not None:
                provenance.append(prov)
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
            errors.append(dict(source_zip=str(path), error=str(exc)))
    require(not errors, 'Artifact validation failed; no outputs written:\n' +
            json.dumps(errors, indent=2))
    keys = [(r['feature_id'], r['seed'], r['dose']) for r, _, _ in entries]
    counts = Counter(keys)
    expected = set(itertools.product(FEATURES, SEEDS, DOSES))
    coverage = dict(selected_logical_rows=len(keys), unique_logical_rows=len(counts),
        missing=sorted(expected-set(counts)), unexpected=sorted(set(counts)-expected),
        duplicates=[dict(feature_id=k[0], seed=k[1], dose=k[2], count=n)
                    for k, n in sorted(counts.items()) if n > 1],
        source_counts=dict(Counter(r['source_type'] for r, _, _ in entries)))
    require(len(keys) == len(counts) == 120 and not coverage['missing']
            and not coverage['unexpected'] and not coverage['duplicates']
            and coverage['source_counts'] == SOURCE_COUNTS,
            'Logical coverage failed; no fallback/deduplication:\n' + json.dumps(coverage, indent=2))
    reference = canonical(provenance[0]['config'])
    for prov in provenance:
        require(canonical(prov['config']) == reference,
                f'Unexpected configuration difference: {prov["source_zip"]}')
    loaded = {k: e for k, e in zip(keys, entries)}
    logical, families, transitions, zero = [], [], [], []
    for key in itertools.product(FEATURES, SEEDS, DOSES):
        row, before, after = loaded[key]
        logical.append(row)
        identity = dict(feature_id=key[0], seed=key[1], dose=key[2])
        for family in FAMILIES:
            metric = stats([(before[c], after[c]) for c in sorted(before)
                            if before[c]['family'] == family])
            families.append(dict(**identity, family=family, aggregation='per_seed', **metric))
        for cid in sorted(before):
            x, y = before[cid], after[cid]
            transitions.append(dict(**identity, case_id=cid, family=x['family'],
                baseline_answer=x['answer'], steered_answer=y['answer'],
                baseline_label=x['label'], steered_label=y['label'],
                transition=x['label'] + '_to_' + y['label'],
                source_type=row['source_type'], source_zip=row['source_zip']))
        if key[2] == 0:
            zero.append(dict(**identity, n_cases=40,
                answer_mismatch_count=sum(before[c]['answer'] != after[c]['answer'] for c in before),
                label_mismatch_count=sum(before[c]['label'] != after[c]['label'] for c in before),
                source_type=row['source_type'], source_zip=row['source_zip']))
    dose_summary = [dict(feature_id=f, dose=d, **aggregate(
        [r for r in logical if r['feature_id'] == f and r['dose'] == d]))
        for f, d in itertools.product(FEATURES, DOSES)]
    family_aggregates = [dict(feature_id=f, seed='all', dose=d, family=family,
        aggregation='equal_seed_weight', **aggregate([r for r in families
            if r['feature_id'] == f and r['dose'] == d and r['family'] == family]))
        for f, d, family in itertools.product(FEATURES, DOSES, FAMILIES)]
    consistency = consistency_rows(loaded)
    consistency_summary = []
    for scope in ('within_feature_across_doses', 'across_features_same_seed'):
        rows = [r for r in consistency if r['scope'] == scope]
        consistency_summary.append(dict(scope=scope, n_pair_case_comparisons=len(rows),
            answer_discordant_comparisons=sum(r['answer_discordant'] for r in rows),
            label_discordant_comparisons=sum(r['label_discordant'] for r in rows),
            discordant_seed_cases=len({(r['seed'], r['case_id']) for r in rows
                if r['answer_discordant'] or r['label_discordant']})))
    summary = dict(validation_status='passed', coverage=coverage,
        run_provenance=provenance, dose_summary=dose_summary,
        family_summary=families + family_aggregates, zero_control_check=zero,
        baseline_consistency=consistency_summary, descriptive_comparisons=descriptive(logical, dose_summary),
        dataset_sha256=digest(dataset_raw), analyzer_sha256=digest(Path(__file__).read_bytes()),
        methods=dict(accuracy_units='proportion', delta_units='percentage points',
            aggregation='Five seeds, equal weight; sample SD of delta_pp, ddof=1.',
            selection='Fixed source rules; all matching ZIPs; duplicate logical rows fail; no fallback.',
            reproduction='home reproduction of matching historical configs; not historical original artifacts',
            zero_control='dose=0 is a zero-vector steering hook control, not baseline; compared with its own no-hook baseline.',
            baseline_policy='Condition-specific no-hook baselines retained. All dose pairs within feature/seed and all feature/dose pairs within seed are compared; no majority relabeling or exclusion.',
            answer_policy='Stored answer stripped only for candidate validation; mismatch comparisons preserve raw strings. Labels are never reclassified.',
            comparisons='Descriptive only: +7 minus +8 within seed; positive (2,4,7,8) versus negative (-8,-4,-2) dose means; matched magnitudes 2,4,8 also reported. Slopes use actual dose spacing on the mean delta curve.',
            limitations=['Seed-case observations repeat cases: 200 observations are not 200 independent unique cases.',
                'Baseline discordance is retained and can affect delta comparisons.',
                'Unequal positive/negative dose grids are descriptive, not a balanced causal contrast.',
                'Slope changes describe the sampled curve, not an inferential nonlinearity test.',
                'No post-hoc best dose, best feature, aggregate feature score, or independence-based significance test.']))
    tables = dict(zip(SUFFIXES[1:], (logical, dose_summary, families + family_aggregates,
                                  transitions, zero, consistency)))
    return summary, tables


def publish(summary, tables, output):
    paths = [output/(PREFIX + suffix) for suffix in SUFFIXES]
    require(not any(p.exists() or p.is_symlink() for p in paths), 'Output exists; refusing to overwrite')
    output.mkdir(parents=True, exist_ok=True)
    published = []
    with tempfile.TemporaryDirectory(dir=output, prefix='.top3-dose-') as temp:
        temp = Path(temp)
        (temp/SUFFIXES[0]).write_text(json.dumps(summary, indent=2,
            ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        for suffix, rows in tables.items():
            fields = list(dict.fromkeys(key for row in rows for key in row))
            with (temp/suffix).open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        try:
            for suffix, path in zip(SUFFIXES, paths):
                # Atomic no-clobber creation, including concurrent writers.
                os.link(temp/suffix, path)
                published.append(path)
        except BaseException:
            for path in published:
                path.unlink()
            raise
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true',
                        help='Validate and compute in memory without writing any output files')
    args = parser.parse_args()
    output = ROOT/'results/analysis'
    if not args.validate_only:
        require(not any((output/(PREFIX+s)).exists() or (output/(PREFIX+s)).is_symlink()
                        for s in SUFFIXES), 'Output exists; refusing to overwrite')
    summary, tables = analyze(ROOT)
    print(json.dumps(dict(validation_status=summary['validation_status'],
                         coverage=summary['coverage'],
                         baseline_consistency=summary['baseline_consistency']), indent=2))
    if not args.validate_only:
        print('CREATED:', *publish(summary, tables, output), sep='\n')


if __name__ == '__main__':
    main()
