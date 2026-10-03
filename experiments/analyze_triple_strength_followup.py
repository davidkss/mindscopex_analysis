"""Descriptive triple-strength response from 15 pinned ZIPs; stdlib only.

Each coefficient applies to each included feature. Seeds reuse cases; seed-case
observations are not independent unique cases. No optimization, mechanistic
interpretation, or significance testing is performed.
"""
from __future__ import annotations

import argparse
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
FEATURE_IDS = [30908, 22552, 14817]
COEFFICIENTS = (2, 4, 7)
SEEDS = tuple(range(5))
FAMILIES = ('crt_difference', 'crt_rate', 'crt_growth')
MEMBERS = ('config.toml', 'manifest.json', 'study_feature.json',
           'behavioral.csv', 'behavioral/generations.json')
PREFIX = 'triple_strength_followup_'
SUFFIXES = ('summary.json', 'condition_summary.csv', 'family_summary.csv',
            'comparison.csv', 'case_transitions.csv', 'baseline_consistency.csv')
# Fixed source directories in seed order. Never search for a newer run or fallback.
SOURCE_DIRECTORIES = {
    2: (
        '20261003-185033_study_behavioral_combined_f30908_f22552_f14817_c2_seed0_2b',
        '20261003-185232_study_behavioral_combined_f30908_f22552_f14817_c2_seed1_2b',
        '20261003-185413_study_behavioral_combined_f30908_f22552_f14817_c2_seed2_2b',
        '20261003-185523_study_behavioral_combined_f30908_f22552_f14817_c2_seed3_2b',
        '20261003-185628_study_behavioral_combined_f30908_f22552_f14817_c2_seed4_2b',
    ),
    4: (
        '20261003-185814_study_behavioral_combined_f30908_f22552_f14817_c4_seed0_2b',
        '20261003-185915_study_behavioral_combined_f30908_f22552_f14817_c4_seed1_2b',
        '20261003-190020_study_behavioral_combined_f30908_f22552_f14817_c4_seed2_2b',
        '20261003-190125_study_behavioral_combined_f30908_f22552_f14817_c4_seed3_2b',
        '20261003-190231_study_behavioral_combined_f30908_f22552_f14817_c4_seed4_2b',
    ),
    7: (
        '20261003-173128_study_behavioral_combined_f30908_f22552_f14817_seed0_2b',
        '20261003-173235_study_behavioral_combined_f30908_f22552_f14817_seed1_2b',
        '20261003-173339_study_behavioral_combined_f30908_f22552_f14817_seed2_2b',
        '20261003-173439_study_behavioral_combined_f30908_f22552_f14817_seed3_2b',
        '20261003-173544_study_behavioral_combined_f30908_f22552_f14817_seed4_2b',
    ),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def close(actual, expected, context):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), expected,
            abs_tol=1e-12, rel_tol=1e-12), context)


def index_coefficients(rows):
    result = {}
    for row in rows:
        coefficient = float(row['coefficient'])
        require(math.isfinite(coefficient) and coefficient not in result,
                'Nonfinite or duplicate coefficient')
        result[coefficient] = row
    return result


def stats(before, after):
    require(bool(before) and set(before) == set(after), 'Empty/mismatched case set')
    n = len(before)
    b = sum(r['label'] == 'correct' for r in before.values())
    s = sum(r['label'] == 'correct' for r in after.values())
    fixes = sum(before[c]['label'] == 'lure' and after[c]['label'] == 'correct' for c in before)
    regressions = sum(before[c]['label'] == 'correct' and after[c]['label'] == 'lure' for c in before)
    require(s-b == fixes-regressions, 'Transition identity failed')
    return dict(n_cases=n, baseline_correct=b, steered_correct=s,
        baseline_accuracy=b/n, steered_accuracy=s/n, delta_pp=100*(s-b)/n,
        fixes=fixes, regressions=regressions)


def aggregate(rows):
    require(len(rows) == 5 and {r['seed'] for r in rows} == set(SEEDS), 'Expected five seeds')
    delta = [r['delta_pp'] for r in rows]
    return dict(n_seeds=5, n_seed_case_observations=sum(r['n_cases'] for r in rows),
        mean_baseline_accuracy=statistics.mean(r['baseline_accuracy'] for r in rows),
        mean_steered_accuracy=statistics.mean(r['steered_accuracy'] for r in rows),
        mean_delta_pp=statistics.mean(delta), sd_delta_pp=statistics.stdev(delta), sd_ddof=1,
        min_delta_pp=min(delta), max_delta_pp=max(delta),
        total_fixes=sum(r['fixes'] for r in rows), total_regressions=sum(r['regressions'] for r in rows))


def case_index(rows, dataset):
    result = {r['case_id']: r for r in rows}
    require(len(rows) == len(result) == 40, 'Expected exactly 40 unique case IDs')
    for cid, row in result.items():
        require(cid in dataset, f'Unknown case: {cid}')
        require(row['family'] in FAMILIES, f'Invalid family: {cid}')
        require(row['label'] in ('correct', 'lure'), f'Invalid label: {cid}')
        require(row['answer'].strip() == dataset[cid][row['label']+'_answer'].strip(),
                f'Candidate answer/label conflict: {cid}')
        # Family disagreement is retained and flagged, not corrected or discarded.
    return result


def output_paths(root):
    return [root/'results/analysis'/(PREFIX+s) for s in SUFFIXES]


def no_overwrite(paths):
    require(not any(p.exists() or p.is_symlink() for p in paths), 'Output exists; refusing to overwrite')


def publish(root, summary, tables):
    paths = output_paths(root)
    no_overwrite(paths)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    published = []
    with tempfile.TemporaryDirectory(dir=paths[0].parent, prefix='.triple-strength-') as temp:
        temp = Path(temp)
        (temp/SUFFIXES[0]).write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                              allow_nan=False)+'\n', encoding='utf-8')
        for suffix, rows in tables.items():
            fields = list(dict.fromkeys(k for row in rows for k in row))
            with (temp/suffix).open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                                  for k, v in row.items()} for row in rows)
        try:
            for suffix, target in zip(SUFFIXES, paths):
                os.link(temp/suffix, target)  # Atomic no-clobber publication.
                published.append(target)
        except BaseException:
            for path in published:
                path.unlink()
            raise
    return paths


def source_map():
    return [dict(coefficient=c, seed=s,
                 source_zip=f'results/runs/{directory}/artifacts.zip',
                 run_name=directory.split('_', 1)[1],
                 source_type='existing_triple' if c == 7 else 'strength_followup')
            for c, directories in SOURCE_DIRECTORIES.items()
            for s, directory in enumerate(directories)]


def coverage(sources):
    counts = Counter((r['coefficient'], r['seed']) for r in sources)
    expected = set(itertools.product(COEFFICIENTS, SEEDS))
    result = dict(expected_logical_runs=15, found_logical_runs=len(sources),
        unique_logical_runs=len(counts), missing=sorted(expected-set(counts)),
        unexpected=sorted(set(counts)-expected),
        duplicates=[dict(coefficient=c, seed=s, count=n) for (c, s), n in counts.items() if n != 1])
    require(len(sources) == len(counts) == 15 and not result['missing']
            and not result['unexpected'] and not result['duplicates'],
            'Coverage failed: ' + json.dumps(result))
    require(len({r['source_zip'] for r in sources}) == 15, 'Duplicate source ZIP')
    return result


def load_run(root, source, dataset):
    path = root/source['source_zip']
    raw_zip = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        require(archive.testzip() is None, f'ZIP CRC failure: {path}')
        for name in MEMBERS:
            require(archive.namelist().count(name) == 1, f'Missing/duplicate ZIP member: {name}')
        raw = {name: archive.read(name) for name in MEMBERS}
    config = tomllib.loads(raw['config.toml'].decode('utf-8'))
    manifest = json.loads(raw['manifest.json'])
    coefficient, seed = source['coefficient'], source['seed']
    feature = dict(layer=5, feature_ids=FEATURE_IDS)
    require(config.get('features') == feature and 'feature' not in config, 'Feature/layer mismatch')
    require(config['run'] == dict(name=source['run_name'], job='research_experiments'), 'Run mismatch')
    require(config['experiment']['kind'] == 'behavioral', 'Wrong experiment kind')
    require(config['model']['profile'] == '2b', 'Wrong model profile')
    require(config['data'] == dict(dataset='hagendorff_crt', limit_per_family=0,
            train_frac=0.6, split_seed=seed, instruction=True), 'Wrong data protocol')
    for key, value in dict(coefficients=[float(coefficient)], max_cases=40,
            max_new_tokens=16, token_position='all', output_mode='binary_choice').items():
        require(config['behavioral'].get(key) == value, f'Behavioral mismatch: {key}')
    for key, value in dict(run_name=source['run_name'], job='research_experiments',
            requested_kind='behavioral', profile='2b', model_id='Qwen/Qwen3.5-2B-Base',
            dataset='hagendorff_crt', train_frac=0.6, split_seed=seed, study_feature=feature).items():
        require(manifest.get(key) == value, f'Manifest mismatch: {key}')
    require(bool(manifest.get('finished_at')), 'Unfinished manifest')
    require(any(r.get('kind') == 'behavioral' and r.get('n_cases') == 40
                and r.get('output_mode') == 'binary_choice' for r in manifest['results']),
            'Manifest result mismatch')
    require(json.loads(raw['study_feature.json']) == dict(feature=feature, source='pinned'),
            'Feature metadata mismatch')
    recorded = index_coefficients(list(csv.DictReader(io.StringIO(raw['behavioral.csv'].decode('utf-8')))))
    details = index_coefficients(json.loads(raw['behavioral/generations.json']))
    require(set(recorded) == set(details) == {coefficient}, 'Coefficient coverage mismatch')
    detail = details[coefficient]
    require(detail['output_mode'] == 'binary_choice', 'Generation mode mismatch')
    require(detail.get('layer') == 5 and detail.get('feature_ids') == FEATURE_IDS,
            'Generation feature/layer mismatch')
    before = case_index(detail['baseline_rows'], dataset)
    after = case_index(detail['steered_rows'], dataset)
    metric = stats(before, after)
    require(all(before[c]['family'] == after[c]['family'] for c in before),
            'Baseline/steered family mismatch')
    require({r['family'] for r in before.values()} == set(FAMILIES), 'Missing CRT family')
    ba, sa = metric['baseline_accuracy'], metric['steered_accuracy']
    for key, value in dict(baseline_accuracy=ba, steered_accuracy=sa, accuracy_delta=sa-ba,
            baseline_lure_rate=1-ba, steered_lure_rate=1-sa, lure_rate_delta=ba-sa).items():
        close(recorded[coefficient][key], value, f'CSV metric mismatch: {key}')
    for key, value in metric.items():
        if key in recorded[coefficient]:
            close(recorded[coefficient][key], value, f'CSV metric mismatch: {key}')
    prov = dict(**source, absolute_source_zip=str(path.resolve()), artifact_sha256=digest(raw_zip),
                member_sha256={n: digest(b) for n, b in raw.items()}, config=config, manifest=manifest)
    return dict(source=prov, before=before, after=after, metric=metric)


def baseline_consistency(loaded, dataset):
    rows, case_set_errors = [], []
    for seed in SEEDS:
        indexes = {c: loaded[c, seed]['before'] for c in COEFFICIENTS}
        all_cases = set.union(*(set(b) for b in indexes.values()))
        if not all(set(b) == all_cases for b in indexes.values()):
            case_set_errors.append(dict(seed=seed, missing_by_coefficient={
                c: sorted(all_cases-set(b)) for c, b in indexes.items()}))
            continue
        for cid in sorted(all_cases):
            values = [indexes[c][cid] for c in COEFFICIENTS]
            flags = {field+'_discordant': len({r[field] for r in values}) != 1
                     for field in ('answer', 'label', 'family')}
            flags['dataset_family_discordant'] = any(r['family'] != dataset[cid]['family'] for r in values)
            for coefficient in COEFFICIENTS:
                row = indexes[coefficient][cid]
                rows.append(dict(seed=seed, case_id=cid, coefficient=coefficient,
                    baseline_answer=row['answer'], baseline_label=row['label'], family=row['family'],
                    expected_family=dataset[cid]['family'], **flags,
                    source_zip=loaded[coefficient, seed]['source']['source_zip']))
    require(not case_set_errors, 'Cross-coefficient case sets differ: ' + json.dumps(case_set_errors))
    flag_names = ('answer_discordant', 'label_discordant', 'family_discordant', 'dataset_family_discordant')
    discordances = [r for r in rows if any(r[k] for k in flag_names)]
    summary = dict(status='discordant' if discordances else 'consistent', case_sets_identical=True,
        discordant_seed_case_counts={k: len({(r['seed'], r['case_id']) for r in rows if r[k]})
                                     for k in flag_names},
        discordances=discordances,
        policy='Retain original raw answers, labels, and families; no harmonization, majority vote, or exclusion.')
    return rows, summary


def validate(root):
    sources = source_map()
    checked_coverage = coverage(sources)
    raw_dataset = (root/'src/mindscopex_analysis/data/hagendorff_crt.json').read_bytes()
    cases = json.loads(raw_dataset)['cases']
    dataset = {r['case_id']: r for r in cases}
    require(len(dataset) == len(cases), 'Duplicate dataset case IDs')
    loaded, errors = {}, []
    for source in sources:
        try:
            loaded[source['coefficient'], source['seed']] = load_run(root, source, dataset)
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
            errors.append(dict(source=source, error=str(exc)))
    require(not errors, 'Artifact validation failed; no outputs written: ' + json.dumps(errors, indent=2))
    coverage([r['source'] for r in loaded.values()])
    consistency, baseline = baseline_consistency(loaded, dataset)
    report = dict(validation_status='passed', coverage=checked_coverage,
        source_provenance=[r['source'] for r in loaded.values()], baseline_consistency=baseline,
        dataset_sha256=digest(raw_dataset), analyzer_sha256=digest(Path(__file__).read_bytes()))
    return loaded, consistency, report


def comparisons(loaded):
    rows = []
    for high, low in ((4, 2), (7, 4), (7, 2)):
        seed_rows = []
        for seed in SEEDS:
            a, b = loaded[high, seed], loaded[low, seed]
            mismatches = {field+'_mismatch_count': sum(a['before'][cid][field] != b['before'][cid][field]
                for cid in a['before']) for field in ('answer', 'label', 'family')}
            seed_rows.append(dict(comparison=f'+{high} vs +{low}', higher_coefficient=high,
                lower_coefficient=low, seed=seed, aggregation='per_seed',
                delta_difference_pp=a['metric']['delta_pp']-b['metric']['delta_pp'],
                steered_accuracy_difference_pp=100*(a['metric']['steered_accuracy']-b['metric']['steered_accuracy']),
                baseline_accuracy_difference_pp=100*(a['metric']['baseline_accuracy']-b['metric']['baseline_accuracy']),
                **mismatches, baselines_identical=not any(mismatches.values())))
        rows.extend(seed_rows)
        aggregate_row = dict(comparison=f'+{high} vs +{low}', higher_coefficient=high,
            lower_coefficient=low, seed='all', aggregation='equal_seed_weight', n_seeds=5,
            sd_ddof=1, baselines_identical=all(r['baselines_identical'] for r in seed_rows))
        for field in ('delta_difference_pp', 'steered_accuracy_difference_pp', 'baseline_accuracy_difference_pp'):
            values = [r[field] for r in seed_rows]
            aggregate_row['mean_'+field] = statistics.mean(values)
            aggregate_row['sd_'+field] = statistics.stdev(values)
        for field in ('answer', 'label', 'family'):
            aggregate_row[field+'_mismatch_count'] = sum(r[field+'_mismatch_count'] for r in seed_rows)
        rows.append(aggregate_row)
    return rows


def analyze(loaded, consistency, report):
    conditions, families, transitions = [], [], []
    for (coefficient, seed), run in loaded.items():
        identity = dict(coefficient=coefficient, seed=seed, aggregation='per_seed')
        before, after = run['before'], run['after']
        conditions.append(dict(**identity, **run['metric']))
        for family in FAMILIES:
            ids = [cid for cid in before if before[cid]['family'] == family]
            families.append(dict(**identity, family=family,
                **stats({cid: before[cid] for cid in ids}, {cid: after[cid] for cid in ids})))
        for cid in sorted(before):
            b, s = before[cid], after[cid]
            transitions.append(dict(coefficient=coefficient, seed=seed, case_id=cid, family=b['family'],
                baseline_answer=b['answer'], steered_answer=s['answer'], baseline_label=b['label'],
                steered_label=s['label'], transition=b['label']+'->'+s['label'], source_zip=run['source']['source_zip']))
    condition_agg = [dict(coefficient=c, seed='all', aggregation='equal_seed_weight',
        **aggregate([r for r in conditions if r['coefficient'] == c])) for c in COEFFICIENTS]
    family_agg = [dict(coefficient=c, seed='all', family=f, aggregation='equal_seed_weight',
        **aggregate([r for r in families if r['coefficient'] == c and r['family'] == f]))
        for c, f in itertools.product(COEFFICIENTS, FAMILIES)]
    comparison = comparisons(loaded)
    tables = dict(zip(SUFFIXES[1:], (conditions+condition_agg, families+family_agg,
                                   comparison, transitions, consistency)))
    summary = dict(**report, condition_summary=conditions+condition_agg, family_summary=families+family_agg,
        comparison=comparison, methods=dict(feature_ids=FEATURE_IDS, layer=5, coefficients_per_feature=list(COEFFICIENTS),
            purpose='Describe behavioral response as triple-intervention strength increases.',
            accuracy_units='proportion', delta_units='percentage points',
            aggregation='Five split seeds equally weighted; sample SD of seed deltas, ddof=1.',
            observations='200 seed-case observations per coefficient reuse cases; not 200 independent unique cases.',
            baseline_policy='Condition-specific baselines retained; raw answer/label/family discordance explicitly reported.',
            family_policy='Use stored families and record dataset/cross-coefficient family discrepancies without correction.',
            label_policy='Candidate validation strips answers only; raw strings and labels remain unchanged.',
            comparison='Higher minus lower coefficient, matched by seed; delta, steered accuracy, and baseline differences reported.',
            interpretation='Descriptive only; no optimal/best coefficient selection, synergy interpretation, or significance calculation.',
            selection='15 explicit ZIP paths, no latest-run search or fallback; select the designated coefficient.'))
    return summary, tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true', help='Validate without writing any output files')
    args = parser.parse_args()
    if not args.validate_only:
        no_overwrite(output_paths(ROOT))
    loaded, consistency, report = validate(ROOT)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if args.validate_only:
        return
    summary, tables = analyze(loaded, consistency, report)
    print('CREATED:', *publish(ROOT, summary, tables), sep='\n')


if __name__ == '__main__':
    main()
