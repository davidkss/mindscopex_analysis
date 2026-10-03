"""Fixed-source Top-3 multi-feature analysis; stdlib only, no model execution.

Arithmetic excess compares accuracy deltas only: it is not evidence of
mechanistic synergy. No significance or automatic synergy decision is made.
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
SEEDS = tuple(range(5))
CONDITIONS = {
    'f30908': (30908,), 'f22552': (22552,), 'f14817': (14817,),
    'f30908_f22552': (30908, 22552), 'f30908_f14817': (30908, 14817),
    'f22552_f14817': (22552, 14817),
    'f30908_f22552_f14817': (30908, 22552, 14817),
}
# Explicit timestamps in seed order. No glob, latest-run selection, or fallback.
SOURCE_TIMESTAMPS = {
    'f30908': ('161313', '161749', '161851', '161949', '162051'),
    'f22552': ('161531', '162154', '162257', '162354', '162456'),
    'f14817': ('162559', '162657', '162759', '162900', '162958'),
    'f30908_f22552': ('175548', '175724', '175827', '175930', '180036'),
    'f30908_f14817': ('171942', '172132', '172237', '172342', '172448'),
    'f22552_f14817': ('172602', '172711', '172812', '172917', '173022'),
    'f30908_f22552_f14817': ('173128', '173235', '173339', '173439', '173544'),
}
FAMILIES = ('crt_difference', 'crt_rate', 'crt_growth')
MEMBERS = ('config.toml', 'manifest.json', 'study_feature.json',
           'behavioral.csv', 'behavioral/generations.json')
PREFIX = 'top3_multifeature_'
SUFFIXES = ('summary.json', 'condition_summary.csv', 'family_summary.csv',
            'combination_comparison.csv', 'case_transitions.csv', 'baseline_consistency.csv')
EXCESS_NOTE = ('Simple arithmetic comparison of accuracy deltas; not evidence of '
               'mechanistic synergy. No statistical significance claim.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def close(actual, expected, context):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), expected,
            abs_tol=1e-12, rel_tol=1e-12), context)


def source_map():
    entries = []
    for condition, times in SOURCE_TIMESTAMPS.items():
        for seed, timestamp in enumerate(times):
            kind = ('combined' if len(CONDITIONS[condition]) > 1 else
                    'fine' if seed == 0 and condition in ('f30908', 'f22552') else 'robust')
            name = f'study_behavioral_{kind}_{condition}'
            if kind != 'fine':
                name += f'_seed{seed}'
            name += '_2b'
            entries.append(dict(condition=condition, seed=seed, run_name=name,
                source_type='home_reproduction_' + kind,
                source_zip=f'results/runs/20261003-{timestamp}_{name}/artifacts.zip'))
    return entries


def coverage(entries):
    counts = Counter((r['condition'], r['seed']) for r in entries)
    expected = set(itertools.product(CONDITIONS, SEEDS))
    result = dict(conditions=len({k[0] for k in counts}), seeds=len({k[1] for k in counts}),
        logical_runs=len(entries), unique_logical_runs=len(counts),
        missing=sorted(expected-set(counts)), unexpected=sorted(set(counts)-expected),
        duplicates=[list(k) for k, n in counts.items() if n != 1])
    require(len(entries) == len(counts) == 35 and not result['missing']
            and not result['unexpected'] and not result['duplicates'],
            'Coverage failed: ' + json.dumps(result))
    require(len({r['source_zip'] for r in entries}) == 35, 'Repeated source ZIP')
    return result


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


def load_run(root, source, dataset):
    path = root/source['source_zip']
    raw_zip = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        for member in MEMBERS:
            require(archive.namelist().count(member) == 1, f'Missing/duplicate ZIP member: {member}')
        raw = {member: archive.read(member) for member in MEMBERS}
    config = tomllib.loads(raw['config.toml'].decode('utf-8'))
    manifest = json.loads(raw['manifest.json'])
    ids, seed = CONDITIONS[source['condition']], source['seed']
    feature = dict(layer=5, **({'feature_id': ids[0]} if len(ids) == 1 else {'feature_ids': list(ids)}))
    section = 'feature' if len(ids) == 1 else 'features'
    require(config.get(section) == feature and ('features' if section == 'feature' else 'feature') not in config,
            'Feature/layer mismatch')
    require(config['run'] == dict(name=source['run_name'], job='research_experiments'), 'Wrong run')
    require(config['experiment']['kind'] == 'behavioral' and config['model']['profile'] == '2b',
            'Wrong experiment/model')
    require(config['data'] == dict(dataset='hagendorff_crt', limit_per_family=0, train_frac=0.6,
            split_seed=seed, instruction=True), 'Data protocol mismatch')
    for key, value in dict(max_new_tokens=16, max_cases=40, token_position='all', output_mode='binary_choice').items():
        require(config['behavioral'].get(key) == value, f'Behavioral protocol mismatch: {key}')
    grid = config['behavioral']['coefficients']
    allowed_grids = ([7], [4, 5, 6, 7, 8]) if source['source_type'].endswith('_fine') else ([7],)
    require(grid in allowed_grids, 'Wrong coefficient grid')
    for key, value in dict(run_name=source['run_name'], job='research_experiments',
            requested_kind='behavioral', model_id='Qwen/Qwen3.5-2B-Base', profile='2b',
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
    require(set(recorded) == set(details) == set(grid), 'CSV/generations/config coefficient mismatch')
    detail = details[7]
    require(detail['output_mode'] == 'binary_choice', 'Wrong generation mode')
    # Older single artifacts may omit feature fields; combined artifacts must carry them.
    for key, value in feature.items():
        if len(ids) > 1 or key in detail:
            require(detail.get(key) == value, f'Generation feature mismatch: {key}')
    before = case_index(detail['baseline_rows'], dataset)
    after = case_index(detail['steered_rows'], dataset)
    metric = stats(before, after)
    require(all(before[c]['family'] == after[c]['family'] for c in before),
            'Baseline/steered family mismatch')
    ba, sa = metric['baseline_accuracy'], metric['steered_accuracy']
    for key, value in dict(baseline_accuracy=ba, steered_accuracy=sa, accuracy_delta=sa-ba,
            baseline_lure_rate=1-ba, steered_lure_rate=1-sa, lure_rate_delta=ba-sa).items():
        close(recorded[7][key], value, f'CSV metric mismatch: {key}')
    for key, value in metric.items():
        if key in recorded[7]:
            close(recorded[7][key], value, f'CSV metric mismatch: {key}')
    prov = dict(**source, absolute_source_zip=str(path.resolve()), artifact_sha256=digest(raw_zip),
        member_sha256={n: digest(b) for n, b in raw.items()}, config=config, manifest=manifest,
        selected_coefficient=7, ignored_coefficients=[d for d in grid if d != 7], config_validation='passed')
    return dict(source=prov, before=before, after=after, metric=metric)


def baseline_consistency(loaded, dataset):
    rows, status = [], {}
    for seed in SEEDS:
        indexes = {c: loaded[c, seed]['before'] for c in CONDITIONS}
        cases = set.union(*(set(b) for b in indexes.values()))
        for cid in sorted(cases):
            values = [b[cid] for b in indexes.values() if cid in b]
            flags = dict(case_set_discordant=len(values) != 7,
                answer_discordant=len({r['answer'] for r in values}) != 1,
                label_discordant=len({r['label'] for r in values}) != 1,
                family_discordant=len({r['family'] for r in values}) != 1,
                dataset_family_discordant=any(r['family'] != dataset[cid]['family'] for r in values))
            for condition, index in indexes.items():
                row = index.get(cid)
                rows.append(dict(seed=seed, case_id=cid, condition=condition, present=row is not None,
                    baseline_answer=row['answer'] if row else None,
                    baseline_label=row['label'] if row else None,
                    family=row['family'] if row else None, expected_family=dataset[cid]['family'],
                    **flags, source_zip=loaded[condition, seed]['source']['source_zip']))
        seed_rows = [r for r in rows if r['seed'] == seed]
        status[seed] = not any(any(r[k] for k in flags) for r in seed_rows)
    flag_names = ('case_set_discordant', 'answer_discordant', 'label_discordant',
                  'family_discordant', 'dataset_family_discordant')
    summary = dict(status='consistent' if all(status.values()) else 'discordant',
        strict_comparison_eligible=all(status.values()), by_seed=status,
        discordant_seed_case_counts={k: len({(r['seed'], r['case_id']) for r in rows if r[k]})
                                     for k in flag_names},
        policy='Original values preserved; no majority label, harmonization, or exclusion.')
    return rows, summary


def validate(root):
    sources = source_map()
    checked_coverage = coverage(sources)
    dataset_raw = (root/'src/mindscopex_analysis/data/hagendorff_crt.json').read_bytes()
    cases = json.loads(dataset_raw)['cases']
    dataset = {r['case_id']: r for r in cases}
    require(len(dataset) == len(cases), 'Duplicate dataset case ID')
    loaded, errors = {}, []
    for source in sources:
        try:
            loaded[source['condition'], source['seed']] = load_run(root, source, dataset)
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
            errors.append(dict(source=source, error=str(exc)))
    require(not errors, 'Validation failed; no outputs written:\n' + json.dumps(errors, indent=2))
    coverage([entry['source'] for entry in loaded.values()])
    consistency, baseline = baseline_consistency(loaded, dataset)
    report = dict(validation_status='passed', coverage=checked_coverage,
        source_provenance=[entry['source'] for entry in loaded.values()], config_validation='passed',
        behavioral_cross_validation='passed', baseline_consistency=baseline,
        dataset_sha256=digest(dataset_raw), analyzer_sha256=digest(Path(__file__).read_bytes()))
    return loaded, consistency, report


def comparisons(loaded, baseline):
    rows = []
    for condition, ids in CONDITIONS.items():
        if len(ids) == 1:
            continue
        singles = [f'f{f}' for f in ids]
        constituents = [c for c, fs in CONDITIONS.items() if set(fs) < set(ids)]
        specs = [('direct_difference', [c]) for c in constituents] + [('arithmetic_excess', singles)]
        for kind, subtract in specs:
            per_seed = []
            for seed in SEEDS:
                left = loaded[condition, seed]['metric']['delta_pp']
                right = sum(loaded[c, seed]['metric']['delta_pp'] for c in subtract)
                row = dict(condition=condition, comparison=kind, subtract_conditions=subtract,
                    aggregation='per_seed', seed=seed, combination_delta_pp=left,
                    subtracted_delta_pp=right, comparison_pp=left-right,
                    baseline_consistency='consistent' if baseline['by_seed'][seed] else 'discordant',
                    strict_comparison_eligible=baseline['by_seed'][seed], interpretation=EXCESS_NOTE)
                per_seed.append(row)
            rows.extend(per_seed)
            values = [r['comparison_pp'] for r in per_seed]
            rows.append(dict(condition=condition, comparison=kind, subtract_conditions=subtract,
                aggregation='equal_seed_weight', seed='all', n_seeds=5,
                mean_comparison_pp=statistics.mean(values), sd_comparison_pp=statistics.stdev(values),
                sd_ddof=1, min_comparison_pp=min(values), max_comparison_pp=max(values),
                baseline_consistency=baseline['status'], strict_comparison_eligible=baseline['strict_comparison_eligible'],
                interpretation=EXCESS_NOTE))
    return rows


def analyze(loaded, consistency, report):
    conditions, families, transitions = [], [], []
    for (condition, seed), run in loaded.items():
        identity = dict(condition=condition, seed=seed, aggregation='per_seed')
        before, after = run['before'], run['after']
        conditions.append(dict(**identity, **run['metric']))
        for family in FAMILIES:
            ids = [c for c in before if before[c]['family'] == family]
            families.append(dict(**identity, family=family,
                **stats({c: before[c] for c in ids}, {c: after[c] for c in ids})))
        for cid in sorted(before):
            b, s = before[cid], after[cid]
            transitions.append(dict(condition=condition, seed=seed, case_id=cid, family=b['family'],
                baseline_answer=b['answer'], steered_answer=s['answer'],
                baseline_label=b['label'], steered_label=s['label'],
                transition=b['label']+'->'+s['label'], source_zip=run['source']['source_zip']))
    condition_agg = [dict(condition=c, seed='all', aggregation='equal_seed_weight',
        **aggregate([r for r in conditions if r['condition'] == c])) for c in CONDITIONS]
    family_agg = [dict(condition=c, seed='all', family=f, aggregation='equal_seed_weight',
        **aggregate([r for r in families if r['condition'] == c and r['family'] == f]))
        for c, f in itertools.product(CONDITIONS, FAMILIES)]
    comparison = comparisons(loaded, report['baseline_consistency'])
    tables = dict(zip(SUFFIXES[1:], (conditions+condition_agg, families+family_agg,
                                   comparison, transitions, consistency)))
    summary = dict(**report, condition_summary=conditions+condition_agg,
        family_summary=families+family_agg, combination_comparison=comparison,
        methods=dict(coefficient_per_feature=7, layer=5, accuracy_units='proportion', delta_units='percentage points',
            seed_aggregation='Five seeds equally weighted; sample SD, ddof=1.',
            observations='200 seed-case observations per condition are repeated observations, not 200 independent cases.',
            provenance='home reproduction of matching historical configs; not historical original artifacts',
            excess_interpretation=EXCESS_NOTE,
            baseline_policy='Consistency is reported before comparisons. Discordant seeds remain in descriptive arithmetic; strict comparison eligibility is false. No relabeling or exclusion.',
            family_policy='Use stored families; dataset/cross-condition family discordances are recorded.',
            label_policy='Validate stripped candidate answers against stored labels; retain raw answers and labels.',
            selection='35 explicitly pinned ZIP paths; coefficient +7 only; no run fallback.'))
    return summary, tables


def output_paths(root):
    return [root/'results/analysis'/(PREFIX+s) for s in SUFFIXES]


def no_overwrite(paths):
    require(not any(p.exists() or p.is_symlink() for p in paths), 'Output exists; refusing to overwrite')


def publish(root, summary, tables):
    paths = output_paths(root)
    no_overwrite(paths)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    published = []
    with tempfile.TemporaryDirectory(dir=paths[0].parent, prefix='.top3-multifeature-') as temp:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true',
                        help='Validate sources/configs/cases/baselines only; write no files')
    args = parser.parse_args()
    if not args.validate_only:
        no_overwrite(output_paths(ROOT))
    loaded, consistency, report = validate(ROOT)
    # Always disclose baseline status before calculating cross-condition comparisons.
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if args.validate_only:
        return
    summary, tables = analyze(loaded, consistency, report)
    print('CREATED:', *publish(ROOT, summary, tables), sep='\n')


if __name__ == '__main__':
    main()
