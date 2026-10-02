"""Offline Top-5 +7 analysis; stdlib only, exact provenance, no overwrites.

Accuracy fields are proportions; delta_pp and std_pp are percentage points.
Family pooled counts are seed-case observations, not independent unique cases.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = (30908, 22552, 14817, 20108, 30475)
FAMILIES = ('crt_difference', 'crt_rate', 'crt_growth')
PREFIX = 'top5_multiseed_behavioral'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def seven(rows):
    coefficients = [float(row['coefficient']) for row in rows]
    require(len(coefficients) == len(set(coefficients)), 'Duplicate coefficient')
    require(all(math.isfinite(x) for x in coefficients), 'Nonfinite coefficient')
    require(7.0 in coefficients, 'Missing +7')
    return rows[coefficients.index(7.0)]


def stats(pairs):
    n = len(pairs)
    require(n > 0, 'Empty metric group')
    b = sum(before['label'] == 'correct' for before, _ in pairs)
    s = sum(after['label'] == 'correct' for _, after in pairs)
    fixes = sum(before['label'] == 'lure' and after['label'] == 'correct' for before, after in pairs)
    regressions = sum(before['label'] == 'correct' and after['label'] == 'lure' for before, after in pairs)
    require(s - b == fixes - regressions, 'Transition identity failed')
    return dict(n=n, baseline_correct=b, steered_correct=s,
                baseline_accuracy=b/n, steered_accuracy=s/n, delta_pp=100*(s-b)/n,
                fixes=fixes, regressions=regressions)


def canonical(config):
    c = copy.deepcopy(config)
    c['run'].pop('name')
    c['feature'].pop('feature_id')
    c['data'].pop('split_seed')
    c['behavioral'].pop('coefficients')
    return c


def validate(path, feature, seed, dataset):
    a = path / 'artifacts'
    names = ('config.toml', 'manifest.json', 'study_feature.json',
             'behavioral.csv', 'behavioral/generations.json')
    for name in names:
        require((a/name).is_file(), f'Missing {a/name}')
    config = tomllib.loads((a/'config.toml').read_text())
    require(tomllib.loads((path/'config.toml').read_text()) == config, 'Run/artifact config mismatch')
    require(config['feature'] == dict(layer=5, feature_id=feature) and 'features' not in config, 'Feature mismatch')
    require(config['run']['job'] == 'research_experiments' and config['experiment']['kind'] == 'behavioral', 'Wrong job/kind')
    require(config['data'] == dict(dataset='hagendorff_crt', limit_per_family=0, train_frac=0.6,
                                   split_seed=seed, instruction=True), 'Data configuration mismatch')
    require(config['model']['profile'] == '2b', 'Wrong profile')
    require(config['behavioral']['max_cases'] == 40 and config['behavioral']['output_mode'] == 'binary_choice', 'Wrong behavioral protocol')
    expected_coefficients = [4.0, 5.0, 6.0, 7.0, 8.0] if feature in (30908, 22552) and seed == 0 else [7.0]
    require(config['behavioral']['coefficients'] == expected_coefficients, 'Unexpected coefficient grid')
    manifest = read_json(a/'manifest.json')
    for key, value in dict(model_id='Qwen/Qwen3.5-2B-Base', profile='2b', dataset='hagendorff_crt',
                           split_seed=seed, train_frac=0.6, requested_kind='behavioral',
                           run_name=config['run']['name'], study_feature=config['feature']).items():
        require(manifest.get(key) == value, f'Manifest mismatch: {key}')
    require(bool(manifest.get('finished_at')), 'Unfinished manifest')
    require(any(r.get('kind') == 'behavioral' and r.get('n_cases') == 40 and r.get('output_mode') == 'binary_choice'
                for r in manifest['results']), 'Manifest result mismatch')
    require(read_json(a/'study_feature.json') == dict(feature=config['feature'], source='pinned'), 'Feature metadata mismatch')
    detail = seven(read_json(a/'behavioral/generations.json'))
    require(detail['output_mode'] == 'binary_choice', 'Generation mode mismatch')
    indexed = []
    for key in ('baseline_rows', 'steered_rows'):
        rows = detail[key]
        index = {r['case_id']: r for r in rows}
        require(len(rows) == len(index) == 40, f'{key}: duplicate/missing cases')
        for cid, row in index.items():
            require(cid in dataset, f'Unknown case {cid}')
            original = dataset[cid]
            require(row['family'] == original['family'] and row['family'] in FAMILIES, f'{cid}: wrong family')
            require(row['label'] in ('correct', 'lure'), f'{cid}: invalid label')
            # Binary-choice answers must equal the recorded candidate. Never relabel.
            require(row['answer'].strip() == original[row['label'] + '_answer'].strip(), f'{cid}: answer/label conflict')
        indexed.append(index)
    before, after = indexed
    require(set(before) == set(after), 'Baseline/steered case set mismatch')
    pairs = [(before[cid], after[cid]) for cid in sorted(before)]
    computed = stats(pairs)
    recorded = seven(csv_rows(a/'behavioral.csv'))
    ba, sa = computed['baseline_accuracy'], computed['steered_accuracy']
    for key, value in dict(baseline_accuracy=ba, steered_accuracy=sa, accuracy_delta=sa-ba,
                           baseline_lure_rate=1-ba, steered_lure_rate=1-sa, lure_rate_delta=ba-sa).items():
        actual = float(recorded[key])
        require(math.isfinite(actual) and math.isclose(actual, value, abs_tol=1e-12, rel_tol=1e-12), f'CSV mismatch: {key}')
    provenance = dict(feature_id=feature, seed=seed, run=str(path.resolve()), status='passed', n=40,
                      coefficient=7.0, config=config, manifest=manifest,
                      artifact_sha256={name: digest(a/name) for name in names},
                      run_config_sha256=digest(path/'config.toml'))
    return before, after, computed, provenance


def analyze(root):
    analysis = root/'results/analysis'
    fixed_path = analysis/'combined_multiseed_primary_summary.json'
    evidence_path = analysis/'topk_feature_evidence_summary.json'
    fixed = read_json(fixed_path)['run_provenance']
    dataset_path = root/'src/mindscopex_analysis/data/hagendorff_crt.json'
    dataset = {r['case_id']: r for r in read_json(dataset_path)['cases']}
    selected, loaded, provenance, errors = {}, {}, [], []
    for feature in FEATURES:
        for seed in range(5):
            try:
                if feature in (30908, 22552):
                    path = Path(fixed[str(seed)][f'feature{feature}'])
                    require(path.parent.resolve() == (root/'results/runs').resolve(), 'Pinned run outside results/runs')
                else:
                    candidates = sorted((root/'results/runs').glob(f'20261002-*_study_behavioral_robust_f{feature}_seed{seed}_2b'))
                    require(len(candidates) == 1, f'Expected exactly one dated run; found {candidates}')
                    path = candidates[0]
                selected[feature, seed] = path
                print(f'SELECT feature={feature} seed={seed}: {path}', flush=True)
                loaded[feature, seed] = validate(path, feature, seed, dataset)
                provenance.append(loaded[feature, seed][3])
            except (ValueError, KeyError, OSError, TypeError) as exc:
                errors.append(dict(feature_id=feature, seed=seed, error=str(exc)))
    require(not errors, 'Validation failed; no aggregate outputs written:\n' + json.dumps(errors, indent=2))
    reference = canonical(provenance[0]['config'])
    for row in provenance:
        require(canonical(row['config']) == reference, f'Unexpected configuration difference: {row["run"]}')
    seed_rows, family_rows, consistency, summaries = [], [], [], []
    for seed in range(5):
        ref = loaded[FEATURES[0], seed][0]
        for feature in FEATURES:
            require(set(loaded[feature, seed][0]) == set(ref), f'Seed {seed}: case sets differ')
        for cid in sorted(ref):
            row = dict(seed=seed, case_id=cid, family=ref[cid]['family'])
            labels, answers = [], []
            for feature in FEATURES:
                b = loaded[feature, seed][0][cid]
                require(b['family'] == ref[cid]['family'], 'Cross-feature family mismatch')
                row[f'f{feature}_baseline_answer'] = b['answer']
                row[f'f{feature}_baseline_label'] = b['label']
                labels.append(b['label']); answers.append(b['answer'])
            row['baseline_discordant'] = len(set(labels)) != 1
            row['baseline_answer_discordant'] = len(set(answers)) != 1
            consistency.append(row)
    for feature in FEATURES:
        all_pairs = []
        for seed in range(5):
            before, after, metric, prov = loaded[feature, seed]
            seed_rows.append(dict(feature_id=feature, seed=seed, **metric, run=prov['run']))
            pairs = [(before[cid], after[cid]) for cid in sorted(before)]
            all_pairs.extend(pairs)
            for family in FAMILIES:
                family_rows.append(dict(feature_id=feature, aggregation='per_seed', seed=seed, family=family,
                                        **stats([p for p in pairs if p[0]['family'] == family])))
        for family in FAMILIES:
            family_rows.append(dict(feature_id=feature, aggregation='pooled_seed_case', seed='all', family=family,
                                    **stats([p for p in all_pairs if p[0]['family'] == family])))
        rows = [r for r in seed_rows if r['feature_id'] == feature]
        acc = [r['steered_accuracy'] for r in rows]
        summaries.append(dict(feature_id=feature, n_seeds=5, n_observations=len(all_pairs),
                              mean_baseline_accuracy=statistics.mean(r['baseline_accuracy'] for r in rows),
                              mean_steered_accuracy=statistics.mean(acc), mean_delta_pp=statistics.mean(r['delta_pp'] for r in rows),
                              std_steered_accuracy_pp=100*statistics.stdev(acc), std_ddof=1,
                              min_steered_accuracy=min(acc), max_steered_accuracy=max(acc),
                              total_fixes=sum(r['fixes'] for r in rows), total_regressions=sum(r['regressions'] for r in rows)))
    evidence = read_json(evidence_path)
    evidence_csv_path = analysis/'topk_feature_evidence_table.csv'
    old_csv = {int(r['feature_id']): r for r in csv_rows(evidence_csv_path)}
    linked = []
    for feature in FEATURES:
        old = next(r for r in evidence['feature_table'] if r['feature_id'] == feature)
        for key in ('discovery_rank', 'discovery_score', 'causal_heldout_result', 'specificity_result',
                    'behavioral_reference_coefficient', 'behavioral_accuracy', 'behavioral_delta'):
            require(math.isclose(float(old_csv[feature][key]), float(old[key]), abs_tol=1e-12), f'Evidence CSV mismatch: {feature}/{key}')
        linked.append(dict(feature_id=feature, previous_evidence=old,
                           current_plus7_robustness=next(r for r in summaries if r['feature_id'] == feature)))
    discordance = [r for r in consistency if r['baseline_discordant']]
    summary = dict(validation_status='passed', validated_run_count=25, run_provenance=provenance,
                   feature_summary=summaries, by_seed=seed_rows, by_family=family_rows,
                   baseline_consistency=dict(n_seed_case_observations=len(consistency),
                       label_discordant_observations=len(discordance),
                       label_discordant_unique_cases=len({r['case_id'] for r in discordance}),
                       answer_discordant_observations=sum(r['baseline_answer_discordant'] for r in consistency),
                       discordances=discordance), evidence_connection=linked,
                   source_sha256={str(p.relative_to(root)): digest(p) for p in (fixed_path, evidence_path, evidence_csv_path, dataset_path)},
                   analyzer_sha256=digest(Path(__file__)),
                   methods=dict(accuracy_units='proportion', delta_units='percentage points',
                       seed_aggregation='equal weight across five seeds', std='sample standard deviation, ddof=1',
                       family_aggregation='pooled seed-case observations; per-seed family metrics also provided',
                       label_validation='exact stripped candidate answer matched to local dataset; original labels preserved',
                       baseline_policy='retain all observations and condition-specific baselines; flag discordances without relabeling/exclusion',
                       selection='fixed finalized provenance for 30908/22552; unique 20261002 robust run per seed for others; no fallback',
                       limitations=['Seeds overlap in cases: 200 observations are not 200 independent cases.',
                                    'Older seed0 evidence uses +8; current robustness uses +7.',
                                    'No composite evidence score or final feature selection.',
                                    'Discordance does not establish a diagnostic tie in any historical run.']))
    return summary, seed_rows, family_rows, consistency


def write_csv(path, rows):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true', help='Validate and compute without writing outputs')
    args = parser.parse_args()
    outputs = [ROOT/'results/analysis'/f'{PREFIX}_{suffix}' for suffix in
               ('summary.json', 'by_seed.csv', 'by_family.csv', 'baseline_consistency.csv')]
    if not args.validate_only:
        require(not any(p.exists() for p in outputs), 'Output exists; refusing to overwrite')
    summary, seeds, families, consistency = analyze(ROOT)
    print(json.dumps(dict(validation_status=summary['validation_status'], feature_summary=summary['feature_summary'],
                          baseline_consistency=summary['baseline_consistency']), indent=2))
    if not args.validate_only:
        with outputs[0].open('x') as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write('\n')
        for path, rows in zip(outputs[1:], (seeds, families, consistency)):
            write_csv(path, rows)
        print('CREATED:', *outputs, sep='\n')


if __name__ == '__main__':
    main()
