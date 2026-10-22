"""Read-only paired analysis of actual unconstrained CRT runs; no model loading."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tomllib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mindscopex_analysis.cases import LureCase
from mindscopex_analysis.open_ended_crt import (
    COEFFICIENTS, FAMILIES, LABELS, MODE, SCORER_VERSION, evaluate,
)
import analyze_crt_choice_letter as choice_analysis
from analyze_crt_negative_dose import write_csv


def analyze(data, seed):
    if not isinstance(data, list) or len(data) != 3:
        raise ValueError(f'Seed {seed}: expected three coefficient blocks')
    indexed = {}
    for block in data:
        c = block.get('coefficient')
        if isinstance(c, bool) or c not in COEFFICIENTS or c in indexed:
            raise ValueError('invalid/duplicate coefficient')
        indexed[c] = block
    if set(indexed) != set(COEFFICIENTS):
        raise ValueError('missing coefficient')
    first = indexed[0]
    cases = [LureCase(**r) for r in first['cases']]
    max_tokens = first['max_new_tokens']
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('invalid token budget')
    for block in data:
        if (block.get('output_mode') != MODE or block.get('scorer_version') != SCORER_VERSION
                or block.get('max_new_tokens') != max_tokens
                or {r['case_id']: r for r in block['cases']} != {r['case_id']: r for r in first['cases']}
                or len(block['cases']) != 40):
            raise ValueError('mode/scorer/budget/original case definitions differ')
    # Reuse the same result validation used during collection, without any generation.
    evaluate(cases, COEFFICIENTS, lambda original, c: indexed[c], max_tokens)
    rows = []
    for c in COEFFICIENTS:
        b = {r['case_id']: r for r in indexed[c]['baseline_rows']}
        a = {r['case_id']: r for r in indexed[c]['steered_rows']}
        for key in sorted(b):
            rows.append(dict(seed=seed, coefficient=c, case_id=key, family=b[key]['family'],
                             baseline_label=b[key]['label'], steered_label=a[key]['label'],
                             baseline_answer=b[key]['answer'], steered_answer=a[key]['answer'],
                             baseline_raw_text=b[key]['raw_text'], steered_raw_text=a[key]['raw_text'],
                             baseline_reason=b[key]['classification']['reason'],
                             steered_reason=a[key]['classification']['reason'],
                             baseline_termination=b[key]['termination_reason'],
                             steered_termination=a[key]['termination_reason'],
                             transition=f"{b[key]['label']}_to_{a[key]['label']}"))
    return rows, {case.case_id: case for case in cases}


def aggregate(rows, coefficient, family='all'):
    if not rows:
        raise ValueError('empty analysis group')
    n = len(rows)
    result = dict(coefficient=coefficient, family=family, total=n)
    for phase in ('baseline', 'steered'):
        counts = Counter(r[f'{phase}_label'] for r in rows)
        if set(counts) - set(LABELS):
            raise ValueError('invalid response label')
        for label in LABELS:
            result[f'{phase}_{label}_count'] = counts[label]
            result[f'{phase}_{label}_rate'] = counts[label] / n
    transitions = Counter(r['transition'] for r in rows)
    for before in LABELS:
        for after in LABELS:
            key = f'{before}_to_{after}'
            result[key] = transitions[key]
    if sum(result[f'{b}_to_{a}'] for b in LABELS for a in LABELS) != n:
        raise ValueError('transition counts do not sum to N')
    result['accuracy_change_pp'] = 100 * (result['steered_correct_rate'] - result['baseline_correct_rate'])
    result['lure_change_pp'] = 100 * (result['steered_lure_rate'] - result['baseline_lure_rate'])
    return result


def summarize(rows):
    seeds, families, variability = [], [], []
    for seed in sorted({r['seed'] for r in rows}):
        for c in COEFFICIENTS:
            for family in ('all', *FAMILIES):
                selected = [r for r in rows if r['seed'] == seed and r['coefficient'] == c
                            and (family == 'all' or r['family'] == family)]
                record = dict(seed=seed, **aggregate(selected, c, family))
                (seeds if family == 'all' else families).append(record)
    for c in COEFFICIENTS:
        for family in ('all', *FAMILIES):
            selected = [r for r in seeds + families if r['coefficient'] == c and r['family'] == family]
            record = dict(coefficient=c, family=family, seed_count=len(selected))
            for metric in ('accuracy_change_pp', 'lure_change_pp'):
                values = [r[metric] for r in selected]
                record.update({f'{metric}_mean': statistics.mean(values),
                               f'{metric}_std': statistics.stdev(values) if len(values) > 1 else None,
                               f'{metric}_min': min(values), f'{metric}_max': max(values),
                               f'{metric}_positive_seeds': sum(v > 0 for v in values),
                               f'{metric}_negative_seeds': sum(v < 0 for v in values),
                               f'{metric}_unchanged_seeds': values.count(0)})
            variability.append(record)
    pooled = [aggregate([r for r in rows if r['coefficient'] == c], c) for c in COEFFICIENTS]
    membership = {}
    for r in rows:
        if r['coefficient'] == 0:
            membership.setdefault(r['case_id'], set()).add(r['seed'])
    overlap = [dict(case_id=k, seed_count=len(v), seeds=';'.join(map(str, sorted(v))))
               for k, v in sorted(membership.items())]
    return dict(seed_summary=seeds, family_summary=families, coefficient_summary=pooled,
                seed_variability=variability, paired_case_transitions=rows, case_overlap=overlap)


def load_open(root, seeds):
    rows, definitions, provenance = [], {}, []
    shared_budget = None
    for seed in seeds:
        name = f'study_open_ended_crt_f30908_seed{seed}_2b'
        paths = sorted(p for p in root.rglob('generations.json')
                       if p.parts[-3:-1] == ('artifacts', 'behavioral')
                       and (p.parents[2].name == name or p.parents[2].name.endswith('_' + name)))
        if len(paths) != 1:
            raise ValueError(f'Seed {seed}: expected exactly one input, found {len(paths)}')
        path = paths[0]
        raw = path.read_bytes()
        artifact = path.parents[1]
        status = json.loads((artifact / 'open_ended_run_status.json').read_text())
        manifest = json.loads((artifact / 'manifest.json').read_text())
        metadata = json.loads((path.parent / 'provenance.json').read_text())
        config_raw = (artifact / 'config.toml').read_bytes()
        config = tomllib.loads(config_raw.decode())
        digest, config_hash = hashlib.sha256(raw).hexdigest(), hashlib.sha256(config_raw).hexdigest()
        if status.get('phase') != 'succeeded' or status.get('returncode') != 0 or not manifest.get('finished_at'):
            raise ValueError(f'Seed {seed}: incomplete/failed run')
        if metadata.get('generations_sha256') != digest or metadata.get('config_sha256') != config_hash or status.get('config_sha256') != config_hash:
            raise ValueError(f'Seed {seed}: result/config hash mismatch')
        if (metadata.get('split_seed') != seed or config['data']['split_seed'] != seed
                or manifest.get('split_seed') != seed or config['data']['dataset'] != 'crt_fresh_v2'
                or metadata.get('model_id') != 'Qwen/Qwen3.5-2B-Base'
                or manifest.get('model_id') != metadata['model_id']
                or config['model']['profile'] != '2b'
                or config['data'].get('instruction') is not True
                or config['data'].get('train_frac') != .6
                or config['behavioral'].get('max_cases') != 40
                or metadata.get('output_mode') != MODE
                or status.get('run_name') != name
                or config['feature'] != {'layer': 5, 'feature_id': 30908}
                or metadata.get('layer') != 5 or metadata.get('feature_id') != 30908
                or config['behavioral']['output_mode'] != MODE
                or config['behavioral']['token_position'] != 'all'
                or config['behavioral'].get('choice_orders') is not None
                or config['behavioral']['coefficients'] != list(COEFFICIENTS)
                or manifest.get('git_commit') != metadata.get('git_commit')):
            raise ValueError(f'Seed {seed}: incompatible experiment metadata')
        budget = config['behavioral']['max_new_tokens']
        if shared_budget is not None and budget != shared_budget:
            raise ValueError('generation budget differs across seeds')
        shared_budget = budget
        records, cases = analyze(json.loads(raw), seed)
        if any(r['max_new_tokens'] != config['behavioral']['max_new_tokens'] for r in json.loads(raw)):
            raise ValueError('config/result budget mismatch')
        rows.extend(records)
        definitions[seed] = cases
        provenance.append(dict(seed=seed, path=str(path.resolve()), sha256=digest,
                               config_sha256=config_hash, model=metadata['model_id'],
                               git_commit=metadata.get('git_commit')))
        print(f'Validated Seed {seed}: 40 cases, three coefficients, complete run, verified hashes')
    return rows, definitions, provenance


def compare_objective(open_rows, objective_rows, encoding):
    indexed = {(r['seed'], r['coefficient'], r['case_id']): r for r in open_rows}
    paired = []
    for r in objective_rows:
        key = (r['seed'], r['coefficient'], r['case_id'])
        if key not in indexed or r['family'] != indexed[key]['family']:
            raise ValueError('objective/open-ended case or family mismatch')
        original = indexed[key]
        record = dict(seed=r['seed'], coefficient=r['coefficient'], choice_order=r['choice_order'],
                      case_id=r['case_id'], family=r['family'], objective_encoding=encoding)
        for phase in ('baseline', 'steered'):
            record[f'open_{phase}_label'] = original[f'{phase}_label']
            record[f'objective_{phase}_label'] = r[f'{phase}_label']
            record[f'{phase}_label_agrees'] = original[f'{phase}_label'] == r[f'{phase}_label']
        paired.append(record)
    summaries = []
    for seed in sorted({r['seed'] for r in paired}):
        for c in COEFFICIENTS:
            for order in choice_analysis.common.ORDERS:
                for family in ('all', *FAMILIES):
                    selected = [r for r in paired if r['seed'] == seed and r['coefficient'] == c
                                and r['choice_order'] == order and (family == 'all' or r['family'] == family)]
                    if not selected:
                        raise ValueError('missing objective comparison group')
                    row = dict(seed=seed, coefficient=c, choice_order=order, family=family,
                               objective_encoding=encoding, total=len(selected))
                    for phase in ('baseline', 'steered'):
                        row[f'{phase}_semantic_agreement'] = sum(r[f'{phase}_label_agrees'] for r in selected) / len(selected)
                        for mode in ('open', 'objective'):
                            row[f'{mode}_{phase}_accuracy'] = sum(r[f'{mode}_{phase}_label'] == 'correct' for r in selected) / len(selected)
                    for mode in ('open', 'objective'):
                        row[f'{mode}_net_pp'] = 100 * (row[f'{mode}_steered_accuracy'] - row[f'{mode}_baseline_accuracy'])
                    row['open_minus_objective_net_pp'] = row['open_net_pp'] - row['objective_net_pp']
                    summaries.append(row)
    return paired, summaries


def run(root, output, seeds=(1, 2, 3, 4, 5), objective_root=None):
    if output.exists():
        raise FileExistsError(f'output already exists: {output}')
    if (not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 1 for s in seeds)
            or 'runs' in output.resolve().parts
            or any(output.resolve().is_relative_to(p.resolve()) for p in (root, objective_root) if p is not None)):
        raise ValueError('invalid seeds/output directory')
    rows, definitions, provenance = load_open(root, seeds)
    data = summarize(rows)
    objective_provenance = []
    if objective_root is not None:
        pairs, summaries = [], []
        for stem, mode in (('study_choice_order_counterbalance', 'binary_choice'),
                           ('study_choice_letter', 'binary_choice_letter')):
            objective, cases, inputs = choice_analysis.load_runs(objective_root, stem, mode, seeds)
            if any({k: LureCase(**v) for k, v in cases[s].items()} != definitions[s] for s in seeds):
                raise ValueError('original case definitions differ between open and objective runs')
            p, s = compare_objective(rows, objective, mode)
            pairs.extend(p); summaries.extend(s); objective_provenance.extend(inputs)
        data.update(objective_case_comparison=pairs, objective_comparison_summary=summaries)
    output.mkdir(parents=True, exist_ok=False)
    for name, records in data.items():
        write_csv(output / f'{name}.csv', records)
    lines = ['# Open-ended CRT Steering Control', '', '## Inputs', '']
    lines += [f'- `{p}`' for p in provenance + objective_provenance]
    lines += ['', '## Validation and metrics', '',
              'Verified successful run status, config/result SHA-256, identity metadata, exact coefficient grid, '
              '40 unique original cases, stable baselines, coefficient zero equality, raw responses, '
              'termination metadata and reproducible classification. Fractions use all 40 responses as '
              'denominator, including other/invalid. Changes are percentage points. All 16 label transitions '
              'are retained; gains are not only lure corrections. Sample seed SD uses ddof=1.', '',
              'Classification uses a single numeric answer and exact Decimal/unit conversion, with raw '
              'text and audit retained. Missing units inherit the requested reference unit. Multiple answers, '
              'unsupported text/units, incomplete generations and ambiguous formats are invalid; a '
              'well-formed numeric answer matching neither reference is other. No substring matching is used.', '',
              '## Pooled seed-case observations', '',
              '| Coefficient | N | Baseline correct | Steered correct | Correct pp | Lure pp |',
              '|---|---|---|---|---|---|']
    lines += [f"| {r['coefficient']} | {r['total']} | {r['baseline_correct_rate']:.4f} | "
              f"{r['steered_correct_rate']:.4f} | {r['accuracy_change_pp']:.2f} | {r['lure_change_pp']:.2f} |"
              for r in data['coefficient_summary']]
    overlap = data['case_overlap']
    lines += ['', f'Unique case IDs: {len(overlap)}; IDs shared across seeds: '
              f'{sum(r["seed_count"] > 1 for r in overlap)}. These are dependent seed-case '
              'observations, not independent questions.', '', '## Objective comparison', '',
              'Both answer-value and explicit-letter controls were paired by seed/coefficient/order/case_id.'
              if objective_root is not None else
              'Not requested. Supply --objective-input-root to require both existing objective experiment sets.', '',
              '## Limits', '', 'No choices or candidate values are added to prompts and no candidate trie is '
              'used. Removing choices, changing instructions and removing decoding constraints are joint '
              'changes relative to objective controls; do not attribute differences to one factor. Numeric '
              'format scoring can produce false negatives; audit raw responses, invalid cases and scorer '
              'reasons before interpretation. Short fixed generation budgets can cause truncation. Greedy '
              'accuracy does not establish reasoning improvement. Steering is not ablation. Five seeds and '
              'overlapping cases limit inference; no independent-observation p-values are reported.', '',
              '## Reproduce', '', f'`python scripts/analyze_crt_open_ended.py --input-root {root} '
              f'--output {output} --seeds {" ".join(map(str, seeds))}'
              + (f' --objective-input-root {objective_root}' if objective_root else '') + '`', '',
              'Use a new output directory. Inputs are read-only. No model execution.']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=Path('results/runs'))
    parser.add_argument('--output', type=Path, default=Path('results/analysis/crt_open_ended_control'))
    parser.add_argument('--objective-input-root', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    try:
        run(args.input_root, args.output, tuple(args.seeds), args.objective_input_root)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f'Analysis failed: {exc}\n')


if __name__ == '__main__':
    main()
