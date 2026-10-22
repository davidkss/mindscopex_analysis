"""Conservative, auditable numeric scoring for unconstrained CRT completions."""
from __future__ import annotations

import re
from dataclasses import asdict
from decimal import Decimal

from mindscopex_analysis.generation import _normalized_text

MODE = 'open_ended_crt'
COEFFICIENTS = (0.0, -7.0, -8.0)
LABELS = ('correct', 'lure', 'other', 'invalid')
FAMILIES = ('crt_difference', 'crt_rate', 'crt_growth')
SCORER_VERSION = 'numeric_single_answer_v1'
# Canonical time unit is seconds. Months/years have no fixed duration here.
UNITS = {'': ('number', Decimal(1)), 'dollar': ('currency_usd', Decimal(1)),
         'usd': ('currency_usd', Decimal(1)), 'cent': ('currency_usd', Decimal('.01')),
         'second': ('time', Decimal(1)), 'minute': ('time', Decimal(60)),
         'hour': ('time', Decimal(3600)), 'day': ('time', Decimal(86400)),
         'week': ('time', Decimal(604800))}
NUMBER = r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)'
PATTERN = re.compile(rf'(?P<currency>\$)?\s*(?P<number>{NUMBER})(?P<ordinal>st|nd|rd|th)?\s*(?P<unit>[a-z]+|¢)?')


def parse_numeric(text, default_unit=''):
    normalized = _normalized_text(text).replace('−', '-')
    normalized = re.sub(r'^(?:answer\s*:|the answer is)\s*', '', normalized)
    if normalized.endswith('.'):
        normalized = normalized[:-1]
    # Accept the common "day 19" ordinal form already recognized by CRT normalization.
    normalized = re.sub(r'^day\s+(' + NUMBER + r')$', r'\1 day', normalized)
    match = PATTERN.fullmatch(normalized)
    if not match:
        raise ValueError('expected one numeric answer with an optional recognized unit')
    unit = match['unit'] or ''
    unit = {'dollars': 'dollar', 'cents': 'cent', '¢': 'cent', 'c': 'cent',
            'sec': 'second', 'secs': 'second', 'min': 'minute', 'mins': 'minute',
            'hr': 'hour', 'hrs': 'hour'}.get(unit, unit.rstrip('s'))
    if match['ordinal'] and unit != 'day':
        raise ValueError('ordinal suffix requires day unit')
    if match['currency']:
        if unit not in ('', 'dollar', 'usd'):
            raise ValueError('conflicting currency and unit')
        unit = 'dollar'
    unit_inferred = bool(not unit and default_unit)
    if not unit:
        unit = default_unit
    if unit not in UNITS:
        raise ValueError(f'unrecognized unit: {unit}')
    dimension, scale = UNITS[unit]
    value = Decimal(match['number'].replace(',', '')) * scale
    if value.is_zero():
        value = abs(value)
    return dict(value=str(value.normalize()), dimension=dimension, unit=unit,
                unit_inferred=unit_inferred,
                normalized_text=normalized)


def validate_case(case):
    if case.family not in FAMILIES or not case.case_id.startswith(f'crt_fresh_v2_{case.family[4:]}_'):
        raise ValueError(f'{case.case_id}: invalid CRT ID/family')
    if not case.prompt.endswith('\nAnswer:') or re.search(r'(?m)^\s*[AB]\s*:', case.prompt):
        raise ValueError(f'{case.case_id}: expected original open-ended prompt without choices')
    correct, lure = parse_numeric(case.correct_answer), parse_numeric(case.lure_answer)
    if correct['dimension'] != lure['dimension']:
        raise ValueError(f'{case.case_id}: inconsistent reference dimensions')
    if correct['value'] == lure['value']:
        raise ValueError(f'{case.case_id}: correct and lure are semantically identical')
    return correct, lure


def classify(text, case, termination_reason='eos'):
    correct, lure = validate_case(case)
    audit = dict(scorer_version=SCORER_VERSION, correct_reference=correct, lure_reference=lure)
    if termination_reason != 'eos':
        return 'invalid', dict(audit, reason=f'unfinished generation: {termination_reason}')
    if not isinstance(text, str) or not text.strip():
        return 'invalid', dict(audit, reason='empty response')
    try:
        parsed = parse_numeric(text, default_unit=correct['unit'])
    except ValueError as exc:
        return 'invalid', dict(audit, reason=str(exc))
    for label, reference in (('correct', correct), ('lure', lure)):
        if parsed['dimension'] == reference['dimension'] and parsed['value'] == reference['value']:
            return label, dict(audit, parsed=parsed, reason=f'exact canonical {label} match')
    return 'other', dict(audit, parsed=parsed, reason='valid numeric answer differs from both references')


def validate_rows(rows, cases, max_new_tokens):
    indexed = {}
    for row in rows:
        case_id = row.get('case_id')
        if case_id not in cases or case_id in indexed:
            raise ValueError(f'unknown/duplicate case_id: {case_id}')
        case = cases[case_id]
        if row.get('family') != case.family or row.get('prompt') != case.prompt:
            raise ValueError(f'{case_id}: family/prompt mismatch')
        if not isinstance(row.get('answer'), str) or not isinstance(row.get('raw_text'), str):
            raise ValueError(f'{case_id}: missing raw response')
        ids, eos = row.get('generated_token_ids'), row.get('eos_token_ids')
        if (not isinstance(ids, list) or not isinstance(eos, list) or not eos
                or any(type(i) is not int or i < 0 for i in ids + eos)
                or row.get('generated_token_count') != len(ids)
                or not 0 < len(ids) <= max_new_tokens or row.get('max_new_tokens') != max_new_tokens):
            raise ValueError(f'{case_id}: invalid generation metadata')
        if any(i in eos for i in ids[:-1]):
            raise ValueError(f'{case_id}: tokens occur after EOS')
        has_eos = ids[-1] in eos
        reason = 'eos' if has_eos else 'max_new_tokens' if len(ids) == max_new_tokens else 'other_stop'
        if row.get('eos_reached') is not has_eos or row.get('termination_reason') != reason:
            raise ValueError(f'{case_id}: inconsistent termination metadata')
        label, audit = classify(row['answer'], case, reason)
        if row.get('label') != label or row.get('classification') != audit:
            raise ValueError(f'{case_id}: inconsistent classification')
        indexed[case_id] = row
    if set(indexed) != set(cases):
        raise ValueError('baseline/steered case sets differ from original cases')
    return indexed


def evaluate(cases, coefficients, generate, max_new_tokens):
    if tuple(coefficients) != COEFFICIENTS or max_new_tokens < 1:
        raise ValueError('expected coefficients 0/-7/-8 and positive generation budget')
    indexed = {c.case_id: c for c in cases}
    if len(cases) != 40 or len(indexed) != 40:
        raise ValueError('expected 40 unique original cases')
    for case in cases:
        validate_case(case)
    blocks, reference = [], None
    for c in coefficients:
        result = generate(cases, c)
        baseline = validate_rows(result['baseline_rows'], indexed, max_new_tokens)
        steered = validate_rows(result['steered_rows'], indexed, max_new_tokens)
        if reference is not None and baseline != reference:
            raise ValueError('baseline differs across coefficients')
        reference = baseline
        if c == 0 and baseline != steered:
            raise ValueError('coefficient zero differs from baseline')
        blocks.append(dict(coefficient=c, output_mode=MODE, max_new_tokens=max_new_tokens,
                           scorer_version=SCORER_VERSION, cases=[asdict(case) for case in cases],
                           baseline_rows=list(baseline.values()), steered_rows=list(steered.values())))
    return blocks
