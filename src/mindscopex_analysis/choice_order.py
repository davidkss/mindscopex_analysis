"""Counterbalanced prompts with unchanged answer-value constrained decoding."""
from dataclasses import asdict, replace

ORDERS = ('correct_first', 'lure_first')
COEFFICIENTS = (0.0, -7.0, -8.0)


def ordered_case(case, order):
    if order not in ORDERS:
        raise ValueError(f'Unknown choice order: {order}')
    if not case.prompt.endswith('\nAnswer:'):
        raise ValueError(f'{case.case_id}: expected Answer delimiter')
    if not case.correct_answer.strip() or not case.lure_answer.strip() or case.correct_answer.strip() == case.lure_answer.strip():
        raise ValueError(f'{case.case_id}: invalid answer pair')
    a, b = (case.correct_answer, case.lure_answer) if order == 'correct_first' else (case.lure_answer, case.correct_answer)
    return replace(case, prompt=case.prompt[:-len('\nAnswer:')] +
                   f'\nA: {a.strip()}\nB: {b.strip()}\nAnswer:')


def choice_letter(answer, case, order):
    if order not in ORDERS:
        raise ValueError(f'Unknown choice order: {order}')
    mapping = {case.correct_answer.strip(): 'correct', case.lure_answer.strip(): 'lure'}
    if answer.strip() not in mapping or len(mapping) != 2:
        raise ValueError(f'{case.case_id}: invalid answer value {answer!r}')
    label = mapping[answer.strip()]
    return ('A' if label == 'correct' else 'B') if order == 'correct_first' else ('B' if label == 'correct' else 'A')


def evaluate(cases, coefficients, generate):
    """generate(cases, coefficient) is the existing steering evaluator."""
    if len(cases) != 40 or len({c.case_id for c in cases}) != 40:
        raise ValueError('counterbalancing requires 40 unique cases')
    if tuple(coefficients) != COEFFICIENTS:
        raise ValueError(f'expected coefficients {COEFFICIENTS}')
    results = []
    baselines = {}
    for order in ORDERS:
        ordered = [ordered_case(c, order) for c in cases]
        for coefficient in coefficients:
            result = generate(ordered, coefficient)
            block = dict(coefficient=coefficient, choice_order=order, output_mode='binary_choice',
                         response_encoding='answer_value', cases=[asdict(c) for c in cases],
                         prompts={c.case_id: c.prompt for c in ordered})
            for field in ('baseline_rows', 'steered_rows'):
                indexed = {r['case_id']: r for r in result[field]}
                if len(indexed) != 40 or len(result[field]) != 40 or set(indexed) != {c.case_id for c in cases}:
                    raise ValueError(f'{order}/{coefficient}: result case IDs differ')
                rows = []
                for case in cases:
                    row = dict(indexed[case.case_id])
                    letter = choice_letter(row['answer'], case, order)
                    expected = 'correct' if letter == ('A' if order == 'correct_first' else 'B') else 'lure'
                    if row['label'] != expected or row['family'] != case.family:
                        raise ValueError(f'{case.case_id}: wrong label/family mapping')
                    rows.append(dict(row, choice=letter))
                block[field] = rows
            baseline = {r['case_id']: r for r in block['baseline_rows']}
            steered = {r['case_id']: r for r in block['steered_rows']}
            if order in baselines and baselines[order] != baseline:
                raise ValueError(f'{order}: baseline differs across coefficients')
            baselines[order] = baseline
            if coefficient == 0 and baseline != steered:
                raise ValueError(f'{order}: coefficient zero differs from baseline')
            results.append(block)
    return results
