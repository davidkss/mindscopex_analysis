"""Counterbalanced prompts with unchanged answer-value constrained decoding."""
from dataclasses import asdict, replace

ORDERS = ('correct_first', 'lure_first')
COEFFICIENTS = (0.0, -7.0, -8.0)


LETTER_MODE = 'binary_choice_letter'
LETTER_INSTRUCTION = 'Answer with exactly one uppercase letter, A or B. Do not output the answer value or any other text.'


def letter_label(letter, order):
    if order not in ORDERS or letter not in ('A', 'B'):
        raise ValueError(f'Invalid letter/order: {letter!r}/{order!r}')
    return 'correct' if letter == ('A' if order == 'correct_first' else 'B') else 'lure'


def ordered_case(case, order, output_mode='binary_choice'):
    if order not in ORDERS:
        raise ValueError(f'Unknown choice order: {order}')
    if not case.prompt.endswith('\nAnswer:'):
        raise ValueError(f'{case.case_id}: expected Answer delimiter')
    if not case.correct_answer.strip() or not case.lure_answer.strip() or case.correct_answer.strip() == case.lure_answer.strip():
        raise ValueError(f'{case.case_id}: invalid answer pair')
    a, b = (case.correct_answer, case.lure_answer) if order == 'correct_first' else (case.lure_answer, case.correct_answer)
    prompt = case.prompt[:-len('\nAnswer:')] + f'\nA: {a.strip()}\nB: {b.strip()}\nAnswer:'
    if output_mode == LETTER_MODE:
        from mindscopex_analysis.prompts import CRT_FINAL_ANSWER_SYSTEM_PROMPT

        prefix = CRT_FINAL_ANSWER_SYSTEM_PROMPT + '\n\n'
        if prompt.startswith(prefix):
            prompt = prompt[len(prefix):]
        return replace(case, prompt=LETTER_INSTRUCTION + '\n\n' + prompt,
                       correct_answer='A' if order == 'correct_first' else 'B',
                       lure_answer='B' if order == 'correct_first' else 'A')
    if output_mode != 'binary_choice':
        raise ValueError(f'Unknown output mode: {output_mode}')
    return replace(case, prompt=prompt)


def choice_letter(answer, case, order):
    if order not in ORDERS:
        raise ValueError(f'Unknown choice order: {order}')
    mapping = {case.correct_answer.strip(): 'correct', case.lure_answer.strip(): 'lure'}
    if answer.strip() not in mapping or len(mapping) != 2:
        raise ValueError(f'{case.case_id}: invalid answer value {answer!r}')
    label = mapping[answer.strip()]
    return ('A' if label == 'correct' else 'B') if order == 'correct_first' else ('B' if label == 'correct' else 'A')


def evaluate(
    cases,
    coefficients,
    generate,
    output_mode='binary_choice',
    *,
    expected_cases=40,
    expected_coefficients=COEFFICIENTS,
):
    """Generate counterbalanced behavioral results."""
    if type(expected_cases) is not int or expected_cases < 1:
        raise ValueError('expected_cases must be a positive integer')
    if len(cases) != expected_cases or len({c.case_id for c in cases}) != expected_cases:
        raise ValueError(f'counterbalancing requires {expected_cases} unique cases')
    if tuple(coefficients) != tuple(expected_coefficients):
        raise ValueError(f'expected coefficients {tuple(expected_coefficients)}')
    results = []
    baselines = {}
    for order in ORDERS:
        ordered = [ordered_case(c, order, output_mode) for c in cases]
        for coefficient in coefficients:
            result = generate(ordered, coefficient)
            block = dict(coefficient=coefficient, choice_order=order, output_mode=output_mode,
                         response_encoding='choice_letter' if output_mode == LETTER_MODE else 'answer_value', cases=[asdict(c) for c in cases],
                         prompts={c.case_id: c.prompt for c in ordered})
            for field in ('baseline_rows', 'steered_rows'):
                indexed = {r['case_id']: r for r in result[field]}
                if len(indexed) != expected_cases or len(result[field]) != expected_cases or set(indexed) != {c.case_id for c in cases}:
                    raise ValueError(f'{order}/{coefficient}: result case IDs differ')
                rows = []
                for case in cases:
                    row = dict(indexed[case.case_id])
                    letter = row['answer'] if output_mode == LETTER_MODE else choice_letter(row['answer'], case, order)
                    letter_label(letter, order)
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
