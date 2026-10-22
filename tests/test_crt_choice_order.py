"""CPU-only synthetic counterbalancing and real case-selection regressions."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.jobs import research_experiments as job
from mindscopex_analysis.cases import LureCase
from mindscopex_analysis.choice_order import COEFFICIENTS, ORDERS, choice_letter, evaluate, ordered_case

SPEC = importlib.util.spec_from_file_location('choice_analysis', job.ROOT / 'scripts/analyze_crt_choice_order.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def cases():
    return [LureCase(f'crt_fresh_v2_{analysis.single.FAMILIES[i%3][4:]}_{i:03}',
                     analysis.single.FAMILIES[i%3], f'Question {i}?\nAnswer:', ' 2', ' 1')
            for i in range(40)]


def generate(items, coefficient):
    baseline = [dict(case_id=c.case_id, family=c.family, answer='1' if i%2==0 else '2',
                     label='lure' if i%2==0 else 'correct') for i,c in enumerate(items)]
    after = copy.deepcopy(baseline)
    if coefficient:
        after[0].update(answer='2',label='correct')
    return dict(baseline_rows=baseline,steered_rows=after)


class ChoiceOrderTests(unittest.TestCase):
    def test_prompts_and_content(self):
        case=cases()[0]
        a,b=(ordered_case(case,o) for o in ORDERS)
        self.assertEqual(a.prompt,'Question 0?\nA: 2\nB: 1\nAnswer:')
        self.assertEqual(b.prompt,'Question 0?\nA: 1\nB: 2\nAnswer:')
        for field in ('case_id','family','correct_answer','lure_answer','note','template_id'):
            self.assertEqual(getattr(a,field),getattr(b,field))
        self.assertEqual(case.prompt,'Question 0?\nAnswer:')

    def test_mapping(self):
        for order,correct,lure in ((ORDERS[0],'A','B'),(ORDERS[1],'B','A')):
            self.assertEqual(choice_letter('2',cases()[0],order),correct)
            self.assertEqual(choice_letter('1',cases()[0],order),lure)
        with self.assertRaises(ValueError):
            choice_letter('A',cases()[0],ORDERS[0])

    def test_paired_summary_and_positions(self):
        blocks=evaluate(cases(),COEFFICIENTS,generate)
        rows=analysis.analyze(blocks,1)
        self.assertEqual(len(rows),240)
        data=analysis.summarize(rows)
        self.assertEqual(len(data['paired_case_transitions']),120)
        r=data['seed_order_summary'][2]  # coefficient -7, correct first
        self.assertEqual(r['baseline_accuracy'],.5)
        self.assertEqual(r['steered_accuracy'],.525)
        self.assertEqual(r['net_improvement_pp'],2.5)
        self.assertEqual(r['lure_to_correct'],1)
        self.assertEqual(r['correct_to_lure'],0)
        self.assertEqual(r['steered_A_rate'],.525)
        b=data['choice_position_bias'][4]
        self.assertEqual(b['steered_label_agreement'],1)
        self.assertEqual(b['steered_choice_agreement'],0)
        self.assertEqual(b['corrections_both_orders'],1)
        self.assertEqual(b['correct_first_correction_retention'],1)
        self.assertEqual(sum(r[k] for k in analysis.single.TRANSITIONS),40)

    def test_order_effect_reversal_and_regression(self):
        def reversed_effect(items, coefficient):
            result = generate(items, coefficient)
            if coefficient and '\nA: 1\n' in items[0].prompt:
                result['steered_rows'] = copy.deepcopy(result['baseline_rows'])
                result['steered_rows'][1].update(answer='1', label='lure')
            return result
        data = analysis.summarize(analysis.analyze(
            evaluate(cases(), COEFFICIENTS, reversed_effect), 1))
        row = data['choice_position_bias'][4]
        self.assertTrue(row['effect_sign_reversed'])
        self.assertEqual(row['order_effect_gap_pp'], 5)
        self.assertEqual(row['corrections_both_orders'], 0)
        self.assertEqual(row['correct_first_correction_retention'], 0)
        summary = data['seed_order_summary'][3]
        self.assertEqual(summary['correct_to_lure'], 1)
        self.assertEqual(summary['regression_rate'], .05)

    def test_undefined_rates(self):
        def all_correct(items, coefficient):
            rows = [dict(case_id=c.case_id, family=c.family, answer='2', label='correct')
                    for c in items]
            return dict(baseline_rows=rows, steered_rows=copy.deepcopy(rows))
        data = analysis.summarize(analysis.analyze(
            evaluate(cases(), COEFFICIENTS, all_correct), 1))
        self.assertTrue(all(r['correction_rate'] is None
                            for r in data['seed_order_summary']))
        self.assertTrue(all(r['correct_first_correction_retention'] is None
                            for r in data['choice_position_bias']))

    def test_invalid_results(self):
        for kind in ('mapping','label','family','missing_case','duplicate_case','missing_block',
                     'duplicate_block','content','prompt','coefficient','baseline','zero'):
            with self.subTest(kind=kind):
                blocks=evaluate(cases(),COEFFICIENTS,generate)
                b=blocks[1]
                if kind=='mapping': b['steered_rows'][0]['choice']='B'
                elif kind=='label': b['steered_rows'][0]['label']='lure'
                elif kind=='family': b['steered_rows'][0]['family']='crt_rate'
                elif kind=='missing_case': b['steered_rows'].pop()
                elif kind=='duplicate_case': b['steered_rows'][-1]=copy.deepcopy(b['steered_rows'][0])
                elif kind=='missing_block': blocks.pop()
                elif kind=='duplicate_block': blocks[-1]=copy.deepcopy(b)
                elif kind=='content': b['cases'][0]['prompt']='Changed\nAnswer:'
                elif kind=='prompt': b['prompts'][cases()[0].case_id]='Changed'
                elif kind=='coefficient': b['coefficient']=-4
                elif kind=='baseline':
                    b['baseline_rows'][0].update(answer='2',label='correct',choice='A')
                elif kind=='zero':
                    blocks[0]['steered_rows'][0].update(answer='2',label='correct',choice='A')
                with self.assertRaises(ValueError): analysis.analyze(blocks,1)

    def test_evaluator_rejects_wrong_mapping(self):
        def bad(items,c):
            result=generate(items,c)
            result['steered_rows'][0]['label']='correct' if c==0 else 'lure'
            return result
        with self.assertRaisesRegex(ValueError,'mapping'): evaluate(cases(),COEFFICIENTS,bad)
        with self.assertRaises(ValueError): evaluate(cases()[:39],COEFFICIENTS,generate)
        with self.assertRaises(ValueError): evaluate(cases(),[0,-7],generate)

    def test_actual_seed_selection_matches_old_runs(self):
        for seed in range(1,6):
            new=job.load_toml(job.ROOT / f'experiments/configs/study_choice_order_counterbalance_f30908_seed{seed}_2b.toml')
            old=job.load_toml(job.ROOT / f'experiments/configs/study_transfer_fresh_v2_negative_dose_f30908_seed{seed}_2b.toml')
            for key in ('model','data','feature'):
                self.assertEqual(new[key],old[key])
            selected=job.family_balanced_subset(job._load_splits(new)['test'],max_cases=40)
            self.assertEqual(selected,job.family_balanced_subset(job._load_splits(old)['test'],max_cases=40))
            self.assertEqual(len(selected),40)
            root=job.ROOT/'results/runs'
            paths=list(root.glob(f'*_study_transfer_fresh_v2_negative_dose_f30908_seed{seed}_2b/artifacts/behavioral/generations.json'))
            if len(paths)==1:
                saved=json.loads(paths[0].read_text())[0]['baseline_rows']
                self.assertEqual({c.case_id for c in selected},{r['case_id'] for r in saved})
            self.assertEqual({ordered_case(c,ORDERS[0]).case_id for c in selected},
                             {ordered_case(c,ORDERS[1]).case_id for c in selected})

    def test_job_routes_to_existing_steering(self):
        config=job.load_toml(job.ROOT/'experiments/configs/study_choice_order_counterbalance_f30908_seed1_2b.toml')
        def fake(model,tokenizer,items,**kwargs): return generate(items,kwargs['coefficient'])
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(job,'steer_generation_labels',side_effect=fake) as steering:
            state={'study_feature':{'feature':{'layer':5,'feature_id':30908},'sae':object()}}
            job.run_behavioral(object(),object(),{'test':cases()},config,{},Path(tmp),state)
            self.assertEqual(steering.call_count,6)
            for call in steering.call_args_list:
                self.assertEqual(call.kwargs['max_new_tokens'],16)
                self.assertEqual(call.kwargs['token_position'],'all')
                self.assertEqual(call.kwargs['output_mode'],'binary_choice')
                self.assertEqual(call.kwargs['feature_id'],30908)
            blocks=json.loads((Path(tmp)/'behavioral/generations.json').read_text())
            analysis.analyze(blocks,1)

    def test_analysis_outputs_missing_duplicate_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'runs'
            for seed in range(1,6):
                p=root/f'test_study_choice_order_counterbalance_f30908_seed{seed}_2b/artifacts/behavioral/generations.json'
                p.parent.mkdir(parents=True)
                p.write_text(json.dumps(evaluate(cases(),COEFFICIENTS,generate)))
            output=Path(tmp)/'analysis'
            analysis.run(root,output)
            self.assertEqual(len(list(output.iterdir())),7)
            before={p.name:p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError): analysis.run(root,output)
            self.assertEqual(before,{p.name:p.read_bytes() for p in output.iterdir()})
            p.unlink()
            with self.assertRaisesRegex(ValueError,'Seed 5'): analysis.run(root,Path(tmp)/'missing')
            duplicate=root/'duplicate_study_choice_order_counterbalance_f30908_seed1_2b/artifacts/behavioral/generations.json'
            duplicate.parent.mkdir(parents=True)
            duplicate.write_text('[]')
            with self.assertRaisesRegex(ValueError,'Seed 1'): analysis.run(root,Path(tmp)/'duplicate')


if __name__=='__main__': unittest.main()
