"""CPU-only five-condition pilot tests; production helper AST, no ML imports."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments import growth019_minimal_pilot as pilot
from experiments import trace_fresh_v2_first_divergence as trace
from experiments.jobs import first_divergence_trace as job
from experiments.runners.config import load_toml
from experiments.runners.launch_colab import resolve_configs
from test_trace_fresh_v2_first_divergence import Tensor, Block, _binary_choice_completion, mock_environment


class PilotTokenizer:
    eos_token_id = 99
    pad_token_id = None
    def __init__(self, plan):
        ref = plan['measurements'][0]['input_reference']
        self.candidates = {ref[f'{label}_answer']:json.loads(ref[f'{label}_token_ids']) for label in ('correct','lure')}
        manifest = json.loads((trace.ROOT/'results/analysis/protocol_b_20261029/manifest.json').read_text())
        self.prompt_ids = next(r['prompt_token_ids'] for r in manifest['prompts'] if r['case_id']==plan['case_ids'][0])
    def encode(self,text,*,add_special_tokens):
        assert not add_special_tokens
        return self.candidates[text]
    def __call__(self,text,*,return_tensors):
        return dict(input_ids=Tensor([self.prompt_ids]), attention_mask=Tensor([[1]*len(self.prompt_ids)]))


class PilotModel:
    def __init__(self, choose_lure=False):
        self.block = Block()
        self.calls = []
        self.choose_lure = choose_lure
        self.generation_config=SimpleNamespace(num_beams=1,num_return_sequences=1)
        self.training=False
    def parameters(self):
        return iter([SimpleNamespace(device='cpu')])
    def generate(self, **kwargs):
        self.calls.append((dict(kwargs),len(self.block._forward_hooks)))
        sequence=kwargs['input_ids'].clone()
        logits,scores=[],[]
        for _ in range(8):
            allowed=kwargs['prefix_allowed_tokens_fn'](0,sequence[0])
            raw=Tensor([[0.]*3000]); processed=Tensor([[float('-inf')]*3000])
            raw.dtype='mock.bfloat16'; processed.dtype='mock.float32'
            for t in allowed:
                raw[0,t]=2. if t == (17 if self.choose_lure else 20) else 1.
                processed[0,t]=raw[0,t]
            selected=max(range(3000),key=lambda t:processed[0,t])
            logits.append(raw); scores.append(processed)
            sequence=Tensor([sequence[0].tolist()+[selected]])
            if selected in kwargs['eos_token_id']: break
        if kwargs.get('return_dict_in_generate'):
            return SimpleNamespace(sequences=sequence,logits=logits,scores=scores)
        return sequence


def mock_capture(plan, choose_lure=False):
    model=PilotModel(choose_lure); tokenizer=PilotTokenizer(plan)
    ref=plan['measurements'][0]['input_reference']
    case=SimpleNamespace(case_id=plan['case_ids'][0],family='crt_growth',prompt='raw question\nAnswer:',
                         correct_answer=ref['correct_answer'],lure_answer=ref['lure_answer'])
    profile=SimpleNamespace(analysis_model_id=plan['model_id'],sae_repo_id=plan['pilot_manifest']['sae_repo_id'],block_path_template='mock.{layer}')
    calls=[]
    def factory(sae, features, *, coefficient, token_position):
        calls.append((features,coefficient,token_position))
        return lambda *_:None
    modules={
        'torch':SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True)),
        'transformers':SimpleNamespace(),
        'mindscopex_analysis.models':SimpleNamespace(default_sae_device=lambda:'mock-device',dtype_from_name=lambda x:x,
            get_qwen35_analysis_profile=lambda _:profile,load_qwen_text_generation_model=lambda *a,**k:(model,tokenizer),
            recommended_dtype_name=lambda:'bfloat16'),
        'mindscopex_analysis.lure_datasets':SimpleNamespace(load_lure_dataset=lambda _:[case]),
        'mindscopex_analysis.prompts':SimpleNamespace(instruct_lure_case=lambda c:c),
        'mindscopex_analysis.qwen_scope':SimpleNamespace(make_feature_steering_hook=factory,
            load_qwen_scope_sae=lambda *a,**k:SimpleNamespace(W_dec=SimpleNamespace(dtype='mock.bfloat16'))),
        'mindscopex_analysis.research':SimpleNamespace(_binary_choice_completion=_binary_choice_completion,
            find_decoder_block=lambda m,layer,**kwargs: model.block if layer==5 else None),
    }
    env,_=mock_environment()
    with mock.patch.dict(sys.modules,modules), mock.patch('experiments.diagnose_rate027_baseline.environment',return_value=env):
        result=trace.run_model(copy.deepcopy(plan),check_equivalence=True)
    return result,model,calls


class MinimalPilotTests(unittest.TestCase):
    def test_manifest_five_conditions_two_blocks_unique_paths_and_twenty_calls(self):
        configs=resolve_configs(trace.ROOT/'experiments/suites/pilot_growth019_minimal_2b.toml')
        self.assertEqual(len(configs),2)
        plans=[job.prepare_plan(load_toml(c)) for c in configs]
        self.assertEqual(sum(len(p['measurements']) for p in plans),10)
        self.assertEqual(sum(p['expected_generate_calls'] for p in plans),20)
        self.assertEqual(len({p['run_id'] for p in plans}),2)
        self.assertEqual(len({m['measurement_id'] for p in plans for m in p['measurements']}),10)
        for p in plans:
            self.assertEqual([(m['feature_id'],m['coefficient']) for m in p['measurements']],
                             [(None,0),(30908,7),(30908,8),(30908,8.5),(12207,8)])

    def test_historical_tf_not_attached_to_new_feature_or_decimal(self):
        p=pilot.build_plan('block-1')
        self.assertEqual([m['score_reference'] is None for m in p['measurements']],[False,False,False,True,True])
        self.assertTrue(all(m['teacher_forced_first_margin'] is None for m in p['measurements'][3:]))
        self.assertFalse(p['seed_references_are_independent_measurements'])
        self.assertEqual(p['slot_comparisons'],[])

    def test_mock_model_receives_features_decimal_and_baseline_without_hook(self):
        p=pilot.build_plan('block-1'); output,model,calls=mock_capture(p)
        pilot.validate_result(output,p)
        self.assertEqual(len(model.calls),10)
        self.assertEqual([n for _,n in model.calls],[0,0,1,1,1,1,1,1,1,1])
        self.assertEqual(calls,[([30908],7,'all')]*2+[([30908],8,'all')]*2+
                         [([30908],8.5,'all')]*2+[([12207],8,'all')]*2)
        self.assertFalse(model.block._forward_hooks)
        for m in output['measurements']:
            self.assertEqual(m['capture']['steps'][1]['raw_logits_dtype'],'mock.bfloat16')
            self.assertEqual(m['capture']['steps'][1]['processed_scores_dtype'],'mock.float32')
            self.assertFalse(m['capture']['greedy_settings']['do_sample'])
        # No requirement forces a positive margin at 8.5.
        self.assertLess(output['measurements'][3]['capture']['actual_raw_margin'],0)

    def test_eos_equivalence_and_prompt_fail_explicitly(self):
        p=pilot.build_plan('block-1'); output,_,_=mock_capture(p)
        for change in ('eos','equivalence','candidate','prompt'):
            altered=copy.deepcopy(output); c=altered['measurements'][0]['capture']
            if change=='eos': c['eos_terminated']=False
            elif change=='equivalence': c['equivalence_check']['equivalent']=False
            elif change=='candidate': c['candidate_token_ids']['correct']=[1]
            else: c['prompt_token_ids']=[1]
            with self.subTest(change=change),self.assertRaises(ValueError):
                pilot.validate_result(altered,p)

    def test_missing_revisions_unverified_and_source_inspection_is_not_execution(self):
        metadata=trace.loading_evidence(SimpleNamespace(),SimpleNamespace())
        self.assertIsNone(metadata['model_revision']['value'])
        self.assertIsNone(metadata['tokenizer_revision']['value'])
        self.assertEqual(metadata['sae_revision']['verification'],'not recorded')
        self.assertIn('unverified',metadata['exact_loaded_state'])
        evidence=trace.generation_source_evidence(PilotModel())
        self.assertIn('execution path unverified',evidence['methods']['generate']['status'])
        self.assertIn('unverified',evidence['actual_processor_order'])

    def test_block_comparison_records_disagreement_without_forcing_successful_reproduction(self):
        left,_,_=mock_capture(pilot.build_plan('block-1'))
        right,_,_=mock_capture(pilot.build_plan('block-2'),choose_lure=True)
        result=pilot.compare_blocks({'block-1':left,'block-2':right})
        self.assertEqual(len(result['rows']),5)
        self.assertTrue(all(not r['answers_equal'] for r in result['rows']))
        with self.assertRaises(ValueError):
            pilot.compare_blocks({'block-1':left})

    def test_adapter_pilot_command_and_equivalence_failure_preserves_output(self):
        p=pilot.build_plan('block-1'); output,_,_=mock_capture(p)
        config=trace.ROOT/'experiments/configs/pilot_growth019_block-1_2b.toml'
        def execute(command,**kwargs):
            self.assertEqual(command[command.index('--pilot-block')+1],'block-1')
            Path(command[-1]).write_text(json.dumps(output))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(job.subprocess,'run',side_effect=execute):
            directory=job.run(config,Path(tmp))
            self.assertEqual(json.loads((directory/'job_manifest.json').read_text())['expected_generate_calls'],10)
            with self.assertRaises(FileExistsError): job.run(config,Path(tmp))
        output['measurements'][0]['capture']['equivalence_check']['equivalent']=False
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(job.subprocess,'run',side_effect=execute):
            with self.assertRaises(ValueError): job.run(config,Path(tmp))
            directory=Path(tmp)/p['run_id']
            self.assertEqual(json.loads((directory/'job_manifest.json').read_text())['status'],'failed')
            self.assertTrue((directory/'actual_first_divergence.json').exists())

    def test_plan_cli_without_torch_and_legacy_fractional_coefficient_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.json'
            code=("import sys; from experiments.trace_fresh_v2_first_divergence import main; "
                  f"main(['--plan','--pilot-block','block-2','--output',{str(path)!r}]); "
                  "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
            r=subprocess.run([sys.executable,'-c',code],cwd=trace.ROOT,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(json.loads(path.read_text())['expected_generate_calls'],10)
            with self.assertRaises(FileExistsError): trace.main(['--plan','--pilot-block','block-2','--output',str(path)])
            with mock.patch('sys.stderr'),self.assertRaises(SystemExit):
                trace.main(['--plan','--coefficient','8.5','--output',str(Path(tmp)/'bad.json')])

    def test_original_runs_and_previous_audits_are_unchanged(self):
        from scripts.compare_growth019_first_divergence import RUNS
        paths=[p for root in list(RUNS.values())+[trace.ROOT/'research/audits']
               for p in root.rglob('*') if p.is_file()]
        before={str(p):trace.digest(p) for p in paths}
        mock_capture(pilot.build_plan('block-1'))
        self.assertEqual(before,{str(p):trace.digest(p) for p in paths})

    def test_invalid_block_and_mixed_legacy_config_rejected(self):
        with self.assertRaises(ValueError): pilot.build_plan('block-3')
        cfg=load_toml(trace.ROOT/'experiments/configs/pilot_growth019_block-1_2b.toml')
        cfg['diagnostic']={'coefficient':8}
        with self.assertRaises(ValueError): job.prepare_plan(cfg)


if __name__=='__main__':
    unittest.main()
