"""Actual cached tokenizer test when available, plus independent failure/gating mocks."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts import validate_cross_case_tokenizer_runtime as audit
from experiments import revision_pinning as pin
from experiments import feature_specificity_cross_case as study


class TokenizerRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.plan=study.build_plan('block-1')
        self.inputs=copy.deepcopy(self.plan['case_inputs'])
        self.cases={}
        for cid,item in self.inputs.items():
            score=next(m['input_reference'] for m in self.plan['measurements'] if m['case_id']==cid)
            item['candidate_texts']={label:score[label+'_answer'] for label in ('correct','lure')}
            self.cases[cid]=SimpleNamespace(prompt=item['prompt'],correct_answer=score['correct_answer'],lure_answer=score['lure_answer'])
        inputs=self.inputs
        class FakeTokenizer:
            eos_token_id=248044;pad_token_id=248044
            def __call__(self,text,**kwargs):return dict(input_ids=next((v['prompt_token_ids'] for v in inputs.values() if v['prompt']==text),[999]))
            def encode(self,text,*,add_special_tokens):
                assert add_special_tokens is False
                return next(v['candidate_token_ids'][label] for v in inputs.values() for label,t in v['candidate_texts'].items() if text==t)
            def decode(self,ids,**kwargs):
                return next((t for v in inputs.values() for label,t in v['candidate_texts'].items() if v['candidate_token_ids'][label]==ids),'unknown')
        self.tokenizer=FakeTokenizer()

    @unittest.skipUnless(importlib.util.find_spec('transformers') and importlib.util.find_spec('tokenizers'),
                         'Use isolated tokenizer CPU environment for the real runtime test')
    def test_real_cached_tokenizer_runtime(self):
        snapshot=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots'/self.plan['study_manifest']['revisions']['tokenizer']
        if not (snapshot/'tokenizer.json').is_file():self.skipTest('Pinned tokenizer cache absent')
        result=audit.run(snapshot)
        self.assertEqual(result['status'],'verified')
        self.assertEqual(len(result['cached_tokenizer_runtime']['cases']),3)
        self.assertEqual(result['network_connection_attempts'],[])
        self.assertFalse(result['model_sae_weights_loaded']);self.assertEqual(result['model_generate_calls'],0)
        self.assertEqual(result['validation_state']['production_processor_runtime_verified'],'unverified')

    def test_prompt_comparison_and_changed_prompt_fail(self):
        result=audit.compare_inputs(self.tokenizer,self.inputs,self.cases)
        self.assertEqual(result['status'],'verified')
        self.cases[study.CASE_IDS[0]].prompt+=' changed'
        result=audit.compare_inputs(self.tokenizer,self.inputs,self.cases)
        self.assertEqual(result['status'],'mismatch')
        self.assertFalse(result['cases'][0]['checks']['prepared_prompt_matches'])

    def test_candidate_and_eos_mismatches_fail(self):
        with mock.patch.object(self.tokenizer,'encode',return_value=[999]):
            with self.assertRaises(ValueError):audit.compare_inputs(self.tokenizer,self.inputs,self.cases)
        self.tokenizer.eos_token_id=99
        result=audit.compare_inputs(self.tokenizer,self.inputs,self.cases)
        self.assertEqual(result['status'],'mismatch')
        self.assertFalse(result['cases'][0]['checks']['eos_id_matches_historical_reference'])

    def test_cli_retains_mismatch_json_and_exits_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'failed.json'
            with mock.patch.object(audit,'run',return_value=dict(status='mismatch',reason='mock mismatch')):
                with self.assertRaises(SystemExit):audit.main(['--snapshot',tmp,'--output',str(output)])
            self.assertEqual(json.loads(output.read_text())['status'],'mismatch')

    def test_cli_refuses_existing_output_before_tokenizer_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'existing.json';output.write_text('preserve')
            with mock.patch.object(audit,'run') as load:
                with self.assertRaises(FileExistsError):audit.main(['--snapshot',tmp,'--output',str(output)])
                load.assert_not_called()
            self.assertEqual(output.read_text(),'preserve')

    def test_cached_verified_cannot_override_production_processor_blocker(self):
        m=copy.deepcopy(self.plan['study_manifest']);m['pinning_validation']='verified'
        state=m['revision_validation']
        for key in pin.RUNTIME_FIELDS:state[key]='verified'
        self.assertEqual(state['cached_tokenizer_runtime_verified'],'verified')
        self.assertEqual(state['production_processor_runtime_verified'],'unverified')
        self.assertTrue(any('production_processor' in s for s in pin.revision_blockers(m)))
        with self.assertRaises(ValueError):study.require_ready(dict(study_manifest=m))

    def test_manifest_runtime_evidence_and_original_inputs(self):
        m=self.plan['study_manifest'];ref=m['cached_tokenizer_runtime_evidence'];path=audit.ROOT/ref['path']
        result=json.loads(path.read_text());pin.validate_cached_runtime_evidence(m,result)
        self.assertEqual(audit.digest(path),ref['sha256'])
        self.assertFalse(self.plan['execution_ready'])
        for c in result['cached_tokenizer_runtime']['cases']:
            self.assertEqual(c['prompt_token_ids'],self.inputs[c['case_id']]['prompt_token_ids'])
            self.assertEqual(c['candidate_token_ids'],self.inputs[c['case_id']]['candidate_token_ids'])

    def test_runtime_evidence_bad_revision_or_result_rejected(self):
        m=self.plan['study_manifest'];result=json.loads((audit.ROOT/m['cached_tokenizer_runtime_evidence']['path']).read_text())
        for field,value in [('tokenizer_revision','a'*40),('status','mismatch'),('model_generate_calls',1)]:
            bad=copy.deepcopy(result);bad[field]=value
            with self.assertRaises(ValueError):pin.validate_cached_runtime_evidence(m,bad)
        bad=copy.deepcopy(result);bad['cached_tokenizer_runtime']['cases'][0]['checks']['prompt_sha_equal']=False
        with self.assertRaises(ValueError):pin.validate_cached_runtime_evidence(m,bad)
        changed=copy.deepcopy(self.inputs);changed[study.CASE_IDS[0]]['prompt']+=' changed'
        with self.assertRaises(ValueError):pin.validate_cached_runtime_evidence(m,result,changed)


if __name__=='__main__':unittest.main()
