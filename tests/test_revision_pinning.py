"""Metadata/Mock revision checks; no network, GPU, model or SAE weights."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments import feature_specificity_cross_case as study
from experiments import revision_pinning as pin
from experiments import trace_fresh_v2_first_divergence as trace
from experiments.jobs import first_divergence_trace as job
from experiments.runners.config import load_toml
from test_feature_specificity_cross_case import extracted, capture_fixture


class RevisionPinningTests(unittest.TestCase):
    def setUp(self):
        self.plan=study.build_plan('block-1')
        self.revisions=self.plan['study_manifest']['revisions']
        self.inputs=copy.deepcopy(self.plan['case_inputs'])
        for cid,evidence in self.inputs.items():
            row=next(m['input_reference'] for m in self.plan['measurements'] if m['case_id']==cid)
            evidence['candidate_texts']={label:row[label+'_answer'] for label in ('correct','lure')}

    def ready_fixture(self):
        plan=copy.deepcopy(self.plan);manifest=plan['study_manifest']
        manifest['pinning_validation']='verified'
        manifest['revision_validation']=dict(validated_revisions=dict(manifest['revisions']),
            revision_metadata_verified=True,revision_loader_forwarding_verified=True,
            runtime_tokenizer_validation='verified',runtime_model_loading_validation='verified',runtime_sae_loading_validation='verified')
        return plan

    def test_exact_sha_format_and_bad_revisions_rejected(self):
        pin.validate_revision_ids(self.revisions)
        for value in (None,True,123,'main','a'*39,'a'*41,'g'*40,'A'*40,'../main','a'*40+'\n'):
            bad=dict(self.revisions,model=value)
            with self.assertRaises(ValueError):pin.validate_revision_ids(bad)
        with self.assertRaises(ValueError):pin.validate_revision_ids(dict(model='a'*40))
        self.assertEqual(self.revisions['model'],'b1485b2fa6dfa1287294f269f5fb618e03d52d7c')
        self.assertEqual(self.revisions['sae'],'132ea3697b591df9ee46d738aa1d528e3c6082f7')

    def test_metadata_and_mock_status_cannot_enable_execution(self):
        self.assertFalse(self.plan['execution_ready'])
        changed=self.ready_fixture();study.require_ready(changed)  # Synthetic status fixture only.
        for key in pin.RUNTIME_FIELDS:
            for status in ('unverified','mismatch','failed',True,None):
                bad=copy.deepcopy(changed);bad['study_manifest']['revision_validation'][key]=status
                with self.assertRaisesRegex(ValueError,key):study.require_ready(bad)
        for key in ('revision_metadata_verified','revision_loader_forwarding_verified'):
            bad=copy.deepcopy(changed);bad['study_manifest']['revision_validation'][key]=False
            with self.assertRaises(ValueError):study.require_ready(bad)
        bad=copy.deepcopy(changed);bad['study_manifest']['revisions']['model']='c'*40
        with self.assertRaisesRegex(ValueError,'stale'):study.require_ready(bad)
        bad=copy.deepcopy(changed);bad['study_manifest'].pop('revision_validation')
        with self.assertRaises(ValueError):study.require_ready(bad)

    def test_actual_sha_forwarded_to_config_model_processor_and_tokenizer(self):
        calls=[];kind=['qwen3_5']
        def auto(label):
            def load(repo,**kwargs):
                calls.append((label,repo,kwargs))
                return SimpleNamespace(model_type=kind[0],eval=lambda:None)
            return SimpleNamespace(from_pretrained=load)
        names=('AutoConfig','AutoModelForCausalLM','AutoTokenizer','AutoModelForMultimodalLM','AutoProcessor')
        fake=SimpleNamespace(**{name:auto(name) for name in names})
        fn=extracted(trace.ROOT/'src/mindscopex_analysis/models.py','load_qwen_text_generation_model',dict(dtype_from_name=lambda _:None))
        for model_type,token_loader,model_loader in [('qwen3_5','AutoProcessor','AutoModelForMultimodalLM'),('qwen2','AutoTokenizer','AutoModelForCausalLM')]:
            kind[0]=model_type;calls.clear()
            with mock.patch.dict(sys.modules,{'transformers':fake}):
                fn(self.plan['model_id'],revision=self.revisions['model'],tokenizer_revision=self.revisions['tokenizer'])
            self.assertEqual([c[0] for c in calls],['AutoConfig',token_loader,model_loader])
            self.assertEqual([c[2]['revision'] for c in calls],[self.revisions['model'],self.revisions['tokenizer'],self.revisions['model']])
            self.assertTrue(all(c[1]==self.plan['model_id'] for c in calls))
        calls.clear()
        with mock.patch.dict(sys.modules,{'transformers':fake}):fn('mock',revision='a'*40,tokenizer_revision='b'*40)
        self.assertEqual([c[2]['revision'] for c in calls],['a'*40,'b'*40,'a'*40])

    def test_sae_exact_sha_layer5_and_default_compatibility(self):
        downloads=[]
        fn=extracted(trace.ROOT/'src/mindscopex_analysis/qwen_scope.py','load_qwen_scope_sae',dict(
            hf_hub_download=lambda **k:downloads.append(k) or 'mock',torch=SimpleNamespace(load=lambda *a,**k:dict(W_enc=1,W_dec=2,b_enc=3,b_dec=4)),
            QwenScopeSAE=lambda **k:SimpleNamespace(to=lambda **k:'loaded'),infer_top_k_from_repo=lambda _:50))
        repo=self.plan['study_manifest']['sae_repo_id']
        self.assertEqual(fn(repo,5,revision=self.revisions['sae']),'loaded')
        self.assertEqual(downloads[-1]['repo_id'],repo);self.assertEqual(downloads[-1]['filename'],'layer5.sae.pt')
        self.assertEqual(downloads[-1]['revision'],self.revisions['sae'])
        fn(repo,5);self.assertNotIn('revision',downloads[-1])

    def test_default_model_loader_omits_revision_and_fallback_uses_model_pin(self):
        calls=[]
        class Auto:
            @staticmethod
            def from_pretrained(repo,**kwargs):calls.append(kwargs);return SimpleNamespace(model_type='qwen3_5',eval=lambda:None)
        fake=SimpleNamespace(**{name:Auto for name in ('AutoConfig','AutoModelForCausalLM','AutoTokenizer','AutoModelForMultimodalLM','AutoProcessor')})
        fn=extracted(trace.ROOT/'src/mindscopex_analysis/models.py','load_qwen_text_generation_model',dict(dtype_from_name=lambda _:None))
        with mock.patch.dict(sys.modules,{'transformers':fake}):fn('mock')
        self.assertTrue(all('revision' not in c for c in calls));calls.clear()
        with mock.patch.dict(sys.modules,{'transformers':fake}):fn('mock',revision=self.revisions['model'])
        self.assertTrue(all(c['revision']==self.revisions['model'] for c in calls))

    def test_configs_and_trace_pass_selected_manifest_revisions(self):
        for block in ('block-1','block-2'):
            config=load_toml(trace.ROOT/f'experiments/configs/cross_case_specificity_{block}_2b.toml')
            plan=job.prepare_plan(config)
            self.assertEqual(plan['study_manifest']['revisions'],self.revisions)
        # Reuse existing fake model/production helper fixture; no real load or generation.
        _,_,_,loads=capture_fixture(self.plan)
        self.assertEqual(loads[0][1]['revision'],self.revisions['model'])
        self.assertEqual(loads[0][1]['tokenizer_revision'],self.revisions['tokenizer'])
        self.assertEqual(loads[1][1]['revision'],self.revisions['sae'])

    def test_git_and_lfs_cache_hashes_not_confused(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot=Path(tmp);files=[]
            for name in pin.TOKENIZER_FILES:
                data=name.encode();(snapshot/name).write_bytes(data)
                item=dict(rfilename=name,size=len(data),blobId=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest())
                if name=='tokenizer.json':item.update(blobId='a'*40,lfs=dict(sha256=hashlib.sha256(data).hexdigest(),size=len(data)))
                files.append(item)
            evidence=pin.inspect_tokenizer_cache(snapshot,files)
            self.assertTrue(evidence['all_tokenizer_files_match'])
            self.assertEqual(evidence['files'][0]['hash_kind'],'lfs_sha256')
            (snapshot/'tokenizer.json').write_bytes(b'changed')
            self.assertFalse(pin.inspect_tokenizer_cache(snapshot,files)['all_tokenizer_files_match'])

    def test_mock_tokenizer_comparison_and_mismatch_remain_blocked(self):
        inputs=self.inputs
        class Tokenizer:
            def __call__(self,text):return dict(input_ids=next(v['prompt_token_ids'] for v in inputs.values() if v['prompt']==text))
            def encode(self,text,*,add_special_tokens):
                assert add_special_tokens is False
                return next(v['candidate_token_ids'][label] for v in inputs.values() for label,t in v['candidate_texts'].items() if t==text)
        tokenizer=Tokenizer();self.assertEqual(pin.validate_tokenizer_inputs(tokenizer,inputs)['status'],'verified')
        tokenizer.encode=lambda *a,**k:[999]
        result=pin.validate_tokenizer_inputs(tokenizer,inputs);self.assertEqual(result['status'],'mismatch')
        plan=self.ready_fixture();plan['study_manifest']['revision_validation']['runtime_tokenizer_validation']=result['status']
        with self.assertRaisesRegex(ValueError,'mismatch'):trace.run_model(plan,check_equivalence=True)

    def test_optional_tokenizer_never_downloads_or_loads_weights(self):
        with mock.patch.object(pin.importlib.util,'find_spec',return_value=None):
            result=pin.validate_cached_tokenizer('unused',dict(all_tokenizer_files_match=True),self.inputs)
            self.assertEqual(result['status'],'unverified')
        recorded=[]
        class Auto:
            @staticmethod
            def from_pretrained(path,**kwargs):recorded.append((path,kwargs));raise OSError('mock missing config')
        with mock.patch.object(pin.importlib.util,'find_spec',return_value=object()),mock.patch.dict(sys.modules,{'transformers':SimpleNamespace(AutoTokenizer=Auto)}):
            result=pin.validate_cached_tokenizer('/mock/snapshot',dict(all_tokenizer_files_match=True),self.inputs)
        self.assertEqual(result['status'],'unverified')
        self.assertEqual(recorded,[('/mock/snapshot',dict(local_files_only=True,trust_remote_code=False))])

    def test_saved_api_metadata_and_manifest_evidence_match(self):
        evidence=json.loads((trace.ROOT/self.plan['study_manifest']['revision_evidence_path']).read_text())
        pin.validate_revision_evidence(self.plan['study_manifest'],evidence)
        self.assertEqual(evidence['queries']['model']['sha'],evidence['queries']['tokenizer']['sha'])
        self.assertNotEqual(evidence['queries']['model']['queried_at_utc'],evidence['queries']['tokenizer']['queried_at_utc'])
        bad=copy.deepcopy(evidence);bad['queries']['sae']['card_metadata']['base_model']='other/model'
        with self.assertRaises(ValueError):pin.validate_revision_evidence(self.plan['study_manifest'],bad)
        bad=copy.deepcopy(evidence);bad['selected_revisions']['model']='a'*40
        with self.assertRaises(ValueError):pin.validate_revision_evidence(self.plan['study_manifest'],bad)
        bad=copy.deepcopy(self.plan['study_manifest']);bad['revisions']['tokenizer']='a'*40
        with self.assertRaises(ValueError):pin.validate_revision_evidence(bad,evidence)

    def test_saved_input_and_original_artifact_bytes_unchanged_by_preflight(self):
        paths=[trace.ROOT/'research/audits/2026-10-30-feature-specificity-cross-case/inputs_verified.json',
               trace.ROOT/'research/audits/2026-10-30-growth019-minimal-pilot/recomputed_comparison.json']
        paths += list((trace.ROOT/'results/runs/20261030-145746_pilot_growth019_block-1_2b').glob('artifacts/*.json'))
        before={p:p.read_bytes() for p in paths}
        study.build_plan('block-1');study.build_plan('block-2')
        self.assertTrue(all(p.read_bytes()==data for p,data in before.items()))

    def test_no_colab_allocation_when_runtime_validation_unverified(self):
        from experiments.runners import launch_colab as launcher
        config=trace.ROOT/'experiments/configs/cross_case_specificity_block-1_2b.toml'
        with mock.patch.object(launcher,'ensure_session') as allocate,mock.patch.object(launcher,'run_cmd') as remote:
            with self.assertRaisesRegex(ValueError,'runtime_tokenizer_validation'):
                launcher.launch_one(config,root=trace.ROOT,session='unused',keep_session=False,default_gpu=None,default_tpu=None,exec_timeout=900,dry_run=False,allocation_attempts=1,allocation_retry_delay=0)
            allocate.assert_not_called();remote.assert_not_called()


if __name__=='__main__':unittest.main()
