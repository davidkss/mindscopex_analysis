# Growth019 A100 0/+7/+8 Artifact comparison

Status: validated

All margins are Lure minus Correct. TF = teacher-forced; actual = observed generation.

| Coefficient | Actual raw | Actual processed | TF first | Full | Actual−TF | Raw Δ vs 0 | Answer | Label | EOS / Equivalence |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 0 | -1.375000000 | -1.374999985 | -1.375000060 | -1.460028678 | 0.000000060 | 0.000000000 | 55 days | correct | passed / passed |
| 7 | -0.375000000 | -0.375000000 | -0.500000000 | 1.573784351 | 0.125000000 | 1.000000000 | 55 days | correct | passed / passed |
| 8 | 0.000000000 | 0.000000000 | -0.125000000 | 0.733229876 | 0.125000000 | 1.375000000 | 28 days | lure | passed / passed |

## Adjacent condition transitions

[{"from_coefficient": 0, "to_coefficient": 7, "raw_margin_delta": 1.0, "from_label": "correct", "to_label": "correct", "answer_changed": false}, {"from_coefficient": 7, "to_coefficient": 8, "raw_margin_delta": 0.375, "from_label": "correct", "to_label": "lure", "answer_changed": true}]


## Environment (full values and input SHA-256 in summary.json)

| Field | Comparison |
|---|---|
| environment.model_id | matched recorded values |
| environment.model_configs.model._commit_hash | not recorded |
| environment.model_configs.text._commit_hash | not recorded |
| environment.tokenizer_name_or_path | matched recorded values |
| environment.tokenizer_revision_metadata.revision | not recorded |
| environment.tokenizer_revision_metadata._commit_hash | not recorded |
| environment.cached_config_snapshot_hint | matched recorded values |
| sae.repo_id | matched recorded values |
| sae.resolved_revision | not recorded |
| sae.dtype | matched recorded values |
| sae.device | matched recorded values |
| environment.torch | matched recorded values |
| environment.transformers | matched recorded values |
| environment.packages | matched recorded values |
| environment.requested_dtype | matched recorded values |
| environment.model_configs | matched recorded values |
| environment.parameter_groups | matched recorded values |
| environment.cuda_runtime | matched recorded values |
| environment.cudnn_version | matched recorded values |
| environment.gpus | matched recorded values |
| environment.generation_config | matched recorded values |
| environment.deterministic_algorithms | matched recorded values |
| environment.environment_variables | matched recorded values |
| config_without_coefficient | matched recorded values |
| source_hashes | matched recorded values |
| input_hashes | matched recorded values |

## Validation failures

[]

## Limits

- Model/tokenizer/SAE revisions are not recorded; matching nulls are not identity proof.
- Cache snapshot path is a hint, not proof of loaded model/tokenizer revision.
- Three single-case conditions; seed 2/4 are historical references, not new repeats.
- Equivalence compares instrumented/uninstrumented paths within each coefficient.
- +8 zero margin is a recorded tie, not Lure logit superiority.
- Teacher forcing and actual generation are separate computation paths.
