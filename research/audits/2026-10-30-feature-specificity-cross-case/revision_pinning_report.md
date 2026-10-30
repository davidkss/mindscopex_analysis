# Phase 5-D-2 — Model / Tokenizer / SAE Revision Pinning

## 결과 및 적용 범위

시작 로컬/원격 main HEAD는 `fad3765e6590cf71ad11ab1123342149d4f94ed1`, 작업 트리는 clean이었다. 운영 규칙 전체·최신 Journal21~22절·Cross-Case plan/Manifest/helper/trace·실제 Model/Tokenizer/SAE loader·기존 Growth019 Audit를 확인했다. 마지막 완료는5-D-1 CPU 준비이며 정확한 revision과 실제 loader/Tokenizer 검증이 blocker였다.

이번에 **새 실험에 사용할 정확한 Commit SHA를 API 메타데이터로 선정하고 Manifest에 고정**했다. Revision 전달은 CPU Mock으로 검증했다. 실제 Tokenizer/모델/SAE 로딩은 검증하지 않아 **execution_ready=false**를 유지한다. 과거 Growth019 실제 loaded revision과 동일했다고 주장하지 않는다. GPU·Colab 세션 생성·모델/SAE/Tokenizer 파일 다운로드·패키지 설치는 수행하지 않았다. 네트워크 사용은 HF API JSON 메타데이터 및 GitHub HEAD 조회뿐이다.

## 조회 근거와 고정 SHA

| 역할 | Repository | main 조회 시각(UTC) | 선택 SHA |
|---|---|---|---|
| Model | Qwen/Qwen3.5-2B-Base |2026-10-30T06:31:35.299899+00:00|`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`|
| Tokenizer | Qwen/Qwen3.5-2B-Base |2026-10-30T06:31:35.626323+00:00|`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`|
| SAE | Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50 |2026-10-30T06:31:35.878634+00:00|`132ea3697b591df9ee46d738aa1d528e3c6082f7`|

Model/Tokenizer는 역할별 별도 요청으로 같은 SHA와 파일 목록을 확인했다. 동일 Repository에 모델과 필요한 Tokenizer 파일이 있어 새 실험에서 같은 immutable revision을 지정할 수 있다. Tokenizer 실행 호환성을 별도로 증명한 것은 아니다.

초기 network sandbox의 요청은 DNS 오류였다. 이후 허용된 metadata-only GET을 수행해 HTTP200을 받았다. 각 역할은 `/api/models/{repo}/revision/main?blobs=true` 조회 후 아래 immutable SHA 경로도 재조회해 SHA·파일 목록의 일치를 확인했다. immutable 재조회 시각은 Model06:32:50.993572, Tokenizer06:32:51.313070, SAE06:32:51.569973 UTC다.

- [Model/Tokenizer immutable API metadata](https://huggingface.co/api/models/Qwen/Qwen3.5-2B-Base/revision/b1485b2fa6dfa1287294f269f5fb618e03d52d7c?blobs=true)
- [SAE immutable API metadata](https://huggingface.co/api/models/Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50/revision/132ea3697b591df9ee46d738aa1d528e3c6082f7?blobs=true)

API의 선택 필드·전체 sibling 파일 목록·Git blob/LFS metadata·시각/URL/HTTP 상태는 신규 `revision_evidence.json`에 저장했다. main은 이후 변경될 수 있으므로 새 실험에서는 위 SHA를 사용한다. 모델 `_commit_hash`가 미기록이었던 과거 Run의 cache hint가 같은 문자열이어도 과거 weights/tokenizer revision 증명이 되지 않는다.

## 파일 목록·SAE 대응 근거

Model과 Tokenizer의 동일 revision은12개 파일을 열거한다. 실제 목록은 evidence JSON이 정본이다. 필요한 설정/입력 파일로 `config.json`, `model.safetensors.index.json`, `preprocessor_config.json`, `video_preprocessor_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`가 존재한다. weights shard도 목록에 있으나 다운로드/로딩하지 않았다. 파일 목록 존재 확인과 config 내용을 실제 loader로 읽은 검증을 구분한다. API config 요약에는 `model_type=qwen3_5`, architecture `Qwen3_5ForConditionalGeneration`이 기록돼 있다.

SAE revision은29개 파일을 열거하며 config.json 및 layer0~23.sae.pt가 있다. API cardData.base_model은 **Qwen/Qwen3.5-2B-Base**, config 요약 model_type은 topk_sae다. 기존 loader는 layer5.sae.pt를 요청한다. 해당 파일의 API metadata는 다음과 같다.

- 크기537,012,593 bytes
- LFS payload SHA-256 `8d65e489c33a9de48b59d8c8c4456758bb21502c19f5d1646d9d18d66d16482f`
- Git LFS pointer blobId `990290214428ab416a904a46180c804958fb45de`

이것은 Repository가 Base 모델·Layer5 파일을 제공한다는 metadata 근거다. SAE 훈련에 사용한 **정확한 Base 모델 commit**, 실제 텐서 shape/dtype와 새 모델의 호환성, checkpoint 로딩 성공을 이번에 입증하지 않았다. 가중치는 내려받지 않았다.

## 로컬 캐시 및 Tokenizer 검사

기본 HF Hub cache를 읽기 전용으로 조사했다. HF_HOME/HF_HUB_CACHE/TRANSFORMERS_CACHE/XDG_CACHE_HOME override는 없었다. `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c/`에 Tokenizer4파일이 이미 있었다.

| 파일 | bytes | 대조 방식 | 결과 |
|---|---:|---|---|
|tokenizer.json|12,807,196|LFS payload SHA-256|API metadata와 일치|
|tokenizer_config.json|16,712|Git blob SHA-1|일치|
|vocab.json|6,722,759|Git blob SHA-1|일치|
|merges.txt|3,353,259|Git blob SHA-1|일치|

LFS의 API blobId는 pointer의 Git hash이므로 payload Git hash와 비교하지 않는다. tokenizer.json payload SHA는 `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`로 일치했다. 전체 SHA/크기/경로는 evidence에 있다. 캐시를 수정하거나 새 파일을 받지 않았다.

로컬에는 Torch/Transformers/Tokenizers/Huggingface Hub 라이브러리가 없다. 따라서 **실제 Tokenizer 재토큰화는0회, runtime_tokenizer_validation=unverified**다. 모델 config/processor config/index 및 weights shard도 이 snapshot에 없고 SAE cache도 없다. 파일 해시 일치로 runtime validation을 대체하지 않는다.

기존 inputs_verified.json의 Growth019/004/016 Prompt SHA와 후보 IDs는 수정하지 않았다. 신규 helper의 비교 함수는 같은 기본 Prompt encoding과 whitespace 보존·add_special_tokens=False 후보 encoding으로 세 문항의 실제 ID를 대조하도록 준비했다. 이 함수는 Mock으로만 검증했다. 선택적 cached-tokenizer 검사는 파일 해시가 모두 맞고 기존 라이브러리가 있을 때만 `AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True, trust_remote_code=False)`를 사용한다. 이번에는 라이브러리 부재를 보고하고 종료했다. 실제 production AutoProcessor/tokenizer 경로의 로딩은 별도로 미검증이다.

## Loader·Config·차단 로직

5-D-1의 loader 전달 구현을 유지하고 실제 선택 SHA를 넣어 AST/Mock 경계 검사를 수행했다.

| 경계 | 검증 범위 |
|---|---|
|AutoConfig|model SHA 전달|
|Multimodal/Causal Model Loader|model SHA 전달, 실제 weights load는 안 함|
|AutoProcessor/AutoTokenizer|각 분기의 tokenizer SHA 전달; model SHA와 다르게 지정하는 synthetic test도 수행|
|Tokenizer revision 생략|model revision fallback 확인|
|모든 revision 생략|기존 호출에 revision keyword가 추가되지 않는 하위 호환성 확인|
|SAE hf_hub_download|SAE SHA 및 filename=layer5.sae.pt 전달; 실제 download 함수는 Mock|
|Cross-Case config 2개|별도 SHA 복제 없이 canonical Manifest의 같은 세 SHA 사용|
|trace.run_model|기존 fake model/helper fixture에서 model/tokenizer/SAE loader 경계에 정확한 SHA 전달|
|Launcher|실제 runtime 미검증이면 ensure_session/remote 호출 전에 실패|

신규 `experiments/revision_pinning.py`가 SHA 형식·API evidence/Manifest 대응·캐시 hash·Tokenizer 입력 비교와 readiness blocker를 담당한다. build_plan은 고정 evidence 경로와 파일 SHA를 기록하고 선정 SHA·repo/file metadata를 evidence에 대조한다. 잘못된 SHA·누락 파일·다른 repo/SAE Base metadata는 거부한다. validation 상태는 검증한 revision mapping에 묶어 stale 상태 전용을 방지한다.

기존 pinning_validation=verified 요구는 **유지**한다. 여기에 metadata/forwarding 검증과 세 runtime 상태가 모두 verified인지도 요구한다. UI의 execution_ready를 바꾸거나 metadata/Mock 성공만으로 통과할 수 없다. Tokenizer mismatch/failed/unverified는 차단한다. 모델 생성 경로·Hook·84호출 예산·기존 Growth019 결과 스키마는 변경하지 않았다.

## 최종 상태

| 상태 | 값 | 의미 |
|---|---|---|
|revision_metadata_verified|true|역할별 HF API main/immutable 조회 및 목록 대조|
|revision_loader_forwarding_verified|true|**CPU Mock 경계 전달**만 확인|
|runtime_tokenizer_validation|unverified|파일은 일치하나 라이브러리 부재로 재토큰화 미실행|
|runtime_model_loading_validation|unverified|실제 weights 미로딩|
|runtime_sae_loading_validation|unverified|실제 SAE checkpoint 미로딩|
|pinning_validation|unverified|기존 종합 gate 유지|
|execution_ready|**false**|GPU/원격 실행 차단 유지|

## CPU 검증 및 보존

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 환경에서 기존6 suite와 신규 revision suite를 실행했다.

| suite | 통과 | 실패/오류 | skip |
|---|---:|---:|---:|
|test_first_divergence_trace_job.py|21|0|0|
|test_trace_fresh_v2_first_divergence.py|35|0|0|
|test_compare_growth019_first_divergence.py|7|0|0|
|test_probe_growth019_precision.py|6|0|0|
|test_growth019_minimal_pilot.py|10|0|0|
|test_feature_specificity_cross_case.py|11|0|1|
|test_revision_pinning.py|12|0|0|
|합계|**102**|**0**|**1**|

신규12개는 잘못된 SHA/증거·stale revision·runtime 미검증/Tokenizer mismatch 차단, 정확한 SHA 전달·하위 호환, Config/Manifest 일치, Git/LFS hash 구분, local_files_only, Artifact 불변성 및 Colab 미호출을 확인한다. 기존 Tensor Hook test skip1은 계속 미검증이다. 기존 pytest 함수형 Launcher 검사3개는 runpy로 별도 직접 실행해 통과했다(102개와 별도). pytest/실제 Tensor 전체 suite를 통과했다고 표현하지 않는다.

실제 실행 방식은 unittest discover loader/runner의 패턴별 집계이며 동등한 재현 명령은 다음과 같다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m unittest discover -s tests -p 'test_revision_pinning.py' -v
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 experiments/trace_fresh_v2_first_divergence.py \
 --plan --cross-case-block block-1 --output <새경로.json>
python3 -m py_compile experiments/revision_pinning.py experiments/feature_specificity_cross_case.py \
 experiments/trace_fresh_v2_first_divergence.py tests/test_revision_pinning.py \
 tests/test_feature_specificity_cross_case.py src/mindscopex_analysis/models.py src/mindscopex_analysis/qwen_scope.py
git diff --check
```

Plan-only도21조건/42예상호출·execution_ready=false와 세 runtime blocker를 확인했다. 실제 generate 호출은0이다. 기존 Run/Audit733파일과 Journal1~22절의 바이트를 보존한다. 이전 inputs_verified.json 및 plan.md를 수정하지 않고 이번 report/evidence/Journal23절로 진행 상태를 추가했다.

## 후속 검증 제안 — 이번에 실행하지 않음

1. 먼저 이미 라이브러리가 있는 CPU 환경에서 **현재 캐시 Tokenizer 파일만** 사용해 세 문항을 재토큰화한다. production AutoProcessor에 필요한 config 파일 부재도 별도 확인해야 한다. 필요한 작은 config/라이브러리 제공은 범위를 따로 정한다. 저장 ID와 다르면 원인을 조사하고 실행 차단을 유지한다.
2. model/SAE loading 검증에는 현재 캐시에 없는 weights가 필요하다. API 파일 목록의 shard 크기와 SAE layer5 약512MiB를 근거로 전송/디스크/RAM 범위를 먼저 계산하고, **가중치 다운로드와 로딩에 대한 별도 지시**를 받은 뒤 검증한다. 이번에는 다운로드/weights load/forward를 하지 않았다.
3. 정확한 immutable snapshot/file hash·로드 class/dtype/device·SAE shape와 모델 block 차원 대응을 기록한다. CPU loading만으로 BF16 A100 runtime 호환성까지 검증됐다고 주장하지 않는다.
4. 실제 검증 근거를 revision별로 기록·리뷰한 뒤 runtime 상태와 기존 pinning_validation을 갱신한다. Cross-Case84 generate 및 GPU/Colab 실행은 그 이후에도 별도 명시적 승인이 필요하다.
