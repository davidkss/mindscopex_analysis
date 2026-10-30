# Phase 5-D-3A — CPU Tokenizer Runtime Validation

## 검증 결과와 범위

고정 revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`의 기존 캐시 Tokenizer를 실제 CPU에서 로딩했다. Growth019·004·016의 prepared Prompt, 전체 Prompt Token IDs/SHA, 공백 보존 후보 IDs와 공통 접두/분기·EOS/trie 규칙이 모두 기존 입력과 일치했다. 실제 결과 JSON은 `tokenizer_runtime_results.json`이다(2026-10-30T06:51:35.120916+00:00).

**cached_tokenizer_runtime_verified=verified**다. 그러나 실제 Production AutoProcessor 경로는 미검증이므로 **production_processor_runtime_verified=unverified, runtime_tokenizer_validation=unverified, execution_ready=false**를 유지한다. 모델/SAE weights loading·Qwen generation·Feature 개입·GPU/Colab 세션은 수행하지 않았다. 과거 Growth019 loaded revision이나 Processor 내부 Tokenizer를 소급 확정하지 않는다.

시작 로컬/원격 main은 `fb1c15a15877bb0c7c506ac234febeca209ca42a`, clean. 운영 규칙 전체·Journal21~23절·Cross-Case plan/Manifest/입력·revision report/evidence·실제 loader를 확인했다. 마지막 완료는5-D-2 metadata/Mock 고정이며 runtime3개가 blocker였다. 과거 문서와 증거는 수정하지 않았다.

## 분리 환경과 설치 범위

현재 WSL 시스템 Python 및 프로젝트 .venv는 Python3.12.3이며 Torch/Transformers/Tokenizers/HF Hub가 없었다. 프로젝트 .venv에는 pytest9.2.0이 있었으나 수정하지 않았다. 별도 `/tmp/mindscopex-5d3a-tokenizer-env`를 만들고 설치 전 범위/버전을 보고했다.

- transformers5.20.2, tokenizers0.23.3, huggingface_hub1.33.0: 기존 A100 기록과 같은 핵심 버전.
- 필수 의존성을 포함한 wheel27개만 설치했다. 정확한27개 버전은 신규 `tokenizer_cpu_requirements.txt`, 실제 설치 환경의 전체 package 버전은 결과 JSON에 보존했다.
- pip --dry-run으로 해석한 목록을 고정한 뒤 --no-deps/--only-binary=:all:로 설치했다. Torch·CUDA·vision extras는 설치하지 않았다. safetensors/hf-xet은 필수 Python 배포판 의존성이며 모델/SAE weight 파일이 아니다.
- 기존 시스템/프로젝트 환경·HF 캐시를 변경하지 않았다. 패키지 wheel 다운로드는 수행했으며, Tokenizer 파일·model/SAE 가중치·Processor config 다운로드는 하지 않았다.

실제 환경은 Python3.12.3, Linux WSL, Transformers5.20.2/Tokenizers0.23.3/HF Hub1.33.0이며 **torch_installed=false**다. package 설치 이후 Tokenizer audit 자체는 HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/local_files_only/네트워크 socket guard 아래 실행했고 **connection attempt0**이었다. CUDA_VISIBLE_DEVICES는 빈 문자열로 실행했다.

## 실제 선택 Tokenizer 및 입력 대조

snapshot은 `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c/`다. 이전 API evidence에 대해4개 캐시 파일의 metadata hash를 재검증했다. 로딩은 `AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)`였다.

실제 클래스는 **transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer**, is_fast=true다. 캐시 tokenizer_config.json의 tokenizer_class=Qwen2Tokenizer, add_bos_token=false, eos/pad=<|endoftext|>, processor_class/auto_map 미지정과 일관된다. 모델명만으로 클래스를 Qwen3_5Tokenizer라고 부르지 않는다.

| Case | Prompt 길이 | Correct ID | Lure ID | 공통 접두 / 최초 분기 | 결과 |
|---|---:|---|---|---|---|
|Growth019|109|[220,20,20,2756]|[220,17,23,2756]|[220] / k=1|모두 일치|
|Growth004|108|[220,17,20,2756]|[220,16,18,2756]|[220] / k=1|모두 일치|
|Growth016|106|[220,19,24,2756]|[220,17,20,2756]|[220] / k=1|모두 일치|

실측 Prompt SHA는 각각:

-019: `145bbd68f373722e8ae38ae183fab16f0ba13c529f315fb824ecf1e68fa45ea2`
-004: `774e80e40a78feee7a6b5b3adb0ba21b71f98bd4c29bbaba09c564bfae836371`
-016: `e9a8a056350b1a89d845a988f34d2c6a7d50d40ae1c6df2fc96b95511c8299a6`

모든 Prompt는 기존 데이터셋의 instruct_lure_case 결과 문자열을 inputs_verified.json과 직접 대조했다. JSON에 실제/기대 Prompt 문자열·전체 IDs·SHA를 함께 저장했다. 후보는 ` 55 days`/` 28 days`, ` 25 days`/` 13 days`, ` 49 days`/` 25 days`의 **앞 공백을 그대로 유지**했다. decode도 공백 포함 원래 문자열과 같았다. 기존 입력을 새 결과에 맞춰 바꾸지 않았다.

Production helper는 Prompt에서 add_special_tokens를 생략(default True), 후보는 명시적False를 쓴다. 이번 실제 실행에서 Prompt 기본 호출과 explicit True가 같았으며 기대 IDs와도 일치했다. False로 인코딩한 Prompt IDs도 별도 기록했다. 후보는 정확히 add_special_tokens=False를 사용했다. Torch 없이 token-list를 확인한 것으로, `return_tensors='pt'` tensor 생성/디바이스 이동까지 실행한 것은 아니다.

EOS/PAD ID는 모두248044였다. EOS가 후보에 없고, 완성 후보에서 기존 allowed_at이 EOS만 허용하며 미완성 prefix에서는 EOS가 허용되지 않음을 확인했다. 후보 prefix collision 없이 k=1 뒤 서로 다른 토큰으로 분기한다. 이것은 Tokenizer metadata와 기존 후보 완성 규칙의 호환성 검증이며 실제 generate의 EOS 종료를 관측한 것이 아니다. 기존 non-letter helper는 EOS 없는 완성 후보도 반환할 수 있지만 study의 capture 검증은 정상 EOS를 별도로 요구한다. 이번에는 그 반환/생성 경로를 실행하지 않았다. 모델 generate0회, SAE 개입0회다.

## Production Processor 경로 — 계속 미검증

실제 `load_qwen_text_generation_model`은 AutoConfig의 qwen3_5 분기에서 AutoProcessor와 AutoModelForMultimodalLM을 선택한다. 이번에 그 모델 loader를 호출하지 않았다. 로컬 snapshot에는 config.json/preprocessor_config.json/video_preprocessor_config.json이 없다. 누락 파일을 임의로 다운로드하거나 대체 config를 만들지 않았다.

설치된 processing_auto.py의 정적 mapping은 qwen3_5→Qwen3VLProcessor다(파일 SHA `3f29c881f41abe7b271112eed82300c3577a01b72ba8a464467830ea66d64321`). 이는 실제 AutoProcessor를 구성하고 내부 Tokenizer를 확인한 증거가 아니다. 과거 A100 metadata의 Qwen3_5Tokenizer와 이번 standalone Qwen2Tokenizer 클래스가 다르므로, 클래스 차이의 원인·Production 선택 동등성은 **unverified**다. 이번 세 문항의 ID 일치와 클래스/전체 Production 경로 동등성을 구분한다.

Torch·추가 Processor 의존성·config 로딩은 이번 최소 환경에서 검증하지 않았다. 전체 runtime_tokenizer_validation을 완료로 처리하지 않았고 모델/SAE runtime도 미검증을 유지한다.

## 상태 갱신과 코드

신규 `scripts/validate_cross_case_tokenizer_runtime.py`는 기존 pinning helper의 비교·hash 및 기존 candidate_layout/allowed_at을 재사용해 실제 결과를 신규 경로에 기록한다. overwrite를 거부하고 Tokenizer 비교 mismatch 결과는 JSON 보존 후 실패로 종료한다. 새로운 generation loop나 Model/SAE load는 없다.

Manifest에는 cached verified/production unverified를 분리하고 runtime 결과 path/SHA를 연결했다. build_plan은 결과 파일 SHA·선택 Tokenizer revision·세 문항·실측 성공 조건 및 현재 prepared 입력과의 대응을 검증한다. stale/mismatch evidence는 거부한다. 기존 종합 pinning/runtime gate에 두 Tokenizer 단계의 명시적 검증도 추가했으며 차단을 완화하지 않았다.

| 상태 | 최종 값 |
|---|---|
|cached_tokenizer_runtime_verified|verified|
|production_processor_runtime_verified|unverified|
|runtime_tokenizer_validation|unverified|
|runtime_model_loading_validation|unverified|
|runtime_sae_loading_validation|unverified|
|pinning_validation|unverified|
|execution_ready|**false**|

## 실행·CPU 테스트·보존

실제 실행 명령:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUDA_VISIBLE_DEVICES='' \
 /tmp/mindscopex-5d3a-tokenizer-env/bin/python scripts/validate_cross_case_tokenizer_runtime.py \
 --snapshot /home/davidkss/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B-Base/snapshots/b1485b2fa6dfa1287294f269f5fb618e03d52d7c \
 --output research/audits/2026-10-30-feature-specificity-cross-case/tokenizer_runtime_results.json
```

재실행은 반드시 새 output 경로를 사용한다. 위 환경은 임시 /tmp 경로이므로 재현 시 필요하면 동일 고정 requirements로 별도 환경을 다시 만든다. 캐시가 없으면 검사를 중단하며 파일을 자동 다운로드하지 않는다.

분리 환경에서 기존7 suite(adapter21/trace35/기존비교7/precision6/pilot10/cross-case11+skip1/revision12)를 모두 재실행했다. 기존 **102통과·0실패·1skip**을 유지했다. 신규 test_tokenizer_runtime.py **8통과·0실패·0skip**(실제 Tokenizer 재실행 포함). 총 **110통과·실패/오류0·1skip**이다. skip1은 기존 실제 Tensor Hook 검사로 Torch 미설치에 따라 계속 미검증이다. Launcher 함수형 테스트3개도 별도 runpy 실행으로 통과했다.

신규 검사는 실제3문항 재실행·Prompt/후보/EOS 불일치·실패 JSON 보존/overwrite 거부·cached verified만으로 production gate를 통과할 수 없음·Manifest와 결과 SHA/ID·stale 입력 증거 거부를 확인한다. Mock과 실제 Tokenizer test를 구분하며 GPU Hook 검증을 통과로 표현하지 않는다. py_compile6파일 및 git diff --check도 통과했다.

기존 Run/Audit735파일과 HF Tokenizer 캐시·inputs_verified.json·기존 report/evidence를 보존했다. Journal1~23절은 바이트 접두 보존하고24절만 추가한다. 과거 GPU 결과·연구 기록은 수정하지 않았다.

## 남은 차단 요인

Production Processor의 실제 내부 Tokenizer와 tensor 입력 경로 확인, 누락 config/의존성 제공 범위 합의, model/SAE 실제 loading 검증이 남았다. 누락 Processor config나 weights를 이번에 다운로드하지 않았다. 전체 실행 준비를 완료로 표시하지 않으며, 향후 로딩·다운로드/GPU 비용 범위는 별도 지시·승인이 필요하다.
