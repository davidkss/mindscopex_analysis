# Phase 5-D-3B-Prep — Pinned A100 Runtime Smoke Test 준비

## 상태·근거

시작 로컬/원격 main=`e4d7ff45defafb9e933371d3f0e4e832433de2b1`, clean. 운영 규칙 전체, Journal21~24절, 기존 Growth019 pilot·Cross-Case plan·revision report/evidence·Tokenizer runtime report와 loader/adapter를 확인했다. 마지막 완료는 CPU cached Tokenizer 검증이다. Production Processor와 pt tensor 입력, 모델/SAE 실제 로딩, 실제 A100 Hook은 미검증이다.

이번은 **CPU 구현·Mock 검증·Dry-run만** 수행했다. `execution_ready=false` 및 별도 Smoke의 `allocation_authorized=false`를 유지한다. 실제 allocation/session/remote 작업·모델/SAE 다운로드는0회다. 기존 Cross-Case Manifest의 runtime 상태·gate를 변경하지 않았다. 과거 Growth019 실제 loaded revision 미기록도 그대로다.

새 canonical Manifest: `experiments/inputs/growth019_runtime_smoke.json` schema1/결과schema4. Config: `experiments/configs/growth019_runtime_smoke_2b.toml`. 실행 helper: `experiments/runtime_smoke.py`. 기존 first_divergence_trace Job/Launcher의 archive·로그·회수 경로에 opt-in으로 연결한다.

## 고정 조건 및 예산

Model/Tokenizer SHA=`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`, SAE SHA=`132ea3697b591df9ee46d738aa1d528e3c6082f7`. 모델 Qwen/Qwen3.5-2B-Base, SAE Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50, Layer5/all. SHA는 기존 canonical Cross-Case Manifest에서 읽고 API evidence에 대조한다.

| 조건 | Feature | α | 계측/비계측 Generate |
|---|---:|---:|---:|
|Baseline|No-hook|0|2|
|Intervention|30908|8.5|2|
|Comparison|12207|8|2|

실제 생성 대상은 Growth019뿐이다. **3관측·6 Generate**이며 각 쌍은 동등성 검사다. 과거 seed/쌍/초기화 검사를 독립 반복으로 세지 않는다. 입력은 기존 Raw Prompt·후보 ID·EOS/trie를 그대로 쓴다.

추가 진단은 **Identity 모듈 Forward2회**(Feature별1회, 실제 CUDA Tensor/기존 steering factory·register_forward_hook), 별도 Qwen model.forward0회, 자동 generation 재시도0회다. generate 내부 forward 수는 토큰/cache에 따라 달라지며0이라고 주장하지 않는다. 실제 Generate 시도 수와 synthetic Hook Forward 시도 수는 실패하더라도 기록한다. validation-only 모드는0 Generate/모듈 Forward2회를 계획한다.

## 엄격한 단계·실제 근거

`execute_stages`는 새 in-process backend에서 아래5단계를 항상 순서대로 실행한다. 이전 JSON을 읽어 실행을 재개하거나 사용자가 입력한 verified 값으로 통과하지 않는다.

| 단계 | 실제 승인 후 수행할 검사 | 실패 시 금지되는 후속 동작 |
|---|---|---|
|1 files|요청 immutable SHA로 모델/Tokenizer 파일 및 SAE config/layer5 체크포인트 확보, snapshot 경로·크기·Git blob/LFS payload hash 스트리밍 대조|Processor/모델/SAE 로딩·생성|
|2 processor|실제 production loader의 processor_only 경로: AutoConfig qwen3_5→AutoProcessor. 내부 Tokenizer로 기존 세 문항 전체 ID·공백·EOS/trie 및 pt tensor IDs 대조|모델/SAE 로딩·생성|
|3 model|같은 검증 snapshot/config/Processor로 production model loader 실행, SHA 인자·config commit(존재할 때)·architecture·eval/greedy·BF16/CUDA 확인, 환경 저장|SAE 로딩·Hook·생성|
|4 sae|동일 pinned layer5 파일로 기존 SAE loader 실행 및 파일 재대조, repo/layer/encoder·decoder/bias shape·feature range·모델 hidden dim·BF16/finite 확인, 실제 Layer5 block 탐색|Hook·생성|
|5 hook|두 Feature에 대해 실제 Tensor+Identity module hook dispatch, 기대 hidden+direction 일치·원본 hidden 불변·finite norm·Hook 제거 확인|생성|

Model/Tokenizer의 현재 SHA/Repository는 같고 경로도 같아야 한다. 다른 snapshot으로 바뀌면 자동 혼합하지 않고 Processor routing 구현이 필요하다는 오류로 중단한다. 현재 선택 모델12파일과 SAE2파일 총14개 unique 파일을 검증하고 동일 Tokenizer snapshot은 재사용한다. API metadata의 모델 weights payload는4,548,221,488bytes, SAE는537,012,593bytes다(약5.085GB 합계, config/tokenizer 등 별도). **향후 files 단계에서 Processor 검사 전에 이 weights 확보가 발생할 수 있으므로 승인 범위에 다운로드·디스크·검증 I/O를 포함해야 한다.** 이번에는 받지 않았다.

무결성은 대형 payload를1MiB chunk로 읽어 메모리에 전체 weights를 복제하지 않는다. LFS pointer Git blobId와 payload SHA-256을 구분한다. actual local snapshot을 production loader에 전달하고 local_files_only=True·정확한 revision 인자를 사용한다. 모델은 use_safetensors=True로 요청한다. SAE loader도 선택적 local_files_only=True를 받아 files 단계 이후 추가 다운로드 없이 검증 캐시를 사용한다(기존 default는 유지). metadata의 cache hint나 null commit을 실제 weights revision 증명으로 쓰지 않는다. 검증 파일·snapshot 경로와 실제 loader 호출/결과를 연결해 기록한다.

Stage2의 세 문항 Tokenizer 비교는 모델 Forward가 없는 입력 검사다. 생성은019로 제한한다. cached Qwen2Tokenizer와 Production Processor 내부 Tokenizer 클래스가 달라도 새 실제 입력 비교가 성공해야 하며 클래스를 가정하지 않는다. Identity 검사는 실제 Model Layer5 전체 forward/cache의 증명이 아니므로 실제 Layer5 Hook dispatch/Norm은 생성 중에도 검증한다.

## 생성 경로 재사용·조건별 실패 중단

기존 `trace.run_model`의 모델 로딩 이후 본문만 `run_loaded_model`로 추출했다. 기존 측정 반복문·RecordingModel·capture_case·compare_uninstrumented·`_binary_choice_completion`을 재사용하며 새 decoding loop를 만들지 않았다. 기존 legacy/pilot/cross-case 호출은 기존 로딩/차단 경로를 그대로 사용한다. Smoke는 새 staged 검증이 실제 끝난 loaded context만 전달하며, global require_ready를 거짓 verified 값으로 우회하지 않는다.

Model loader에는 선택적 processor_only/prepared_config/prepared_processor/local_files_only 인자를 추가했다. default 호출 동작은 유지하고, Processor 검사가 끝나기 전에 weights를 로딩하거나 두 번 모델을 로딩하지 않도록 분리했다.

Smoke의 raw callback은 캡처 JSON을 먼저 보존한 뒤 Prompt/후보 ID·EOS·분기 관측/finite Margin·후보 완성 및 실제 activation norm을 검사한다. 여기서 실패하면 비계측 비교도 호출하지 않는다. comparison callback은 계측/비계측 token IDs/답변 및 compared/세 True를 검사하고, 실패하면 다음 조건을 실행하지 않는다. Baseline은 기존0조기 반환으로 Hook을 등록하지 않는다.

Raw/Processed Margin(Lure−Correct), baseline delta, 선택 토큰·label, tie, EOS, 계측 동등성, requested/cast vector norm·unit decoder norm·actual delta·before/after residual norm을 구분한다. 계수와 실제 변화 크기를 같다고 가정하지 않는다. 과거 +8.5/+8 수치나 Lure 답변은 성공 기준이 아니다. 일관된 새 음의 Margin/Correct도 유효하다. TF/Full은 가능한 역사적 참조만 별도 표시한다.

## 상태·실패·Artifact

할당 허가는 runtime 검증 상태와 별개다. 현재 Manifest allocation_authorized=false이므로 실제 Launcher는 세션 할당 전에, CLI는 Torch/backend 생성 전에 실패한다. CPU unit tests의 fake backend는 테스트 경계일 뿐 production에서 선택할 옵션이 없다. Config는 through 외 verified/force 등의 필드를 거부한다.

미래 명시적 GPU 승인 후에만 할당 허가를 별도 변경·리뷰할 수 있다. 그래도5단계 검증은 매번 새로 수행하며 through=generate에서도 생략하지 않는다. validation-only 결과를 verified 플래그로 주입해 생성을 재개하지 않는다. **이번에 허가를 변경하거나 Cross-Case gate를 해제하지 않았다.** 성공 상태는 같은 실행/Manifest SHA/Revision/loader 증거가 연결된 sidecar에만 기록하며 canonical runtime 상태 승격은 후속 Artifact 감사 후 별도 작업이다.

예상 Artifact: config.toml/plan.json/job_manifest.json/diagnostic.log/job.log, stage_1_files.json~stage_5_hook.json, runtime_validation.json, runtime_status.json, 조건별 capture_*.json/measurement_*.json, actual_first_divergence.json. 실패 시 runtime_failure.log·partial_file_integrity.json도 보존한다. 전 단계 성공 파일은 유지하고 부모 Job은 failed/비정상 exit를 전달한다. 기존 Launcher/bootstrap은 실패해도 partial artifact ZIP 및 로그 회수를 시도한다. validation-only 성공은 equivalence=not_run이며 생성 동등성 성공으로 기록하지 않는다.

출력/claim/Manifest binding·exclusive-create를 검사하고 기존/심볼릭 링크 artifact 경로는 거부한다. JSON allow_nan=False를 유지한다. 파일 무결성/Processor/모델 revision/SAE shape/Hook·비유한 값/생성 불일치가 나면 후속 단계로 진행하지 않는다. 실패는 성공으로 바꾸거나 자동 재실행하지 않는다.

## CPU 테스트 및 Dry-run 실제 결과

기존8 suite **110통과·실패0·skip1**을 유지했다. 신규 `test_runtime_smoke.py` **15통과·실패0·skip0**. 합계 **125통과·실패/오류0·skip1**이다. 기존 skip은 Torch 부재의 실제 Tensor Hook 검사다. 이번 새 Stage5 Tensor 코드는 작성·Mock/수학·경계 검증만 했으며 실제 CUDA Tensor 검증 완료가 아니다. 기존 CPU Tokenizer test는 분리 환경에서 실제 재실행해 통과했다. Launcher assertion3개도 별도 직접 실행했다.

새 검사는3조건/6호출·revision 전달/processor-only/model 지연·allocation 차단·모든 단계 실패 중단·실패 JSON/로그 보존·stream hash·SAE shape·revision mismatch·finite activation·첫 비교 불일치 시 다음 조건 미호출·출력 재사용 거부·이전 수치와 다른 정상 결과 허용을 확인한다. py_compile9파일 및 git diff --check도 실행했다.

실제 Dry-run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m experiments.runners.launch_colab \
 experiments/configs/growth019_runtime_smoke_2b.toml --dry-run \
 --session mindscopex-pinned-runtime-smoke --gpu A100
```

신규 `results/runs/20261030-163513_growth019_pinned_runtime_smoke_2b/` 준비만 성공했다. 출력에3관측/6 Generate·세 SHA·추가 Forward 예산·allocation_authorized=false/execution_ready=false를 확인했다. source.zip에 adapter/Smoke Manifest/Protocol B manifest/revision evidence가 포함됨을 검사했다. 실제 원격 호출0회다. Dry-run archive는 그 시점의 미커밋 준비본이며 실제 승인 후 새 Run/Archive를 만들어야 한다.

## 승인 후 명령 형태와 남은 위험 — 이번에 실행하지 않음

아래 명령은 **현재 할당 차단으로 실행되지 않으며**, 명시적 GPU 승인 및 할당 허가 변경/리뷰 후에만 사용할 형태다.

```bash
bash experiments/run_colab.sh experiments/configs/growth019_runtime_smoke_2b.toml \
 --session mindscopex-pinned-runtime-smoke --gpu A100 --exec-timeout 900 --allocation-attempts 1
```

validation-only는 config.runtime_smoke.through=validate로 별도 검토한 config에서 가능하며0 Generate지만 모델/SAE 로딩과2개 synthetic module forward는 수행한다. 다른 단계 재개용으로 JSON을 넘기는 기능은 없다. 이전 remote artifact 경로를 재사용하지 말고 새 session/output root를 사용한다.900초 exec 대기는 네트워크/할당/정리 포함 비용 상한이 아니다.

실제 Processor 의존성·native Tensor/cache·모델/SAE memory peak·GPU Source/Kernel 상태는 미검증이다.6 Generate와2개의 작은 모듈 Forward 외 별도 model Forward/재시도는 계획하지 않았으며 실제 GPU 결과는 없다. 과거 미기록 revision·일반화/specificity 한계를 유지한다. 기존 Run/Audit738파일·기존 설정과 Journal1~24절의 해시/바이트 보존을 확인하고25절만 추가했다.
