# MindScopeX 연구 진행 현황

**기준일:** 2026년 10월 30일

**주제:** Growth019 실제 생성 First-Divergence 관측과 Feature Specificity / Cross-Case 검증 준비

**작성 시작 HEAD:** `bdac1d518c4fe5731f6a7703e9db811174aacc5e` (main, 작업 트리 clean)

10월 29일에는 Teacher-forced First-Divergence와 Behavioral 응답의 일치를 분석했다. 30일에는 **Growth019 실제 A100 생성에서 개입 계수에 따른 Margin·응답 전환을 관측하고 두 블록의 Artifact를 감사**했다. Feature 12207에서도 Lure가 관측되어 Feature 30908의 고유성은 아직 확증되지 않았다.

그 후 새 실험용 Revision 고정과 CPU Tokenizer 검증을 진행했으나, **새 고정 Revision의 Runtime Validation-only 실행은 TLS 오류로 GPU 할당 전에 중단**됐다. 현재 `execution_ready=false`다. 이 상태는 앞선 Growth019 GPU Pilot이 실행되지 않았다는 뜻이 아니다. 이 보고서 작성에서는 기존 파일 조회·대조와 신규 Markdown 작성만 수행했다.

## 0. 핵심 용어 사전

| 용어 | 의미 | MindScopeX에서의 사용 |
|---|---|---|
| CRT / Hagendorff CRT | 인지 반성 문제 / 초기 탐색 문항군 | 직관적 오답과 관련된 후보 Feature 탐색 |
| CRT Fresh v2 | 탐색 문항과 구분되는 후속 CRT 문항군 | 행동 전이 및 신규 사례 검증; 현재 비교 대상은 Growth 유형 |
| Correct / Lure | 정답 / 직관적으로 매력적인 오답 | Growth019는 `55 days` / `28 days` |
| Layer / Residual Stream | 모델 내부 계산 층 / 층 사이에 전달되는 hidden representation | Layer 5 출력에서 SAE 방향 개입 |
| SAE Feature | Sparse Autoencoder의 특징 방향·활성 | 주 대상 30908과 비교 Feature 12207 |
| Activation / Intervention | 내부 표현의 활성 / 인위적 변경 | residual에 계수 α를 곱한 SAE decoder 방향을 가산 |
| Coefficient / Baseline | 개입 계수 / 무개입 기준 | α=7,8,8.5 및 Hook을 등록하지 않는 α=0 |
| Behavioral | 실제 생성된 응답의 관찰 | 후보로 제한된 Greedy 생성의 Correct/Lure 판정 |
| Token / Candidate | 모델의 처리 단위 / 허용 답안 | 앞 공백을 보존한 Correct/Lure 후보 Token ID |
| Full-Answer Likelihood | 전체 후보 토큰 로그확률 합 | 역사적 Protocol B는 EOS 제외 |
| First-Divergence | 후보 Token ID가 처음 달라지는 위치 | 공통 접두 길이 k 뒤 Correct/Lure의 경쟁 |
| Teacher-forced (TF) | 정해진 후보 경로에서 점수 계산 | 실제 생성 경로의 관측과 별도 측정 |
| Raw / Processed | 제약 적용 전 logits / processor 적용 후 scores | 각각 별도로 저장하며 상호 대체하지 않음 |
| Margin / Tie | Lure 점수−Correct 점수 / 저장된 차이0 | 양수는 Lure 우위, 음수는 Correct 우위; 0은 우위가 아님 |
| Constrained Greedy | 허용 후보 경로에서 다음 토큰 선택 | 기존 `generate()`와 후보 prefix trie 사용 |
| Equivalence | 동일 조건의 계측/비계측 token·답변 비교 | Baseline 대비 개입 효과나 독립 반복을 의미하지 않음 |
| Slot / Block | 과거 `(case_id, split_seed)` 참조 / 실행 묶음 | 과거 seed 슬롯과 신규 블록 관측을 구분 |
| Immutable Revision | Repository의 고정40자리 Commit SHA | 요청 SHA, 파일 무결성, 실제 로딩 검증을 구분 |
| Runtime Validation / Dry-run | 실제 로딩·Tensor 검사 / 로컬 실행 준비 | CPU 준비 성공으로 GPU 검증을 완료 처리하지 않음 |
| Bootstrap | 문항 단위 복원추출 | 29일 감사의 case-weighted 불확실성 추정 |

Raw Tie는 저장된 점수의 동률이다. 고정밀 계산의 잠재 차이가 없거나 특정 수치 오차가 원인이라고 단정하지 않는다. 같은 α도 Feature별 방향 Norm·실제 Activation 변화량이 같음을 뜻하지 않는다.

## 1. 연구 목적과 진행 단계

### 1.1 목적과 가설

`Qwen/Qwen3.5-2B-Base`의 CRT Correct/Lure 응답에서 Layer 5 SAE Feature 30908 개입이 실제 첫 분기 선택과 응답을 어떻게 바꾸는지 분석한다. 초기에는 억제로 Correct가 개선될 가능성을 탐색했지만 Fresh v2에서 안정적 개선 근거를 확보하지 못했다. 현재는 **개입·첫 분기 점수·제한적 Greedy 선택 사이의 관계**를 검증한다.

Feature 30908이 일반적인 직관 오류나 추론 기능을 담당한다는 인과적 해석, 다른 문항으로의 일반화, Feature 12207 대비 의미적 특이성은 아직 확증되지 않았다. 관측된 개입 효과와 그 기제·고유성의 확증을 구분한다.

### 1.2 단계별 상태

| 연구 단계 | 10월30일 기준 상태 | 근거와 한계 |
|---|---|---|
|1 Hagendorff CRT 탐색|후보30908 선정|탐색 결과이며 확증이 아님|
|2 Fresh v2 Behavioral|기존 실행·집계 완료|Correct 안정적 개선 미입증|
|3 Full-Answer Likelihood|기존 TF 분석 완료|전체 점수와 실제 응답 불일치|
|4 TF First-Divergence|기존 감사 완료|Tie 제외 높은 일치, 실제 generate logits와 구분|
|5-B~5-C 실제 생성 추적|구현·CPU 검증 및 Growth019 A100 관측/감사 완료|단일 문항, 과거 정확한 loaded revision 미기록|
|5-D-1 Cross-Case|설계·구현·CPU/Dry-run 준비 완료|신규42관측·84 Generate 실행은 미승인·미실행|
|5-D-2 Revision 고정|API metadata·CPU 전달 검증 완료|실제 Model/SAE 로딩 검증과 다름|
|5-D-3A CPU Tokenizer|캐시 Tokenizer3문항 실측 완료|Production Processor·pt tensor 경로 미검증|
|5-D-3B-Prep Smoke Test|CPU 준비·Dry-run 완료|3조건·6 Generate는 실행되지 않음|
|5-D-3B Validation-only|실행 시도 실패: 할당 전 TLS 중단|새 Runtime Validation5단계 모두 미도달|

진행 이력은 [29일 보고서](MindScopeX_연구진행현황_2026-10-29.md), [30일 Journal1~28절](../journal/2026-10-30.md)에 보존돼 있다. 일지 앞부분의 당시 실행 대기를 후속 완료 기록과 혼동하지 않는다.

## 2. 10월29일까지의 연구 결과

### 2.1 탐색과 Fresh v2 행동 검증

Hagendorff CRT에서 Layer5/30908을 후보로 탐색했다. 과거 Hagendorff의 개선 수치를 새로운 Fresh v2의 결과로 전용하지 않는다. [30일 Journal1절](../journal/2026-10-30.md)에 정리된 10월19~23일 Fresh v2 결과는 Baseline109/200(54.5%), +7 110/200(55.0%), +8 97/200(48.5%)다. Growth는 각각34/64,48/64,34/64 슬롯이었다. 문항 유형별 개선·악화가 공존하며, 전체 Correct를 안정적으로 개선한다는 증거로 해석하지 않는다. 이 수치는 과거 집계 기록 인용이며 이번 문서 작업에서 전체 Behavioral 실험을 재실행하지 않았다.

### 2.2 Full-Answer와 First-Divergence

Protocol A(chat template·strip 후보)와 B(raw prompt·앞 공백 보존 후보)는 조건이 다르다. B의33 Growth 문항×0/+7/+8=99 score행과 계수별64 Behavioral 슬롯을 결합했다. 후보 전체 TF 로그확률 합과 실제 응답이 자주 어긋나 첫 분기 점수를 분석했다.

| 기존 감사 지표 | +7 | +8 |
|---|---:|---:|
|TF First-Divergence Tie|11슬롯·4문항|6슬롯·2문항|
|TF First-Divergence 일치(Tie 제외)|53/53 (100%)|56/58 (96.55%)|
|Full-Answer 전체 일치|28/64 (43.75%)|36/64 (56.25%)|
|같은 비Tie 슬롯의 Full-Answer 일치|19/53 (35.85%)|30/58 (51.72%)|

표는 [29일 감사 보고서](../audits/2026-10-29-first-divergence/report.md), [summary.json](../audits/2026-10-29-first-divergence/summary.json), [slots.csv](../audits/2026-10-29-first-divergence/slots.csv)에 근거한다. 서로 다른 분모의 일치율을 무조건 직접 비교하지 않는다. Bootstrap은 문항 단위이며 같은 문항의 여러 seed 슬롯을 독립 문항으로 세지 않는다.

+8의 비Tie 불일치2슬롯은 Growth019의 과거 seed2·4다. TF First=-0.125는 Correct 방향인데 과거 Behavioral은 Lure였다. 이 예외가 실제 생성 logits 추적의 주요 동기가 됐다. 29일의 점수는 실제 generate 중 캡처한 값이 아니다.

## 3. 10월30일 실제 생성 관측과 해석

### 3.1 기존 생성 경로 재사용

5-B는 Rate027 진단의 `RecordingModel`을 재사용했다. 기존 `_binary_choice_completion`의 Raw Prompt·앞 공백 후보·prefix trie와 Feature Steering Hook을 사용하며 새 decoding loop를 만들지 않았다. 실제 `generate()`에 `output_logits=True`, `output_scores=True`, `return_dict_in_generate=True`를 요청하고 선택 ID는 실제 `sequences`에서 읽는다. Raw가 없으면 Scores로 대체하지 않고 실패한다.

분기 k에 실제로 도달하고 이전 생성 prefix가 후보 공통 접두와 같으며 두 다음 후보 토큰이 허용되고 Raw가 존재할 때만 관측으로 인정한다. 분기 미도달은 null Margin과 사유로 기록한다. EOS 관측·후보 완성·정상 응답·분기 관측도 각각 구분한다. 계측 동등성은 같은 문항·Feature·계수의 원래 생성 경로와 비교한다. 구현 경위와 실패 처리·직렬화 보완은 [Journal8~17절](../journal/2026-10-30.md)에 있다.

### 3.2 Growth019 Baseline/30908 +7/+8: 최초 A100 비교

실제 Run은 Baseline [131401](../../results/runs/20261030-131401_trace_fresh_v2_growth019_c0_2b/artifacts/actual_first_divergence.json), +7 [131523](../../results/runs/20261030-131523_trace_fresh_v2_growth019_c7_2b/artifacts/actual_first_divergence.json), +8 [104935](../../results/runs/20261030-104935_trace_fresh_v2_growth019_c8_2b/artifacts/actual_first_divergence.json)이다. [비교 감사](../audits/2026-10-30-growth019-generation-comparison/report.md)와 [summary.json](../audits/2026-10-30-growth019-generation-comparison/summary.json)이 세 Artifact를 검증했다.

Margin은 모두 Lure−Correct다. 아래 TF/Full은 **기존 Protocol B의 역사적 참조**이며 이 생성 실행에서 새로 측정한 점수가 아니다.

| 계수 | Actual Raw | Actual Processed | TF First | Full Answer | 응답 |
|---|---:|---:|---:|---:|---|
|0 no-hook|-1.375|-1.3749999850988388|-1.3750000596046448|-1.4600286781787872|55 days / Correct|
|30908 +7|-0.375|-0.375|-0.5|+1.573784351348877|55 days / Correct|
|30908 +8|0|0|-0.125|+0.7332298755645752|28 days / Lure|

세 Job succeeded, Equivalence passed, 측정 captured, EOS248044 정상 종료를 기존 감사에서 확인했다. +7의 Full이 양수여도 실제 Greedy 응답은 Correct다. +8의 Lure 응답은 **Raw/Processed 동률에서 선택된 결과**로 Lure Logit 우위가 아니다. Actual−TF는0에서 약5.96e-8, +7/+8에서 각각+0.125다.

### 3.3 Growth019 최소 재현 Pilot: 두 블록

별도 최소 파일럿은 [조건 Manifest](../../experiments/inputs/growth019_minimal_pilot.json)의5조건을 두 블록에서 실행했다. 총 **10조건별 관측·20 Generate**이며 각 관측의2호출은 계측/비계측 동등성 검사다.

| 조건 | Raw B1 / B2 | Processed (두 블록 동일) | Baseline 대비 Raw 변화 | 선택 ID | 응답 (두 블록 동일) |
|---|---:|---:|---:|---:|---|
|no-hook0|-1.375 / -1.375|-1.3749999850988388|0|20|55 days / Correct|
|30908 +7|-0.375 / -0.375|-0.375|+1.0|20|55 days / Correct|
|30908 +8|0 / 0|0|+1.375|17|28 days / Lure|
|30908 +8.5|+0.125 / +0.125|+0.125|+1.5|17|28 days / Lure|
|12207 +8|+0.625 / +0.625|+0.6250000298023224|+2.0|17|28 days / Lure|

수치는 이번에도 두 실측 JSON과 대조했다. 공통 Prompt는109tokens, Correct `[220,20,20,2756]`, Lure `[220,17,23,2756]`다. 앞 공백 `[220]` 뒤 k=1이 분기이며 허용ID는 `[17,20]`, EOS는248044다. 두 Job succeeded/Equivalence passed, 모든10관측의 EOS 종료·후보 완성·응답 유효성을 기존 감사와 실제 JSON에서 확인했다.

+8 Raw는 Lure/Correct 모두16.375다. +8.5는16.125/16.0으로 양의 Margin을 관측했다. 12207 +8도 Lure 우위여서 **12207은 검증된 Neutral Control이 아닌 비교 Feature**다. 새 +8.5/12207 조건의 TF/Full은 unavailable이며 Baseline 점수를 전용하지 않는다.

원본: [Block1 actual JSON](../../results/runs/20261030-145746_pilot_growth019_block-1_2b/artifacts/actual_first_divergence.json), [Block2 actual JSON](../../results/runs/20261030-145912_pilot_growth019_block-2_2b/artifacts/actual_first_divergence.json). 근거: [Pilot 감사](../audits/2026-10-30-growth019-minimal-pilot/report.md), [독립 대조 결과](../audits/2026-10-30-growth019-minimal-pilot/verification.json), [재계산 비교](../audits/2026-10-30-growth019-minimal-pilot/recomputed_comparison.json), [Journal21절](../journal/2026-10-30.md). 원본 Run은 통상 Git 관리 밖이므로 별도 checkout에서는 보관 Artifact 복원이 필요하다.

### 3.4 Tie·정밀도·해석 한계

작은 Token ID17 선택은 낮은 인덱스 우선 argmax와 부합한다. Journal15절의 별도 CPU 환경(Torch2.10.0+cpu)은 동일값17/20의 argmax=17을 실측했다. 반면5-C-9 로컬 환경은 Torch가 없어 표준 라이브러리로 BF16 표현과 기준 first-argmax를 검증했다. 이들을 같은 실행이나 실제 A100 선택 경로의 동적 증명으로 합산하지 않는다.

BF16에서 사라진 차이는 FP32 사후 변환으로 복원되지 않는다. 16.375 부근 BF16 간격0.125와 TF/Actual 차이의 일치만으로 원인을 특정할 수 없다. TF의 전체 후보 no-cache forward와 실제 generate의 cache 설정·연산 길이, SAE device/dtype/cast, 별도 forward 정규화, kernel·미기록 Revision이 후보 원인이며 **unverified**다.

후속 Pilot의 저장 Raw/Processed tensor dtype은float32였지만 모델/SAE는BF16이었다. 이는 FP32 모델 forward 실행이 아니다. 설치된 Transformers 소스의 processor→argmax 행은 보존됐으나 실제 분기 경로·processor 순서의 동적 trace는 아니다. 과거 Run의 미기록 dtype을 후속 값으로 보충하지 않는다. [Tie-breaking 감사](../audits/2026-10-30-growth019-tie-breaking/report.md)와 [Pilot 환경 기록](../audits/2026-10-30-growth019-minimal-pilot/report.md)에 범위를 명시했다.

사용자가 동일 VM 재사용을 보고한 두 블록에서 값·token·응답이 일치했지만, 한 문항의 고정 순서 반복이다. 고유 VM ID·종료 상태의 독립 증명은 미기록이며 통계적 독립성·유의성·일반화 근거가 아니다. 과거 seed2/4도 새 독립 측정이 아니다. 실제 Activation 변화 Norm은 이 Pilot에서 미측정이므로30908의 의미적 특이성을 확증하지 않는다.

## 4. Cross-Case·Revision·Runtime 준비

### 4.1 Phase 5-D-1: Feature Specificity & Cross-Case

RQ1은 문항 간 일반화, RQ2는30908과12207/추가 대조의 특이성, RQ3는 계수 증가에 따른 Margin·응답 변화다. 기존 데이터와 후보를 대조해 다음을 설계했다.

| 대상 | Correct / Lure (실제 후보는 앞 공백 포함) | Correct IDs / Lure IDs | 과거 Baseline 참조 |
|---|---|---|---|
|Growth019|55 / 28 days|[220,20,20,2756] / [220,17,23,2756]|seed2,4 Correct|
|Growth004|25 / 13 days|[220,17,20,2756] / [220,16,18,2756]|seed1 Correct|
|Growth016|49 / 25 days|[220,19,24,2756] / [220,17,20,2756]|seed1,4 Correct|

문항마다 no-hook0,30908의7/8/8.5,12207의7/8/8.5로7조건이다. **3문항×7조건×2블록=42관측·84 Generate**가 CPU 설계용 잠정 예산이다. 추가 Feature/random direction·별도 forward·재시도·진단은 포함하지 않으며 과거234 Generate 확장안과 별개다. GPU 승인이나 비용 상한이 아니다. 004/016은 저장 입력과 CPU Tokenizer상 호환되지만 Production Runtime 미검증으로 최종 실행 확정은 아니다. 알려진 Growth 문항이며 새로운 held-out 대표 표본으로 부르지 않는다.

실제 개입 지점의 before/after residual norm, actual `||Δactivation||₂`, 계수 적용 전 decoder 방향 Norm, requested/cast vector norm을 분리해 기록하는 CPU 구현을 준비했다. 무Hook Baseline의 residual norm은 미측정으로 둔다. 동일α나 Norm 일치만으로 의미적 특이성을 주장하지 않는다. 추가 대조 Feature는 사전 선정 기준 제안 단계이며 실행 ID를 확정하지 않았다.

[계획](../audits/2026-10-30-feature-specificity-cross-case/plan.md), [검증 입력](../audits/2026-10-30-feature-specificity-cross-case/inputs_verified.json), [현행 Manifest](../../experiments/inputs/feature_specificity_cross_case.json), [Journal22절](../journal/2026-10-30.md). 당초 계획의 Revision 미설정은 당시 기록이며 다음 절의 고정 SHA와 구분한다.

### 4.2 Phase 5-D-2: Immutable Revision Pinning

새 실험의 고정 기준은 다음과 같다. Model/Tokenizer는 역할별 HF API 조회로 같은 SHA를 확인했다.

| 역할 | Repository | 고정 Commit SHA |
|---|---|---|
|Model|Qwen/Qwen3.5-2B-Base|`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`|
|Tokenizer|Qwen/Qwen3.5-2B-Base|`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`|
|SAE|Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50|`132ea3697b591df9ee46d738aa1d528e3c6082f7`|

2026-10-30 API main/immutable metadata·파일 목록, SAE Base 모델 대응과 Layer5 파일 존재를 확인했다. AutoConfig·Model·AutoProcessor/Tokenizer·SAE hf_hub_download의 SHA 전달은 CPU Mock/AST로 검증했다. Metadata·인자 전달과 실제 weights 로딩 성공은 다르다. 과거 Growth019 Artifact의 exact revision은 미기록으로, 새 SHA와 같았다고 소급 확정하지 않는다.

근거: [Revision 보고서](../audits/2026-10-30-feature-specificity-cross-case/revision_pinning_report.md), [API 증거JSON](../audits/2026-10-30-feature-specificity-cross-case/revision_evidence.json), [Journal23절](../journal/2026-10-30.md). 새 Model/SAE 파일 무결성·shape/runtime 호환성은 미검증이다.

### 4.3 Phase 5-D-3A: 실제 CPU Cached Tokenizer와 Production Processor

기존 캐시4파일 hash를 API metadata와 대조하고 고정 SHA의 AutoTokenizer를 CPU에서 실행했다. Python3.12.3, Transformers5.20.2, Tokenizers0.23.3, HF Hub1.33.0의 분리 환경이며 Torch는 설치하지 않았다. 실제 클래스는Qwen2Tokenizer(is_fast=true)로, local_files_only/offline/socket guard 아래 연결0회·Model/SAE 로딩0회였다.

| 문항 | Prompt tokens | prepared 문자열·전체 Prompt IDs/SHA·Correct/Lure IDs | Prefix/EOS 규칙 |
|---|---:|---|---|
|Growth019|109|기존 값과 일치|[220] / k=1, EOS/PAD248044|
|Growth004|108|기존 값과 일치|같음|
|Growth016|106|기존 값과 일치|같음|

후보 앞 공백과 add_special_tokens=False를 유지했다. Prompt는 Production helper의 default encoding에 맞췄다. 실제 pt tensor 생성, AutoConfig→AutoProcessor Production 경로는 미실행이며 필요한 config도 당시 캐시에 없었다. 과거 A100 metadata의 Tokenizer 클래스와 차이도 미해결이다. 따라서 **cached_tokenizer_runtime_verified=verified**이나 **production_processor_runtime_verified/runtime_tokenizer_validation은unverified**다.

[Tokenizer 보고서](../audits/2026-10-30-feature-specificity-cross-case/tokenizer_runtime_report.md), [실측JSON](../audits/2026-10-30-feature-specificity-cross-case/tokenizer_runtime_results.json), [환경 버전 목록](../audits/2026-10-30-feature-specificity-cross-case/tokenizer_cpu_requirements.txt), [Journal24절](../journal/2026-10-30.md).

### 4.4 Phase 5-D-3B-Prep과 Validation-only 실패

Smoke 준비는 Growth019 no-hook0 /30908+8.5 /12207+8의3관측·6 Generate를 계획했다. 현재 [Smoke Manifest](../../experiments/inputs/growth019_runtime_smoke.json)는allocation_authorized=false, execution_ready=false이며 생성 모드는 허용되지 않았다.

사용자의 예외적 명시 승인으로 Validation-only용 별도 [Config](../../experiments/configs/growth019_validation_only_2b.toml)와 [승인 범위 Manifest](../../experiments/inputs/growth019_validation_only_authorization.json)를 추가했다. through=validate, 최대A100 신규 할당1회, Generate0회·추가 Qwen Forward0회·합성 CUDA Identity Hook Forward2회 범위였다. 기존6 Generate Config나 Cross-Case 실행 권한이 아니다.

Codex의 Launcher1회 실행은 최초 colab status에서 TLS `self-signed certificate in certificate chain` 오류로 세션 부재를 확인하지 못해 할당 전에 멈췄다. **colab new0회·원격 실행0회·Generate0회·Qwen Forward0회·실제 합성 Hook Forward0회**다. 자동 재시도·재할당·수정 후 GPU 재실행은 하지 않았다.

Run [20261030-170836](../../results/runs/20261030-170836_growth019_pinned_validation_only_2b/)의 로그·Launcher Manifest·source.zip은 보존했다. 소스 Archive이며 결과 ZIP이 아니다. stage1~5/runtime_validation/runtime_status/결과 ZIP은 생성되지 않았다. stop은Session not found를 반환했고 마지막 status도TLS 오류여서 당시 서버 상태는 미확인으로 남겼다.

[Smoke 계획](../audits/2026-10-30-feature-specificity-cross-case/smoke_test_plan.md), [실행 실패 보고서](../audits/2026-10-30-feature-specificity-cross-case/validation_only_execution_report.md), [실행 증거JSON](../audits/2026-10-30-feature-specificity-cross-case/validation_only_execution_evidence.json), [Journal25~26절](../journal/2026-10-30.md). 이 실패를 과거 Pilot 실패나 Generation 불일치로 해석하지 않는다.

## 5. 현재 검증 수준과 연구 질문

### 5.1 CPU와 Runtime의 경계

최종 Validation-only 실행 보고서의 관련 CPU 검사는 **133 passed / 0 failed / 1 skipped (총134개, error0)**다. 기존125통과와 신규8통과를 포함한다. skip은 로컬Torch 부재의 실제 Tensor Hook 검사이며 Launcher 함수 assertion3개는 별도 집계다. 전체 저장소 테스트 통과나 새 CUDA Hook 성공을 의미하지 않는다. 이번 보고서 작성에서 해당 테스트를 재실행하지 않았다.

| 검증 대상 | 현재 판단 |
|---|---|
|과거Growth019 A100 Pilot·EOS·계측 동등성|저장 Artifact에서 검증 완료, 단일 문항의 제한적 결과|
|새 Revision metadata / Loader 인자 전달|verified / CPU Mock 범위에서verified|
|CPU 캐시 Tokenizer3문항|verified|
|새 Runtime1: 파일/SHA 무결성|미검증, 할당 전 중단으로not reached|
|새 Runtime2: Production Processor/Tokenizer tensor|미검증, not reached|
|새 Runtime3: Qwen Model 로딩|미검증, not reached|
|새 Runtime4: Layer5 SAE 로딩·shape|미검증, not reached|
|새 Runtime5: CUDA Tensor/Identity Hook|미검증, not reached|
|종합pinning/runtime_tokenizer/model/sae|unverified|
|Cross-Case execution_ready|**false**|
|새 Revision의 Generation·EOS·Equivalence|미실행·미검증, 별도 승인 대상|

### 5.2 연구 질문별 현재 판단

| 연구 질문 | 10월30일 판단 |
|---|---|
|30908은Lure 관련 후보인가?|탐색상 후보이며Growth019 개입에서 점수·응답 변화 관측|
|Fresh v2 Correct를 안정적으로 개선하는가?|근거 부족|
|Full-Answer가 실제 Greedy 응답을 설명하는가?|제한적. +7의 점수 방향과 응답 불일치|
|TF First-Divergence와Actual은 같은가?|다수 과거 비Tie에서 일치하지만Growth019 수치 차이 존재|
|+8 Lure는Lure Logit 우위인가?|아니다. 저장Raw/Processed Margin은0인Tie|
|30908 효과가12207 대비 특이적인가?|미확증.12207+8에서도Lure 우위 관측|
|다른 문항으로 일반화되는가?|Cross-Case 설계·CPU 준비 단계, 새 Runtime 실측 없음|
|Feature의 의미·기제나 통계적 유의성이 확정됐는가?|미확정|

## 6. 기술 환경·운영·재현성

### 6.1 NNsight, Residual, SAE의 역할

NNsight는 탐색 시 `lm.trace`로 Layer residual을 얻는 데 사용한다. 실제 Behavioral/First-Divergence 생성은 Production Transformers `generate()`, Layer block forward hook의 SAE decoder 방향 가산, RecordingModel의logits/scores 캡처를 사용한다. NNsight 후보 경로 trace와 실제 generate 중 캡처를 혼동하지 않는다.

참조 구현: [activations.py](../../src/mindscopex_analysis/activations.py), [research.py](../../src/mindscopex_analysis/research.py), [qwen_scope.py](../../src/mindscopex_analysis/qwen_scope.py), [trace 진단](../../experiments/trace_fresh_v2_first_divergence.py). Residual 가산은 Feature 활성값을 특정 값으로clamp하는 개입이 아니다.

기존 Pilot 기록 환경은A100-SXM4-40GB, Python3.13.16, Torch2.11.0+cu130, Transformers5.20.2, CUDA13.0/cuDNN92700, Model/SAE BF16·저장Raw/Processed float32였다. 과거 실측 환경이며 TLS 중단된 새 Validation-only 환경을 뜻하지 않는다. 과거 exact Model/Tokenizer/SAE revision은not recorded, 새 Revision의 로딩 환경·GPU 메모리는 미기록이다.

### 6.2 회사CA와 실행 책임

[회사CA TLS 가이드](../guides/colab_cli_corporate_ca_tls.md)와 [Journal27절](../journal/2026-10-30.md)에 따르면 사용자 WSL은 회사CA 환경변수 적용 후 Colab CLI0.7.4의colab sessions에 성공해 `No active sessions found on server.`를 반환했다. **사용자 보고에 따른 해당 조회 시점의 확인**이며 이번 원격 재확인 결과가 아니다. 기존TLS 실패를 성공으로 재분류하지 않고 환경 차이는 가능한 원인으로 미확정 상태를 유지한다.

회사CA는 `/usr/local/share/ca-certificates/sk-holdings-root.crt`, 시스템 번들은 `/etc/ssl/certs/ca-certificates.crt`다. REQUESTS_CA_BUNDLE은 회사WSL의HTTPS 검증용이며Codex 전용이 아니다. 사용자 연결이 성공한 현재는 인증서 재설치·CLI 업데이트가 불필요하다고 정리했다. TLS 우회를 금지하고 인증서 원본·인증정보를Git에 추가하지 않는다.

[운영 규칙9절](../../MINDSCOPEX_WORKFLOW_RULES.md)에 따라 ChatGPT는 연구 설계·분석·지시, Codex는 구현·CPU/Mock/Dry-run·Config/Manifest·문서·Git, **사용자WSL은Colab CLI·세션·GPU 실행·로그 회수**, Google Colab은실제Model/SAE/CUDA를 담당한다. 예외적으로Codex에 원격 실행을 맡기려면 주체·범위의 별도 명시 승인이 필요하며 과거 승인은 재실행의 포괄적 권한이 아니다. WSL Bash에서Jupyter 전용!git/%cd를 사용하지 않는다.

### 6.3 Git과Artifact 보존

[Journal28절](../journal/2026-10-30.md)은 사용자WSL에서33335de까지Push 성공·HEAD/origin/main/origin/HEAD 동기화를 확인한 기록이다. 이번 시작HEAD는 이후Journal28절Commit `bdac1d518c4fe5731f6a7703e9db811174aacc5e`다. 과거 동기화 확인을 이후 모든Commit의Push 성공으로 전용하지 않는다. 본 보고서Commit/Push 결과는 완료 응답에 별도로 남긴다.

기존Run·실패 로그·Audit·Journal을 삭제/덮어쓰지 않는다. Prompt/후보ID·SHA, 실행Git/dirty 상태, 소스Archive, 요청Revision과 실제 로딩 근거를 구분해 보존한다. 새 출력 경로와 실패 시 부분Artifact/로그 보존, 기존main 중심Git 운용을 유지한다. 이번에는 사용자의 **신규 보고서만 작성** 지시에 따라Journal도 변경하지 않는다.

## 7. 남은 연구 과제와 다음 단계

1. 최신Git HEAD/Journal과 사용자WSL 회사CA 설정·읽기 전용 연결을 확인한다. **별도 사용자 승인 후** A100 Runtime Validation-only를 사용자WSL에서 실행한다.
2. 고정Revision의 파일/SHA→Production Processor/3문항 입력→Model→SAE→CUDA Identity Hook을 단계 검증한다. Generate0회·추가Qwen Forward0회·합성Hook Forward2회 범위를 유지하고 실패 시 자동 재시도 없이 중단한다.
3. 실제Artifact·부분 실패 로그·환경/시간/메모리/캐시 정보를 회수·감사한다. Identity Hook 성공만으로 실제Model Layer5 dispatch나Generation 성공을 주장하지 않는다. 상태 변경은Revision에 연결된 근거를 검토한 뒤 수행한다.
4. 후속 별도 승인 후Growth019 최소 생성 실험(3조건·6 Generate)의 필요성·성공 기준을 검토한다. 새Revision에서 과거와 다른 값이 나와도 자동 실패로 처리하지 않고EOS·후보 제약·Equivalence·입력 정합성으로 판단한다.
5. Runtime과 승인 조건을 충족한 뒤Cross-Case specificity·dose response·강건성을 검증한다. 추가 대조 선정·실제Activation 변화/방향Norm·순서/VM 효과·다른 문항과token ID 영향을 명시한다.

**Validation-only 재실행,6 Generate,84 Generate는 현재 향후 실행이 승인된 상태가 아니다.** 234 Generate는 과거 확장 설계로 보존하고 직전 실행 계획과 혼동하지 않는다.

## 8. 참고 근거와 이번 문서 검증

| 단계 | 주요 근거 | 판단 대상 |
|---|---|---|
|10/29 이전|[기존 보고서](MindScopeX_연구진행현황_2026-10-29.md), [10/29 Journal](../journal/2026-10-29.md)|과거 탐색·Behavioral·TF 집계|
|First-Divergence|[report](../audits/2026-10-29-first-divergence/report.md), [summary](../audits/2026-10-29-first-divergence/summary.json)|Tie/분모를 구분한 기존 감사|
|실제0/+7/+8|[비교report](../audits/2026-10-30-growth019-generation-comparison/report.md), [summary](../audits/2026-10-30-growth019-generation-comparison/summary.json)|기존A100 Artifact 검증|
|두 블록Pilot|[report](../audits/2026-10-30-growth019-minimal-pilot/report.md), [verification](../audits/2026-10-30-growth019-minimal-pilot/verification.json)|실제10관측·20호출, 제한적 반복 일치|
|Tie/정밀도|[report](../audits/2026-10-30-growth019-tie-breaking/report.md)|관측·CPU 수치 예시·미검증 기제|
|Cross-Case|[plan](../audits/2026-10-30-feature-specificity-cross-case/plan.md), [Manifest](../../experiments/inputs/feature_specificity_cross_case.json)|설계·CPU 준비, 실행 미승인|
|Revision|[report](../audits/2026-10-30-feature-specificity-cross-case/revision_pinning_report.md), [evidence](../audits/2026-10-30-feature-specificity-cross-case/revision_evidence.json)|Metadata·CPU 인자 전달|
|Tokenizer|[report](../audits/2026-10-30-feature-specificity-cross-case/tokenizer_runtime_report.md), [results](../audits/2026-10-30-feature-specificity-cross-case/tokenizer_runtime_results.json)|실제CPU 입력 일치,Production 미검증|
|Smoke/Validation-only|[plan](../audits/2026-10-30-feature-specificity-cross-case/smoke_test_plan.md), [실패report](../audits/2026-10-30-feature-specificity-cross-case/validation_only_execution_report.md), [evidence](../audits/2026-10-30-feature-specificity-cross-case/validation_only_execution_evidence.json)|CPU 준비·할당 전TLS 실패·Runtime 미검증|
|운영/종료|[Workflow](../../MINDSCOPEX_WORKFLOW_RULES.md), [CA 가이드](../guides/colab_cli_corporate_ca_tls.md), [Journal1~28절](../journal/2026-10-30.md)|역할·승인·보존·Git 동기화 시점|

이번 검증은 저장 근거/실제JSON의 수치 대조, 상대 링크 존재, 기존 문서/Run/Audit SHA 불변, 신규Markdown1개 및git diff --check 확인으로 한정한다. 새CPU 연구 테스트·분석 실험·Dry-run·GPU/Colab·모델 다운로드·Generate는 실행하지 않았다. 미실행을 성공으로 표현하지 않는다.
