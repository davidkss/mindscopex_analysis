# Phase 5-C-9 — Growth019 Tie-breaking 및 TF/Generation 수치 분석

기준 HEAD `f01c326`, clean main에서 시작했다. workflow rules, Journal 18절과 관련 14~17절, README, 기존 비교 보고서, adapter·실제 helper·capture·TF 코드를 확인했다. 마지막 완료 단계는 5-C-8 Artifact 감사다. 이번에는 GPU/Colab·모델 다운로드·새 모델 생성·의존성 설치를 수행하지 않았다. 이전 A100 원본 42파일과 5-C-8 산출물은 보존했다.

## 1. 관측 사실

Margin은 모두 Lure−Correct다. `research/audits/2026-10-30-growth019-generation-comparison/summary.json` 및 원본 Run의 실제 step 값에 근거한다.

| Coefficient | TF First | Actual Raw | Actual Processed | Actual−TF | 응답 |
|---|---:|---:|---:|---:|---|
| 0 | -1.3750000596046448 | -1.375 | -1.374999985 | +0.0000000596046448 | Correct `55 days` |
| +7 | -0.5 | -0.375 | -0.375 | +0.125 | Correct `55 days` |
| +8 | -0.125 | 0 | 0 | +0.125 | Lure `28 days` |

+8 step 1(0-based)은 공통 접두 `[220]` 뒤이며, 허용 token `[17,20]`, 실제 선택 17이다. Lure 17과 Correct 20의 저장 Raw는 각각 16.375, Processed는 각각 -0.6931471824645996이다. **저장된 raw/processed 동률과 Lure 선택**이 확인됐으며 Lure Logit 우위가 아니다. EOS 248044, 완성 응답 및 동일 +8 내 계측/비계측 일치는 5-C-8에서 검증됐다. 과거 seed 2·4와 계측/비계측 호출은 신규 독립 반복이 아니다.

## 2. 실제 코드 경로와 확인 한계

| 경로 | 직접 확인한 동작 |
|---|---|
| `experiments/jobs/first_divergence_trace.py::run` | 선택 조건을 기존 진단 CLI에 전달, 결과/Equivalence/입력 SHA 검증; 모델 선택 알고리즘을 직접 구현하지 않음 |
| `experiments/trace_fresh_v2_first_divergence.py::run_model/capture_case` | Base 모델/SAE, Layer 5·30908/all Hook; 0은 무Hook. 기존 `_binary_choice_completion`을 RecordingModel로 호출 |
| `src/mindscopex_analysis/research.py::_binary_choice_completion` | raw prompt, 공백 보존 standalone 후보, prefix trie. `allowed_tokens`는 set을 `sorted(allowed)`로 반환. do_sample=False, prefix_allowed_tokens_fn, renormalize_logits=True; max_new_tokens는 최대(16, 후보 길이+1). EOS 이후 후보 ID 일치를 검사 |
| `experiments/diagnose_rate027_baseline.py::RecordingModel.generate` | 원래 prefix callback을 기록용으로 감싸고 return_dict_in_generate/output_logits/output_scores=True 추가; 실제 sequences 반환 |
| `trace::inspect_capture` | 선택을 재실행하지 않고 `output.sequences`의 실제 ID와 `output.logits`/`output.scores`의 해당 step 값을 읽어 margin 계산 |
| `trace::compare_uninstrumented` | 동일 계수의 production helper를 원래 인자로 한 번 더 호출; token IDs/답변 동일성 대조 |

**로컬 helper에는 argmax 호출이 없다. 선택은 Transformers `model.generate`에 위임된다.** 현재 Python 3.12.3에는 torch와 transformers가 없고, A100에서 설치된 Transformers 5.20.2의 내부 소스는 기존 source ZIP에 포함되지 않았다. 따라서 실제 호출된 HF 함수의 argmax 문장, 추가 processor 목록·순서, logits/scores의 변환 dtype을 이번에 직접 확인한 것으로 쓰지 않는다. 불필요한 라이브러리 설치/외부 소스 추정으로 대체하지 않았다.

로컬에서 확인 가능한 processor 요청은 후보 prefix 제한과 renormalization이다. 기록된 generation_config에는 use_cache=true, EOS248044가 있고 추가 penalty/억제 옵션은 null이지만 null만으로 실제 내부 processor가 전혀 없었다고 확정할 수 없다. PrefixConstrainedLogitsProcessor/LogitNormalization 등의 구체적 클래스·호출 순서는 **unverified**다. 다만 이 분기에서 Raw·Processed는 함께 동률이고 실제 17이 선택됐다.

`diagnose_rate027_baseline.py::run`에 raw/processed 허용 부분집합의 argmax를 사후 계산하는 코드가 있어도, 이는 이번 trace의 실제 생성 선택 코드가 아니다. 테스트의 Python Mock argmax도 production HF 구현의 증거가 아니다.

## 3. CPU 정밀도 및 동률 검증

재현 스크립트: `scripts/probe_growth019_precision.py`; 실제 출력: 같은 폴더 `cpu_probe.json`.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/probe_growth019_precision.py \
  --output research/audits/2026-10-30-growth019-tie-breaking/cpu_probe.json
```

출력 파일은 미존재 경로만 허용한다. 재실행은 새로운 출력 경로를 사용한다. 이번 환경의 **실제 Torch Tensor 사례 실행 수는 0**이다. Torch 부재를 명시한 표준 라이브러리 실험이며, 과거 Journal 15절의 별도 Torch 2.10.0+cpu 결과를 이번 실행 결과로 재사용해 통과 처리하지 않았다. Torch가 이미 설치된 환경에서는 스크립트의 선택적 검사가 명시적 CPU tensor만 생성한다.

`struct`로 FP32 값을 표현하고, FP32 비트에서 BF16 round-to-nearest-even을 구현했다. `first_argmax`는 명시적으로 첫 최대값을 택하는 **기준 규칙**이며 실제 HF 선택을 재현했다고 주장하지 않는다. BF16의 추가 단위 간격은 이 값 구간에서 0.125다.

| Lure/Correct 입력 | BF16 표현 | BF16→FP32 | 기준 FP32 선택 | BF16 및 widening 후 기준 선택 |
|---|---|---|---:|---:|
| 16.375 / 16.375 | 16.375 / 16.375 | 동일 | 17 | 17 |
| 16.375 / 16.385 | 16.375 / 16.375 | 동일 | 20 | 17 |
| 16.375 / 16.365 | 16.375 / 16.375 | 동일 | 17 | 17 |
| 16.375 / 16.5 | 16.375 / 16.5 | 동일 | 20 | 20 |

16.375는 BF16과 FP32 양쪽에서 정확히 표현된다. **이미 양자화된 동일 BF16 값을 FP32로 확장해도 같은 값**이다. FP32 모델 forward나 양자화 전 FP32 logits 측정은 이 확장과 다르며, 손실된 차이를 복구하지 않는다. 작은 FP32 차이가 BF16에서 동률로 바뀔 수 있다는 수치적 가능성은 검증했지만 실제 +8이 그 경로로 동률이 됐다는 원인은 **unverified**다.

중요하게도 A100 weights/SAE dtype이 bfloat16이라는 기록만으로 실제 선택 시점 `output.logits`/scores가 BF16이었다고 확정하지 않는다. trace는 `.item()`으로 숫자를 저장했고 원래 tensor dtype은 별도 저장하지 않았다. HF가 이미 float로 변환했는지까지 이번 Artifact에서 알 수 없다.

### 허용 토큰 순서

전체 vocabulary 인덱스를 보존한 기준 mask+first-argmax는 허용 목록 `[17,20]`과 `[20,17]` 모두 17을 선택한다. 반대로 허용 부분집합을 주어진 순서로 gather한 뒤 첫 위치를 고르는 별도 알고리즘은 각각 17/20을 택한다. 두 규칙은 다르다. Production callback은 항상 sorted ID를 반환하며, 실제 HF가 어느 형태로 argmax를 구현했는지는 이번에 직접 확인하지 않았다. 관측 17은 낮은 vocabulary ID 우선 가설에 부합하지만 가설을 내부 코드 검증으로 격상하지 않는다.

## 4. TF와 실제 Generation 차이: 사실과 가설

| 항목 | 확인한 사실 | 확인할 수 없는 부분 |
|---|---|---|
| Prompt / 후보 | 동일 Prompt token hash `145bbd68f373722e8ae38ae183fab16f0ba13c529f315fb824ecf1e68fa45ea2`, P=109. Correct `[220,20,20,2756]`, Lure `[220,17,23,2756]` | 정확한 weights/tokenizer/SAE revision은 not recorded |
| 공통 접두 | `[220]`, k=1. 분기는 110-token 인과 context(P+접두) 다음을 예측 | 실제 각 cache forward의 tensor shape·cache 내용은 기록되지 않음 |
| TF 길이·정렬 | `scripts/diagnose_fresh_v2_full_answer_likelihood.py::teacher_force`: prompt+candidate[:-1], 각 112-token 전체 입력, use_cache=False. 첫 분기는 logits index109(P−1+k)에서 gather | 후보별 별도 forward raw logits는 저장하지 않음 |
| 실제 생성 길이·cache | 기록된 generation_config.use_cache=True. 동일 prefix에 도달한 실제 step의 scores/logits 관측 | 통상 prefill109→접두1-token 증분 처리가 예상되나 해당 HF/하이브리드 cache 내부 경로·모든 forward 길이는 unverified; 단순 KV cache만 사용했다고 단정하지 않음 |
| TF 점수 | logits에서 관련 위치를 `.float()`한 뒤 full-vocab log_softmax·target gather. 두 별도 후보 경로의 첫 token logprob 차이 | 저장되지 않은 TF raw dtype·두 normalization 차이를 분리 불가 |
| 실제 점수 | 같은 실제 forward의 두 raw `.item()` 차이와 processor 이후 두 scores 차이를 따로 저장 | 선택 직전 raw/scores tensor dtype 및 processor 전체 순서 미기록 |
| SAE | TF manifest는 device=cpu, loader에 dtype override 없음. Generation은 cuda·torch.bfloat16 기록 | TF SAE 실제 dtype은 기록되지 않아 FP32였다고 확정하지 않음; 캐스팅 전 방향 차이 가능성 unverified |
| Model dtype | TF manifest 요청은 auto; Generation은 bfloat16 및 BF16 parameters 기록. Torch는 둘 다 2.11.0+cu130 | TF의 resolved dtype 및 모든 package/kernel 상태 동등성은 미확인 |

이상적인 동일 인과 context의 두 토큰 logprob 차이는 raw logit 차이와 같다. 그러나 여기서는 TF의 두 별도 전체 forward/정규화와 실제 cached 경로가 다른 측정이다. Baseline의 약 5.96e-8 차이는 FP32 log-softmax/subtraction 반올림과 양립하지만 단독 원인 입증은 아니다. +7/+8의 0.125는 16~32 BF16 구간의 한 간격과 같지만 그것만으로 BF16 반올림·cache·SAE 중 원인을 특정할 수 없다.

가능 원인은 전체 길이 vs cache 증분 경로의 수치 차이, 모델/SAE dtype·캐스팅, normalization을 별도 forward에서 뺀 효과, kernel 및 미기록 revision 차이다. 모두 **unverified**다. 실제 generation raw/processed margin 자체는 +7 -0.375/+8 0으로 같아, 단순 processor 순위 역전만으로 TF↔Actual 차이를 설명할 근거는 없다.

## 5. 후속 A100 계획 — 미구현·미승인·미실행

### A. 실행 전 필요한 코드/기록

1. 실제 설치된 Torch/Transformers 버전·소스 파일/hash를 저장하고 그 환경의 generate 호출 경로, 실제 argmax와 processor 순서를 직접 확인한다. tiny CPU tensors로 실제 라이브러리 동률 규칙을 검증한 후, 실제 generate의 selected ID·전체 vocab index 대응과 대조한다.
2. raw/processed logits의 원래 dtype·선택 이전 dtype, forward 입력 길이, cache 활성/shape·관련 모듈 경로를 기록한다. 진단이 출력을 바꾸지 않는지 계측 동등성을 다시 검증한다. dtype 변경은 `.float()` 사후 변환과 분리한다.
3. model/tokenizer/SAE exact revision과 checkpoint hash를 고정한다. 동일 raw prompt·후보 토큰·EOS·generation kwargs를 고정하고 GPU metadata/peak memory/시간을 기록한다.
4. 현재 adapter는 Growth019와 계수 0/7/8만 허용한다. 아래 확장 대상·중간 계수·대조 Feature·dtype는 현재 CLI로 실행할 수 없으며 별도 최소 구현/CPU 테스트/명시적 GPU 승인이 선행돼야 한다. 기존 코드를 몰래 변경하거나 이번 작업에서 실행하지 않는다.

### B. 사전 선정 조건 및 반복 단위

- 사례: Growth019(주 대상), `crt_fresh_v2_growth_004`(후보 25/13 days), `crt_fresh_v2_growth_016`(49/25 days). 기존 B CSV에서 두 대조 사례의 존재를 확인했고 추가 결과를 보기 전에 대상 목록을 고정한다. 신규 독립 dataset 일반화 검증을 대신하지 않는다.
- 개입: 목표 Feature30908, 기존 비교 Feature12207(유의미한 효과가 없다고 가정하지 않는 대조), 같은 Layer5/all. 각 feature당 α=7,7.5,8,8.5,9. Baseline은 Feature를 지정한 α0 hook 대신 **공통 no-hook α0** 한 조건이다.
- 독립 실행 블록 3개: 별도 프로세스/모델 재로드로 상태를 초기화하고 사전 고정한 순서 무작위화 일정을 저장한다. 같은 문항/계수의 새 실행을 반복하되 seed2/4·계측/비계측 한 쌍을 반복으로 세지 않는다. 결정적 greedy의 반복은 재현성 측정이며 서로 다른 문항이나 독립 모집단 표본이 아니다. 프로세스 분리만으로 통계적 독립성을 보장하지 않음도 명시한다.

### C. 작업량 산정

| 단계 | 고유 조건 | 실행 블록 | 실제 반복 관측 | 계측/비계측 쌍을 포함한 generate |
|---|---:|---:|---:|---:|
| BF16 주 실험 | 3사례 × (no-hook 1 + 2features×5계수)=33 | 3 | 99 | 198 |
| FP32 추가(주사례·목표Feature만) | Growth019 × (no-hook1 + 30908×5계수)=6 | 3 | 18 | 36 |
| 합계 | 정밀도별 조건 합 39 | 각3 | 117 | **234** |

BF16/FP32 비교의 BF16 해당 18관측은 주 실험 99관측에 이미 포함돼 중복 계수하지 않는다. 3개 BF16 프로세스(Job당66 generate)와 3개 FP32 프로세스(Job당12 generate), 총 6개 Job으로 설계한다. 실제 HF 소스·tiny tensor CPU 검사에는 모델 generate/GPU가 필요 없으므로 이 표에 추가하지 않는다. 예외가 발생하면 결과를 보존하고 중단하며 유리한 결과를 얻을 때까지 임의 반복하지 않는다.

기존 run_model 시작~완료는 0/+7/+8에서 약 33.36/32.01/41.02초였으나 모델 로딩을 포함하고 개별 decode 시간을 분리하지 않아 234회 소요 시간을 선형 추정할 수 없다. 계획상 각 Job remote exec timeout 900초, 6개 합계 **90 A100 실행분의 예약 상한 목표**를 먼저 검토한다. 세션 할당·전송·결과 회수·종료 지연 등은 별도이며 이는 과금의 엄격한 상한이 아니다. 승인 시 실제 시간·메모리와 허용 총비용을 고정하고 초과 시 중단해야 한다. 이번에는 비용 발생 작업을 하지 않았다.

### D. FP32 메모리 타당성

Artifact의 parameter numel은 2,213,241,664다. weights만 BF16 약 **4.1225 GiB**, FP32 약 **8.2450 GiB**이며, Layer5 SAE 두 2048×32768 행렬은 FP32 약 **0.5 GiB**(+bias)다. FP32 weights+SAE만 약 8.75 GiB이므로 A100 40GB에서 시도 가능성이 있지만 activations·하이브리드 cache·temporary logits/scores·CUDA allocator/workspace·동적 offload를 제외한 하한 추정이다. 실제 peak memory는 **unverified**이며 보장하지 않는다.

먼저 batch1·동일 짧은 후보·no-grad·한 dtype 모델만 상주하는 제한을 사용하고 max_memory_allocated/reserved를 측정하도록 별도 구현한다. BF16/FP32 모델 동시 적재는 계획하지 않는다. 전체 forward를 FP32로 수행하려면 모델/SAE dtype 자체를 바꿔야 하며 BF16 logits `.float()`만으로 대신하지 않는다. GPU 부하·속도 및 커널 지원은 실제 승인된 준비 단계에서 확인해야 한다.

## 6. 검증 및 결론

현재 CPU 검증은 representation emulation/reference-rule 검증이며 Torch/Transformers 실측 선택 검증이 아니다. 기존 A100 결과는 17 선택·동률을 실제로 보여주지만 정확한 HF tie-breaking 내부 구현은 여전히 unverified다. BF16 값을 FP32로 확장하는 것과 FP32 계산을 명확히 구분하며, 실제 +8이 양자화로 동률이 된 원인을 확정하지 않는다. 정확한 model/tokenizer/SAE revision은 not recorded 상태를 유지한다.

신규 테스트 6개와 기존 adapter21·진단35·5-C-8분석7개를 실행한다. 실제 개수·결과와 원본 보존 확인은 Journal 19절에 기록한다. 기존 Run/Audit·Journal은 수정하지 않으며 새 산출물만 저장한다.

CPU 실제 결과: adapter21 + trace35 + 기존 비교7 + 신규 정밀도6 = **69개 통과·실패/skip 0개**. 실제 Torch tensor 사례는 미설치로 **0건**이다. `py_compile`, `git diff --check`도 통과했다. 재현 명령은 Journal 19절에 수록했다.
