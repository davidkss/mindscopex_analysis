# Phase 5-D-3B 연구 범위 계획

작성일은 2026-11-21이며 기준 Git HEAD는 `3e5a9a9a0981340f260f959f7bfb53bf4bd6eb4f`다.
B·C는 설계 단계이며 향후 실행 전에 [운영 규칙](../../MINDSCOPEX_WORKFLOW_RULES.md)의 일반
검토 경로와 별도 사용자 승인을 거친다. 표본 수, 효과 기준, Null 반복 수와 통계 임계값은
B 단계 실행 전에 확정한다.

## A. Attempt3 — 2B A100 Validation-only

2026-11-16 Attempt3 B안의 CPU 검증과 로컬 Dry-run이 완료됐다. 앞선 10월 30일 시도는
할당 전 TLS 오류로 중단됐고, 11월 14일 시도와 Attempt2는 A100 할당 단계에서 실패했다.
Attempt3의 실제 GPU 성공과 Runtime 5단계는 **미검증**이며 `execution_ready=false`다.
[10월 30일 일지](../journal/2026-10-30.md), [11월 14일 일지](../journal/2026-11-14.md),
[11월 16일 일지](../journal/2026-11-16.md).

기존 승인은 사용자 회사 WSL에서 2B 모델의 A100 논리적 할당 최대 1회, 파일/SHA 확인,
Production Processor, Qwen 모델 로딩, Layer 5 SAE 로딩, CUDA Synthetic Hook 검증에
한정된다. Generate 0회, 추가 Qwen Model Forward 0회, Synthetic Hook Forward 최대 2회,
자동 재할당 0회이며 불확실하거나 실패한 뒤 새 할당을 시작할 수 없다. 기존 가이드의 위험 수락과
사전·사후 점검 조건을 적용한다. 엄격한 HTTP POST 1회 정책의 NO-GO도 유효하다. 승인 JSON의
SHA 확인으로 승인 사용 상태, GPU 세션 종료 또는 Runtime 성공을 판정할 수 없다.
[Attempt3 승인 가이드](../guides/attempt3_approved_validation.md).

## B. 2B 재현·특이성 검증 — 별도 승인 대상

Growth019에서 무개입 Baseline과 Layer 5 Feature 30908의 Positive Steering을 재현하는
조건을 계획한다. 무개입 Baseline, 비교 Feature 12207, Norm-matched Random Null,
Peer Feature, 문항별 Matched Control과 독립 확증 평가를 설계한다.
과거 Pilot의 정확한 로딩 Revision은 미기록이다. 사전 고정 Revision에서의
실행과 과거 설정의 재현 범위는 증거에 따라 해석한다.
[10월 30일 연구 보고서](../reports/MindScopeX_연구진행현황_2026-10-30.md),
[기존 Cross-Case 계획](../audits/2026-10-30-feature-specificity-cross-case/plan.md).

### B.1 Margin 부호와 개입 방향

- `margin = logprob(lure) - logprob(correct)`이며 양수는 Lure 선호다.
- 기존 `effects.py`와 `research.py`의 `margin_delta`는 `baseline_margin - intervened_margin`
  방향이다. `effects.py`의 Ablation 결과에서는 개입값이 `ablated_margin`이다. 같은 답안
  매핑에서 양수는 개입 후 Lure 선호 감소를 뜻한다. Ablation은 Feature 제거 또는 억제를
  시험한다.
- Growth019의 기존 30908 +7/+8 조건은 양수 계수로 SAE 방향을 주입한 Positive Steering이다.
  현재 `make_feature_steering_hook` 소스는 계수를 곱한 decoder 방향을 residual에 가산한다.
  실행 전에 실제로 사용할 코드 경로의 모드, 계수 부호, Hook 계산식과 지표 산출식을 대조해
  최종 정의를 고정한다.
- 독립 평가의 Primary Metric은 문항별 연속 변화량
  `steer_shift = intervention_margin - baseline_margin`이다. 양수는 Lure 쪽 이동, 음수는 Correct
  쪽 이동을 뜻한다. 30908의 공통 기대 방향은 가정하지 않는다. 양측 검정 또는 데이터셋별로
  근거를 갖춰 사전에 지정한 방향 검정을 사용한다. 생성 답 전환은 보조 지표로 기록한다.
- 두 지표가 같은 문항·후보·답안 매핑·개입 조건과 측정 프로토콜의 동일한 Baseline·개입 Margin 쌍을
  가리키면 `steer_shift = -margin_delta`다. 이 조건에서 기존 `margin_delta` 열을 B 지표로
  가져올 때 부호를 한 번만 반전하고 두 열의 이름과 출처를 구분한다. 부호 변환은 Ablation
  또는 다른 점수 프로토콜의 결과를 Positive Steering 측정으로 바꾸지 않는다.
- 후보 전체 Teacher-forced Logprob, 실제 생성의 첫 분기 Raw/Processed 점수, 제한 생성
  응답 및 자유 생성 응답은 측정 정의를 각각 기록한다. 연구 결론의 범위는 사용한 생성
  프로토콜에 따른다. Thinking On/Off 비교를 추가할 경우 같은 모델 체크포인트인지 확인한다.

[Margin 및 개입 코드](../../src/mindscopex_analysis/effects.py),
[기존 연구 지표 코드](../../src/mindscopex_analysis/research.py),
[Steering Hook 코드](../../src/mindscopex_analysis/qwen_scope.py),
[통제 연구 설계](../../docs/study_design.md).

기존 관측은 데이터셋과 평가 프로토콜별로 해석한다.

- **Hagendorff CRT** — 두 주간 일지의 30908 +8 Dose-response 보고값은 39.5%→82.5%
  (+43.0%p)다. 평가 분류는 **MIXED / PARTIALLY EXPOSED — selection-contaminated split held-out evaluation**이다.
  Split Seed 0~4별로 같은 데이터셋의 held-out 40문항을
  평가하는 설계였고, 평가 결과는 Feature·Coefficient 선택에도 사용됐다. 고정 +7 원본은
  총 200슬롯에 112개 고유 문항이었고 Seed 쌍별로
  9~14개 문항이 겹쳤다. Seed 0의 +8 원본 40문항은 당시 Seed 0 discovery 표본과 분리됐으며
  정답 수는 17/40→33/40(42.5%→82.5%, +40.0%p)이었다. 재구성한 Seed 1~4 평가 세트와 Seed 0의
  discovery 24문항은 각각 8·9·8·4문항이 겹친다. Seed 1~4의 +8 원본과 전체 집계의 재계산은
  `NOT VERIFIED`이며, 계획한 120개 조건의 완전성도 미확정이다. +8은 스윕에서 가장 높은
  관측 조건이었으며 선택 이후의 독립 확증 평가는 확인되지 않았다. 이 결과는 독립 데이터셋
  일반화 근거에서 제외한다.
  [10월 1~10일 일지](../journal/2026-10-01_2026-10-10.md),
  [10월 11~18일 일지](../journal/2026-10-11_2026-10-18.md).
- **Fresh v2** — 전체 정답률은 Baseline 54.5%, +7 55.0%, +8 48.5%였다. 유형별로도
  개선과 악화가 함께 관측됐다.
- **Growth019 Pilot** — 30908 +7은 Margin을 움직였고 Correct를 유지했다. +8에서는
  Baseline의 Correct(`55 days`)가 Lure(`28 days`)로 전환됐다. 첫 분기 Raw/Processed
  Margin은 모두 0으로, 저장 Logit은 동률이었다.
  30908 +8.5와 12207 +8의 Raw Margin은 각각 +0.125, +0.625였고 둘 다 Lure가 생성됐다.
  12207은 비교 Feature로 쓰며 중립성은 검증되지 않았다.

[10월 30일 일지](../journal/2026-10-30.md),
[10월 30일 연구 보고서](../reports/MindScopeX_연구진행현황_2026-10-30.md),
[Growth019 Pilot Audit](../audits/2026-10-30-growth019-minimal-pilot/report.md).

### B.2 Norm-matched Null과 Matched Control

- 30908 +8과 비교할 Random Null도 동일한 Layer·Hook·토큰 위치에 동일한 계수 규칙으로
  가산한다. 계수 적용 전 방향의 L2 Norm을 맞추고 요청 벡터 Norm, dtype 변환 후 벡터 Norm,
  실제 residual 변화량을 기록할 방법과 허용 오차를 정한다. 제거 방식의 Null은 대응하는
  Ablation 조건에서만 사용한다.
- 같은 문항의 Baseline·Steering·Null에는 Prompt, Tokenizer, Model/SAE Revision과
  Generation Protocol을 고정한다. 동일한 계수만으로 방향 Norm이 같다고 간주할 수 없다.
  12207과 추가 Peer Feature의 선정 기준 및 Norm 비교 방법을 사전에 고정한다.
- Matched Control은 가능한 한 동일한 과제·답안 형식에서 함정 단서를 제거한 문항 쌍으로
  구성한다. 답안 매핑, 비교 가능한 채점 규칙, 양측 또는 사전 지정 방향에 대응하는 쌍별
  효과 차이와 통제군 허용 효과를 사전에 고정한다.

기존 `control_specificity_rows`는 Hostile과 Matched Control에 원래 Hostile 문항의
`correct_answer`와 `lure_answer` 문자열을 동일하게 적용한다. `hagendorff_crt`와
`crt_fresh_v2`는 각각 150/150문항에 Matched Control이 있으며, 모든 쌍에서 Hostile의
`lure_answer` 값이 Control 문항의 의미상 정답이다.

기존 Margin은 두 조건 모두 원래 Hostile 문자열의 `logprob(lure_answer) - logprob(correct_answer)`다.
양의 Margin은 Hostile에서는 의미상 오답인 후보를, Control에서는 의미상 정답인 후보를
더 높게 채점했다는 뜻이다. 따라서 같은 Margin 부호라도 조건에 따라 의미상 정오가 달라진다.
기존 `specificity_gap = hostile_delta - control_delta`는 이 고정된 후보 문자열에 대한 개입 효과의
조건 간 차이를 측정한다. 기존 계산 방식과 결과를 보존한다. 의미상 정답률 개선을 판정하려면
조건별 의미상 정답을 기준으로 응답 정확도를 별도로 평가한다.

B의 Control 분석 전에는 조건별 의미상 정답·오답 후보와 채점 방향, 정답 재매핑의 필요 여부,
매핑이 모호한 쌍의 처리, 기존 지표와 구분할 출력명을 확정한다. Random Direction Null은
동일 개입 조건에서 방향을 바꾼 비교이고, Matched Control은 함정 단서를 바꾼 문항 쌍의
비교다. 두 비교가 각각 어떤 판정에 쓰이는지와 계산 규칙을 사전에 고정한다.
[기존 Specificity 계산 코드](../../src/mindscopex_analysis/research.py),
[Matched Control 구성](../../docs/datasets.md),
[Hagendorff 원본 JSON](../../src/mindscopex_analysis/data/hagendorff_crt.json),
[Fresh v2 원본 JSON](../../src/mindscopex_analysis/data/crt_fresh_v2.json).

[기존 Cross-Case 계획의 Norm 계측](../audits/2026-10-30-feature-specificity-cross-case/plan.md),
[통제 연구 설계](../../docs/study_design.md).

### B.3 Selection Bias와 다중 비교

30908 선정 전후에 검색한 Layer 수, 후보 Feature 수, 실제 비교 계수 수, 사후 선정한 문항과
유형, 반복 실험 및 중간 결과에 따른 변경, 기타 탐색적 판단을 기록과 대조해 복원한다. 9월 30일
기록의 Layer 5 후보 27개는 복원에 사용할 한 단계의 근거다. `selection_k`를 후보 Feature
수만으로 정하거나 독립적인 전체 Cartesian Product를 실제 검색했다는 근거 없이 Feature ×
Coefficient × 문항 수를 곱해 보정값으로 삼지 않는다. 가능하면 각 Null 반복에서 전체 적응적
선택 절차를 재현한다. 이력을 복원할 수 없으면 해당 분석은 탐색적 근거로 표시하고 독립 문항의
사전 고정 평가를 우선한다.
[9월 탐색 일지](../journal/2026-09-21_2026-09-30.md),
[Fresh v2 비교 일지](../journal/2026-10-23.md), [통제 연구 설계](../../docs/study_design.md).

### B.4 Held-out 평가의 실행 전 확정 항목

B의 Primary Confirmatory Evaluation에는 Feature 30908, +8 계수, 문항 유형 또는 가설
선택 과정에 사용되지 않은 문항만 배정한다. `hagendorff_crt`, `crt_fresh_v2`,
`crt_fresh_v2_rate_order_pilot`는 주 평가에서 제외한다. 세 데이터셋의 Hostile, Matched
Control, 동일 문항, 표현·답안 순서 변형과 사실상 동일한 Template 기반 문항까지 평가 노출을
확인한다.

`crt7_classic`, `crt2`, `crt_fresh_v1`은 평가 후보로만 둔다. Hagendorff와 기존 평가
데이터에 대해 문항 ID, 정답·Lure 구조, Template, 의미적 중복과 Feature·Coefficient 선택
노출을 확인하고 사용 가능 여부를 판정한다. 검토를 통과해 결과 확인 전에 고정한 문항에만
Held-out 지위를 부여한다. [데이터셋과 중복 기록](../../docs/datasets.md).

Fresh v2에 속하는 Growth019/004/016은 기존에 노출된 문항이다. 이 세 문항의 Cross-Case
분석은 `Secondary / Exploratory / Reproduction`에 배정하며 B의 독립 일반화 주 평가에서
제외한다. 기존 실험 계획과 Artifact는 보존한다. 세 문항의 Bootstrap은 일반화된 효과의
정밀한 신뢰구간을 뒷받침할 수 없다.

GPU 실행 전에 아래 항목을 확정하고 평가 입력과 분석 기준을 고정한다.

- 주 평가 문항의 출처와 Case ID, 기존 실험 노출 여부, 중복·Paraphrase·동일 Template 점검
- 최소 독립 문항 수와 그 근거, 가족·유형별 평가 기준과 집계 방법
- 문항별 `steer_shift`의 측정 시점과 전체 문항에 적용할 **Primary Test Statistic**. 부호
  있는 평균 Shift, 문항별 절댓값의 평균 Shift, 방향과 크기를 함께 고려하는 사전 지정 분석을
  후보로 두고 B 단계 GPU 실행 전에 확정한다. 절댓값 평균은 변화 크기를 요약하므로 방향은
  별도로 기록한다. 어느 통계량이든 관측 조건, Null 및 비교군에 동일한 계산 규칙을 적용한다.
- 양측 검정 또는 데이터셋별 사전 지정 방향과 그 근거, 최소 실질 효과 크기 및 Matched Control
  대비 특이성 기준
- Random Null 반복 횟수, 양쪽 꼬리를 반영하는 Null 판정 통계량과 임계값, 적용할 다중
  비교·선택 보정 방법과 범위. 데이터셋별 방향을 사전 지정한 경우 대응하는 꼬리도 고정한다.
- 문항 단위 대응 Bootstrap과 신뢰구간 계산 방법. 동일 문항의 여러 Seed·조건·계측 호출을
  함께 재표집하며 문항 수를 중복 계상하지 않는다.
- 정확한 동률과 동률 근처(`|margin|`이 작은 경우)를 구분할 절댓값 임계값. Baseline이나
  개입 조건 중 하나라도 해당하면 표시하고 연속 변화량과 함께 기록한다.
- Rate Control 분석. 기존 Fresh v2의 Rate 정답 수는 Baseline 54/66, 30908 +7 1/66,
  +8 2/66이었다. 답변 형태·출력 토큰·후보 위치 편향이 이 변화를 설명할 가능성을 검증
  가설로 둔다. 의미상 정답과 Lure를 구별하는 채점 기준, 선택지 순서·표면 문자열·Template
  변형에 대한 민감도, 실제 Margin 변화와 Behavioral 정답률 변화의 관계, 동일 조건의
  Matched Control 및 Norm-matched Random Direction 비교 방법을 사전에 확정한다.
  과거 수치로 형태 편향의 원인은 확정되지 않았다. [Fresh v2 유형별 관측](../journal/2026-10-23.md).
- 분기 미도달, 무효 응답, 누락 Artifact의 처리와 실패·불확실 판정 기준

[기존 First-Divergence 감사](../audits/2026-10-29-first-divergence/report.md),
[기존 Cross-Case 계획](../audits/2026-10-30-feature-specificity-cross-case/plan.md).

### B.5 판정 기준

| 판정 | 조건 |
|---|---|
| GO | 독립 문항의 `steer_shift`에 적용한 사전 지정 주 검정통계량이 선택한 검정의 판정 기준과 최소 실질 효과 기준을 충족하고, 다중 비교·선택 보정 후 Null 비교 및 Matched Control 대비 특이성 기준을 모두 통과함 |
| NO-GO | 충분히 검증했으나 사전 정의한 가설의 핵심 조건을 충족하지 못함 |
| INCONCLUSIVE | 실행 오류, 결과 누락 또는 통계적 불확실성으로 기준 충족 여부를 판정할 수 없음 |

GO / NO-GO / INCONCLUSIVE의 세부 판정 기준은 B 단계 GPU 실행 전에 고정한다. 사후
기준 변경을 GO 판정에 사용하지 않는다. 2B에서 부정적 결과를 얻으면 결과를 보존하고 추가
탐색은 별도 연구로 다룬다.

## C. 27B 확장 — 설계 및 비용 검토

| 현재 설정 | 계획 기준 |
|---|---|
| [study_27b.toml](../../experiments/configs/study_27b.toml) | H100 80GB, `study_margin`, SAE는 CPU에 배치, 자유 생성은 설정 범위 밖 |
| [phenomenon_27b.toml](../../experiments/configs/phenomenon_27b.toml) | 40GB A100에서 CPU Offload와 GPU 메모리 30GiB 상한을 적용한 Baseline `phenomenon` forward, SAE 개입 없음 |

27B SAE 통제 연구는 H100 80GB 설정을 기준으로 설계한다. A100 CPU Offload의 Baseline은
별도 목적의 실행 경로다. 2B 판정과 함께 H100 가용성, 예상 Compute Unit 소모와 비용, 로딩 및
실험 소요 시간, 모델의 GPU·호스트 메모리 사용량과 여유를 확인해 전환을 결정한다. 이 조건과
별도 실행 승인이 충족될 때까지 27B 본실험은 미확정이다.
