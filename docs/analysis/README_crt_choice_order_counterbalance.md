# CRT Answer Choice Order Counterbalancing Experiment

구현과 CPU 테스트만 수행했다. 모델, GPU, Colab을 실행하지 않았으며 실제 성능 결과나
`results/analysis/crt_choice_order_counterbalance/`를 생성하지 않았다.

## 기존 코드 확인

| 항목 | 기존 위치 및 동작 |
|---|---|
| 문항 로딩 | `lure_datasets.py::load_lure_dataset`, `research_experiments.py::_load_splits`; crt_fresh_v2 원본 질문·정답·Lure 로딩 |
| 프롬프트 | `cases.py::_answer_prompt`: 질문 뒤 `Answer:`; `prompts.py::instruct_lure_cases`: 기존 답값·단위 출력 지시문 추가 |
| 선택지 배치 | 기존에는 A/B 선택지 표시 및 순서 제어 기능 없음 |
| 출력 및 판정 | `research.py::_binary_choice_completion`: 정답값·Lure값만 허용하는 토큰 trie; `_generate_labels`: `classify_lure_answer` 판정 |
| 문항 선정 | `research.py::split_lure_cases`: case_id와 Seed의 stable hash, train_frac 0.6; `family_balanced_subset`: held-out에서 유형별 round-robin 40문항 |
| Steering | `research.py::steer_generation_labels`에서 layer block forward hook 등록; `qwen_scope.py::make_feature_steering_hook`가 coefficient × SAE decoder 방향을 residual에 더함 |
| 디코딩 | Base checkpoint, chat template 없음, do_sample=False, renormalize_logits=True, eos까지 후보값 제약, max_new_tokens 최소 16 또는 최장 후보 토큰 수+1 |

## 설계 및 구현

사용자가 확인한 방식대로 **답값 출력과 기존 디코딩을 유지하고 A/B 위치를 파생**한다.
모델은 A/B 문자를 생성하지 않는다. 기존 답변 지시문은 변경하지 않는다.

`choice_order.py::ordered_case`는 기존 `Answer:` 바로 앞에 다음 선택지만 추가한다.

```text
correct_first:       lure_first:
A: <Correct>        A: <Lure>
B: <Lure>           B: <Correct>
Answer:             Answer:
```

두 조건은 동일 원본 LureCase에서 만들어지며 case_id, 본문, 숫자, 단위, 정답값, Lure값과
모든 메타데이터가 동일하다. 프롬프트에서 선택지 값의 위치만 교환한다.
정답값·Lure값의 공통 앞뒤 공백은 표시에서 제거하고 디코딩 후보는 기존 그대로 유지한다.

기존 behavioral 설정에 `choice_orders`가 있을 때만 신규 분기로 진입한다.
없으면 기존 코드 경로를 그대로 실행한다. 신규 분기는 기존 모델 로딩, pinned SAE,
`steer_generation_labels`, feature 30908, layer 5, token_position all을 재사용한다.
Baseline은 기존과 같이 각 coefficient 호출에서 재평가한다. 두 순서를 동일한 모델로
순차 실행하며 hook은 기존 finally 구문에서 제거된다.

각 block에는 원본 case 정의, 실제 렌더링 프롬프트, 순서, coefficient, 응답 인코딩,
baseline/steered 답값·의미 라벨·파생 A/B 위치를 저장한다. 실행 시 잘못된 답값·라벨 매핑,
문항 누락·중복, baseline 변동 및 coefficient 0 불일치는 오류로 중단한다.

## Seed별 평가 건수

- Seed당 고유 원본 40문항 × 2순서 × 3계수 = 240개 steered 관측.
- 기존 evaluator가 각 계수에서 baseline을 재실행하므로 baseline 240개를 더해 **480회 생성**.
- 중복 실행을 제외한 baseline 조건은 40 × 2 = 80개이며, coefficient 0의 steered 결과도 baseline과 동일해야 한다.
- 5개 Seed 합계는 2,400회 생성, steered 1,200개 관측. 독립적인 1,200문항이라는 뜻은 아니다.

CPU 회귀 테스트에서 새 설정의 Seed 1~5 문항 집합이 기존 Negative Dose 결과의
각 40개 case_id 집합과 정확히 일치함을 확인했다.

## 추후 Colab A100 실행

프로젝트 루트의 WSL에서, 기존 Colab CLI 인증·환경 준비 후 실행한다.
아래 명령은 이번 작업에서 실행하지 않았다.

```bash
bash experiments/run_colab.sh experiments/suites/crt_choice_order_counterbalance.toml \
  --session mindscopex-choice-order --gpu A100 --exec-timeout 14400
```

한 Seed만 실행하려면:

```bash
bash experiments/run_colab.sh \
  experiments/configs/study_choice_order_counterbalance_f30908_seed1_2b.toml \
  --session mindscopex-choice-seed1 --gpu A100 --exec-timeout 14400
```

기존 run_colab.sh → launch_colab → research_experiments 경로를 유지한다.
각 Seed는 두 순서를 모두 실행한다. A100 할당 및 실행 시간은 실행 환경에 의존한다.
기존 실험 디렉터리를 실행 대상으로 지정하지 않는다.

## 실제 실행 후 분석

다섯 run의 `artifacts/behavioral/generations.json`이 `results/runs/`로 회수된 후:

```bash
MPLCONFIGDIR=/tmp/mindscopex-mpl .venv/bin/python scripts/analyze_crt_choice_order.py \
  --input-root results/runs \
  --output results/analysis/crt_choice_order_counterbalance
```

입력은 Seed마다 정확히 하나여야 한다. 두 순서·세 계수·40개 고유 문항, 원본 문항 정의,
렌더링 프롬프트, coefficient별 문항 및 baseline 일관성, coefficient 0 동일성,
정답값·의미 라벨·위치 매핑을 검증한다. 누락·중복·불일치는 중단하며 출력 디렉터리는
기본적으로 덮어쓰지 않는다. 입력 경로와 SHA-256을 README에 기록한다.

생성하도록 구현한 파일:

| 파일 | 내용 |
|---|---|
| seed_order_summary.csv | Seed·계수·순서별 baseline/steered 정답률, 전환, 교정·회귀율, A/B 선택률 |
| family_order_summary.csv | 위 지표의 Difference/Rate/Growth 유형별 집계 |
| choice_position_bias.csv | Seed·계수·유형별 문자/의미 일치율, 두 순서의 교정 빈도, 공통 교정, 교정 유지율, 효과 부호 반전 및 순서 간 pp 차이 |
| paired_case_transitions.csv | 동일 문항의 두 순서에서 baseline/steered 의미·위치·전환과 일치 여부 |
| order_accuracy_comparison.png | 순서·Seed·계수별 정답률 |
| order_net_improvement.png | 순서·Seed·계수별 pp 변화 |
| README.md | 입력 해시, 검증, 지표, 실제 결과 표, 한계, 재실행 명령 |

정답률·A/B 비율·교정률·회귀율은 0~1 비율이고 net_improvement_pp는 percentage point다.
교정률=L→C / baseline Lure, 회귀율=C→L / baseline Correct이며 분모 0은 빈 CSV 셀이다.
전환 합계는 그룹 문항 수와 일치해야 한다. 문자 일치율은 파생된 A/B 위치 일치율이며,
의미 일치율은 Correct/Lure 라벨 일치율이다. Correct-first 교정 유지율은
두 순서 모두 교정된 문항 수 / Correct-first 교정 문항 수다.

순서 효과 반전 여부는 두 순서의 net pp 곱이 음수인지로 정의한다. 0과 양수/음수의
조합은 반전으로 분류하지 않는다. -7/-8 안정성은 유형별 효과 방향, 순서 간 pp 차이와
Seed별 변동을 함께 검토하며 특정 계수를 자동으로 우월하다고 결론내리지 않는다.

## 검증 및 변경 파일

```bash
MPLCONFIGDIR=/tmp/mindscopex-mpl .venv/bin/python -m unittest discover -s tests -v
```

최종 전체 테스트: 477개 실행, 실패 0, skip 4. `git diff --check` 통과.
Suite 설정 5개가 기존 launcher에서 정상 해석되는 것을 확인했다.

합성 fixture는 임시 디렉터리에서만 사용한다. 테스트는 프롬프트·원본 동일성, 답값 매핑,
문항 집합, 전환 집계, 위치 비율 및 pp, 쌍 비교, 잘못된 매핑·입력·baseline 검출,
기존 문항 선정과 기존 steering 호출 호환성, 출력과 덮어쓰기 방지를 포함한다.

수정: `experiments/jobs/research_experiments.py` (명시적 choice_orders 분기만 추가).

신규: `src/mindscopex_analysis/choice_order.py`, `scripts/analyze_crt_choice_order.py`,
`tests/test_crt_choice_order.py`, 이 문서, Seed 1~5 신규 설정,
`experiments/suites/crt_choice_order_counterbalance.toml`.

기존 실험 TOML·결과·Negative Dose 분석·mindscopex_inventory.py는 수정하거나 삭제하지 않았다.
Git commit/push를 수행하지 않았다.

## 남아 있는 한계와 다음 연구

이번 실험 내부 비교에서는 선택지 순서만 달라진다. 기존 Negative Dose 실험에는 표시된
선택지가 없었으므로 기존 결과와 이번 결과 사이에는 선택지 표시 자체의 차이도 있다.
따라서 두 연구 간 차이를 순서만의 효과로 해석할 수 없다.

A/B는 생성된 문자가 아니라 답값에서 파생한 위치다. 실제 문자 생성 편향을 직접
검증하려면 별도 A/B 디코딩 실험이 필요하다. 답값 자체의 토큰화·길이·빈도 편향은 남는다.
이 실험의 constrained binary_choice 정확도는 실제 추론 개선을 입증하지 않으며,
음수 steering은 feature ablation과 다르다. Seed 간 문항 중복과 Seed 5개의 한계가 있다.

실제 실행 후 두 순서 공통 교정과 효과 반전, Difference 개선·Growth 악화의 유지 여부,
-7/-8의 순서·Seed 간 변동을 먼저 확인한다. 이후 선택지 표시 없는 대조군,
A/B 출력 대조군, 중복 없는 held-out 문항 및 자유 생성 조건을 검토할 수 있다.
