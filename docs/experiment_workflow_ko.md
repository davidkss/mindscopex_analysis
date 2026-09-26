# MindScopeX 실험 흐름: 탐색 Notebook에서 통제 연구까지

이 문서는 연구자가 실험의 목적, 실행 순서, 결과의 의미를 파악하기 위한 한국어 가이드다.
코드 사용법이나 상세 설계 문서를 대체하지 않는다.

작성 기준은 2026-09-26에 확인한 `main` 커밋 `88bdb9f`이다. 작성 당시 로컬 `main`과
원격 `main`의 커밋이 같았다. 아래 설명은 Notebook 00~13, 현재 TOML 설정, job 및 공통
연구 코드를 대조한 것이다. Validation 수치는 로컬 실행
`results/runs/20260926-131348_study_2b_validation/artifacts/`의 실제 산출물과 대조했다.
이 결과 디렉터리는 버전 관리 대상이 아니므로 다른 clone에는 없을 수 있다.

## 목차

1. 연구의 최종 목표
2. Notebook과 Experiments의 역할
3. Notebook 00~13과 현재 실험의 대응
4. 실험 디렉터리와 실행 흐름
5. Controlled-study의 다섯 단계
6. 최근 2B A100 validation의 설정과 결과
7. 앞으로의 full study
8. 연구자가 결과에서 확인할 질문
9. 중요한 해석 원칙
10. 기존 문서와의 관계 및 현재 코드와의 차이

## 1. 연구의 최종 목표

MindScopeX의 핵심 연구 목표는 CRT 문제에서 모델이 선택하는 직관적인 오답(Lure)과
관련된 내부 SAE Feature를 발견하고, 해당 Feature에 인과적으로 개입했을 때 Lure 응답이
감소하고 Correct 응답이 증가하는지를 검증하는 것이다. SAE는 모델의 내부 활성 벡터를
분해해 관찰·개입할 후보 방향을 제공한다. Feature 번호만으로 그 의미가 확정되지는 않는다.

최종 질문은 다음과 같다.

> 직관적인 오답을 유발하거나 지지하는 내부 Feature를 찾아 억제했을 때, 모델이 Thinking
> 여부와 관계없이 Lure 대신 Correct 답을 선택하도록 만들 수 있는가?

이를 주장하려면 특정 문제 하나의 성공을 넘어 여러 CRT 문제와 family, discovery에 쓰지
않은 held-out 문제, matched control, random/null intervention, Thinking 조건과
Non-thinking 조건에서 효과를 검토해야 한다. 현재 validation이 이 목표를 이미 증명한 것은 아니다.

특히 **2B의 SAE 분석 모델은 `Qwen/Qwen3.5-2B-Base`**이고, Notebook 00과 행동 baseline
config에서 사용하는 post-trained `Qwen/Qwen3.5-2B`와 구분해야 한다. Base에서 발견한
feature의 효과를 post-trained 모델의 Thinking/Non-thinking 효과로 곧바로 해석할 수 없다.
Non-thinking도 내부 계산이 없다는 뜻은 아니다.

## 2. Notebook과 Experiments의 역할

### Notebook: 탐색용 연구 노트 / 현미경

`notebooks/`는 아이디어 탐색, 코드와 모델 내부 동작 이해, 작은 실험, 디버깅, 시각화,
새 가설의 빠른 확인을 담당한다. 한 문항의 활성과 개입 결과를 가까이서 살펴보고,
예상과 다른 결과의 원인을 추적하기에 적합하다. 삭제하거나 폐기하는 대상이 아니다.

### Experiments: 정식 실험 파이프라인

`experiments/`는 조건을 TOML로 기록하고 같은 절차를 반복 실행하는 구조다.
그중 `research_experiments`의 controlled study는 train/held-out 분리, random/null 비교,
matched-control specificity, behavioral generation을 연결한다. Runner는 Colab GPU 실행과
artifact 수집을 자동화한다. 모든 job이 자동으로 이 통제를 전부 수행하는 것은 아니므로
어떤 job과 kind를 실행하는지 확인해야 한다.

```text
Notebook
  -> 탐색
  -> 의미 있는 가설 발견
  -> experiments로 정식 구현
  -> Colab GPU 실행
  -> artifacts
  -> 통계 분석
  -> 논문
```

탐색에서 좋은 결과가 나온 조건만 골라 본 실험 결과처럼 보고하면 안 된다. 탐색으로 정한
가설·설정과 평가에 사용할 데이터를 구분하고, 결과가 기대와 달라도 같은 절차로 기록한다.
통계 분석과 논문 반영까지 runner가 자동으로 완성해 주는 것은 아니다.

## 3. Notebook 00~13과 현재 실험의 대응

아래 상태는 연구 목적의 이전 범위다. **정식 실험으로 이전**되어도 원래 Notebook은 남는다.
**일부 이전**은 질문의 일부만 batch 경로에 반영되었다는 뜻이고, **Notebook 유지**는 직접
비교할 탐색 도구로 남는 경우, **탐색 전용**은 해당 비교를 현재 표준 study가 수행하지 않는 경우다.

| Notebook | 기존 연구 목적 | 현재 experiments 대응 | 현재 역할 | 상태 |
|---|---|---|---|---|
| [00 CRT text responses](../notebooks/00_qwen_crt_text_responses.ipynb) | 실제 CRT 응답과 Thinking/Non-thinking 행동 baseline | `crt_text_responses` job, `00_hagendorff_*` config 등 | 생성 원문·응답 형식·행동 기준선 확인. Feature 개입 study와 별도 실험 | 정식 실험으로 이전 |
| [01 Activation MVP](../notebooks/01_qwen_scope_activation_mvp.ipynb) | residual activation 캡처와 SAE 활성 관찰 | `discover`가 활성 캡처 후 layer별 후보를 평가 | 활성 이해·시각화는 Notebook, 다문항 localization은 job | 일부 이전 |
| [02 Lure feature ablation](../notebooks/02_bat_ball_lure_feature_ablation.ipynb) | bat-and-ball 한 문항에서 feature 제거 효과 탐색 | `discover`의 train 다문항 feature discovery | 단일 문항 후보 탐색을 반복 활성·다문항 평가로 확장 | 정식 실험으로 이전 |
| [03 Layer sweep](../notebooks/03_layer_sweep_feature_search.ipynb) | layer별 강한 후보 탐색 | `discover`의 layer localization과 null screen | 정식 후보 비교·선택, `study_feature.json` 저장 | 정식 실험으로 이전 |
| [04 Coefficient dose response](../notebooks/04_coefficient_dose_response.ipynb) | 한 prompt의 coefficient별 teacher-forced margin 곡선 | `behavioral`의 coefficient별 생성 답변 비교 | 강도 민감도라는 질문은 반영; 원래 margin dose sweep은 표준 study에 없음 | 일부 이전 |
| [05 Intervention mode comparison](../notebooks/05_intervention_mode_comparison.ipynb) | 제거·projection 제거·방향 추가 등 방식 비교 | `[discover].intervention_mode`로 한 방식을 지정 가능 | 현재 study는 `remove_activation` 한 방식. 여러 mode 자동 비교는 미채택 | 탐색 전용 |
| [06 Control specificity](../notebooks/06_control_prompt_specificity.ipynb) | 함정 문항과 matched control의 개입 효과 비교 | `control_specificity` | held-out의 hostile/control 쌍에서 특이성 평가 | 정식 실험으로 이전 |
| [07 Paraphrase robustness](../notebooks/07_paraphrase_robustness.ipynb) | 같은 문제를 다른 표현으로 바꿔 효과 확인 | `causal_heldout`은 다른 문항 일반화; `feature_falsification`은 조건부 paraphrase 진단 지원 | held-out은 같은 문제의 표현 강건성과 다름. 기존 paraphrase 검사는 계속 필요 | Notebook 유지 |
| [08 Answer format sensitivity](../notebooks/08_answer_format_sensitivity.ipynb) | 숫자·단위·문장형 답 표기에 따른 margin 민감도 | 표준 study에 답 표면형 sweep 없음 | `binary_choice` 지정만으로 표현 민감도를 검증한 것은 아님 | 탐색 전용 |
| [09 Token position sweep](../notebooks/09_token_position_sweep.ipynb) | prompt 내 개입 위치별 효과 | behavioral의 `token_position` 설정, 별도 `feature_diagnostics`의 위치 진단 | 위치 선택·활성 진단은 존재하지만 원래 인과적 위치 sweep을 대체하지 않음 | 일부 이전 |
| [10 CRT transfer](../notebooks/10_crt_transfer.ipynb) | bat-and-ball 후보를 다른 CRT에 적용 | `causal_heldout` | train에서 찾은 후보를 미사용 test 문항에 적용. Family 전체를 제외하는 검증은 별도 | 정식 실험으로 이전 |
| [11 Control delta bypass](../notebooks/11_control_delta_bypass.ipynb) | control residual − lure residual 방향을 더해 우회 | 표준 study에 대응 stage 없음 | SAE feature 제거와 다른 개입 가설을 탐색 | 탐색 전용 |
| [12 Decoder geometry](../notebooks/12_decoder_geometry.ipynb) | 후보 decoder 방향의 cosine 유사도 확인 | `feature_diagnostics`의 siblings, `cross_layer_siblings`의 기하·활성·효과 비교 | 기하 분석 일부는 별도 batch 진단으로 확장; 원래 후보 간 시각화는 Notebook 유지 | 일부 이전 |
| [13 Semantic/logic specificity](../notebooks/13_semantic_logic_specificity.ipynb) | 같은 CRT feature를 semantic/logic 문제에 적용 | `[data].dataset` 교체 및 고정 feature 평가 경로 | 다른 dataset 재발견과 동일 feature의 전이를 구분해야 함 | 일부 이전 |

이 표는 [Notebook–논문 감사 문서](notebook_paper_audit.md)를 출발점으로 삼되 실제 코드를
우선했다. 특히 07의 held-out을 paraphrase의 완전 대체로, 13의 dataset 변경을 동일 feature의
전이 증명으로 보지 않는다. `feature_falsification`의 현재 코드는 진짜 paraphrase에 동일
시나리오의 서로 다른 표현을 요구하며, 현재 제공 데이터에는 그 검사를 위한 쌍이 없다고
명시한다. Family 간 차이를 paraphrase 강건성이라고 부르면 안 된다.

12의 기하 분석 및 09의 위치 관찰에는 후속 batch job이 있지만, 표준 `study` suite에 포함되지
않으며 원래 Notebook의 모든 비교를 재현하지 않는다. **Experiments는 Notebook 00~13을
100% 대체하지 않는다.**

## 4. 실험 디렉터리와 실행 흐름

| 경로 | 연구자가 이해할 역할 |
|---|---|
| `experiments/suites/` | 여러 config를 순서대로 묶는 실행 단위. `study.toml`처럼 config 하나만 묶을 수도 있음 |
| `experiments/configs/` | 모델, dataset, layer, 후보 수, coefficient, GPU 등의 조건 기록 |
| `experiments/jobs/` | 어떤 연구 단계를 어떤 순서로 실행할지 정하는 코드 |
| `experiments/runners/` | Colab session·hardware 확인, source upload, 원격 실행, artifact download 자동화 |
| `src/mindscopex_analysis/` | Notebook과 Experiments가 공유하는 모델·SAE·활성·margin·개입·null·생성 로직 |
| `results/runs/` | 실행별 설정 사본, source archive, 로그, manifest, 내려받은 artifacts |

```text
Suite
  |
  v
Config
  |
  v
Job
  |
  v
src/mindscopex_analysis  (공통 연구 로직)
  |
  v
Google Colab GPU에서 Job + 공통 로직 실행
  |
  v
Artifacts
  |
  v
Results / Analysis / Paper
```

이는 설정과 실행의 관계를 나타낸 그림이다. 실제로는 로컬 runner가 source와 설정을 Colab으로
보내고, Colab 안에서 job과 공통 로직이 실행된다. `[remote].source = "archive"`는 현재
workspace를 전송하므로 미커밋 코드·설정도 실행 대상에 포함된다. `.git`과 결과·캐시 디렉터리
등은 제외된다. 재현성을 위해 실행별 config, source archive, manifest와 로그를 함께 보관한다.

확인할 대표 산출물은 다음과 같다.

- `manifest.json`: 모델·SAE·데이터 분할·완료 상태와 단계별 요약.
- `phenomenon.csv`: 개입 전 문항별 margin.
- `discover_localization.csv`, `discover_features.csv`, `study_feature.json`: 후보 비교와 선택 결과.
- `control_specificity.csv`, `causal_heldout.csv`: control 및 held-out의 문항별 효과.
- `behavioral.csv`, `behavioral/generations.json`: coefficient별 성능과 개입 전후 실제 답변.
- `job.log`, `colab_log.md`, `colab_log.jsonl`: 실행·경고·실패 진단 기록.

PNG는 빠른 확인용이다. 현재 job은 모든 CSV에 대응 PNG를 만드는 것은 아니므로 그림 유무만으로
계산 성공을 판단하지 않는다. 수치 검토는 CSV·JSON·manifest를 기준으로 한다.

## 5. Controlled-study의 다섯 단계

`research_experiments`의 `kind = "study"`는 다음 순서로 실행된다.

```text
phenomenon
    |
    v
discover
    |
    v
control_specificity
    |
    v
causal_heldout
    |
    v
behavioral
```

| 단계 | 쉬운 연구 질문 | 데이터와 해석 범위 |
|---|---|---|
| `phenomenon` | 이 모델이 개입 전 실제로 Lure를 선호하는가? | 선택한 전체 데이터의 teacher-forced margin. 생성 정확도와 동일하지 않음 |
| `discover` | 어느 layer의 어떤 feature를 건드리면 Lure 선호가 줄어드는가? 임의 방향과 다른가? | train의 discovery 부분에서 후보 탐색, 별도 train 부분에서 null screen을 시도 |
| `control_specificity` | 함정 없는 matched control에도 똑같이 영향을 주는가? | held-out의 hostile/control 쌍을 비교. 두 효과 차이가 specificity gap |
| `causal_heldout` | 발견에 쓰지 않은 CRT에서도 개입 효과가 유지되는가? | 같은 선택 feature를 test에 적용한 margin 효과 |
| `behavioral` | 확률 점수뿐 아니라 실제 생성 답도 바뀌는가? | held-out subset에서 coefficient별 baseline/steered 답변 비교 |

처음 네 단계는 NNsight 기반 margin 측정 경로이고, 마지막은 Hugging Face 생성 모델 경로다.
같은 분석 checkpoint를 사용하면서 모델을 순차 로드한다. 발견한 feature와 SAE를 후속 단계가
이어받는 연결도 validation의 검증 대상이다.

부호는 다음처럼 읽는다.

```text
margin       = logprob(Lure) - logprob(Correct)
margin_delta = baseline margin - intervened margin
```

Margin이 양수이면 두 답 중 Lure 선호가 더 크다. Margin delta가 양수이면 개입 후 Lure의
상대적 우위가 줄었다. 이것만으로 Correct 응답이 늘었다거나 reasoning이 개선되었다고
결론 내리지 않는다.

**계수 부호도 경로별로 구분한다.** Notebook 04와 discovery의 `remove_activation`은
양의 계수로 활성 성분을 제거한다. 반면 behavioral은 decoder 방향을 더하는 steering
hook을 사용하므로 음의 계수가 억제 방향이다. 두 계수의 수치와 강도를 동일시하지 않는다.

## 6. 최근 2B A100 validation의 설정과 결과

### 목적과 실제 설정

이 실행의 목적은 **전체 controlled-study pipeline이 처음부터 끝까지 실제로 실행되는지
저비용으로 검증**하는 것이다. 논문 결과를 확정하는 실행이 아니다.

설정은 [study_2b_validation.toml](../experiments/configs/study_2b_validation.toml),
실행 묶음은 [validation.toml](../experiments/suites/validation.toml)이다.

| 항목 | 실제 설정 또는 확인값 | 검증 의도 |
|---|---|---|
| Job / kind | `research_experiments` / `study` | 다섯 단계 모두 실행 |
| 모델 / SAE | `Qwen/Qwen3.5-2B-Base` / `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50` | 대응 checkpoint에서 발견과 개입 |
| GPU / source | A100 / `archive` | 실제 실행 GPU는 A100-SXM4-40GB |
| 데이터 | `hagendorff_crt`, family당 10개 | 총 30개, matched controls 포함 |
| 분할 | `train_frac = 0.6`, seed 0 | 실제 train 16개 / held-out 14개 |
| Discovery | layer `[11]`, `max_cases = 9` | family별 3개로 단일 layer 경로 검증 |
| 후보 | `candidate_top_n = 10`, `min_active_cases = 2`, `max_candidates = 8` | 문항별 탐색 폭 유지, 반복 활성 필터와 scoring 비용 축소 |
| Margin 개입 | `coefficient = 1.0`, `remove_activation` | 기존 study와 같은 방식 |
| Null | 4 draws, `null_cases = 6`, seed 0 | discovery 제외 train 7개 중 6개 평가 |
| Behavioral | held-out 6개, `[0.0, -4.0]` | family별 2개, 무개입과 억제 설정 비교 |
| 생성 | `binary_choice`, `token_position = "all"`, 최대 16 tokens | Correct/Lure 답변 선택과 개입 연결 확인 |

Train 비율은 정확히 개수를 나누는 보장이 아니다. 현재 분할은 case ID와 seed의 안정적인
해시로 정해지므로 30개가 18/12가 아니라 16/14로 나뉜다. Discovery 9개와 별도 null 6개가
확보되었고 artifact의 `screen_in_sample = false`다. `min_active_cases = 2`는 발견 자체를
불가능하게 만드는 과도한 기준을 피하면서 한 문항만의 활성 후보는 제외하려는 설정이다.

### 실제 관측 결과

아래 수치는 `manifest.json`, `study_feature.json`, 각 단계의 CSV/summary JSON과
`behavioral/generations.json`에서 확인했다. 표의 positive는 margin delta가 양수인 문항 비율이다.

| 측정 | 결과 |
|---|---|
| 선택 feature | Layer 11, Feature 19069 |
| Discovery | 9 cases, mean margin delta **+0.059426**, positive **66.7%** |
| Control specificity | n = 14, hostile delta **+0.059053**, control delta **−0.005893**, gap **+0.064946** |
| Held-out causal | n = 14, mean margin delta **+0.059053**, positive **71.4%** |
| Null screen | `screen_null_z = +0.3371`, `screen_null_percentile = 0.75` |

Control specificity의 hostile 측정과 causal held-out은 같은 test 문항·같은 feature에 대한
효과다. 같은 +0.059053이 두 곳에 나온 것을 독립된 두 번의 재현으로 세지 않는다.

| Behavioral coefficient | Accuracy: baseline → steered | Lure rate: baseline → steered |
|---|---|---|
| 0 | 33.3% → 33.3% | 66.7% → 66.7% |
| −4 | 33.3% → 50.0% | 66.7% → 50.0% |

Coefficient −4의 6개 문항 중 Lure → Correct는 **2개**, Correct → Lure는 **1개**,
나머지 **3개**는 변화가 없었다. 따라서 정답 수는 2/6에서 3/6으로 순수하게 한 개 늘었다.
좋아진 두 사례만 제시하면 부작용을 숨기게 된다. Coefficient 0에서는 모두 그대로였다.

Null screen의 실제 관측 평균은 `screen_hostile_delta = +0.058296`,
무작위 방향 평균은 `screen_null_mean = +0.042401`이다. Discovery 평균 +0.059426과
screen 평균은 평가 문항이 다르므로 섞지 않는다. 네 null draws의 percentile 0.75는
관측 효과보다 작은 draw가 3개였다는 뜻이지, 75% 확률로 인과 feature라는 뜻이 아니다.
작은 z와 적은 draws로 강한 random/null 우위를 주장할 수 없다.

### 무엇을 검증했고, 무엇이 남았는가

Feature 19069는 현재 **“Lure-associated causal candidate”**이며 확정된 Lure causal
feature가 아니다. 작은 표본에서 개입 방향의 효과와 일부 답변 전환이 관측되었지만,
random/null 대비 근거가 강하지 않고 일부 Correct 답을 Lure로 바꾸는 부작용도 있었다.
본 실험에서 여러 layer와 더 많은 후보를 대상으로 다시 discovery 해야 한다.

실행 로그와 완료 manifest는 다섯 단계의 연결, 모델/SAE 사용, null 평가, held-out 개입,
생성 steering과 artifact 수집이 동작했음을 보여준다. 당시 `discover_localization.png`는
`null_mean` 필드 참조 오류로 생성되지 않았으나 CSV·JSON과 이후 연구 계산은 완료되었다.
현재 main의 plotting은 canonical field인 `screen_null_mean`을 읽도록 수정되어 있다.
이는 기존 run에서 모든 그림까지 생성되었다는 뜻은 아니며, 이번 문서 작업에서 결과를
재생성하거나 GPU 실험을 다시 실행하지 않았다.

또한 이 behavioral 결과는 **Base 모델의 Correct/Lure 제한 생성**이다. 자유로운 답변 전체나
Thinking/Non-thinking 두 조건에서의 인과 개입 성공을 검증한 결과가 아니다.

## 7. 앞으로의 full study

### Feature 19069를 고정하지 않고 다시 발견한다

Validation에서 나온 번호를 본 실험의 정답처럼 고정하지 않는다. 현재
[study_2b.toml](../experiments/configs/study_2b.toml)에는 `[feature]` 고정값이 없으며,
[study suite](../experiments/suites/study.toml)는 이 config를 실행한다.

```text
여러 Layer
  -> 다수 SAE Feature
  -> Discovery
  -> Random/null comparison
  -> Feature selection
  -> Matched control
  -> Held-out causal validation
  -> Coefficient intervention
  -> Behavioral Correct/Lure 변화
```

현재 기본 discovery는 layer별 후보를 평균 margin delta로 평가한 뒤 대표 feature의
별도 train null screen을 계산하고, 기본 `select_by = "null_z"`에 따라 layer 간 대표를
선택한다. 모든 후보 각각에 강한 selection-adjusted null을 적용하는 것과는 다르다.

| 항목 | Validation | 현재 `study_2b.toml` |
|---|---|---|
| 데이터 | 30개 | Hagendorff CRT 150개 전체 |
| Layer | 11 | 2B profile 기본 `[5, 11, 17, 23]` |
| Discovery 문항 상한 | 9 | 24 |
| 문항별 활성 후보 / 최소 반복 활성 | 10 / 2 | 10 / 3 |
| Layer별 scoring 후보 상한 | 8 | 24 |
| Null draws | 4 | 32 |
| Null 평가 문항 | 6 | 기본 6 |
| Behavioral 문항 상한 | 6 | 40 |
| Behavioral coefficients | `[0.0, -4.0]` | `[0.0, -2.0, -4.0, -8.0]` |
| 출력 / GPU / source | `binary_choice` / A100 / `archive` | 동일 |

현재 full config도 `[null].selection_adjusted`를 켜지 않는다. 코드에 peer-feature 및
selection-adjusted 검증 기능이 있다고 해서 이 suite가 그것까지 자동 실행한다고 보면 안 된다.
본 실험의 주장 강도에 따라 더 강한 null, 여러 split/seed, 표현·family 전이 검증을 별도로
계획해야 한다. 더 큰 config를 실행했다는 사실만으로 최종 인과 기제가 입증되지는 않는다.

인증된 Colab CLI 환경에서 저장소 루트 기준 실행 예시는 다음과 같다.

```bash
# 실행 준비 점검: Colab에 접속하지 않음
bash experiments/run_colab.sh experiments/suites/study.toml --dry-run

# 실제 2B full study 실행 예시
bash experiments/run_colab.sh experiments/suites/study.toml \
  --session mindscopex --gpu A100 --exec-timeout 9000
```

9000초는 대기 제한이며 예상 소요 시간이 아니다. Runner는 기본적으로 종료 시 VM을 중지하고,
`--keep`을 주면 유지한다. 이 문서 작성 과정에서는 위 실험을 실행하지 않았다.

### 본 실험에서 지킬 평가 순서

먼저 모델·SAE·dataset·분할과 후보 탐색 범위를 정하고 train에서 발견한다. 계수를 선택하거나
튜닝한다면 train 내부의 별도 검증 절차로 정하며, test 결과를 보고 가장 좋은 계수만 골라
확인적 결과처럼 보고하지 않는다. **현재 job은 TOML에 적힌 coefficient 목록을 held-out에서
비교할 뿐, 별도의 train 기반 최적 계수 탐색을 자동 수행하지 않는다.**

이후 matched control, held-out margin, behavioral 전환을 문항별로 분석한다. 부정적 효과와
불확실성도 함께 보고하고, 같은 문항의 여러 coefficient·seed 결과를 독립 표본으로 세지 않는다.
Validation에서 이미 살펴본 문항을 포함한 full 실행은 규모 확대이지 완전히 새로운 독립
재현은 아니다. 확인적 주장에는 손대지 않은 평가 데이터나 새 표현·문제 집합을 확보한다.

`[data].dataset`을 바꾸면 새로운 집합에서 연구를 실행할 수 있지만, 재발견했다면 기존
feature의 전이와 구별한다. 전이를 검증하려면 source에서 선택한 feature를 target에서
고정 평가하고 target 결과로 다시 선택하지 않아야 한다. Dataset마다 정답/Lure 쌍과
matched control 지원 여부도 다르다. 예를 들어 premise-rejection 방식의 semantic illusion
데이터를 CRT의 두 답 margin 경로에 그대로 넣는 것은 같은 검증이 아니다.

Thinking/Non-thinking 행동 baseline은 `crt_text_responses`와 관련 suite로 조사할 수 있고,
`reasoning_trajectory`는 생성 trace를 따라 feature 활성을 관찰하는 별도 경로다. 현재 2B
study는 Thinking 토글 실험이 아니며, trajectory 관찰도 두 조건에서 인과 steering이
성공했다는 증거는 아니다. 동일 checkpoint·SAE 대응과 생성 조건을 확인하는 후속 설계가 필요하다.

## 8. 연구자가 결과에서 확인할 질문

Python 구현보다 먼저 다음 질문에 답할 수 있는지 확인한다.

1. **Baseline에서 실제 Lure가 발생하는가?** `phenomenon`의 margin과 행동 baseline의 실제 답을 구분한다.
2. **어느 Layer / Feature가 관련되는가?** Localization과 문항별 효과를 보고 한 사례의 최고값에 의존하지 않는다.
3. **개입이 random/null보다 특별한가?** Screen의 문항 수·draws·in-sample 여부와 강한 null 사용 여부를 확인한다.
4. **Control에도 같은 효과가 나타나는가?** Hostile 효과와 specificity gap을 함께 본다.
5. **Discovery 미사용 held-out CRT에서도 유지되는가?** Train/test 효과와 family별 편차를 비교한다.
6. **실제 생성에서 Lure가 Correct로 바뀌는가?** `behavioral/generations.json`에서 개별 답변을 확인한다.
7. **Correct → Lure 부작용은 없는가?** 개선·악화·불변 전환 수를 함께 센다.
8. **Coefficient에 따라 효과가 일관적인가?** 0 계수 대조와 전체 sweep을 보고 유리한 점만 고르지 않는다.
9. **Thinking / Non-thinking에서도 유지되는가?** 현재 실험이 그 조건을 실제 실행했는지부터 확인한다.
10. **다른 CRT family / 표현에서도 재현되는가?** 같은 family 안의 held-out, 새로운 family, paraphrase를 구별한다.

## 9. 중요한 해석 원칙

- Margin 변화만으로 확정된 causal feature라고 결론 내리지 않는다. Correct/Lure 각각의
  logprob 변화와 모델 손상 가능성도 함께 본다.
- 한 문제에서 발견한 feature를 다른 문제·모델·SAE로 일반화하지 않는다. Feature ID는
  해당 checkpoint와 SAE layer에 묶인 번호다.
- Feature discovery와 evaluation 데이터를 분리한다. Null screen도 discovery에 사용한
  문항을 재사용했는지 확인한다.
- Random/null intervention 및 matched control과 비교한다. 작은 control 평균만으로
  무효과가 입증되는 것은 아니므로 문항별 분포와 불확실성도 필요하다.
- Teacher-forced margin뿐 아니라 실제 생성 행동을 확인한다. 장기 목표를 위해 자유 생성
  (free generation)도 검증하되, 현재 `binary_choice` 결과를 자유 생성으로 부르지 않는다.
  공통 생성 코드에는 `output_mode = "free"` 경로가 있지만 두 study config는 이를 사용하지 않는다.
- Lure → Correct 개선과 Correct → Lure 악화를 함께 기록한다. 자유 생성에서는 `both`/`other`,
  잘림과 응답 형식도 확인한다.
- Validation 결과와 논문 본 실험 결과를 구분한다. 실행 성공, 후보 발견, 통계적 근거,
  일반화된 인과 메커니즘은 서로 다른 주장이다.

## 10. 기존 문서와의 관계 및 현재 코드와의 차이

| 문서 | 역할 |
|---|---|
| [README.md](../README.md) | 프로젝트 전체 소개와 진입점 |
| [experiments/README.md](../experiments/README.md) | Runner 사용법과 artifact 수집 |
| [study_design.md](study_design.md) | Controlled-study 상세 설계와 후속 검증 |
| [notebook_paper_audit.md](notebook_paper_audit.md) | Notebook과 논문 실험의 관계 |
| `experiment_workflow_ko.md` (이 문서) | 연구자를 위한 전체 실험 흐름과 validation 해석 |

기존 설명을 읽을 때 다음 차이는 현재 코드를 기준으로 해석한다.

- `experiments/README.md`와 감사 문서의 일부 **free-generation** 표현과 달리 현재 두
  2B study 설정은 `binary_choice`다. `study_design.md`의 behavioral 절은 이 제한을 설명한다.
- `study_design.md` E2의 예전 `null_mean`/`null_z`/`null_percentile` 설명과 달리 localization의
  현재 필드는 `screen_null_mean`/`screen_null_z`/`screen_null_percentile`이다. 내부
  `null_summary()`의 `null_mean`과 저장되는 localization schema를 구분한다.
- 같은 문서의 **첫 discovery 문항 하나에서 null 평가**라는 설명은 현재 구현과 다르다.
  지금은 discovery 밖 train 문항 여러 개의 효과를 평균낸 screen이며, 부족할 때만 discovery
  문항으로 fallback하고 `screen_in_sample`을 기록한다.
- Localization 그림의 feature 평균은 discovery 문항, null 평균은 screen 문항에서 나온다.
  그림의 간격만으로 유의성을 판단하지 말고 동일 screen에서의 `screen_hostile_delta`와
  `screen_null_*`를 함께 확인한다.
- 설계 문서의 **train에서 계수 선택**은 연구 원칙이다. 현재 설정의 고정 coefficient sweep이
  그 튜닝 절차까지 자동 구현했다는 뜻은 아니다.
- 감사 문서의 07·13 “대체”와 12 “탐색 전용”은 현재의 부분 대응·추가 진단 job을 모두
  설명하지 못한다. 3절의 표는 연구 질문이 실제로 동일한지까지 구분한 것이다.

기존 문서는 이번 작업에서 수정하지 않았다. 추후 `docs/README.md`나 `experiments/README.md`에
이 가이드 링크를 추가하면 연구자의 진입점으로 활용할 수 있다.
