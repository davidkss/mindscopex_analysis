# Feature 30908 Steering Stability Sweep

코드·설정·CPU 테스트만 수행했다. Colab/GPU/실제 모델 추론, commit/push를 수행하지 않았다.
기존 실험 설정·결과 및 mindscopex_inventory.py를 수정하지 않았다. 사용자 보고의 강한 음수
steering 생성 손상을 조사하기 위한 설계이며, 이번 작업에서 안정적인 계수를 판정한 것은 아니다.

## 신규 실행 조건

현재 검증 대상은 **Seed 1**이다. 기존 nothink64 Seed 1의 모델·SAE·문항 선정·지시문·template을
그대로 사용한다. Qwen/Qwen3.5-2B-Base, SAE layer 5, feature 30908, crt_fresh_v2,
thinking_mode=disabled, max_new_tokens=64, do_sample=False, token_position=all.

| 설정 | 계수 | 문항 |
|---|---|---|
| study_steering_stability_sweep_f30908_seed1_2b.toml | 0, -0.25, -0.5, -1, -2, -3 | 기존과 동일한 40개 LureCase |
| study_steering_stability_smoke_f30908_seed1_2b.toml | 같은 6개 | 동일 40문항 중 deterministic 처음 6개, 유형별 2개 |

두 설정은 별도 run 이름을 사용하며 기존 설정을 변경하지 않는다. smoke 문항이 full의 부분집합이고
full의 원본 문항 정의가 기존 nothink64와 같음을 테스트한다. smoke는 빠른 동작 점검용이며 40문항의
성능을 대표한다고 해석하지 않는다. 추가 Seed는 이번 실행 설정에 포함하지 않았다.

기존 open_ended_crt → greedy completion → steer_generation_labels 경로를 그대로 사용한다.
공식 checkpoint template의 enable_thinking=False를 실제 입력 토큰까지 검증한다. 선택지·후보값을
추가하지 않고 constrained decoder도 사용하지 않는다. 분류와 잘린 응답 invalid 처리도 유지한다.

새 설정의 analysis_protocol은 stability_sweep 또는 stability_smoke다. 기존 standard 프로토콜의
계수 0/-7/-8 및 40문항 검증은 유지하면서, 신규 프로토콜에서만 다른 grid·6문항 smoke를 허용한다.
Stability 프로토콜은 thinking disabled 및 64토큰을 요구한다. 문항 누락·중복, baseline drift,
0계수의 baseline/steered 응답 및 원문·종료 메타데이터 불일치는 기존처럼 오류로 중단한다.

## 기존 Steering의 수학적 정의

`qwen_scope.py::sae_decoder_direction`은 checkpoint의 W_dec에서 feature 방향을 꺼낸다.
W_dec가 (d_model, d_sae)이면 열, (d_sae, d_model)이면 행을 사용한다.

단일 feature에 대해 d = d_30908, coefficient = c라 하면:

```text
Δ_requested = c × d
Δ_cast = Δ_requested.to(hidden.device, hidden.dtype)
h' = h + Δ_cast
```

`make_feature_steering_hook`는 decoder block 5의 출력 residual에 이를 적용한다.
`token_position=all`이면 해당 forward의 모든 batch/token 위치에 같은 벡터를 더한다.
캐시를 사용하는 생성에서는 prompt prefill의 모든 토큰과 이후 각 decode forward의 토큰에
계속 적용된다. 관측되는 h는 그 forward에서 이미 steering의 영향을 받은 생성 상태에 속한다.

이 함수는 d를 단위 L2 norm으로 정규화하거나 residual RMS/norm에 맞춰 rescale하지 않는다.
Decoder bias는 이 방향 합에 들어가지 않는다. 모델 내부의 일반적인 norm 연산과 이 hook의
별도 정규화 여부는 구분해야 한다. Checkpoint decoder 벡터 자체가 학습 결과 단위 norm인지 여부는
가중치로 측정해야 하며, 이번 작업에서는 가중치를 로딩하거나 실제 norm을 추정하지 않았다.
정확 연산이라면 ||Δ||₂ = |c| ||d||₂이지만 실제 low-precision 덧셈은 반올림 영향을 받는다.

### Feature 활성화와의 관계

이 프로젝트의 `encode_qwen_scope_topk`는 다음을 계산한다.

```text
preactivation = h @ W_enc.T + b_enc
TopK(preactivation)  # 별도 ReLU clamp 없음
```

TopK support 밖의 feature는 sparse reconstruction에서 0이다. 선택 feature의 raw preactivation과
실제 TopK sparse activation은 같다고 간주할 수 없다. 해당 계산은 steering hook에서 호출되지 않는다.

현재 행동 실험의 edit는 **활성화 크기 z_30908(h)를 곱하지 않는 상수 방향 이동**이다.
음수 c는 feature를 반드시 0으로 만들거나 완전히 제거하지 않는다. 예를 들어 활성화 제거 형태의
`h − z_30908(h) d_30908`과 다르다. 따라서 c=-3을 "활성화 3배 제거"나 "3% 감소"로 해석할 수 없다.
이번 진단은 vector/residual 크기를 기록하며 feature 활성화 값을 별도로 측정했다고 주장하지 않는다.

## Norm 진단 구현

새 설정에서만 steering_diagnostics=true를 켠다. 기본값은 false이므로 기존 실험 hook 동작과 비용은
유지한다. Hook이 원래와 동일하게 edited를 만든 뒤 detach된 값에서 통계만 계산한다.
출력 residual·cache를 바꾸거나 steering을 다시 정규화하지 않는다.

각 steered forward마다 coefficient와 forward_index, feature_ids, token_position,
affected_token_count 및 다음을 저장한다.

- requested_vector_norm: SAE dtype에서 coefficient를 적용한 방향을 float32로 관측한 norm
- cast_vector_norm: hidden dtype/device로 변환한 벡터의 norm
- residual_before / residual_after: 개입 대상 토큰별 ||h||₂, ||h'||₂의 평균·최소·최대
- actual_delta: float32로 관측한 ||h' − h||₂의 평균·최소·최대
- delta_residual_ratio: 실제 개입량 / 개입 전 residual norm의 평균·최소·최대
- 0 residual 및 비유한 residual/post/delta 수, ratio가 정의된 토큰 수

분모가 0이거나 비유한 토큰은 ratio 계산에서 제외하고 개수를 기록한다. 정의되는 값이 하나도
없으면 None/빈 CSV 셀로 기록하며 0으로 바꾸지 않는다. Norm은 토큰별 hidden 차원 기준이며,
batch×sequence 전체 Frobenius norm과 혼동하지 않는다. 낮은 정밀도에서는 cast vector가 0이 아니어도
실제 덧셈이 반올림으로 사라질 수 있으므로 두 값을 구분한다.

Baseline에서는 hook을 설치하지 않는다. c=0의 hook 진단도 기록하지만 응답 동일성 검증은 원래의
baseline/steered 생성 응답끼리 수행한다. Norm은 steered trajectory 내부의 전후 비교이며, 같은
문항의 별도 baseline trajectory hidden state와 직접 대응한 비교가 아니다.

기록은 coefficient별 generations.json block의 steering_diagnostics에 포함되고 기존 결과 SHA-256
대상에 들어간다. 분석에서는 steering_norm_diagnostics.csv로 펼친다. forward_index는 계수 안의
forward 순번이며 case_id나 생성 token index라고 해석하지 않는다. Prefill과 decode를 함께 담으므로
토큰 수가 다른 forward 평균을 단순 평균하여 전체 토큰 평균이라고 부르지 않는다.

진단은 추가 reduction과 device 동기화를 유발한다. 예상 생성 횟수에는 진단 비용이 포함되지 않으며
실행 시간은 실제 GPU·토큰 길이·라이브러리 조건에 따라 달라진다. 진단이 켜진 full/smoke 간 비교를
사용하고, norm 비율만으로 생성 손상의 인과적 원인을 확정하지 않는다.

## 결과 지표와 분석

기존 seed_summary.csv / family_summary.csv / coefficient_summary.csv에 4개 범주 수·비율,
모든 전환, EOS count/rate, repetition count/rate를 저장한다. Repetition은 repeated_character,
repeated_token, repeated_phrase 중 하나 이상인 응답의 비율이며 중복 플래그를 합산하지 않는다.
Symbol-heavy와 생성 Thinking 태그 등은 기존 response_diagnostics_summary.csv에서 별도 집계한다.
Invalid를 분모에서 제외하지 않는다. 원문·종료 사유·판정 근거·개별 진단은 기존 paired CSV/JSON에 남긴다.

기존 성공/실패 상태, config/result 해시, 실행 디렉터리와 ZIP 보호를 재사용한다.
분석은 protocol에 맞는 계수·문항 수·thinking/budget을 검증하며 norm 기록 누락도 오류로 처리한다.
이 sweep의 계수는 기존 객관식 -7/-8 grid와 다르므로 objective comparison 옵션으로 혼합하지 않는다.

## 예상 GPU 실행량

기존 evaluator는 각 coefficient에서 baseline을 다시 생성한다. 동일 출력이어도 생성을 건너뛰지 않는다.

| 실행 | Baseline 생성 | Steered 생성 | 총 생성 | 최대 신규 생성 토큰 예산 |
|---|---:|---:|---:|---:|
| Smoke 6문항 × 6계수 | 36 | 36 | 72 | 4,608 |
| Full 40문항 × 6계수 | 240 | 240 | 480 | 30,720 |
| 둘 다 | 276 | 276 | 552 | 35,328 |

위 값은 각 생성 64토큰의 상한이며 prompt prefill 토큰·SAE 로딩·norm 진단 비용은 별도다.
EOS로 일찍 끝나면 실제 생성 토큰은 줄어든다. 모델 실행 전이므로 소요 시간을 추정하지 않는다.

## 실행 명령 (이번 작업에서는 실행하지 않음)

깨끗한 새 Colab A100 세션에서 최신 소스 archive로 실행한다.

```bash
# Smoke first
bash experiments/run_colab.sh experiments/configs/study_steering_stability_smoke_f30908_seed1_2b.toml \
  --session mindscopex-stability-smoke --gpu A100 --exec-timeout 14400

# Full Seed 1
bash experiments/run_colab.sh experiments/configs/study_steering_stability_sweep_f30908_seed1_2b.toml \
  --session mindscopex-stability-sweep --gpu A100 --exec-timeout 14400
```

실제 결과 회수 후:

```bash
.venv/bin/python scripts/analyze_crt_steering_stability.py --smoke
.venv/bin/python scripts/analyze_crt_steering_stability.py
```

기본 입력은 results/runs이며 기본 출력은 각각
results/analysis/crt_steering_stability_smoke_seed1 및
results/analysis/crt_steering_stability_sweep_seed1이다. --input-root / --output으로 지정할 수 있다.
분석 결과 경로가 이미 있으면 중단한다. 이번 작업에서는 실제 연구 결과를 생성하지 않았다.

## 검증 및 변경 파일

```bash
MPLCONFIGDIR=/tmp/mindscopex-mpl .venv/bin/python -m unittest discover -s tests -v
git diff --check
```

CPU 테스트는 full 문항 동일성·smoke 부분집합, 여섯 계수 검증, c=0 동일성, 네 범주·EOS·반복 집계,
비정규화 벡터 norm, 진단 on/off hook 출력 동일성, all/last 위치, tuple cache 보존,
0 residual ratio, float16 반올림, hook 실패 시 해제, smoke job→provenance→분석 및 덮어쓰기 거부를 검증한다.
합성 응답은 임시 디렉터리에서만 사용하며 성능 결과로 보고하지 않는다.
최종 전체 CPU unittest 513개 실행, 실패 0, skip 4. git diff --check 통과.

신규:

- experiments/configs/study_steering_stability_sweep_f30908_seed1_2b.toml
- experiments/configs/study_steering_stability_smoke_f30908_seed1_2b.toml
- scripts/analyze_crt_steering_stability.py
- tests/test_crt_steering_stability.py
- docs/analysis/README_crt_steering_stability.md

수정: experiments/jobs/research_experiments.py, src/mindscopex_analysis/research.py,
src/mindscopex_analysis/qwen_scope.py, src/mindscopex_analysis/open_ended_crt.py,
scripts/analyze_crt_open_ended.py. 기존 standard 프로토콜을 기본값으로 유지한다.
작업 시작 시 있던 scripts/analyze_crt_choice_order.py 변경은 건드리지 않았다.
