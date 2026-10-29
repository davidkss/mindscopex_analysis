# MindScopeX — Behavioral 생성과 Protocol B Likelihood 조건 감사

**기록:** 2026-10-29 · **상태:** GitHub 소스·Protocol B Manifest 기반 정적 감사 완료. GPU 동작 동일성 재실행 검증은 미완료.

## 1. Lure Likelihood 증가의 의미

CRT(Cognitive Reflection Test)의 **Lure**는 직관적이지만 틀린 함정 답안이며, **Correct**는 정답이다. 현재 실험은 Qwen3.5-2B-Base의 Layer 5 SAE Feature 30908에 개입 계수 0, +7, +8을 적용한다.

후보 전체 답변의 점수는 다음과 같다.

```text
S(answer) = Σ log P(answer_token_i | prompt, answer_tokens_before_i)
Full-Answer Margin = S(lure) − S(correct)
ΔMargin(+7) = Margin(+7) − Margin(0)
ΔMargin(+8) = Margin(+8) − Margin(0)
```

**Lure Likelihood 증가**는 정확히 어떤 값을 비교했는지 구분해야 한다.

1. **Lure 후보 자체의 likelihood 증가:** `S(lure, intervention) > S(lure, baseline)`. 이는 lure 후보 경로의 로그확률이 증가한 것이다.
2. **Lure 상대 선호 증가(양의 ΔMargin):** `[S(lure)−S(correct)]_intervention > [S(lure)−S(correct)]_baseline`. Lure의 로그확률이 실제로 증가하지 않더라도, Correct가 더 크게 감소하면 ΔMargin은 양수가 될 수 있다.
3. **양의 Margin:** 두 후보 중 Lure의 전체 토큰 로그확률 합계가 Correct보다 크다. 이 결과가 곧 실제 생성에서 Lure가 선택되었다는 뜻은 아니다.

**본 연구에서 반복적으로 보고한 'Lure Margin 증가'는 2번, 즉 Correct 대비 Lure의 상대적 likelihood 우위가 증가한 것을 뜻한다.** 'Lure의 절대 likelihood가 증가했다'거나 '모델이 더 많이 오답을 생성했다'는 표현으로 바꾸면 안 된다.

예를 들어 Baseline에서 `S(lure)=-5, S(correct)=-4`라면 Margin은 -1이다. 개입 후 `S(lure)=-6, S(correct)=-8`이라면 Margin은 +2, ΔMargin은 +3이다. **Lure 자체 점수는 -5에서 -6으로 감소했지만, Correct 점수가 더 많이 감소해 상대적 Lure 선호는 증가한다.**

또한 점수는 후보 토큰별 **로그확률 합계**이고 EOS를 포함하지 않는다. 후보 토큰 길이와 토큰화가 다르면 합계에 길이 편향이 발생할 수 있다. 두 후보로 제한한 정규화 선택확률도 계산하지 않았으므로 Full Margin을 실제 생성 정답률이나 모델의 일반적 사고 성향으로 해석하지 않는다.

## 2. Protocol A/B 정의와 입력 조건

| 항목 | Protocol A | Protocol B |
|---|---|---|
| 식별자 | `diagnostic_compatible` | `behavior_compatible` |
| Prompt 구성 | chat template, system_prompt="", enable_thinking=False | `text_tokenizer(case.prompt, return_tensors="pt")` |
| 후보 답안 | `answer.strip()` | 원본 Correct/Lure 문자열, 선행 공백 유지 |
| 후보 토큰화 | `add_special_tokens=False` | `add_special_tokens=False` |
| 점수 | teacher-forced 후보 전체 로그확률 합계 | 동일 |
| EOS | 제외 | 제외 |
| 목적 | 기존 First-Divergence 진단 조건과 비교 | Behavioral binary-choice 입력 경로에 더 가깝게 비교 |

근거: `src/mindscopex_analysis/full_answer_protocols.py`, `docs/analysis/README_fresh_v2_full_answer_likelihood.md`, `results/analysis/protocol_b_20261029/manifest.json`.

## 3. 생성 코드와 Likelihood 코드의 조건 감사

| 감사 항목 | 실제 Behavioral 생성 `research.py:_binary_choice_completion` | Protocol B | 판정 |
|---|---|---|---|
| Prompt 입력 | raw `text_tokenizer(case.prompt, return_tensors="pt")` | 동일 경로를 참조하고 CPU capture로 후보 ID 검증 | **입력 경로 정합성 확인** |
| 후보 답안 | 원본 Correct/Lure | 원본 Correct/Lure | **정합성 확인** |
| 후보 토큰화 | `encode(..., add_special_tokens=False)` | 동일 | **정합성 확인** |
| 선택 방식 | 두 후보 prefix trie의 허용 토큰만 남기는 **greedy constrained generation**; `do_sample=False`, `renormalize_logits=True` | 후보별 **teacher-forced full logprob 합산**, `generate()` 사용 안 함 | **핵심적인 평가 방식 차이** |
| 종료 조건 | 완성 후보 이후 EOS 허용 | EOS 점수 미포함 | **차이 존재** |
| Seed 반복 | Behavioral 결과 64개 슬롯, Seed 1–5 | 33개 문항 × 3계수로 문항별 점수 한 번씩 | **반복 구조 다름** |
| 모델/개입 | 기존 Behavioral 실험의 실제 run manifest 대조 필요 | Manifest: Qwen3.5-2B-Base, Layer 5, Feature 30908, token_position=all | **실행 메타데이터의 완전 동등성 미검증** |
| 체크포인트 revision | 과거 Behavioral run의 revision 검증 필요 | Protocol B manifest `resolved_revision=null` | **revision 동일성 미확정** |

### 왜 Full Margin과 생성 결과가 다를 수 있는가?

**Full-Answer Margin은 각 후보의 전체 경로 확률을 누적 비교**한다. 반면 constrained greedy 생성은 매 시점 허용된 다음 토큰 중 점수가 가장 높은 토큰을 선택하며, **첫 분기에서 선택된 후보 경로가 뒤집히지 않을 수 있다**. 따라서 전체 시퀀스의 누적 로그확률이 Lure 쪽으로 높아도 greedy 생성은 Correct를 선택할 수 있다. EOS 종료 규칙과 후보 길이도 차이를 만든다.

이는 **코드 구조로부터 설명 가능한 불일치 원인**이지, 현재 33개 문항에서 관측된 불일치의 유일한 원인으로 입증된 것은 아니다.

## 4. 실제 데이터 교차 검증 결과

Behavioral 원본 `results/analysis/fresh_v2_f30908_cplus7_vs_cplus8.csv`와 Protocol B `full_answer_scores.csv`의 Growth 33개 문항을 대응했다. Behavioral 64개 평가 슬롯 × +7/+8 = 128개 결합 행이다.

| 지표 | +7 | +8 |
|---|---:|---:|
| 실제 Lure 응답 (슬롯 기준) | 16/64 | 30/64 |
| Likelihood의 Lure 선호 (슬롯 기준) | 52/64 | 46/64 |
| 실제 응답과 선호 부호 일치 (슬롯 기준) | 28/64 | 36/64 |
| 실제 Lure 응답률 (문항 균등 가중) | 21.2% [95% CI 9.1%, 36.4%] | 39.4% [24.2%, 54.5%] |
| Likelihood Lure 선호율 (문항 균등 가중) | 75.8% [60.6%, 90.9%] | 63.6% [45.5%, 78.8%] |
| 응답–선호 일치율 (문항 균등 가중) | 45.5% [27.3%, 63.6%] | 57.6% [39.4%, 72.7%] |

문항 단위 cluster bootstrap 20,000회, seed 20261029. 실제 Lure 응답률의 **+8 − +7 차이는 +18.2%p [95% CI +6.1%p, +33.3%p]**. 이는 계수 0 대비 효과가 아니다.

## 5. 후속 검증 순서

1. 실제 Behavioral 실행에 사용한 `config.toml`, `manifest.json`, `generations.json`을 Protocol B Manifest와 직접 대조해 모델 revision, SAE 체크포인트, 개입 hook, Prompt 및 생성 규칙의 차이를 명시한다.
2. 동일 33문항의 첫 분기 토큰 부호와 실제 greedy 선택을 직접 대응시켜, **first-branch 선택과 full-answer 점수의 불일치**를 정량화한다.
3. EOS 포함 및 후보 길이 정규화 민감도 분석을 별도 프로토콜로 실행한다. 기존 A/B 결과와 혼합하지 않는다.
4. 신규 독립 CRT 문항과 추가 Seed에서 재현성을 확인한다.

**연구 해석의 핵심:** Feature 30908 개입은 두 후보의 상대적 로그확률을 변화시켰다. 하지만 후보 전체 로그확률 우위와 constrained greedy 생성 답변은 동일한 측정치가 아니다. **Lure Margin 증가 = 실제 Lure 오답 증가**라는 등식은 성립하지 않는다.
