# MindScopeX 연구 진행 현황

**기준일:** 2026년 10월 29일  
**주제:** Hagendorff CRT에서 탐색한 SAE Feature 30908의 Fresh v2 검증과 후속 분석

## 0. 핵심 용어 사전

| 용어 | 의미 | MindScopeX에서의 사용 |
|---|---|---|
| CRT (Cognitive Reflection Test) | 직관적인 첫 답을 재검토하도록 요구하는 인지 반성 문제 | Qwen의 직관적 오답과 정답 비교 |
| Hagendorff CRT | 연구 초기 후보 Feature 탐색에 사용한 CRT 문항군 | Feature 30908 탐색 단계 |
| CRT Fresh v2 | 기존 탐색 문항과 구분되는 후속 CRT 검증 문항군 | 신규 문항에서 개입 효과 검증 |
| Correct | 올바른 정답 | 정답 후보 |
| Lure | 직관적으로 매력적이지만 틀린 답 | 오답 후보 |
| Layer | 모델 내부 계산 단계 | Layer 5에서 SAE Feature 분석 |
| SAE Feature | Sparse Autoencoder로 표현한 내부 특징 신호 | Feature 30908 |
| Activation | 특정 내부 특징 신호가 나타나는 정도 | Feature 30908의 활성화 관찰 |
| Intervention | 내부 신호에 인위적으로 변화를 주는 실험 | Feature 30908 강화·억제 |
| Coefficient | 개입 강도를 정하는 계수 | +7, +8 등의 개입 조건 |
| Baseline | 개입하지 않은 비교 기준 | Coefficient 0 |
| Behavioral | 실제 모델이 출력한 답을 관찰하는 실험 | Correct/Lure 생성 결과 |
| Token | 모델이 처리·생성하는 텍스트 단위 | 후보 답변을 토큰 단위로 비교 |
| Likelihood | 특정 후보 답변에 모델이 부여하는 확률적 선호를 나타내는 점수 | Correct/Lure 점수 비교 |
| Full-Answer Likelihood | 후보 답변 전체 토큰의 로그확률을 합산한 점수 | 전체 Correct/Lure 답변 비교 |
| First-Divergence | 두 후보의 토큰열이 처음 달라지는 위치 | 첫 분기점의 상대적 점수 비교 |
| Tie | 비교 점수가 같은 경우(저장된 값 기준) | First-Divergence Margin 0인 슬롯 |
| Margin | 두 후보 점수 차이 | Lure 점수 − Correct 점수 |
| Greedy Decoding | 각 생성 단계에서 점수가 가장 높은 다음 토큰을 선택하는 방식 | 허용된 후보 토큰으로 제한된 생성 실험 |
| Teacher-forced | 후보 답변 토큰을 정해 놓고 각 토큰의 점수를 계산하는 방식 | Full-Answer·First-Divergence 진단 |
| Bootstrap | 표본을 반복 재추출해 추정치의 불확실성을 계산하는 통계 방법 | 문항 단위 신뢰구간 검증 |
| Slot | 반복 측정으로 생성된 개별 관측 단위 | 조건별 64개 Behavioral 관측 |

> **주의:** First-Divergence, Full-Answer, Teacher-forced 등은 LLM 분석에 쓰이는 기술적 표현이며, 이 연구에서의 구체적인 점수 정의와 계산 방법은 실험 프로토콜에 따라 달라질 수 있습니다. Tie는 저장된 계산값이 0이라는 뜻으로, 실제 확률이 수학적으로 정확히 같다고 단정할 수 없습니다.

## 1. MindScopeX 연구 진행 단계

### 1단계. Hagendorff CRT — 의심 Feature 탐색

Hagendorff CRT 문제를 기반으로 Qwen의 직관적 오답(Lure)과 관련된 내부 활성화를 분석했습니다. 이 과정에서 **Layer 5의 SAE Feature 30908**을 주요 후보로 선정했습니다.

**현재 결과:** Lure 관련 후보 Feature 확보. 단, 후보 선정 자체가 인과적 역할의 확증은 아닙니다.

### 2단계. CRT Fresh v2 — 새로운 문항으로 행동 검증

기존 탐색 문항과 구분되는 CRT Fresh v2 문항에 Feature 30908 개입을 적용했습니다. 현재까지 **Correct 응답을 안정적으로 개선한다는 근거를 확보하지 못했습니다.**

**현재 결과:** Correct 개선 가설 미입증. 이는 Feature 30908이 응답에 전혀 영향을 주지 않는다는 뜻은 아닙니다.

### 3단계. Full-Answer Likelihood — 응답 전체 점수 분석

실제 정답률만 보는 대신, 모델이 Correct와 Lure 전체 답변에 부여하는 로그확률 점수를 비교했습니다. 개입 +7/+8에서 Lure의 상대적 점수가 증가하는 현상이 관찰되었지만, 이 변화가 실제 Lure 생성 빈도와 일치하지는 않았습니다.

**현재 결과:** 전체 답변 점수 변화와 실제 응답 행동 사이의 불일치 확인.

### 4단계. First-Divergence — 불일치 원인 탐색

Correct와 Lure의 토큰이 처음 달라지는 위치에서의 상대적 선호를 실제 Behavioral 응답과 비교했습니다. Codex가 보고한 CPU 분석 결과는 다음과 같습니다.

| 지표 | Coefficient +7 | Coefficient +8 |
|---|---:|---:|
| First-Divergence 일치율(Tie 제외) | 53/53 (100%) | 56/58 (96.55%) |
| Full-Answer 일치율 | 28/64 (43.75%) | 36/64 (56.25%) |
| First-Divergence Tie | 11슬롯·4문항 | 6슬롯·2문항 |

**해석:** 첫 분기점의 점수 방향이 실제 제한적 Greedy 생성 결과와 높은 일치율을 보였습니다. 다만 두 지표의 표본 분모가 다르므로 직접 비교 시 공통 유효 표본 기준 분석도 필요합니다. 또한 이 점수는 실제 생성 도중 직접 수집한 logits가 아닌 teacher-forced 진단값이므로 인과관계가 입증된 것은 아닙니다.

### 5단계. 실제 생성 과정과 인과관계 검증 — 다음 단계

실제 Greedy Decoding 중 첫 분기점의 토큰 점수를 직접 확인하고, Feature 30908의 개입이 해당 선택과 실제 Correct/Lure 전환을 어떻게 바꾸는지 검증해야 합니다. Tie와 불일치 사례도 개별 분석해야 합니다.

**현재 위치:** 4단계 분석까지 수행했으며, 5단계 검증을 준비하는 시점입니다.

## 2. 현재 연구에서 정확히 밝혀진 것

| 연구 질문 | 현재 판단 |
|---|---|
| Feature 30908이 Lure와 관련된 후보인가? | 탐색 결과상 유력한 후보 |
| 새로운 CRT 문항에서도 Correct를 개선하는가? | 안정적인 개선 근거 부족 |
| Feature 개입이 모델의 점수를 변화시키는가? | 상대적 Likelihood 변화 관찰 |
| Full-Answer 점수로 실제 답변을 설명할 수 있는가? | 설명력이 제한적 |
| First-Divergence 점수는 실제 답변과 일치하는가? | Tie 제외 관측에서 높은 일치율 |
| Feature 30908이 직관적 오답을 유발하는 인과적 Feature인가? | 아직 미입증 |

### 연구 가설의 변화

**초기 가설:** Feature 30908을 억제하면 모델의 직관적 오답이 줄고 Correct 응답이 증가할 것이다.

**현재 검증 중인 가설:** Feature 30908 개입은 Correct/Lure 후보의 상대적 점수를 변화시키지만, 실제 응답 선택에 미치는 영향은 첫 분기 토큰의 선호와 생성 방식에 따라 달라질 수 있다.

이는 초기 가설을 입증했다는 뜻이 아니라, 관찰된 불일치를 설명하기 위해 연구 질문을 정교화한 것입니다.

## 3. Git 상태 및 연구 기록 관리 (2026-10-30 확인)

### 3.1 최신 확인 Git 상태

사용자가 2026-10-30 WSL에서 확인한 작업 상태:

```text
On branch main
Your branch is up to date with 'origin/main'.
nothing to commit, working tree clean
main
/home/davidkss/ai/mindscopex_analysis  057fb7b [main]
```

- 위 `057fb7b`는 **2026-10-30 문서 수정 작업 시작 전 사용자 로컬에서 확인한 기준 Commit**이다. 이후 GitHub 문서 갱신 Commit이 발생할 수 있으므로 항상 현재 `git status`와 원격 최신 이력을 다시 확인한다.
- 과거 `26bd44d` 및 미커밋 파일 목록은 2026-10-29 중간 상태였으며 최신 상태로 오인하지 않는다.
- GitHub 반영 대상은 사용자 본인 저장소 `davidkss/mindscopex_analysis`이며, 원본 `JeonDongJun/mindscopex_analysis`에 반영하지 않는다.

### 3.2 연구 기록의 기준

- **1차 상세 연구 기록:** [research/journal/](../journal/) — 날짜별 연구 목적, 기존 실험과의 연결, 실행 조건·명령, 원본 산출물, 결과, 한계, 다음 작업을 기록한다.
- **전체 진행 현황:** `research/reports/` — Journal 및 검증 보고서를 참조하여 단계별 진행 상황과 핵심 결론을 요약한다.
- **감사·재현성 근거:** [First-Divergence 감사](../audits/2026-10-29-first-divergence/report.md) — 검증 수치와 슬롯별 근거를 확인한다.
- 이전 일지를 삭제·이동하거나 당시의 '실행 대기'를 소급 수정하지 않고, 완료 사실은 후속 날짜가 명시된 항목으로 추가한다.
- 새로운 실험 전에 기존 Journal, 감사 결과, 코드·산출물을 대조하여 이미 수행한 작업을 반복하지 않는다.
- 기존의 `main` 중심 Git 작업 방식을 유지하며, 사용자 확인 없이 별도 브랜치·Worktree·Pull Request 절차를 도입하지 않는다. 파일 변경 및 Commit/Push 전에는 현재 상태와 대상 저장소를 확인한다.

### 3.3 4단계 감사 완료 후 5단계 준비

- +7: First-Divergence 일치 53/53 (Tie 제외), Tie 11슬롯.
- +8: First-Divergence 일치 56/58 (Tie 제외), Tie 6슬롯, 불일치 2슬롯.
- Full-Answer 전체 슬롯 일치: +7은 28/64, +8은 36/64.
- 실제 생성 시점의 logits 직접 추적과 Feature 30908의 인과적 역할은 아직 미검증이다.
- 다음 작업은 기존 생성 코드와 Tie/불일치 사례를 먼저 점검한 뒤 실제 생성 logits 검증을 설계하는 것이다.

## 4. 결론 — 우리는 어느 시점에 있는가?

**MindScopeX는 Lure 관련 SAE Feature 탐색과 신규 CRT 행동 검증을 마쳤으며, 기대했던 Correct 개선이 재현되지 않은 이유를 토큰별 생성 점수와 실제 응답 선택의 관계에서 규명하는 단계에 있습니다.**

새로운 Feature를 찾는 초기 탐색 단계는 지났지만, Feature 30908의 인과적 역할을 최종 입증한 단계는 아닙니다.

현재 가장 중요한 다음 작업은 **First-Divergence에서 관찰된 높은 일치율이 실제 Greedy Decoding의 토큰 선택 과정에서도 재현되는지 검증하는 것**입니다. 재현된다면 Feature 30908의 역할을 더 정확하게 설명할 수 있고, 재현되지 않더라도 Likelihood 분석이 실제 생성 행동을 어디까지 설명하는지 규명하는 결과가 됩니다.

---

*작성 근거: 현재 대화에서 공유된 연구 진행 내역, Codex 검증 요약, `git status --short` 및 `git log -1 --oneline` 출력. Codex의 상세 로컬 보고서 내용은 직접 열람하지 않았으므로 별도 검증이 필요한 부분은 명시했습니다.*
