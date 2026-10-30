# MindScopeX 프로젝트 작업 원칙 — ChatGPT / Codex / Colab

> 작성일: 2026-10-30  
> 상태: 프로젝트 운영 규칙 (향후 작업에서 우선 참조)

## 1. 기본 원칙

MindScopeX의 **소스코드 구현, 수정, 테스트, Git Commit 및 Push는 Codex를 통해 진행**한다. ChatGPT는 연구 설계, 실험 결과 해석, 오류 원인 분석, Codex에 전달할 구체적인 작업 지시서 작성 및 결과 검토를 담당한다.

**사용자가 별도로 명시적으로 요청하지 않는 한 ChatGPT는 GitHub 저장소의 소스코드·설정·테스트·연구 일지를 직접 변경하거나 Commit/Push하지 않는다.** 문서 기록을 포함한 저장소 변경도 원칙적으로 Codex를 통해 수행한다.

## 2. 역할 분담

| 작업 | 기본 담당 | 운영 기준 |
|---|---|---|
| 연구 질문·가설·실험 설계 | ChatGPT | 인과관계와 관찰 결과를 구분 |
| 소스코드 구현 및 수정 | Codex | 기존 코드 기준으로 최소 변경 |
| CPU 테스트·회귀 테스트·오류 수정 | Codex | 실제 실행 결과를 확인한 뒤 보고 |
| Git Commit / Push | Codex | 변경 범위·검증 결과를 보고 |
| Colab CLI / A100 GPU 실행 | 기존 Colab CLI 워크플로 | **비용 발생 전 사용자의 명시적 승인 필요** |
| JSON/로그/그래프 해석 | ChatGPT | 실제 Artifact를 근거로 해석 |
| 연구 일지 작성·갱신 | Codex (ChatGPT가 초안·지시서 작성) | 기존 기록 보존, append-only |

## 3. ChatGPT가 따라야 할 작업 순서

1. 사용자의 연구 목표와 현재 실험 결과를 확인한다.
2. 실제 저장소 코드와 Artifact를 검토하여 문제를 특정한다.
3. Codex에 전달할 **작업 지시서**를 작성한다. 수정 대상 파일, 변경 내용, 보존 조건, 테스트 명령, 완료 기준을 명시한다.
4. Codex가 수행한 변경 및 테스트 결과를 확인한다. **실행하지 않은 테스트를 통과했다고 표현하지 않는다.**
5. CPU 검증이 끝난 뒤 GPU 실험이 필요하면 비용·범위를 설명하고 사용자의 명시적 승인을 받는다.
6. 실험 결과를 해석하고 다음 연구 방향을 결정한다.

## 4. 변경·실험 안전 규칙

- 기존 성공/실패 Run 디렉터리와 Artifact를 삭제·덮어쓰지 않는다.
- `research/journal/YYYY-MM-DD.md`는 **기존 내용 유지 + 신규 섹션 추가(append-only)** 원칙을 적용한다.
- 실험의 Prompt, Tokenizer, 모델/SAE 버전, Dtype, 후보 제약, 계측 조건을 명확히 기록한다.
- 관찰된 Lure 생성, Margin 동률, Teacher-forced 점수, 인과적 개입 효과를 혼동하지 않는다.
- 과거 Seed 참조를 신규 독립 반복 측정으로 표현하지 않는다.
- Equivalence 검증(계측/비계측)을 Baseline/Intervention 인과 비교로 해석하지 않는다.
- GPU 사용, 모델 다운로드, 비용 발생 가능 작업은 명시적 승인 없이 실행하지 않는다.
- 사용자가 명시적으로 직접 Git 변경을 요청한 예외 상황이 아니면 ChatGPT는 저장소를 직접 수정하지 않는다.

## 5. 이번 사례 및 후속 조치

2026-10-30 Phase 5-C-5 진행 중 ChatGPT가 Codex를 거치지 않고 GitHub에 직접 어댑터·설정·테스트 코드와 연구 일지를 반영했다. 이는 합의한 작업 방식과 맞지 않았다.

당시 최신 Git Commit: `75578f59a4cd83cd187c43317ccd7578ddea0bc6`.

관련 변경 파일:

- `experiments/jobs/first_divergence_trace.py`
- `experiments/configs/trace_fresh_v2_growth019_c0_2b.toml`
- `experiments/configs/trace_fresh_v2_growth019_c7_2b.toml`
- `tests/test_first_divergence_trace_job.py`
- `research/journal/2026-10-30.md`

**후속 처리:** 변경을 임의로 되돌리지 않는다. Codex가 최신 코드를 검토하고 CPU 회귀 테스트를 실행한 뒤, 필요할 경우 수정·Commit/Push하도록 지시한다. 이 변경에 대한 CPU 테스트와 신규 A100 실험은 검증 완료로 간주하지 않는다.

## 6. Codex 작업 지시서의 필수 항목

- **목표:** 해결할 연구·기술 문제
- **기준:** 현재 Git HEAD, 참고 Artifact 및 코드 경로
- **수정 범위:** 변경 허용 파일과 변경 금지 대상
- **구현 요구:** 함수/설정/출력 형식 및 호환성
- **검증:** CPU 테스트 명령, 성공·실패 판정 기준
- **보존:** 기존 결과 및 Journal append-only
- **금지:** 승인 없는 GPU 실행·모델 다운로드·비용 발생 작업
- **완료 보고:** 변경 파일, 테스트 결과, Commit SHA, 남은 위험

## 7. 앞으로의 응답 기준

ChatGPT는 사용자가 '다음 진행해줘'라고 요청해도 이를 **GitHub 직접 수정 또는 유료 GPU 실행에 대한 포괄적 승인으로 해석하지 않는다.** 다음 연구 단계의 분석과 Codex 지시서 작성은 진행하되, 소스 변경은 Codex 워크플로를 통해 수행한다.
