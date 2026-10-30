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

## 8. 작업 시작 전 MD 이력 확인 — ChatGPT·Codex 공통 필수 규칙

**ChatGPT와 Codex는 MindScopeX의 어떤 작업을 시작하기 전에도 저장소의 MD 기록을 먼저 읽고 현재까지의 진행 이력을 확인한다.** 사용자가 `다음 진행해줘`, `계속`, `수정해줘`라고 요청한 경우에도 예외 없이 적용한다.

### 필수 사전 확인 순서

1. 저장소 최신 상태와 현재 Git HEAD를 확인한다. 오래된 로컬 파일이나 이전 대화 요약만으로 최신 상태를 단정하지 않는다.
2. 루트 `MINDSCOPEX_WORKFLOW_RULES.md`를 읽어 역할·승인·보존 규칙을 확인한다.
3. `research/journal/`의 최신 날짜 MD와 현재 단계 관련 과거 일지 MD를 읽는다. 마지막 완료 단계, 진행 중 작업, 미검증 항목, 다음 계획을 구분한다.
4. 작업과 관련된 `README.md`, 실험 계획 MD, 결과 요약 MD가 있으면 함께 확인한다. MD에 기록된 결과와 실제 코드·Artifact가 충돌하면 이를 명시하고 근거를 확인한다.
5. 작업 시작 전 **현재 기준 Git Commit / 마지막 완료 단계 / 미완료·미검증 항목 / 이번 작업 목표**를 간단히 정리한다.
6. 확인한 이력에 근거해 중복 실행·기존 결과 덮어쓰기·이미 해결된 문제의 재작업을 방지하고, 다음 단계의 범위와 검증 기준을 정한다.

### 작업 종료 시

- Codex가 변경 파일, 테스트 실행 여부와 실제 결과, Commit SHA, 남은 문제를 보고한다.
- 연구 일지는 **append-only**로 최신 진행 내용을 추가한다. 이전 기록을 임의로 삭제·수정하지 않는다.
- GPU 실험 및 비용 발생 작업은 기존과 같이 별도의 명시적 승인이 필요하다.
- MD 파일에 접근할 수 없거나 최신 이력을 확인할 수 없으면 **확인하지 못한 사실을 확인한 것처럼 말하지 않는다.** 확인 가능한 범위를 밝히고 필요한 자료 또는 접근을 요청한다.

### Codex 지시서에 항상 포함할 문구

> 코드 수정·테스트·Commit 전에 반드시 최신 Git HEAD, `MINDSCOPEX_WORKFLOW_RULES.md`, `research/journal/`의 최신 및 관련 MD를 읽고 이전 진행 이력·미검증 항목을 요약하라. 확인된 이력에 따라 작업을 수행하고, 기존 결과를 보존하며, 완료 후 연구 일지를 append-only로 갱신하라.

## 9. Codex와 Colab CLI 실행 역할 분리

2026-10-30 Phase 5-D-3B 후속 운영 결정이다. 기존 실행 이력과 위 규칙은 보존하며, Colab 실행 주체가 불명확한 표현이나 과거 예외 승인과 관계없이 **향후 기본 운영 방식은 이 절을 따른다.** 기존 README·실험 계획의 원격 실행 명령은 사용자 WSL 터미널에서 실행할 명령으로 해석한다.

| 주체 | 책임 |
|---|---|
| ChatGPT | 연구 설계, 결과 분석, Codex 지시서 작성; Colab 실행 명령과 사전 점검 절차 안내; GPU 실행 결과 및 Artifact 해석 |
| Codex | 코드 구현·수정, CPU 테스트·Mock 검증·Dry-run, Git Commit/Push, 연구 일지·기술 문서 작성, GPU 실행용 Config/Manifest 준비 |
| 사용자 WSL 터미널 | 회사 인증서가 적용된 환경에서 Colab CLI 실행, A100 세션 생성 및 실험 실행, GPU 실행 로그·Artifact 확보 |
| Google Colab | Qwen 모델·SAE 실제 로딩, CUDA Tensor/Hook 검증, 승인된 범위의 GPU 실험 수행 |

### 필수 운영 원칙

1. Codex는 기본적으로 `colab new`, `colab exec`, `colab stop` 등 원격 세션 제어 명령을 실행하지 않는다.
2. GPU 실험은 사용자의 명시적 승인 후 **사용자 WSL 터미널**에서 수행한다.
3. Codex는 실행 가능한 명령과 사전 검증 결과를 제공한다. 원격 연결 없는 CPU Dry-run과 실제 GPU 실행을 구분한다.
4. ChatGPT는 GPU 실험 지시를 Codex의 코드 수정 지시와 분리한다.
5. 예외적으로 Codex에 원격 실행을 맡기려면 사용자가 **실행 주체와 승인 범위**를 별도로 명시해야 한다. 과거 승인은 후속 실행·재시도·범위 확대의 포괄적 승인이 아니다.
6. 기존 GPU 실행 결과와 실패 기록을 수정하거나 성공으로 재분류하지 않는다.

Phase 5-D-3B에서는 사용자의 예외적 명시 승인에 따라 Codex가 Validation-only Launcher를 실행했으나 TLS 오류로 GPU 할당 전에 중단했다. 이 실패 기록은 유지한다. 후속 사용자 WSL 연결 성공은 모델/SAE 런타임 검증 성공을 의미하지 않는다. 회사 CA 사전 점검은 [Colab CLI 회사 CA TLS 운영 가이드](research/guides/colab_cli_corporate_ca_tls.md)를 따른다. 인증서 재설치·CLI 업데이트·TLS 검증 우회는 이번 후속 조치에 포함하지 않는다.

## 10. ChatGPT / Codex 혼합 운영 방식

MindScopeX의 연구 효율성과 실험 재현성을 높이기 위해 소스코드 개발과 연구 문서 작성을 분리한다. **이 절은 승인된 연구 문서에 대한 향후 예외 및 역할 분담 기준이다.** 기존1~9절의 이력은 삭제하지 않으며, 1~8절의 ChatGPT 직접 문서 변경 제한 및 기존 문서·Git 담당 표현과 충돌할 경우 해당 승인 문서 범위에는 이 절을 적용한다. 소스코드 변경 책임과9절의 GPU 승인·실행 원칙은 유지한다.

### 10.1 역할별 담당 범위

| 작업 | 기본 담당 |
|---|---|
| 연구 설계 및 가설 수립 | ChatGPT |
| 연구 결과 분석 및 해석 | ChatGPT |
| 연구 진행 현황 보고서 | ChatGPT |
| 논문 초안 및 연구 요약 MD | ChatGPT |
| 소스코드 구현 및 수정 | Codex |
| Notebook / Config 수정 | Codex |
| CPU 테스트 및 회귀 검증 | Codex |
| 실험 기술 문서 및 실행 결과 Journal | Codex |
| Git Commit / Push | 각 작업 담당자 |
| Colab CLI GPU 실행 | 사용자 WSL |
| Qwen / SAE CUDA 실행 | Google Colab |

### 10.2 ChatGPT의 GitHub 직접 수정 허용 범위

사용자가 명시적으로 승인한 경우 ChatGPT는 다음 문서를 GitHub에서 직접 작성하거나 수정할 수 있다.

- `research/reports/`의 연구 진행 현황 보고서
- 논문 초안 및 연구 요약 문서
- 실험 결과를 바탕으로 작성하는 분석 중심 MD 문서

직접 수정 전에 반드시 최신 Git HEAD와 관련 Journal, Audit, Artifact를 확인한다. 직접 수정 시에도 Git Commit SHA와 변경 파일을 사용자에게 보고한다. 사용자의 단순한 분석·설명 요청을 GitHub 수정 승인으로 해석하지 않는다.

### 10.3 Codex 담당 범위

Codex는 다음 작업을 기본적으로 담당한다.

- `src/`
- `experiments/`
- `tests/`
- Jupyter Notebook
- 실험 Config / Manifest
- 실험 실행과 재현성에 직접 연결되는 기술 문서
- 실험 결과 및 검증 이력을 기록하는 `research/journal/`

소스코드 변경 시 필요한 CPU 테스트와 회귀 검증을 수행하고 실제 결과를 보고한다.

### 10.4 Journal 운영 원칙

`research/journal/YYYY-MM-DD.md`는 Codex가 실험 실행 및 코드 변경 이력을 기록하는 것을 기본으로 한다. ChatGPT는 Journal을 검토하고 연구적 의미를 해석하여 `research/reports/`에 반영한다.

Journal은 기존 내용 보존 및 append-only 원칙을 유지한다. 예외적으로 ChatGPT가 Journal을 직접 수정해야 할 경우 사용자의 별도 명시적 승인을 받는다.

### 10.5 GitHub 변경 안전 규칙

1. 모든 변경 전 최신 Git HEAD를 확인한다.
2. 기존 Run/Audit/Artifact를 덮어쓰지 않는다.
3. 기존 연구 결과를 새로운 실험 결과로 표현하지 않는다.
4. GitHub 직접 변경과 Codex 변경을 동시에 진행하여 충돌시키지 않는다.
5. 원격 Commit과 로컬 WSL 상태가 달라지면 다음 작업 전에 동기화한다.
6. Commit SHA와 Push 성공 여부를 명확히 구분하여 보고한다.
7. 문서만 변경한 작업을 코드 테스트 또는 GPU 검증 완료로 표현하지 않는다.
8. 사용자가 별도로 지정한 작업 방식이 있다면 해당 요청의 승인 범위 내에서 수행한다.

### 10.6 GPU 실험 운영 원칙

기존9절을 유지한다. Codex와 ChatGPT는 기본적으로 Colab GPU 실험을 직접 실행하지 않는다. GPU 실험은 별도 사용자 승인 후 사용자 WSL에서 Colab CLI를 통해 실행한다. 기존 승인이나 문서 변경 승인을 GPU 재실행 승인으로 해석하지 않는다.
