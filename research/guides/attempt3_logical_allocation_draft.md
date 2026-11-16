# Attempt3 B안 — 논리적 할당 1회 CPU 전용 승인 초안

기준일: 2026-11-16. 준비 기준 HEAD: `36fdd7eb3f65ccffadc30db1787f1905f947b476`.
**Draft 준비만 완료하는 작업이며 GPU 실행 상태는 NO-GO다. 위험 수락과 실제 승인은 아직 없다.**

## 1. 배경과 보존된 결과

11월14일과15일 Attempt2는 Colab 할당 단계에서 실패했고 Runtime5단계/모델·SAE 실제 로딩/
Generate에 도달하지 못했다. 사용자 제공 실행 보고와 원본 확인 범위는 기존 Journal을 따른다.
과거 승인은 소비됐으며 ACTIVE_AUTHORIZATION=None으로 재사용을 차단한다.

설치 CLI0.7.5에서 assign retries0은 오류 루프를 제한하지만 인증 갱신·Redirect의 POST 재호출까지
막지 못한다. 엄격한 단일 HTTP POST 정책의 이전 NO-GO는 그대로 보존한다.
관련 근거는 [할당 진단 가이드](colab_cli_allocation_diagnostics.md)의8~9절이다.
B안 초안 작성 동의는 이 위험을 수락하거나 이전 NO-GO를 성공으로 재분류하는 동의가 아니다.

## 2. 정책 비교

| 항목 | 엄격한 단일 POST 정책 | B안 논리적 할당 정책 초안 |
|---|---|---|
| 기준 | HTTP 할당 POST≤1 | 외부 세션생성 프로세스≤1 |
| Launcher 자동 재할당 | 0 | 0 |
| CLI502/503/504·ReadTimeout 추가 루프 | 0 | 0 |
| 인증 갱신에 따른 POST 재호출 | 금지 필요 | 발생 가능, 명시적 위험 수락 필요 |
| 307/308 Redirect POST | 금지 필요 | 발생 가능, 명시적 위험 수락 필요 |
| HTTP POST 총횟수≤1 보장 | 요구했으나 미충족 | 보장하지 않음 |
| 서버 측 GPU 할당수≤1 보장 | 입증되지 않음 | 보장하지 않음 |
| 현재 실행 상태 | NO-GO | NO-GO, 미승인 Draft |

B안의 논리적1회는 `colab new` 프로세스 최대1회, Launcher 자동 재할당0,
COLAB_CLI_ASSIGN_RETRIES=0 및 COLAB_CLI_RETRY_BACKOFF=0을 뜻한다.
오류·타임아웃·불확실한 성공 뒤 새로운 할당을 자동 시작하는 권한은 없다.
인증·Redirect에 따른 하위 HTTP 요청은 이 프로세스 횟수와 다르며 추가 요청 위험은 남는다.
401·307·308에서 서버가 요청을 어떻게 처리했는지 확정하지 않는다.
중복 GPU 할당/비용 발생 가능성이 완전히 제거됐다는 뜻이 아니다.

## 3. 초안 식별과 실행 분리

- JSON: `experiments/inputs/growth019_validation_only_authorization_20261116_attempt3_draft.json`
- Authorization ID: `validation-only-20261116-attempt3`
- Run ID: `growth019_pinned_validation_only_20261116_attempt3_2b`
- Session: `mindscopex-valonly-20261116-a3`
- Draft SHA-256: `f82f4a3f518a04ef0a1cfb5ce2e6c2550c8f8620e31df7db1b449f9e124a610a`

SHA는 초안 검토용 식별값이다. 최종 GPU 승인 지문이 아니다.
생성 전 해당 ID·경로가 저장소에 없음을 확인했다.

schema1의 기존 핵심 필드/예산/Revision을 유지하고, B안 설명은 draft_metadata로 분리했다.
strict_http_post_limit=`not_guaranteed`, server_side_single_allocation_guaranteed=false,
authentication_post_replay_possible=true, redirect_post_replay_possible=true다.
risk_acceptance_status=`pending_explicit_user_approval`, gpu_execution_risk_accepted=false다.
authorization_status=pending/allocation_authorized=false/execution_ready=false를 유지한다.

현재 실행 스키마에 등록된 정책이 아니다. 독립 CPU 테스트가 JSON을 검증하며 Runtime/Launcher/
Job 및 기존 Attempt2 draft_plan은 변경하지 않았다. Attempt3는 AUTHORIZATIONS 밖이고
ACTIVE_AUTHORIZATION=None이다. JSON만 변경하거나 approved로 위조해도 실행/claim은 허용되지 않는다.
실행용 TOML/최종 승인 JSON/새로운 활성화 코드는 만들지 않았다.

## 4. 제안 Runtime 범위와 실패 처리

제안 범위는 A100/user_wsl/through=validate의 다음5단계다.
Files·SHA → Production Processor → Qwen Model → Layer5 SAE → CUDA Synthetic Hook.
Generate0/추가 Qwen Model Forward0/Synthetic Hook 최대2회이며 실험 생성 범위를 확대하지 않는다.

고정 Model/Tokenizer Revision=`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`,
SAE Revision=`132ea3697b591df9ee46d738aa1d528e3c6082f7`.
향후 Files 단계에서는 모델·SAE weights 약5.085GB와 부속파일 다운로드 및 GPU/디스크/메모리
사용이 발생할 수 있다. 이번 작업에서는 다운로드·실제Forward·Generate·GPU 할당을 하지 않는다.

Draft의 failure_policy는 다음 순서를 향후 정책 요구로 정의한다. 현재 실행 기능을 추가한 것은 아니다.

1. 할당 전에 별도 승인 상태/정확한 실행환경을 확인한다. 기존 또는 불확실한 세션이면 중단한다.
2. 최초 할당 전에 새 승인의1회 사용 기록을 소비한다. 과거승인은 사용하지 않는다.
3. 실패·불확실한 할당 뒤 자동 신규 할당/재할당은0이다. 인증·Redirect의 POST 위험은 별도다.
4. 해당 세션 상태를 확인하고 가능한 Artifact를 회수한 뒤 세션 종료를 시도한다.
5. 종료 명령 성공과 실제 서버 종료 확인을 구분하고 사용자 WSL에서 최종 세션/CU를 확인한다.
6. Claim은 실패해도 보존하며 삭제/재사용으로 승인을 복구하지 않는다.

기존 Launcher의 회수/finally 종료는 보존돼 있지만 새 상태조회·승인 절차가 모두 실행 가능하다고
판정한 것은 아니다. 향후 최종 정책 준비 시 실제 코드와 위 초안 요구의 일치 여부를 점검해야 한다.

## 5. CPU 검증 및 실제 승인 전 조건

신규 test_validation_attempt3_draft.py는 core schema/Revision/예산, 논리적1회와 POST차이,
위험수락pending/NO-GO, failure_policy, 등록되지 않은 실행·claim·Generate 차단,
기존 Attempt2 정책 지문·Draft 동작 보존을 확인한다. 기존 환경/회수·종료 회귀도 실행한다.
실제 테스트 명령/전체 smoke/보존 확인 결과는11월16일 Journal의 이번 작업 절에 기록한다.

다음은 이후 사용자 검토용 제안이며 이번에 승인하지 않았다.

- B안이 HTTP POST 총횟수나 서버 측 단일 할당을 보장하지 않는다는 명시적 위험 수락.
- 인증 갱신/307·308 재호출 및 잠재 중복 할당·비용/불확실한 서버 처리 위험 확인.
- 정확한 실행checkout/CLI환경/CA/세션상태/CU/디스크·의존성·Revision 점검.
- 신규 승인 ID와 별도 소비 기록, 최종 정책의 승인범위·SHA 고정 및 별도 실행 Config 검토.
- 현재 Draft ID의 최종 사용 여부도 별도 승인에서 확정. 과거 ID/claim은 재활성화하지 않음.
- 논리적1회/CLI루프0/Generate0/추가Forward0/Hook≤2/다운로드범위/회수·정리·서버확인을 모두 승인범위에 포함.
- CPU 회귀검증 이후에도 실제 실행 주체는 별도 승인받은 사용자 WSL이며 Codex는 자동 실행하지 않음.

이번 초안 파일만으로 위 조건을 충족한 것이 아니다. 사용자 위험 수락, 최종 정책/Config의 별도
준비 및 명시적 실행 승인 전까지 GPU 실행은 NO-GO로 유지한다. Push도 사용자 WSL 담당이다.
