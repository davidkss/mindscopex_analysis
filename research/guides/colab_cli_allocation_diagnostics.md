# Colab CLI A100 할당 진단 및 validation-only 단일 시도 안전성

기준일: 2026-11-16. 조사 기준 HEAD: `3f80a312a5ee8a5df1d07ceb453886824d196f6a`.
실제 Colab 명령/원격 API/GPU 실행은 수행하지 않았다. 공식 GitHub 소스만 조회했다.

## 1. 장애 이력과 근거 범위

| 날짜 | 확인된 실행 보고 | Runtime | 근거 한계 |
|---|---|---|---|
| 10월30일 | 상태 조회 TLS 인증서 오류, 할당 명령 0회 | 5단계 미도달 | 당시 기록 기준, 후속 회사 CA 적용 조회 성공과 구분 |
| 11월14일 | CLI 할당 요청 1회, HTTP 503 Service Unavailable | 5단계 미실행, Generate 0 | 사용자 보고 및 과거 Journal; 내부 POST 횟수 미확인 |
| 11월15일 Attempt2 | CLI 할당 호출 1회, ColabRequestError: Service Unavailable, exit 1 | 5단계 미실행, 실제 모델/SAE 로딩 및 Generate 0 | 원문 숫자503 없음, 실패 Run/claim 현재 checkout에 없음 |

11월15일 Run은 `results/runs/20261115-183312_growth019_pinned_validation_only_20261114_attempt2_2b`,
승인 ID는 `validation-only-20261114-attempt2`다. 실패 원문 로그 저장/claim 생성/회수 시도/
종료 명령 정상 반환은 사용자 보고 근거이며 로컬 원본을 검증한 것으로 표현하지 않는다.
기존 allocation_attempt.json의 category=unknown은 소급 수정하지 않는다.
두 날짜의 내부 원인이나 CLI 버전/환경이 같다고 가정하지 않는다.

## 2. 회사 WSL 점검 결과 (사용자 제공)

2026-11-16 보고: `/home/davidkss/.local/bin/colab`, uv tool의 google-colab-cli 0.7.5,
업데이트 안내 0.8.0, `colab --version` 미지원. 두 재시도 환경변수는 모두 unset.
세션 조회는 No active sessions found on server, CU 1754.49, 사용률 0.00/hr, 활성 할당 0.
이는 해당 조회 시점의 결과이며 11월14~15일 설치 버전/환경을 증명하지 않는다.
조회 성공은 할당 권한/가용 용량/향후 할당 성공을 보장하지 않으며 CU 청구 원장을 검증한 것도 아니다.
이번 Codex 작업에서 CLI 실행이나 사용자 인증정보 조회는 하지 않았다.

## 3. 공식 릴리스 소스 비교

직접 확인한 공식 태그 소스:

- [v0.7.4 client.py](https://github.com/googlecolab/google-colab-cli/blob/v0.7.4/src/colab_cli/client.py#L265)
- [v0.7.5 client.py](https://github.com/googlecolab/google-colab-cli/blob/v0.7.5/src/colab_cli/client.py#L274)
- [v0.8.0 client.py](https://github.com/googlecolab/google-colab-cli/blob/v0.8.0/src/colab_cli/client.py#L274)

| 동작 | 0.7.4 | 0.7.5 | 0.8.0 |
|---|---|---|---|
| assign 최초 동작 | GET, Assignment이면 반환 | 동일 | 동일 |
| COLAB_CLI_ASSIGN_RETRIES 기본값 | 읽지 않음, 루프 없음 | 3, 음수는0으로 제한 | 동일 |
| COLAB_CLI_RETRY_BACKOFF 기본값 | 읽지 않음 | 10초, 음수는0으로 제한 | 동일 |
| 502/503/504 | 예외 전파 | 추가 최대3회, 10/20/40초 대기 | 동일 |
| ReadTimeout | 예외 전파 | 추가 최대3회 | 동일 |
| 412 | TooManyAssignmentsError로 변환, 재시도 없음 | 동일 | 동일 |
| 재POST 전 확인 | 추가 POST 없음 | GET으로 재조회, Assignment면 즉시 반환 | 동일 |
| assign의 _post_assignment 호출 상한 | 1 (기존 Assignment면0) | 기본4, retries=0이면1 | 동일 |

상한은 이 함수의 `_post_assignment` 호출 수이며 HTTP 전송 계층 전체의 wire-level 요청 수를
실측한 값이 아니다. POST 실패 후 서버 할당이 생성됐을 가능성도 남는다.
GET 자체의 예외, 비재시도 오류, 재조회에서 Assignment 반환/예외 발생은 더 일찍 종료한다.
0.7.5/0.8.0의 초기 GET은 루프 밖이므로 그 GET의 ReadTimeout은 이 루프로 재시도하지 않는다.
재시도 시 최신 GET의 token으로 POST한다. `_get_assignment`는 GET 요청과 응답 union 검증,
`_post_assignment`는 XSRF header를 붙인 POST 요청과 응답 검증을 수행한다.
0.7.5/0.8.0은 `_issue_request`에서 선택적 COLAB_CLI_REQUEST_TIMEOUT도 읽지만 이번에 변경하지 않는다.

조회한 client.py SHA-256:

```text
0.7.4 af6cb28c96ac9dc3d1d1a0a7587ad87acc8e0eed36b63de301ce4b9f026755b1
0.7.5 7081d2c9decbc9a4cafb4ab89202c7e7a644f72bd74a7e846bffce5344839a06
0.8.0 7081d2c9decbc9a4cafb4ab89202c7e7a644f72bd74a7e846bffce5344839a06
```

### 오류 문구의 의미

세 버전의 `_issue_request`는 오류 메시지에 response.reason을 사용하며 숫자 status를 반드시
포함하지 않는다. [get_status_code](https://github.com/googlecolab/google-colab-cli/blob/v0.7.5/src/colab_cli/utils.py#L24)는
문자열 대신 예외 response.status_code 등을 조회한다. Service Unavailable만으로 실제 숫자503을
독립 확인했다고 할 수 없다. 향후 민감정보를 제거한 응답 status/시간/요청별 로그가 필요하다.
인증 헤더·토큰·전체 debug 로그를 그대로 공유하지 않는다.

## 4. Launcher 경계와 최소 안전장치

`--allocation-attempts 1`은 외부 CLI 호출 예산이다. validation-only의
`ensure_validation_session_once`는 상태 부재 확인 후 `run_cmd`를 1회 호출하며 외부 재시도하지 않는다.
기존에는 CLI 자식이 부모 환경을 상속해 0.7.5 기본 내부 재시도가 가능했다.
이번 변경은 해당 할당 자식에만 다음을 명시 전달한다.

```text
COLAB_CLI_ASSIGN_RETRIES=0
COLAB_CLI_RETRY_BACKOFF=0
```

부모 os.environ 복사본을 사용하므로 사용자 값3/기타값도 자식에서만0으로 덮어쓴다.
CA/Proxy/PATH 등 나머지는 유지한다. 공통 run_cmd의 env 인자는 선택적이며 일반 실험 동작은 유지한다.
로그 원문/분류/기존 진단 JSON 필드와 실패 재전파는 유지하고, Runtime 진입 전 claim 및
불확실한 할당 실패 후 Artifact 회수/finally 종료도 유지한다. 정리는 할당 재시도가 아니다.

`ACTIVE_AUTHORIZATION=None`은 유지한다. Launcher→claim, Job→require_allocation_authorized,
Runtime→동일 gate에서 기존 모든 승인 실행을 거부한다. 신규 승인/Config를 만들지 않는다.

### 보장 범위와 지원 불명 환경의 실행 거부

확인한 릴리스 assign 루프에서 내부 추가 POST를0으로 만드는 설정이다.
Mock은 자식 argv/env와 외부 호출1회만 확인하며 실제 CLI의 내부 HTTP 요청 횟수는 실측하지 않는다.
변경된 CLI/wrapper, 임의 빌드, 인증 transport의 재전송/redirect까지 일반적으로 막았다고 주장하지 않는다.
미확인 CLI까지 허용하는 자동 버전 감지/우회 옵션은 추가하지 않았다. 현재 모든 실제 경로는
비활성 승인으로 fail-closed다. 향후 활성화 전에 설치 실행파일과 해당 소스/의존 transport를
CPU에서 대조하고 환경변수 적용을 무시하거나 확인할 수 없는 환경은 **실행 거부 상태를 유지**한다.
버전 문자열만으로 검증을 대체하지 않는다. 더 강한 wire-level 단일 POST 보장이 필요하면
별도 승인 전 CPU transport 검증과 추가 gate 설계를 먼저 수행해야 한다.

## 5. 원인 후보별 평가

| 후보 | 확인된 증거 | 미확인 | 추가 GPU 없이 가능한 확인 |
|---|---|---|---|
| Colab 서비스 오류 | 할당 단계 오류 보고; 공식 저장소 [이슈47](https://github.com/googlecolab/google-colab-cli/issues/47)에 유사 서비스 오류 사용자 보고 | 당시 서버 내부 장애/광역 장애 여부 | 시각·상태코드가 있는 기존 원문 및 공식 이슈 비교 |
| A100 용량 부족 | 할당이 완료되지 않았다는 보고 | 재고·용량 부족이라는 직접 증거 없음 | 기존 응답 body의 구조화 사유 유무 확인 |
| CLI 버전 차이 | 공식0.7.4와0.7.5 루프 차이 | 과거 실제 실행 버전·코드 | 과거 Manifest/source archive/설치 기록 대조 |
| 내부 자동 재시도 | 0.7.5/0.8.0 기본3회 추가 가능 | 과거 실제 POST 수·그 환경변수 | 기존 요청별 로그/프로세스 환경 기록 대조 |
| 회사 Proxy/TLS | 10월30일 TLS 오류 기록 | 이번 실패와 동일 원인인지 불명 | 비밀정보 제외 기존 CA/Proxy 설정·traceback 검토 |
| 인증/세션 | 사용자 현재 조회 성공/활성0 보고 | 할당 시점 인증·권한·서버 상태 | 당시 인증 오류 여부 및 기존 세션 기록 대조 |
| MindScopeX | 실패는 Runtime 전, 외부 CLI1회; 내부 재시도 설정 누락 위험 발견 | 내부 서버 원인 및 Runtime 호환성 | CPU Mock·코드 경계·원문 보존 검증 |

공식 저장소 이슈는 다른 사용자의 사례이며 Google의 이번 장애 확인 공지가 아니다.
[이슈73](https://github.com/googlecolab/google-colab-cli/issues/73)은412 예외명이
실제 서버 사유를 확정하지 못하는 사례도 보고한다. 412/503만으로 quota/재고를 단정하지 않는다.

## 6. CPU 검증

관련46개 테스트가 통과했다. 신규3개 메서드에서 부모 retries 미설정/3/0 × backoff 미설정/25 ×
report 유무, 실제 subprocess 인자 전달/부모환경 불변, 일반 명령 환경 상속을 검사한다.
503/Service Unavailable/TLS/인증/unknown/ReadTimeout의 원문 및 기존 분류를 검사한다.
ReadTimeout은 기존 unknown 분류를 유지한다. 생명주기 Mock에 환경 확인과 timeout 후 회수·종료를
추가했다. 소진 승인 차단 및 claim 미생성 회귀도 통과했다. 실제 CLI/HTTP 호출은0회다.
전체 smoke와 정적 검증의 최종 결과/실행 명령은 같은 날짜 Journal 후속 절에 기록한다.

## 7. 다음 Runtime 검증 전 조건과 범위 제안

1. 사용자가 Commit을 회사 WSL에서 Push하고 실행 checkout의 HEAD/clean 상태를 확인한다.
2. 기존 실패 Run/승인 claim은 보존하고 당시 CLI 버전·원문 상태코드·요청 횟수는 가능한 범위만 대조한다.
3. 실제 CLI 실행파일/설치 릴리스/소스와 재시도 환경 적용 및 transport 동작을 CPU에서 확인한다.
   적용을 확인할 수 없으면 활성 승인 없이 중단한다. 이번에 CLI 설치/업그레이드하지 않는다.
4. 그 뒤 별도 신규 승인을 받아 신규 ID/Run/정책을 준비한다. 현재는 생성·활성화하지 않는다.
5. 제안 범위: A100 할당 CLI 최대1회, 확인한 assign 루프 POST 최대1회/자동 추가 재시도0,
   Files/SHA→Production Processor→Model→Layer5 SAE→CUDA Synthetic Hook,
   Generate0/추가 Qwen Model Forward0/합성 Hook 최대2, Artifact 회수 후 종료 및 사용자 서버/CU 확인.
6. Model/Tokenizer `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`,
   SAE `132ea3697b591df9ee46d738aa1d528e3c6082f7`와 SHA/디스크/의존성/메모리 예산을 재확인한다.
   파일 확보에는 약5.085GB weights와 부속파일이 필요할 수 있으므로 이 범위도 승인에 포함한다.

Runtime5단계는 여전히 미검증이다. 서비스 가용성·실제 GPU·네트워크 및 서버 종료/청구는
CPU Mock이 보장하지 않는다. 이번 작업 성공은 할당 서비스 복구나 Runtime 성공을 의미하지 않는다.
코드/테스트/Journal/Commit은 Codex, Push는 사용자 WSL, GPU 실행은 별도 승인 후 사용자 WSL 담당이다.
