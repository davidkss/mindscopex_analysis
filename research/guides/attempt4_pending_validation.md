# Attempt4 A100 Runtime Validation-only 승인 초안

2026-11-21 기준 `bc3bc59b4876460924c465a91aebb7fcb6abb3c5`에서 준비했다.
**PENDING / GPU NO-GO.** Attempt3의 사용 여부는 `UNKNOWN`이다. Attempt4는 별도
사용자 승인 대상이다. [Attempt3 승인 및 실행 기록](attempt3_approved_validation.md).

## 실행 범위와 승인할 잔여 위험

| 항목 | 제안 |
|---|---|
| Authorization ID | `validation-only-20261121-attempt4` |
| Run ID | `growth019_pinned_validation_only_20261121_attempt4_2b` |
| Session ID | `mindscopex-valonly-20261121-a4` |
| 실행 주체와 GPU | 사용자 회사 WSL, A100 논리적 할당 최대 1회 |
| 검증 | Files/SHA → Production Processor → Qwen Model → Layer 5 SAE → CUDA Synthetic Hook |
| 호출 예산 | Generate 0회, 추가 Qwen Model Forward 0회, Synthetic Hook Forward 최대 2회 |
| 재시도 | 외부 세션 생성 프로세스 최대 1회, 자동 재할당·CLI 할당 오류 재시도·자동 실행 재시도 0회 |
| 고정 Model·Tokenizer Revision | `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |
| 고정 SAE Revision | `132ea3697b591df9ee46d738aa1d528e3c6082f7` |

새 승인은 위 Validation-only 범위, 가중치 약 5.085GB와 부속 파일 다운로드, 실패 시 Artifact
회수 및 세션 정리 시도를 포함한다. Growth019 행동 재현, Cross-Case 및 Generate는 범위 밖이다.
CLI 오류 재시도는 0회지만 401 인증 갱신과 307/308 Redirect에서 POST가 재전송될 수 있다.
논리적 할당 1회의 HTTP 요청 수와 서버의 GPU 할당 수에는 보장된 상한이 없다. 중복·잔류 할당,
추가 CU 차감 및 청구 지연이 가능하다. A100 가용성, CU 사용량·예상 비용·비용 상한은 **미확정**이다.
엄격한 HTTP POST 1회 조건은 기존 NO-GO를 유지한다.
[Attempt4 승인 초안 JSON](../../experiments/inputs/growth019_validation_only_authorization_20261121_attempt4_draft.json).

## 승인과 활성 전환

사용자는 위 범위와 POST 재전송·중복 할당·추가 CU 비용 위험을 별도로 명시 승인해야 한다.
승인 후 실제 승인일과 위험 수락을 최종 정책에 기록하고, CPU 검증과 검토를 거쳐 고정 SHA를
등록한다. `ATTEMPT4_APPROVED_SHA` 등록이 단일 활성 슬롯을 Attempt4로 전환한다. 전환 때
기존 세 승인과 동일한 방식으로 Attempt3의 실제 실행 경로를 차단하며, 최종 정책의
`retirement_on_activation`에 Attempt3의 사용 여부 `UNKNOWN`을 보존한다. 기존 Attempt3
JSON·Config·Run·Claim의 기록은 그대로 남긴다.

Attempt4는 `results/.authorization_claims/validation-only-20261121-attempt4.json`을
별도로 사용한다. 실제 Launcher가 환경 검사 후 할당 전에 Claim을 소비하며, 실패한 경우에도
Claim을 보존한다. Ledger는 checkout 간 공유되지 않으므로 승인된 한 사용자 checkout에서만
시도한다. `execution_ready=false`는 검증 준비 상태를 나타낸다. 현재 Config의 CPU Dry-run은
원격 호출·GPU 할당·Claim 없이 PENDING 계획을 출력한다.
[승인 검증 코드](../../experiments/runtime_smoke.py),
[Attempt4 Config](../../experiments/configs/growth019_validation_only_20261121_attempt4_2b.toml).

## 사용자 WSL에서 실행 전 확인

1. 최종 정책이 승인되고 Attempt4만 활성화된 실행 checkout인지 확인한다. `git rev-parse HEAD`,
   `git status --short`와 승인 JSON SHA-256을 확인한다. 새 소스 Archive는 Launcher에서 생성한다.
2. 사용자 WSL에서 `command -v colab`과 `uv tool list`로 기존 검증 경로와 CLI 0.7.5를 확인한다.
   업데이트 알림이 표시돼도 CLI 0.7.5를 유지한다. Launcher가 검증된 설치 파일 지문을
   확인한다. 디스크·메모리 여유도 점검한다.
3. 아래 네트워크·TLS 점검을 실행 직전에 완료한다. CU 잔액·사용률·활성 할당을 기록한다.
   관련 활성 세션이나 귀속이 불분명한 할당이 있으면 새 할당을 보류한다.
4. 현재 승인으로 시도한 적이 없고 모든 실행 환경에서 Attempt4 Claim이 사용되지 않았음을
   확인한다. 기존 승인이나 오래된 Attempt3 checkout·Archive로 재시도하지 않는다.

### 네트워크·TLS 점검

회사망과 휴대폰 테더링은 각각 확인할 수 있다. 실제 실행에 사용할 네트워크에서 같은 사용자
WSL 셸과 Colab 계정으로 점검한다. 네트워크를 바꿨다면 새 연결에서 다시 확인한다. 테더링에서도
셸 초기화 파일이 설정한 회사 CA 및 Proxy 환경변수가 남을 수 있으므로 현재 셸을 확인한다.

- `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE`, `SSL_CERT_FILE`의 설정 여부를 확인한다.
  설정된 값은 해당 WSL에서 존재하고 읽을 수 있는 CA 파일이어야 한다. 파일 경로와 가독성을
  확인한 뒤, 선택한 네트워크에서 HTTPS 인증서 체인을 신뢰할 수 있도록 CA 번들이 구성됐는지
  확인한다. `SSL_CERT_FILE`은 이 절차에서 수동으로 점검한다.
- `HTTPS_PROXY`, `HTTP_PROXY`, `NO_PROXY`의 설정 여부를 확인하고 실제 네트워크에 맞는
  경로와 예외가 적용됐는지 확인한다. Proxy URL의 인증정보와 토큰·쿠키는 로그에서 제외한다.
- 실행 직전에 아래 읽기 전용 조회에서 정상적인 세션 목록과 각 전용 세션의 명시적 부재 응답을
  확인한다. 인증서 오류·Proxy 오류·세션 조회 실패·응답 불확실이 있으면 GPU 할당 전에 중단한다.
  `verify=False`, `curl -k` 또는 TLS 검증 비활성화는 금지한다.

```bash
colab sessions
colab status -s mindscopex-valonly-20261116-a3
colab status -s mindscopex-valonly-20261121-a4
```

CA 파일의 존재는 파일 접근 확인이고, `colab sessions` 성공은 해당 조회 경로의 TLS 검증만
확인한다. 이후 인증 갱신·리디렉션·할당·업로드 등 다른 HTTPS 요청은 대상이나 경로가 달라질
수 있어 각 요청의 인증서 검증 성공을 보장할 수 없다.
[회사 CA 운영 가이드](colab_cli_corporate_ca_tls.md).

## 활성화 후 실행할 단일 명령

아래 명령은 최종 승인과 활성화, 사전 점검을 마친 사용자가 WSL에서 한 번만 실행한다.

Attempt4의 `experiments/run_colab.sh`는 사용자 회사 노트북 WSL의 지정 checkout에서만 실행한다. 집 PC 등 다른 checkout에서는 실행을 금지한다. Claim은 checkout 간 공유되지 않는다.

```bash
bash experiments/run_colab.sh experiments/configs/growth019_validation_only_20261121_attempt4_2b.toml --session mindscopex-valonly-20261121-a4 --gpu A100 --allocation-attempts 1 --allocation-retry-delay 0 --exec-timeout 900
```

`--exec-timeout 900`은 원격 실행 명령의 대기 설정이다. 할당·회수·종료를 포함한 전체
세션 시간과 CU 비용은 별개다.

## 사후 확인

- Claim, `authorization_execution.json`, `allocation_attempt.json`, `artifact_recovery.json`,
  `session_cleanup.json`과 확보한 Runtime 산출물을 보존한다. 성공은 같은 Authorization ID·SHA의
  다섯 단계가 모두 통과하고 호출 예산을 지킨 결과로 확인한다.
- 사용자 WSL에서 `colab sessions`, `colab status -s mindscopex-valonly-20261121-a4`와
  CU 사용률·잔액·활성 할당을 조회한다. 종료 명령의 성공은 서버 종료 확인과 구분한다.
- 실패나 상태 불확실성이 발생하면 새 할당을 시작하지 않는다. Claim이 소비된 뒤에는 결과와
  관계없이 같은 승인으로 다시 실행하지 않으며, 정리 상태와 CU 이력을 확인한다.

## 최종 정책 확정 절차

2026-11-22 기준 HEAD는 `728f37093e2238409a7b5c529a3981a587473e6b`다.
현재 Attempt4는 **PENDING / GPU NO-GO**다. 최종 정책을 구성하는 규칙은
`expected_attempt4_policy(approval_date)`에 정의돼 있다. 이 함수는 초안의 고정 SHA를
검증하고 승인일을 포함한 정책 내용을 반환한다. 승인 근거는 사용자의 명시적 의사 표시로 확인한다.
[정책 검증 코드](../../experiments/runtime_smoke.py).

### 필드별 변경

| Draft 필드 | 최종 정책 필드 | 승인 후 값 |
|---|---|---|
| `authorization_status` | 동일 | `"pending"` → `"approved"` |
| `allocation_authorized` | 동일 | `false` → `true` |
| `proposed_scope` | `approved_scope` | 기존 필드를 제거하고 코드의 `ATTEMPT4_SCOPE` 문자열 추가 |
| `draft_metadata` | `approval_metadata` | 객체를 옮기고 아래 내부 필드 적용 |
| `draft_metadata.cpu_preparation_only` | `approval_metadata.cpu_preparation_only` | `true` → `false` |
| `draft_metadata.risk_acceptance_status` | `approval_metadata.risk_acceptance_status` | `"pending_explicit_user_approval"` → `"accepted_explicit_user_approval"` |
| `draft_metadata.gpu_execution_risk_accepted` | `approval_metadata.gpu_execution_risk_accepted` | `false` → `true` |
| `draft_metadata.gpu_execution_status` | `approval_metadata.gpu_execution_status` | `"NO-GO"` → `"validation_only_authorized"` |
| 해당 없음 | `approval_metadata.user_approval_date` | 실제 승인일 `YYYY-MM-DD`. 코드는 2026-11-21 이상을 요구 |
| 해당 없음 | `approval_metadata.source_draft_sha256` | `d67d23c5712daf4c5831bee1343ff33a58957b1ad12a5fe70832473b2b744929` |

스키마, 승인 ID·Run ID·Session, 실행 주체, `through`, GPU, 고정 Revision, 할당·호출·재시도
예산, 자동 세션 정리와 `execution_ready=false`는 초안 값을 사용한다. 위 표에 없는 메타데이터는
값 그대로 `approval_metadata`에 옮긴다. 여기에는 `failure_policy`, 엄격한 단일 POST 정책의
`NO-GO`, Attempt3 사용 여부 `UNKNOWN`을 담은 `retirement_on_activation`이 포함된다.
Attempt4 Config는 현재 승인 ID로 최종 정책을 찾는다.

### 사용자 승인과 최종 파일의 SHA-256

사용자가 실제 승인일과 Attempt4 ID를 명시하고, 위의 회사 WSL Validation-only 범위와 약
5.085GB 가중치 및 부속 파일 다운로드, 실패 시 Artifact 회수 및 세션 정리 시도를 승인해야
한다. 401 인증 갱신과 307/308 Redirect에 따른 추가 POST, HTTP POST 및 서버 할당 수 상한
미보장, 중복·잔류 할당, 추가 CU 차감·청구 지연과 비용 상한 미확정도 명시적 수락 대상이다.
엄격한 HTTP POST 1회가 조건이면 GPU NO-GO다.

승인을 기록한 뒤 저장소 루트에서 아래 절차를 사용한다. `ATTEMPT4_USER_APPROVAL_DATE`에는
실제 승인일을 넣는다. 파일은 코드에 지정된 최종 경로에 UTF-8 JSON으로 한 번 생성한다.
최종 SHA는 줄바꿈과 공백을 포함해 디스크에 저장된 전체 바이트에서 계산한다.

```bash
python3 -B - "${ATTEMPT4_USER_APPROVAL_DATE:?실제 승인일을 지정해야 합니다}" <<'PY'
import json
import sys

from experiments.runtime_smoke import ATTEMPT4_POLICY, expected_attempt4_policy

policy = expected_attempt4_policy(sys.argv[1])
with ATTEMPT4_POLICY.open("x", encoding="utf-8", newline="\n") as final:
    json.dump(policy, final, indent=2, ensure_ascii=False)
    final.write("\n")
PY
sha256sum experiments/inputs/growth019_validation_only_authorization_20261121_attempt4.json
```

파일 생성 성공을 확인한 뒤 최종 정책 전체와 SHA를 검토하고 활성화 변경에 해당 SHA를 등록한다.
최종 파일이 경로에 존재해도 `ATTEMPT4_APPROVED_SHA = None`이면 Attempt4는 미등록이며
CPU 계획은 고정 Draft를 읽는다. PENDING에서는 코드의 활성 슬롯이 Attempt3에 있고 그 사용
이력은 `UNKNOWN`이다. Attempt3 승인으로 Attempt4를 실행할 수 없다.
SHA를 등록하면 레지스트리에 최종 파일 경로·Run ID·SHA가 들어가고 활성 슬롯이 Attempt3에서
Attempt4로 전환된다. 활성 정책은 파일 SHA와 고정 Draft에서 재구성한 전체 JSON 및 Revision이
일치해야 통과한다. 기존 Attempt3 정책은 이력 조회에 남고, Launcher·Job·Runtime·Claim의
실행 검사는 활성 ID 불일치로 `historical approval retired`를 반환한다. 로컬 Claim이 없어도
같은 차단이 적용된다. 이전 checkout·Archive에는 해당 전환이 반영되지 않으므로 재사용하지 않는다.

### 활성화 변경의 CPU 검증

1. PENDING에서 최종 경로의 파일만으로 활성화할 수 없는지 확인한다. 레지스트리의 SHA 미설정,
   파일 누락·SHA 불일치, Draft 대입, 승인일·위험 수락·Revision·예산·`UNKNOWN` 변조를 거부해야
   한다. 올바른 최종 파일과 SHA 조합은 검토본과 일치해야 한다.
2. 메모리 Mock이나 격리본에서 Attempt4만 활성이고, Attempt3와 나머지 과거 승인 세 개가
   새 Ledger와 기존 Claim 유무에 관계없이 차단되는지 확인한다. 잘못된 Run·Session·GPU,
   TPU·`--keep`·재시도 옵션, Generate도 거부해야 한다.
3. CLI 0.7.5 환경 Gate 실패, 기존 또는 불확실 세션, 중복 Claim에 대한 차단을 확인한다.
   Launcher가 환경 Gate 다음에 Claim을 소비하고, 할당 실패나 불확실한 응답에도 같은 승인으로
   재할당 없이 회수·정리를 시도하며 Claim을 보존하는 흐름을 Mock으로 검사한다.
4. Mock Backend에서 Files/SHA → Processor → Qwen Model → Layer 5 SAE → CUDA Synthetic
   Hook 순서와 Generate 0·추가 Qwen Forward 0·Synthetic Hook Forward 최대 2를 확인한다.
   Dry-run에서 최종 SHA와 Config, 소스 Archive CRC·파일 바이트 및 원격 호출·Claim 0을
   확인하고 전체 CPU 회귀를 실행한다.

CPU 증거와 활성화 Diff는 운영 규칙 §12에 따라 검토한다. 사용자 Commit·Push 후 회사 WSL의
지정 checkout에서 위 사전 점검을 완료하고 단일 실행 명령을 사용한다. 승인 이후에도 실제
A100 가용성, 서버 처리·종료, CU와 Runtime 5단계는 회사 WSL 실행 및 사후 확인으로 검증한다.
[운영 규칙](../../MINDSCOPEX_WORKFLOW_RULES.md#12-2026-11-21-ai-협업-및-독립-검토).
