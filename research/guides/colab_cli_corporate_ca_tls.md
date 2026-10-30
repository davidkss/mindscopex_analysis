# Colab CLI 회사 CA TLS 운영 가이드

## 목적과 실행 주체

회사 CA는 Codex 전용 설정이 아니다. 회사 WSL 환경에서 Colab CLI의 HTTPS 인증서 검증에 사용하는 신뢰 인증서다. 향후 Colab CLI 연결 확인·세션 제어·승인된 GPU 실행은 **사용자 WSL 터미널**에서 수행한다. Codex는 코드·Config/Manifest·CPU 검증·원격 연결 없는 Dry-run 및 명령 안내를 담당한다. 예외 원격 실행은 실행 주체와 범위를 명시한 별도 사용자 승인이 필요하다.

기준 운영 규칙: [MINDSCOPEX_WORKFLOW_RULES.md 9절](../../MINDSCOPEX_WORKFLOW_RULES.md). 이 문서의 명령은 사용자 WSL용 안내이며 이번 문서화 작업에서는 실행하지 않았다. WSL Bash에서는 `!git`, `%cd` 같은 Jupyter 전용 구문을 사용하지 않는다.

## 확인된 환경과 근거

다음 환경 및 연결 성공은 **사용자가 제공한 확인 결과**다. Codex가 이번 작업에서 사용자 셸 설정을 읽거나 Colab 연결을 재검증한 결과가 아니다.

| 항목 | 사용자 확인 내용 |
|---|---|
| 회사 CA | `/usr/local/share/ca-certificates/sk-holdings-root.crt` |
| 시스템 CA 번들 | `/etc/ssl/certs/ca-certificates.crt` |
| 셸 설정 | `~/.profile`, `~/.bashrc`에 `REQUESTS_CA_BUNDLE` 설정 존재 |
| Colab CLI | 0.7.4 |
| 연결 결과 | 사용자 WSL에서 CA 환경변수 적용 후 `colab sessions` 성공 |
| 서버 응답 | `No active sessions found on server.` |

이 응답은 **사용자 조회 시점에 활성 Colab 세션이 없었다**는 후속 확인이다. 과거 Codex 실행 당시 서버 상태를 소급 확정하거나, 기존 실패 Run을 성공으로 변경하는 근거가 아니다.

## Phase 5-D-3B 실패와 후속 판단

기존 Codex 실행은 최초 `colab status`에서 `SSLCertVerificationError: self-signed certificate in certificate chain`으로 실패했고, 세션 부재를 확인하지 못해 GPU 할당 전에 중단했다. `colab new`·원격 작업·Generate·Qwen Forward·합성 Hook Forward 모두0회였다. 당시 정리 명령은 Session not found를 반환했으나 마지막 상태 조회도 TLS 오류여서 서버 상태는 미확인으로 기록했다. 원본 [실행 보고서](../audits/2026-10-30-feature-specificity-cross-case/validation_only_execution_report.md)는 그대로 보존한다.

이후 사용자 WSL에서는 회사 CA 환경변수를 적용한 연결이 성공했다. Codex 실행과 사용자 WSL 실행 사이의 환경변수 전달·셸 초기화·CA 참조 차이는 **가능한 원인이며 확정되지 않았다.** 이번 작업에서는 이를 비교 실험하거나 TLS 오류의 단일 원인을 확정하지 않았다.

현재 사용자 WSL 연결이 성공했으므로 **인증서 재설치와 CLI 업데이트는 현재 불필요하다.** Colab CLI의 업데이트 알림만으로 이 실패 원인을 버전 문제로 판단하지 않는다. `verify=False`, `curl -k`, CA 검증 비활성화 등 TLS 검증 우회는 금지한다. 인증서 원본, 개인 키, 토큰, 쿠키, 인증정보 및 셸 설정 파일 원본을 Git에 추가하지 않는다.

## GPU 실행 전 사용자 WSL 사전 점검

아래는 인증서 경로·환경변수·읽기 전용 연결을 확인하는 절차다. **GPU 실행 승인이 아니며 세션을 생성하지 않는다.** 사용자 WSL Bash에서 수행한다.

```bash
cd ~/ai/mindscopex_analysis

test -r /usr/local/share/ca-certificates/sk-holdings-root.crt
test -r /etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
printf 'REQUESTS_CA_BUNDLE=%s\n' "$REQUESTS_CA_BUNDLE"
test -r "$REQUESTS_CA_BUNDLE"
colab --version
colab sessions
```

각 `test`의 종료 코드0과 실제 CLI 출력을 확인한다. `~/.profile`·`~/.bashrc`에 설정이 있어도 현재 셸에 적용됐다고 가정하지 않는다. 인증서 내용을 출력하거나 인증정보를 공유할 필요는 없다. 연결 오류가 남으면 GPU 실행을 시작하지 말고 오류와 현재 CA 경로를 기록한다. 예상 밖 활성 세션이 보이면 소유·사용 상태부터 확인하며 자동으로 종료하지 않는다.

연결 확인 후에도 별도 사용자 승인을 받아 승인된 Config·Revision·실행 범위·Generate 예산·할당 횟수·실패 중단/로그 회수·정리 절차를 확인한 뒤 실행한다. 기존 Validation-only 시도의 승인은 자동 재실행 권한이 아니다. Validation-only는 Generate0회이며 3조건·6 Generate Smoke Test 또는 84 Generate Cross-Case 실행을 포함하지 않는다.

## 연구 검증 상태

CA 연결 성공은 Colab 제어 API 접근 확인이다. 파일/SHA, Production Processor, 모델 로딩, SAE 로딩, CUDA Hook의 Runtime Validation5단계는 모두 미검증 상태를 유지한다. `execution_ready=false`이며 Generation·EOS·계측 동등성도 미검증이다. 이번 문서화 작업의 원격 Colab 명령·GPU 할당·모델/SAE 다운로드·Generate는0회다.
