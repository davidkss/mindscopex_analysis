# Attempt3 B안 최종 승인 및 사용자 WSL 실행 전 점검

2026-11-16 명시적 사용자 승인. 준비 시작 HEAD=`20efe394dee228870335356616053beb0ee9dd32`.
이 문서는 기존 Draft/엄격한 HTTP POST1회 NO-GO 기록을 대체하거나 수정하지 않는다.

## 승인 식별 및 범위

- 최종 JSON: `experiments/inputs/growth019_validation_only_authorization_20261116_attempt3.json`
- 최종 SHA-256: `90b5ad68f4871b69d1e1625e7acc7af9dd19bba8126e055b601ab6f08c86e26e`
- Config: `experiments/configs/growth019_validation_only_20261116_attempt3_2b.toml`
- ID: `validation-only-20261116-attempt3`
- Run ID: `growth019_pinned_validation_only_20261116_attempt3_2b`
- Session: `mindscopex-valonly-20261116-a3`
- 실행 주체: 사용자 회사 WSL. Codex의 실제 원격 명령 실행은 허용하지 않는다.

A100 논리적 할당 최대1회/외부 세션생성 프로세스1회/Launcher 자동 재할당0/CLI 오류루프0이다.
COLAB_CLI_ASSIGN_RETRIES=0/COLAB_CLI_RETRY_BACKOFF=0은 해당 할당 자식에만 강제한다.
401 인증갱신·307/308 Redirect 추가 POST, HTTP POST 총횟수 및 서버 GPU 개수 상한 미보장,
중복 GPU/추가 CU 비용 가능성을 사용자가 명시적으로 수락했다. TLS·인증 기능은 변경하지 않는다.
불확실하거나 실패한 뒤 새 할당을 시작할 권한은 없다.

Files/SHA → Production Processor → Qwen Model → Layer5 SAE → CUDA Synthetic Hook만 허용한다.
Generate0/추가Qwen Model Forward0/SyntheticHook≤2, weights 약5.085GB와 부속파일 다운로드가
승인 범위다. Model/Tokenizer Revision=`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`,
SAE Revision=`132ea3697b591df9ee46d738aa1d528e3c6082f7`.
execution_ready=false는 유지하며 Generation/Cross-Case 승인이나 Runtime 성공을 의미하지 않는다.

## 코드 경계와 남은 한계

Attempt3만 ACTIVE_AUTHORIZATION에 연결한다. 과거 승인3개는 계속 거부한다.
고정 최종 SHA 및 고정 Draft 지문에서 재구성한 전체정책을 엄격히 비교한다.
Attempt3 Draft 자체의 pending 바이트는 최종 SHA와 불일치해 거부되며 기존 Attempt2 CPU Draft는 보존된다.
세션/GPU/실행예산/keep 옵션을 검사하고, 실제 Launcher에서는 Claim 전에 환경 gate를 통과해야 한다.
새 gate는 WSL kernel/WSL_DISTRO_NAME, CLI 경로·shebang, metadata 버전 및 사전검증한
client/auth/HTTP전송 핵심파일 지문을 검사한다. Python import 경로 override와 누락 CA bundle을
거부한다. 이 검사는 로컬 Python metadata/파일만 읽으며 Colab CLI/인증 loader/원격 API를 호출하지 않는다.
설치환경 drift는 자동 업데이트로 고치지 말고 중단한다. 다른 일반실험/원격 Runtime Backend에는
회사 WSL gate를 적용하지 않는다. 이 gate는 환경 확인이며 사람의 신원 증명 장치가 아니다.
CA내용/Proxy경로/계정상태·외부망·모든 의존파일의 무변경까지 보증하지 않는다.

Claim은 repository root의 `results/.authorization_claims/validation-only-20261116-attempt3.json`에
배타 생성하고 실패해도 유지한다. 동일 root의 다른 Run은 재사용할 수 없다.
다른 checkout/VM/root에는 ledger가 공유되지 않으므로 정책파일 복사만으로 전역1회를 보장하지 않는다.
**하나의 사용자 checkout에서만 실행하고, 시도 후 어느 checkout에서도 같은 승인을 다시 사용하지 않는다.**
소비 후에는 별도 후속작업에서 승인을 retire해야 한다. Claim 삭제/정책복구로 재실행하지 않는다.

불확실 실패 뒤 Artifact 회수/finally 종료를 시도한다. stop_command_succeeded는 명령 반환 성공이며
서버종료 증거와 다르다. 로컬 명명세션이 없더라도 서버 assignment가 남을 수 있으므로 사용자
사후 sessions/CU 확인이 필수다. 네트워크 단절·강제종료 때 회수/종료는 실패할 수 있다.

## 사용자 WSL에서 실행 전 확인 (Codex는 실행하지 않음)

1. 이 Commit을 사용자 WSL에서 Push하고 실제 실행checkout의 HEAD/clean 상태를 확인한다.
   오래된 Dry-run source.zip으로 실행하지 않는다. 실제 실행 시 최신 소스 Archive를 생성한다.
2. `command -v colab` 및 `uv tool list`로 `/home/davidkss/.local/bin/colab`,0.7.5를 확인한다.
   환경 gate의 설치지문 대조가 통과해야 한다. 실패하면 할당하지 말고 CPU 검토로 돌아간다.
3. 회사 CA/Proxy환경과 CA파일 접근성을 확인한다. 토큰·쿠키·Proxy비밀번호/환경 전체를 로그에 남기지 않는다.
4. 사용자 WSL에서 `colab sessions`, `colab status -s mindscopex-valonly-20261116-a3`로
   요청 세션 부재·다른 할당 유무를 확인하고 CU잔액/사용률/활성할당도 확인한다.
   해당 세션이 있거나 응답이 불확실하면 중단한다. 자동 교체/재할당은 하지 않는다.
5. `df -h .` 등으로 다운로드·압축·모델로딩에 충분한 디스크/메모리를 확인한다.
   실제 가중치 및 부속파일 합계/압축 복제에 필요한 여유를 고려한다.
6. 최종 JSON SHA와 Config를 확인하고 모든 실행환경에서 승인 미사용인지 확인한다.
   로컬 Claim이 없어도 이전 시도가 있었다면 실행하지 않는다.
7. 이상이 없을 때에만 사용자 본인이 아래 승인된 단일 명령을 실행한다.

```bash
bash experiments/run_colab.sh experiments/configs/growth019_validation_only_20261116_attempt3_2b.toml --session mindscopex-valonly-20261116-a3 --gpu A100 --allocation-attempts 1 --allocation-retry-delay 0 --exec-timeout 900
```

8. 성공/실패에 관계없이 같은 승인으로 재실행하지 않는다. Run의 승인소비/할당원문/회수/정리파일을 보존한다.
9. 사용자 WSL에서 sessions/해당 status/CU잔액·사용률·활성할당을 다시 확인한다.
   종료 실패/불확실 시 기존 정책의 수동정리 안내를 따른다. 청구지연 및 orphan assignment도 고려한다.

## CPU 준비 결과와 판정

관련56개 테스트 및 전체785개 중779통과/6skip. 실제 실행 명령/환경/보존 검증은11월16일 Journal 참조.
신규 환경 gate/정책변조/Config 예산/과거승인 및 Draft 차단, Attempt3 Claim 중복과
503/Service Unavailable/401/307/308/TLS/ReadTimeout/회수실패/stop실패 Mock을 포함한다.
테스트의401/Redirect 예외는 실패경로 검증이며 추가 POST를 차단한 결과가 아니다.

CPU Dry-run=`results/runs/20261116-162143_growth019_pinned_validation_only_20261116_attempt3_2b`.
최종 ID/SHA/Session/A100/고정 Revision/validate/Generate0/추가Forward0/Hook2 확인,
원격호출0/GPU할당0/Claim생성0. ZIP CRC와 현재 실행코드/최종 정책 바이트 일치 확인.
Archive는 로컬 준비물이며 Runtime 결과 Artifact가 아니다.

**B안 CPU 준비 GO — 위 사용자 WSL 사전점검을 통과한 경우에 한함.**
엄격한 HTTP POST1회 정책은 계속 NO-GO다. 서버가용성/실제Runtime5단계는 미검증이며,
GPU실행·모델다운로드는 Codex가 수행하지 않았다. 실제 A100 성공이나 비용 상한을 보장하지 않는다.
