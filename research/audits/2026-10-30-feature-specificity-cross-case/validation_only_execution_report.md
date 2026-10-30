# Phase 5-D-3B — Approved Validation-only 실행 기록

## 범위와 사전 상태

시작 HEAD는 `e51be0c7f8990a446be81fdcaadefa58821fbb2f`이며 시작 main 작업 트리는 clean이었다. Workflow Rules 전체, Journal 25절과 관련 이전 기록, Smoke Test Plan, runtime_smoke/Adapter/Launcher를 확인했다. 이전 단계는 CPU 준비·cached Tokenizer 3문항 검증까지 완료했고 Production Processor/모델/SAE/CUDA Hook은 미검증이었다.

사용자는 A100 신규 할당 최대1회, pinned 파일 다운로드·실제 로딩과 합성 Identity Hook Forward2회를 승인했다. Qwen generate/forward와 6 Generate Smoke Test는 승인 범위 밖이다. 별도 `growth019_validation_only_2b.toml`과 명시적 승인 정책 JSON을 도입했다. 기존 6 Generate Config 및 Cross-Case Manifest는 수정하지 않았다. 두 기존 allocation/readiness 차단 상태를 유지한다.

## 코드와 안전 경계

- runtime_smoke: validate 전용 승인 식별자와 정책 SHA 확인, 생성 조건0/Generate0, generate·model.forward 실행 방지, 단계별 UTC/소요시간·메모리·캐시/파일 크기 기록. 모델의 실제 Layer5 Forward는 수행하지 않는다.
- job adapter: 전용 승인 식별자 전달, validate 결과와 생성 결과 구분. Equivalence는 not_run이며 검증 성공을 생성 성공으로 간주하지 않는다.
- Launcher: 전용 경로에서 A100/할당 시도1/자동 정리 강제, 기존 세션 재사용 금지, 상태 불명 시 실패. 새 할당 명령은 최대1회이며 원격 실행 자동 재시도와 전송 재시도를 금지한다. 기존 작업 경로는 유지한다.

## CPU 검증과 Dry-run

기존125개+신규8개 **133통과·실패0·오류0·1skip (총134개)**. skip은 Torch 미설치로 실제 Tensor Hook 미검증이다. CPU Tokenizer는 기존 분리 환경의 실제 라이브러리로 실행했다. Mock으로 실제 CUDA 동작이 검증되었다고 해석하지 않는다.

명령: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /tmp/mindscopex-5d3a-tokenizer-env/bin/python -m unittest discover -s tests -p '<pattern>' -v`. 실행한 패턴은 test_first_divergence_trace_job, test_trace_fresh_v2_first_divergence, test_compare_growth019_first_divergence, test_probe_growth019_precision, test_growth019_minimal_pilot, test_feature_specificity_cross_case, test_revision_pinning, test_tokenizer_runtime, test_runtime_smoke, test_validation_only_authorization의 각 `.py`다. Launcher 기존 assertion3개도 별도 통과했다. 변경 Python py_compile 및 git diff --check 통과.

신규8개는 승인 예산·Revision, Generate 차단, 잘못된 GPU/재시도/keep 거부, 할당1회 및 상태 불명 차단, model 실행 방지, 단계별 계측, adapter 전달·덮어쓰기 거부, 실패 단계 중단·부분 Artifact 보존을 확인한다.

실제 CPU Dry-run:
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m experiments.runners.launch_colab experiments/configs/growth019_validation_only_2b.toml --dry-run --session mindscopex-valonly-20261030-approved --gpu A100 --allocation-attempts 1 --allocation-retry-delay 0 --exec-timeout 900
```
최종 Dry-run `results/runs/20261030-170636_growth019_pinned_validation_only_2b/`: 생성 조건0, Generate0, 추가 모델 Forward0, 예정 합성 Hook Forward2, execution_ready=false. 실제 원격 호출0회.

## 승인된 실제 실행 시도와 실패

다음 Launcher를 **1회** 실행했다. 이 명령은 이번 실패 기록이며 재실행 지시가 아니다.
```bash
bash experiments/run_colab.sh experiments/configs/growth019_validation_only_2b.toml --session mindscopex-valonly-20261030-approved --gpu A100 --allocation-attempts 1 --allocation-retry-delay 0 --exec-timeout 900
```

신규 Run: `results/runs/20261030-170836_growth019_pinned_validation_only_2b/`, Launcher 준비 시각 `2026-10-30T08:08:36.536650+00:00`.

Colab CLI0.7.4의 최초 `colab status`가 `SSLCertVerificationError: self-signed certificate in certificate chain`으로 실패했다. Launcher는 세션 부재가 확인되지 않았으므로 **할당 전에 중단**했다. `colab new`0회, 원격 upload/exec0회, 실제 Generate0회, 실제 Qwen Forward0회, 실제 합성 Hook Forward0회다. HTTP 라이브러리의 `Max retries exceeded`는 TLS 요청 내부 메시지이며 Launcher의 GPU 재할당/실험 재시도는 없었다. 인증서 검증을 끄거나 환경을 변경해서 재시도하지 않았다.

|단계|실제 결과|
|---|---|
|1 파일 다운로드/SHA|not reached|
|2 Production Processor/3문항|not reached|
|3 Qwen 모델 로딩|not reached|
|4 Layer5 SAE 로딩|not reached|
|5 CUDA Identity Hook2회|not reached|

Model/Tokenizer 요청 SHA는 `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`, SAE 요청 SHA는 `132ea3697b591df9ee46d738aa1d528e3c6082f7`이다. **실제 런타임 로딩 SHA 검증은 수행되지 않았다.** 원격 CUDA/Torch/Transformers, 메모리 사용량·단계별 소요시간·캐시 상태·다운로드량은 not recorded다. 이번 경로로 모델/SAE 다운로드는 시작되지 않았다. 과거 Artifact의 revision 미기록 사실도 유지한다.

## 로그·Archive·정리

위 신규 Run의 `launcher_execution.log`, `final_session_status.log`, `launcher_manifest.json`, `config.toml`, `remote_bootstrap.py`, `source.zip`을 보존했다. source.zip CRC와 현재 미커밋 runtime_smoke/승인 정책 포함을 검증했다. Launcher Manifest는 실행 Git HEAD와 미커밋 작업 트리를 기록한다. **source.zip은 로컬 소스 묶음이며 결과 Artifact ZIP이 아니다.** 원격 작업 미실행으로 stage_1~5, runtime_validation/status, job_manifest 및 결과 ZIP은 생성되지 않았다. 이를 성공/부분 런타임 검증으로 대체 기록하지 않았다. 각 보존 파일 SHA와 실제 호출 수는 `validation_only_execution_evidence.json`에 기록했다.

Launcher finally의 `colab stop`1회는 `Session not found`를 반환했다. 마지막 읽기 전용 `colab status`도 TLS 오류로 실패했다. 신규 할당 명령이 호출되지 않은 것은 확인되지만 서버 측 전체 세션 상태는 unverified이며 종료 성공으로 주장하지 않는다. 자동 재실행/재할당은 수행하지 않았다.

## 남은 차단 요인

먼저 Colab CLI의 신뢰 CA/TLS 연결 문제를 별도 해결해야 한다. 그 후 **사용자 새 승인 없이 이번 시도를 재개하지 않는다.** Production Processor·실제 pinned Model/SAE 로딩·CUDA Hook과 파일 무결성은 전부 미검증이다. Generation/EOS/계측 동등성/실제 Layer5 dispatch도 미검증이며 6 Generate는 별도 승인 대상이다. Cross-Case execution_ready=false와 기존 미검증 상태를 유지한다.
