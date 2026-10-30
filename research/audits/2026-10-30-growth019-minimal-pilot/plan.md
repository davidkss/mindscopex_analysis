# Growth019 최소 재현 파일럿 — CPU 준비

기준: main `938ab48581214d73317d5005d3b18e621950698c` (시작 시 원격 main 동일). Phase 5-C-9의 후속 축소 설계이며 GPU 실행 승인이 아니다. 기존 +8 동률은 Lure Logit 우위가 아니다.

## 범위 및 측정 단위

정식 Manifest: `experiments/inputs/growth019_minimal_pilot.json`.

공통 조건: Qwen/Qwen3.5-2B-Base, Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50, crt_fresh_v2_growth_019, Layer 5, token_position=all. 기존 Raw Prompt/Tokenizer/후보 trie 및 constrained greedy 생성 경로를 재사용한다.

| 블록별 조건 ID | Feature | α | 기존 TF First 참조 |
|---|---:|---:|---:|
| baseline | 없음 (Hook 미등록) | 0 | -1.3750000596046448 |
| f30908-c7 | 30908 | 7 | -0.5 |
| f30908-c8 | 30908 | 8 | -0.125 |
| f30908-c8p5 | 30908 | 8.5 | unavailable |
| f12207-c8 | 12207 | 8 | unavailable |

block-1/block-2 각각 위 순서로 실행한다. 블록마다 별도 Python 진단 프로세스에서 모델을 새로 로딩한다. 같은 Colab VM에서 실행할 수 있으므로 환경의 통계적 독립성을 보장하지 않는다. **5조건 × 2블록 = 10개 조건별 관측, 계측/비계측 각 1회 = 총 20 generate**. 블록별 expected_generate_calls=10. 계측 동등성의 두 호출과 기존 seed2/4는 독립 반복 수가 아니다. 과거 슬롯은 새 파일럿 비교 슬롯 수에 합산하지 않는다.

## 구현 및 실패 조건

기존 `run_model()` 측정 반복문, `capture_case()`, `compare_uninstrumented()`, `RecordingModel`, `_binary_choice_completion` 및 steering hook을 재사용한다. 별도 생성 루프는 없다. 기존 0/7/8 CLI/config의 의미는 유지하고 `--pilot-block`으로만 새 범위를 연다. run_id/measurement_id에 block을 포함한다.

모든 조건은 생성 전과 후 기존 Protocol B Prompt SHA/후보 Token ID와 대조한다. 새 feature/계수에 과거 TF/Full 점수를 전용하지 않는다. 결과는 각 step의 원본 Raw/Processed 값·tensor dtype·허용/선택 Token, 공통 접두·분기 관측, EOS, 최종 응답, Hook 조건 및 equivalence를 저장한다. Raw/Processed Margin은 Lure−Correct다. EOS·후보/trie 제약·분기 관측 또는 동등성 검증 실패, 입력 SHA 불일치, 결과 누락은 실패 Manifest가 된다. 결과 JSON/로그는 보존한다. 기존 경로 재사용·심볼릭 링크·출력 덮어쓰기는 거부한다.

loading_evidence는 모델 config commit 및 tokenizer metadata가 있으면 저장하지만 실제 weights/tokenizer 동일성을 입증하지 않는다. SAE 정확한 revision은 not recorded다. 모델 eval 상태·기존 환경/model config와 새 Raw/Processed tensor dtype을 기록한다. 설치된 generate/_sample/_get_logits_processor의 소스 위치·SHA·관련 행을 가능하면 수집한다. 이는 실제 선택 메서드/processor 실행 순서의 동적 추적이 아니며 unverified로 남긴다. revision/metadata가 없으면 미확인으로 표시한다.

## 실행 전 CPU 검증

다음 5개 suite를 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 환경에서 unittest discover로 실제 실행했다.

| tests 패턴 | 통과 | 실패/오류 | skip |
|---|---:|---:|---:|
| test_first_divergence_trace_job.py |21|0|0|
| test_trace_fresh_v2_first_divergence.py |35|0|0|
| test_compare_growth019_first_divergence.py |7|0|0|
| test_probe_growth019_precision.py |6|0|0|
| test_growth019_minimal_pilot.py |10|0|0|

총 **79통과/0실패/0skip**. 새 테스트는 Mock으로 5조건 파싱·8.5/12207 전달·두 블록·출력 고유성·Baseline 무Hook·기존 호환·실패 보존·EOS/후보/SHA·revision 미확인·원본 불변성을 확인한다. Torch/Transformers가 설치되지 않아 실제 Tensor/HF 내부 선택 경로 검증은 별도 미검증이다(테스트 skip 수와 구분). py_compile 및 git diff --check도 통과했다.

CPU Plan: `python3 experiments/trace_fresh_v2_first_divergence.py --plan --pilot-block block-1 --output <신규경로.json>` (block-2도 동일).

실제 Launcher `--dry-run`은 20261030-145254_pilot_growth019_block-{1,2}_2b 신규 경로에 준비 파일만 만들었다. 모델 import/loading·원격 실행·세션 생성은 수행하지 않았다. 이 Archive는 당시 미커밋 변경을 포함한 준비본이므로 실제 승인 후에는 새 Run/Archive를 생성해야 한다.

## 승인 후에만 실행할 명령

WSL Bash에서 실행하며 Jupyter 전용 `!git`, `%cd`는 사용하지 않는다. 아래 GPU 명령은 이번에 실행하지 않았다.

```bash
bash experiments/run_colab.sh experiments/suites/pilot_growth019_minimal_2b.toml \
  --gpu A100 --session mindscopex-growth019-minimal-pilot --exec-timeout 900
```

2 jobs, 각10회 generate. timeout900초/job은 최대1800초 원격 exec 대기 예산이며 할당/전송/정리 시간과 비용의 상한이 아니다. 기존 로딩 포함 run 시간으로 새 20회 비용을 단순 외삽하지 않는다. 실패 시 기본 suite 중단 정책 유지. 예상 산출물: 각 새 Run의 artifacts 아래 plan.json, config.toml, actual_first_divergence.json, job_manifest.json, diagnostic.log 및 Launcher 환경/실행 로그. 실패도 결과/로그를 회수한다.

향후 회수 후 CPU 비교(경로는 실제 Run 확인 후 지정):

```bash
python3 -m experiments.growth019_minimal_pilot \
  --block-1 <block1-artifacts/actual_first_divergence.json> \
  --block-2 <block2-artifacts/actual_first_divergence.json> \
  --output <새-비교결과.json>
```

## 성공 기준과 해석 한계

각 조건에서 관측/제약/EOS/equivalence가 유효한지 먼저 확인한 뒤 Raw/Processed Margin, TF 참조(존재할 때), 선택 토큰·응답·동률, 블록 간 정확한 값/응답 일치를 보고한다. 블록 간 불일치도 유효한 연구 결과로 보존하며 강제로 성공/일치시키지 않는다. +8.5 양의 Margin과 12207 중립성을 요구하지 않는다. BF16 동률의 FP32 변환은 정밀도를 복원하지 않는다. 실제 설치 버전의 선택 규칙·정확한 revision·실측 dtype·메모리·Lure 전환 재현은 GPU 미실행으로 아직 미검증이다. 2블록 반복은 일반화나 통계적 유의성 증거가 아니다.
