# CRT Explicit A/B Choice Decoding Control

이번 작업은 코드와 CPU 테스트만 수행했다. 실제 모델 추론·GPU·Colab 실행, 성능 결과 생성,
Git commit/push는 수행하지 않았다. `mindscopex_inventory.py`와 기존 결과를 수정하지 않았다.

## 목적과 고정 조건

기존 답값 출력 Choice Order 실험의 선택지 순서 효과가 실제 A/B 문자 생성에서도 나타나는지
검증한다. Qwen/Qwen3.5-2B-Base, SAE Layer 5 / Feature 30908, crt_fresh_v2,
Seeds 1~5, Seed당 40개 원본 문항, coefficients [0, -7, -8], token_position all을 유지한다.
기존 split_seed·train_frac 0.6·family-balanced held-out 선정과 모델/SAE 설정을 재사용한다.

각 Seed에서 동일한 문항에 correct_first(A=Correct, B=Lure)와
lure_first(A=Lure, B=Correct)를 적용한다. 생성된 문자는 실제 출력이며 답값에서 파생하지 않는다.

## 조사한 기존 코드와 구현

- `research.py::_binary_choice_completion`은 후보값을 독립 토큰화하고 prefix trie로 허용 토큰을
  제한한다. 기존 답값 모드의 후보 순서·토큰화·EOS 처리 경로를 유지했다.
- 신규 `output_mode="binary_choice_letter"`는 동일 함수에서 후보를 항상 `["A", "B"]`로 고정한다.
  선택지 순서나 의미상 정답 위치에 따라 디코딩 후보 순서를 교환하지 않는다.
- tokenizer 또는 processor 내부 tokenizer로 `encode(..., add_special_tokens=False)`를 사용한다.
  후보는 앞 공백 없는 대문자 A/B이며, 공백 접두사·소문자·설명은 허용하지 않는다. 여러 토큰으로
  인코딩되어도 trie를 그대로 사용한다. 공통 토큰 접두사는 허용하되, 한 후보 토큰열 전체가 다른
  후보의 접두사인 충돌은 명시적으로 거부한다. 빈 토큰열, 동일 토큰열, 후보 내 EOS도 거부한다.
- EOS ID는 단일 값과 목록을 모두 지원한다. 후보가 완성되기 전에는 EOS를 허용하지 않고,
  완성 후에는 EOS만 허용한다. 실제 반환 토큰열의 EOS 종료를 확인하고 EOS를 제거한 뒤 후보와
  정확히 비교한다. 최소 생성 예산은 기존처럼 max(설정 16, 최장 후보 토큰 수 + 1)이다.
  pad_token_id가 없으면 기존처럼 첫 EOS ID를 사용한다. do_sample=False와 renormalize_logits=True를 유지한다.
- `choice_order.py::ordered_case`가 선택지를 렌더링하고 문자 출력 지시문을 추가한다.
  기존 공통 답값·단위 지시문이 있으면 문자 출력 지시문으로 교체하여 상충하는 지시를 피한다.
  문제 본문·선택지 값·원본 case_id는 유지한다. 기존 답값 모드의 프롬프트는 변경하지 않는다.
- 생성용 임시 LureCase에는 순서별 정답/오답 문자를 설정한다. `_generate_labels`에서 생성된 문자를
  이 매핑으로 의미 라벨에 연결한다. 원본 숫자 답을 대상으로 A/B 문자를 분류하지 않는다.
- `steer_generation_labels`의 baseline 및 forward hook 등록·finally 해제 경로를 그대로 재사용한다.
  신규 모드도 동일 hook으로 계수 × SAE decoder 방향을 residual에 더한다. 음수 steering은 ablation이 아니다.
- `research_experiments.py::run_behavioral`은 명시적 choice_orders와 신규 모드가 있을 때 문자 실험을 실행한다.
  모드와 순서가 없는 기존 설정은 기존 동작을 유지한다. 문자 모드에서 빈 디렉터리와 검증된 현재 bootstrap 준비 상태만 허용하며,
  기존 결과·불명 상태 및 generations.json 덮어쓰기를 거부한다.

## 저장 형식과 검증

새 run 이름은 `study_choice_letter_f30908_seed{1..5}_2b`로 기존 답값 run 이름과 다르다.
기존 Colab launcher의 timestamp run 디렉터리 아래 `artifacts/behavioral/generations.json`을 사용한다.

순서 × 계수의 6개 block에 다음을 저장한다.

- output_mode=binary_choice_letter, response_encoding=choice_letter
- coefficient, choice_order
- 원본 숫자 답·Lure와 본문을 포함한 case 정의, case_id별 실제 렌더링 프롬프트
- baseline_rows, steered_rows: 실제 생성 문자(answer), 동일 문자(choice), 의미 label, case_id, family

실행 및 분석에서 40개 고유 문항, 두 순서·세 계수, 잘못된 문자·의미 매핑·유형,
문항 누락·중복, coefficient 0 불일치와 계수 간 baseline 변동을 검증한다.
분석에서는 원본 문항 정의와 프롬프트를 재구성하여 순서/계수 간 동일성을 검증한다.
답값 실험과 비교할 때도 원본 case 정의와 조건별 case_id가 정확히 같아야 한다.

## 추후 Colab A100 실행

아래 명령은 이번 작업에서 실행하지 않았다. 기존 Colab CLI 인증·환경 준비가 필요하다.

```bash
bash experiments/run_colab.sh experiments/suites/crt_choice_letter_control.toml \
  --session mindscopex-choice-letter --gpu A100 --exec-timeout 14400
```

한 Seed만 실행하려면:

```bash
bash experiments/run_colab.sh experiments/configs/study_choice_letter_f30908_seed1_2b.toml \
  --session mindscopex-letter-seed1 --gpu A100 --exec-timeout 14400
```

Seed당 steered 관측은 40 × 2순서 × 3계수 = 240개다. 기존 evaluator가 매 계수에서 baseline을
재생성하므로 baseline 240개를 포함해 480회 생성, 5개 Seed 합계 2,400회 생성이다.
중복 실행을 제외한 baseline 조건은 Seed당 80개다. 이 수는 독립 문항 수가 아니다.

## 실제 결과 회수 후 분석

신규 문자 실험과 기존 답값 Choice Order 실험의 Seed 1~5 결과가 모두 필요하다.
경로가 다르면 `--value-input-root`를 별도로 지정한다. 입력이 없으면 결과를 만들지 않고 오류로 종료한다.

```bash
MPLCONFIGDIR=/tmp/mindscopex-mpl .venv/bin/python scripts/analyze_crt_choice_letter.py \
  --input-root results/runs \
  --value-input-root results/runs \
  --output results/analysis/crt_choice_letter_control
```

출력 디렉터리가 이미 존재하면 중단한다. 입력 경로와 SHA-256을 README에 기록한다.
이번 작업에서는 위 연구 결과 디렉터리를 생성하지 않았다. 합성 결과는 테스트 임시 디렉터리에서만 생성한다.

| 생성 예정 파일 | 내용 |
|---|---|
| seed_order_summary.csv | 문자 실험의 Seed·순서·계수별 baseline/steered 정확도, 전환, 교정·회귀율, 실제 A/B 비율 |
| family_order_summary.csv | Difference / Rate / Growth별 위 지표 |
| choice_position_bias.csv | 두 순서 사이 문자·의미 일치, 공통 교정, 순서별 효과와 반전 여부 |
| paired_case_transitions.csv | 문자 실험 내부의 순서별 문항 대응 |
| encoding_case_comparison.csv | 두 출력 방식의 동일 문항 baseline/steered 문자·의미·전환과 일치 여부 |
| encoding_summary.csv | Seed·순서·계수·유형별 출력 방식 간 일치율, 정확도 차이와 net pp 차이 |
| seed_effect_variability.csv | Seed별 개선·악화·동일 개수, 평균·표본 표준편차·최소·최대 pp |
| encoding_net_improvement.png | 출력 방식·선택지 순서·계수별 Seed 평균 pp 비교 |
| README.md | 입력 해시·검증·정의·실제 결과 표·한계·재실행 명령 |

정확도·A/B 비율·교정률·회귀율·일치율은 0~1 비율이며 변화량은 percentage point다.
교정률=L→C / baseline Lure, 회귀율=C→L / baseline Correct이고 분모 0은 빈 CSV 셀이다.
Net pp=100 × (steered accuracy − baseline accuracy). 전환 합계는 그룹 문항 수와 일치해야 한다.
Seed 표준편차는 ddof=1이며 Seed가 하나면 정의되지 않는다.

두 출력 방식 비교는 seed/coefficient/choice_order/case_id로 대응한다.
문자 일치율은 실제 A/B와 답값에서 파생한 A/B의 위치 일치이며,
의미 일치율은 Correct/Lure 일치다. 같은 순서 내에서 유효한 이진 매핑을 사용하므로
두 일치율은 같을 수 있지만 컬럼과 정의를 구분한다. Baseline 차이와 steered 차이를 따로
기록하고, net 변화량 차이도 별도로 기록한다.

## CPU 검증

```bash
MPLCONFIGDIR=/tmp/mindscopex-mpl .venv/bin/python -m unittest discover -s tests -v
```

전체 488개 실행, 실패 0, skip 4. 신규 테스트 11개와 기존 회귀 테스트를 포함한다.
`git diff --check` 통과. 실제 모델·토크나이저 체크포인트를 로딩하지 않는다.

CPU 스텁으로 단일/다중 토큰 후보, 공통 접두사, 접두사 충돌, 빈/동일 토큰열, EOS 후보 충돌,
다중 EOS ID, 누락된 EOS, 잘못된 생성 토큰, 고정 후보 순서, do_sample=False,
정상 A/B 라벨 매핑과 실패 시 hook 해제를 확인했다. 합성 fixture로 정확도·A/B 비율·교정·회귀·pp,
문항별 출력 방식 대응, 불일치 검출, 파일 생성과 덮어쓰기 방지를 검증했다.
Seed별 40문항과 모델·SAE·데이터 설정은 기존 답값 실험과 동일함을 확인했다.

## 변경 파일

수정:

- src/mindscopex_analysis/research.py
- src/mindscopex_analysis/choice_order.py
- experiments/jobs/research_experiments.py
- scripts/analyze_crt_choice_order.py (기존 기본값을 유지하는 공통 검증 확장)

신규:

- experiments/configs/study_choice_letter_f30908_seed1_2b.toml
- experiments/configs/study_choice_letter_f30908_seed2_2b.toml
- experiments/configs/study_choice_letter_f30908_seed3_2b.toml
- experiments/configs/study_choice_letter_f30908_seed4_2b.toml
- experiments/configs/study_choice_letter_f30908_seed5_2b.toml
- experiments/suites/crt_choice_letter_control.toml
- scripts/analyze_crt_choice_letter.py
- tests/test_crt_choice_letter.py
- docs/analysis/README_crt_choice_letter_control.md

## 비교 가능 범위와 한계

동일 Seed의 원본 문항·계수·선택지 순서·모델·SAE·개입을 대응하여 선택과 효과 차이를
기술적으로 비교할 수 있다. 다만 **출력 지시문과 디코딩 후보가 함께 변경**되므로 두 실험 간
차이를 순수한 토큰 출력 형식 효과로 단정하지 않는다. 실제 토크나이저 및 GPU 생성 동작은
추후 Colab 실행에서 확인해야 하며 CPU 스텁은 모델 성능이나 실제 토큰화를 검증하지 않는다.

제약된 A/B 정확도만으로 추론 능력 개선을 입증할 수 없다. Seed 간 중복 문항은 독립 관측이
아닐 수 있고 5개 Seed만으로 일반적 인과 효과를 확정할 수 없다. 모든 계수의 개선과 악화를
함께 보고하며, 실제 실행 전에는 성능이나 재현 여부를 추정하지 않는다.

다음 단계는 실제 두 실험의 결과를 회수하여 의미 일치율·순서 효과 반전·유형별 효과·Seed 변동을
확인하는 것이다. 출력 지시문과 디코딩 효과를 분리하려면 추가 요인 대조 실험을 별도로 설계한다.


## Colab 최초 실행 디렉터리 오류 수정

원인: `run_colab.sh` → `launch_colab.py` → 렌더링된 `remote_bootstrap.py` 순서로 실행된다.
Bootstrap은 원격 run_dir를 먼저 만들고 `run_capturing`이 `job.log`를 쓰기 모드로 연 다음
research_experiments.py를 실행한다. 따라서 기존 run_dir.exists() 검사는 정상 최초 실행도 거부했다.
`collect_artifacts.py`는 소스 ZIP 생성과 내려받은 ZIP 해제만 담당하며 원격 run_dir를 만들지 않는다.

CPU 회귀 테스트를 먼저 추가해 빈 준비 디렉터리에서 동일 FileExistsError를 재현한 후 수정했다.
새 `experiments/runners/letter_run_directory.py`가 다음 정책을 적용한다.

| 상태 | 처리 |
|---|---|
| 미존재 또는 빈 디렉터리 | 허용, 새 실행 상태 생성 |
| 현재 bootstrap의 준비 상태와 job.log | 실행 토큰·설정 SHA-256·run 이름·prepared 상태 검증 후 한 번만 허용 |
| job.log만 있는 과거 실패 디렉터리 | 거부 |
| 결과·manifest·config·알 수 없는 파일이 있는 디렉터리 | 거부 |
| 심볼릭 링크, 잘못된 상태, 이전 실행 토큰, 이미 사용된 준비 상태 | 거부 |
| 기존 ZIP 또는 fallback ZIP | bootstrap에서 로그를 열기 전에 거부 |

Bootstrap은 비어 있는 디렉터리에서만 `letter_run_status.json`을 만들고 실행 토큰을 자식
프로세스 환경으로 전달한다. job은 이를 검증하고 배타적 `.letter_job_claim`을 생성하여 중복
인수를 막는다. 기존 파일을 삭제하지 않는다. 기존 Choice Order·Negative Dose 경로는
이 준비/검증 분기에 진입하지 않는다.

상태 파일은 ZIP에 포함된다. phase는 prepared → running → succeeded/failed이며 returncode를
함께 기록한다. Bootstrap은 정상 종료뿐 아니라 비정상 종료와 프로세스 시작 예외에서도 상태를
기록한 후 로그와 함께 패키징한다. 기존 JOB_FAILED 및 launcher의 비정상 종료 보고도 유지한다.
Bootstrap 자체가 강제 종료되면 prepared/running 상태가 남을 수 있으며 성공으로 간주하지 않는다.
로컬 직접 실행 중 예외도 running 상태가 남을 수 있으므로 역시 자동 재사용하지 않는다.

최종 전체 CPU unittest: 492개 실행, 실패 0, skip 4. `git diff --check` 통과.
새 회귀 테스트는 빈 디렉터리, 현재 로그 인계, 재사용 거부, 과거 로그·결과·불명 상태·링크 보존,
렌더링된 bootstrap의 성공/실패 ZIP과 종료 코드, 기존 두 실험 모드의 패키징 경로를 검증한다.
Bootstrap 테스트의 /content 경로와 프로세스는 임시 디렉터리와 CPU 스텁으로 대체했다.
실제 Colab/GPU/모델 추론은 실행하지 않았다.

수정 파일: `experiments/jobs/research_experiments.py`, `experiments/runners/remote_bootstrap.py`,
`tests/test_crt_choice_letter.py` (기존 거부 테스트를 실제 결과가 있는 디렉터리로 수정), 이 문서.
신규 파일: `experiments/runners/letter_run_directory.py`, `tests/test_letter_run_directory.py`.
기존 작업 중이던 `scripts/analyze_crt_choice_order.py` 변경은 이번 수정에서 건드리지 않았다.

이미 실패한 VM 디렉터리는 자동 삭제하거나 재사용하지 않는다. 새 Colab 세션/깨끗한 원격
출력 위치에서 최신 소스 archive로 Seed 1을 다시 실행한다. 기존 실패 아티팩트는 보존한다.

```bash
bash experiments/run_colab.sh experiments/configs/study_choice_letter_f30908_seed1_2b.toml \
  --session mindscopex-letter-seed1-fixed --gpu A100 --exec-timeout 14400
```
