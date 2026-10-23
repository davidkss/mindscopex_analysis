# Fresh v2 Growth 전체 대조군 margin

## 확인된 입력과 Inventory

- 행동 CSV: `results/analysis/fresh_v2_f30908_cplus7_vs_cplus8.csv`
- 기존 margin: `results/runs/20261023-152801_diagnose_fresh_v2_matched_branch_margin_f30908_2b/artifacts/diagnostics/fresh_v2_f30908_matched_branch_margin.csv`
- 행동 컬럼: seed, case_id, family, steered_7, steered_8, correct_answer, lure_answer 등.
- margin 컬럼: case_id, family, correct_answer, lure_answer, baseline_margin, plus7_margin, plus8_margin, branch/token 정보 및 차이값.

| 전이 | 평가 슬롯 | 고유 문항 |
| --- | ---: | ---: |
| correct_to_lure | 14 | 6 |
| correct_to_correct | 34 | 20 |
| lure_to_lure | 16 | 7 |
| 합집합 | 64 | 33 |

기존 측정 6개를 보존하고 27개를 추가 측정한다. 현재 그룹 간 중복 문항은 없지만 코드는 Seed 간 반복 및 그룹 간 중복을 허용한다.

생성한 결과:

- `results/analysis/fresh_v2_growth_control_inventory.csv`: 문항별 그룹, Seed, 슬롯 수, 추가 측정 여부.
- `results/analysis/fresh_v2_growth_control_inventory_summary.json`: 그룹 집계 및 입력 경로·컬럼·SHA-256.
- `experiments/inputs/fresh_v2_growth_controls_f30908.json`: 위 Inventory의 원격 전송용 JSON. 33문항, 64슬롯, 신규 27개 ID, 기존 6개 측정값과 출처를 포함한다.

`results/`는 source archive에서 제외된다. 따라서 원격 Job은 `results/`에 의존하지 않고 이 JSON을 읽는다. TOML에 JSON의 SHA-256이 고정되어 있어 누락·변조 시 모델 실행 전에 실패한다. 실제 dry-run source.zip에서 JSON 포함 및 해시 일치를 검증했다.

Inventory 재생성 예시(기존 출력은 덮어쓰지 않으므로 새 경로 사용):

```bash
.venv/bin/python scripts/analyze_fresh_v2_growth_controls.py inventory \
  --behavior results/analysis/fresh_v2_f30908_cplus7_vs_cplus8.csv \
  --existing-margin results/runs/20261023-152801_diagnose_fresh_v2_matched_branch_margin_f30908_2b/artifacts/diagnostics/fresh_v2_f30908_matched_branch_margin.csv \
  --output-dir /tmp/mindscopex-growth-inventory-review \
  --selection /tmp/mindscopex-growth-inventory-review/inventory.json
```

## Colab 실행

```bash
bash experiments/run_colab.sh \
  experiments/configs/diagnose_fresh_v2_growth_controls_f30908_2b.toml --dry-run

bash experiments/run_colab.sh \
  experiments/configs/diagnose_fresh_v2_growth_controls_f30908_2b.toml
```

Model Qwen/Qwen3.5-2B-Base, Layer 5, Feature 30908, coefficients 0/+7/+8, A100을 사용한다. 기존 진단의 instruct_lure_case, chat template 입력 준비, enable_thinking=False, first-branch 토큰화 및 token_position=all 개입은 유지한다. 이는 기존 진단과 같은 조건이며 행동 생성 전체 경로와의 완전한 동등성을 새로 입증하지는 않는다.

신규 CSV와 로그는 다음에 다운로드된다.

```text
results/runs/<timestamp>_diagnose_fresh_v2_growth_controls_f30908_2b/
  artifacts.zip
  artifacts/job.log
  artifacts/diagnostics/fresh_v2_growth_controls_f30908_branch_margin.csv
  artifacts/diagnostics/config.toml
  artifacts/diagnostics/inventory.json
  artifacts/diagnostics/manifest.json
  colab_log.md
  colab_log.jsonl
```

27문항 × 3계수 = 81개 margin을 계산하며, 기존 6문항과 Rate는 재측정하지 않는다. 동일 원격 run 폴더·ZIP 또는 기존 로컬 결과가 있으면 덮어쓰기를 거부한다.

## 다운로드 후 전체 결합

다음 명령의 `<timestamp>`를 실제 다운로드 폴더명으로 바꾼다.

```bash
.venv/bin/python scripts/analyze_fresh_v2_growth_controls.py compare \
  --inventory experiments/inputs/fresh_v2_growth_controls_f30908.json \
  --new-margin results/runs/<timestamp>_diagnose_fresh_v2_growth_controls_f30908_2b/artifacts/diagnostics/fresh_v2_growth_controls_f30908_branch_margin.csv \
  --bootstrap 2000 --seed 0
```

출력:

- `results/analysis/fresh_v2_growth_controls_behavior_margin_join.csv`: 전체 Growth 슬롯과 margin, 출처, 그룹 및 부호 변화.
- `results/analysis/fresh_v2_growth_controls_behavior_margin_summary.json`: 그룹별 고유 문항/슬롯 통계, margin 평균·중앙값·사분위수·전체 값, 증감 비율, 부호 변화, bootstrap 구간.

신규 27개 중 하나라도 빠지거나 중복·예상 밖 ID·비유한 수치·답안 불일치가 있으면 실패한다. 기존 문항이 신규 CSV에도 있으면 값과 branch/token 메타데이터 일치를 검사하고 기존 값을 유지한다. 정상 실행은 중복 재측정을 하지 않으므로 6개 기존 문항의 실행 배치 간 수치 일관성을 직접 검증한 것으로 표현하지 않는다.

## 통계와 한계

고유 문항 통계는 각 그룹의 문항을 한 번씩 센다. 슬롯 통계는 Seed에 따른 반복 횟수로 가중된 기술통계이며 독립 관측 추론에 사용하지 않는다.

Bootstrap은 Case ID 합집합을 2,000회 공동 재표집하며 각 문항의 그룹 소속을 유지한다. 그룹별 고유 문항 평균 delta와 그룹 간 평균 delta 차이에 대한 탐색적 95% percentile 구간을 계산한다. 그룹에 표본이 없는 반복은 제외하고 유효 반복 수를 기록한다. 그룹 중복·소표본·문항 선택 편향을 고려해야 하며 이 구간은 인과적 검정이 아니다.

Margin은 Logit(Lure) − Logit(Correct)의 조건부 first-divergence 측정이다. 전체 답변 우도나 추론 향상 자체를 의미하지 않는다. 신규 GPU 결과가 아직 없으므로 전체 그룹 margin 통계 및 CI는 이번 구현 작업에서 연구 결과로 생성하지 않았다.

## CPU 검증

```bash
for pattern in test_growth_controls.py test_matched_branch_margin_job.py \
  test_fresh_v2_growth_behavior_margin.py test_letter_run_directory.py; do
  MPLCONFIGDIR=/tmp/mindscopex-mpl .venv/bin/python \
    -m unittest discover -s tests -p "$pattern" || break
done
```

합성 margin을 사용하는 테스트는 임시 폴더에만 결과를 쓰며 연구 결과와 혼합하지 않는다.
