# CRT Negative Dose Multi-Seed Integrated Analysis

신규 분석 스크립트: `scripts/analyze_crt_negative_dose_multiseed.py`

```bash
MPLCONFIGDIR=/tmp/mindscopex-mpl .venv/bin/python scripts/analyze_crt_negative_dose_multiseed.py
MPLCONFIGDIR=/tmp/mindscopex-mpl .venv/bin/python -m unittest discover -s tests -v
```

기본 입력은 `results/runs`, Seed는 1~5이며 `--input-root`, `--output`, `--seeds`,
`--seed1-reference`로 지정할 수 있다. 출력 디렉터리가 존재하면 중단한다.
Seed 1 선택 시 기존 `results/analysis/crt_negative_dose_seed1_final`의 세 CSV가 필요하며,
공개된 정답률과 전체·유형별·문항별 결과를 대조한다. 입력 누락·중복·검증 실패는
출력 생성 전에 오류로 처리한다. 기존 단일 Seed 분석 함수를 재사용한다.

실제 결과 디렉터리는 `results/analysis/crt_negative_dose_multiseed_seed1to5/`이다.
상세 입력 경로, SHA-256, 지표 정의, Seed별·유형별 결과와 한계는 그 안의 README에 기록했다.
기존 `.gitignore` 정책상 결과 디렉터리는 Git 상태에 나타나지 않는다.

| Coefficient | Baseline (%) | Steered (%) | 변화 (pp) |
|---|---:|---:|---:|
| 0 | 54.5 | 54.5 | 0.0 |
| -1 | 54.5 | 49.5 | -5.0 |
| -2 | 54.5 | 28.0 | -26.5 |
| -4 | 54.5 | 6.0 | -48.5 |
| -7 | 54.5 | 61.0 | +6.5 |
| -8 | 54.5 | 59.5 | +5.0 |

5개 Seed × 40문항 = 200 Seed-문항 관측치이며 고유 case_id는 99개,
여러 Seed에 출현하는 case_id는 63개이다. 200개의 독립 문항으로 해석하지 않는다.
고유 문항 분석은 관측 횟수·전환 빈도·Seed별 결과를 보존한다.

- -7 변화: Seed 1~5 순서로 +7.5, +2.5, +12.5, -5.0, +15.0 pp.
  개선 4개, 악화 1개; 평균 +6.5 pp, 표본 표준편차 8.02 pp.
- -8 변화: +10.0, 0.0, +10.0, -5.0, +10.0 pp.
  개선 3개, 동일 1개, 악화 1개; 평균 +5.0 pp, 표본 표준편차 7.07 pp.
- 두 coefficient 모두 Difference는 5개 Seed에서 개선, Growth는 5개에서 악화했다.
  Rate는 4개에서 악화, 1개에서 개선했다. 전체 개선은 유형별로 균일하지 않다.
- Seed 1 기존 CSV의 모든 필드와 공개된 정답률이 일치했다.

산출물은 `seed_summary.csv`, `family_summary.csv`, `coefficient_summary.csv`,
`case_overlap.csv`, `seed_reproducibility.csv`, `unique_case_summary.csv`,
`multiseed_accuracy.png`, `multiseed_correction_rate.png`, `README.md`이다.

검증: 기존·신규 unittest 전체 467개 실행, 실패 0, skip 4.
합성 테스트는 실제 연구 데이터와 분리했다. 정상 검색·집계, 누락·중복 입력,
계수 누락·중복, 문항 불일치, coefficient 0 불일치, baseline 변동, 필수 필드,
문항 수, 전환 합계, 유형별 집계, 중복 문항, 반복 관측과 결과 불일치 보존,
분모 0, pp 계산, 실제 Seed 1 대조 및 덮어쓰기 방지를 확인했다.
Ruff는 로컬 환경에 설치되어 있지 않아 실행하지 못했다.

실험은 binary_choice 조건이며 답 선택 편향과 출력 형식 영향을 배제하지 못한다.
음수 steering은 완전한 feature ablation과 같지 않으며, 정답률 향상만으로 추론 능력
향상을 입증하지 못한다. 5개 Seed 및 중복 문항의 기술 통계로 일반적 인과 효과를
확정할 수 없다. 후속 연구에서는 문항 중복 없는 평가, 선택지 순서 교차 검증,
자유 생성 및 ablation 대조 조건을 별도 실험으로 검토할 수 있다.

기존 코드·결과·TOML·mindscopex_inventory.py를 수정하지 않았고,
모델·Colab·GPU 실행 및 Git commit/push를 수행하지 않았다.
