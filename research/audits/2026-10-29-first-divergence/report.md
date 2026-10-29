# First-Divergence × Behavioral 검증 (2026-10-29)

기존 CSV만 사용한 CPU 분석이다. GPU 실행 및 기존 실험 결과 수정은 수행하지 않았다.

## 발견 및 수정

- 기존 코드는 First-Divergence Margin=0에서 예외로 종료했다. Full-Answer Margin=0은 `correct`로 취급했다. 두 Margin 모두 정확히 0(−0 포함)이면 `tie`로 분류하며, 작은 비영 값은 Tie로 근사하지 않는다.
- Tie는 관련 지표에서만 제외한다. First-Divergence–실제 응답은 First-Divergence Tie를, Full-Answer–실제 응답은 Full-Answer Tie를, 두 선호 간 일치와 대응 차이는 어느 쪽이든 Tie인 슬롯을 제외한다. CSV에서 제외된 지표는 빈 셀, JSON에서 정의 불가 값은 null이다.
- 지표마다 유효 슬롯 수, 제외 슬롯 수, 유효/제외 문항 수를 기록한다. 공통 유효 슬롯에서 `First-Divergence 일치 − Full-Answer 일치` 대응 차이를 추가했다.
- CSV 필수 열/행 형태, 유한 수치, 실제 응답 label, 프로토콜/EOS, score 중복, 문항×계수 완전성, 슬롯 중복 및 계수 간 `(case_id, seed)` 동일성을 검증한다. 계수를 정수로 잘라 +7.9를 +7로 취급하던 경로도 제거했다. Margin 대조는 상대 오차 없이 절대 오차 1e−5를 사용한다.
- 기존 출력은 덮어쓰지 않는다. 두 출력 경로가 같으면 거부하고, exclusive-create 방식으로 생성한다. 쓰기 중 OSError가 발생하면 이번 실행이 생성한 파일을 제거한다. 여러 파일의 생성이 프로세스 강제 종료까지 원자적으로 보장되지는 않는다.

## 중복 구조 및 Bootstrap

Score는 33문항 × 3계수(0/+7/+8)=99행이다. Behavioral은 계수별 64개 `(case_id, seed)` 슬롯으로 두 계수 합계 128행이다. 문항별 슬롯 수는 1개인 문항 14개, 2개 10개, 3개 6개, 4개 3개로, 합계 33문항/64슬롯이다. 두 계수의 슬롯 ID 집합이 동일함을 확인했다.

Slot-weighted는 유효 슬롯의 단순 평균이다. Case-weighted는 문항 안의 유효 슬롯 평균을 먼저 계산한 뒤 유효 문항을 동등하게 가중한다. Bootstrap은 원래 33개 case_id에서 33개를 복원추출하며 반복 추출된 문항의 중복을 유지한다. 같은 추출 목록을 두 계수와 모든 지표에 사용한다. 지표상 모든 슬롯이 Tie인 문항은 해당 추출의 분모에서 제외한다. 전부 제외된 추출은 버리고 유효 추출 수를 기록한다. 따라서 신뢰구간은 이 '33문항 재표집 후 유효 문항만 평균' 규칙에 해당하며, 처음부터 유효 문항 수만큼 재표집하는 방식과는 다를 수 있다.

재표집 20,000회, seed=20261029. 이번 데이터에서는 모든 지표에서 20,000개의 유효 추출을 얻었다. 기존 코드의 '문항별 평균 → 문항 재표집' 자체는 case-weighted 추정에 적합했으나 Tie를 처리하지 못했다. Slot-weighted 신뢰구간은 산출하지 않는다.

## 실제 응답과의 일치율

| 지표 | +7 | +8 |
|---|---:|---:|
| First-Divergence Tie 슬롯 / 문항 | 11 / 4 | 6 / 2 |
| Full-Answer Tie 슬롯 | 0 | 0 |
| First-Divergence 슬롯 일치율 | 53/53 = 100.00% | 56/58 = 96.55% |
| Full-Answer 슬롯 일치율 | 28/64 = 43.75% | 36/64 = 56.25% |
| 두 선호 간 슬롯 일치율 | 19/53 = 35.85% | 28/58 = 48.28% |
| First-Divergence 문항 일치율 [95% CI] | 100.00% [100.00, 100.00] (29문항) | 96.77% [89.66, 100.00] (31문항) |
| Full-Answer 문항 일치율 [95% CI] | 45.45% [27.27, 63.64] (33문항) | 57.58% [39.39, 72.73] (33문항) |

같은 비Tie 슬롯에서 비교하면 Full-Answer 일치율은 +7에서 19/53=35.85%, +8에서 30/58=51.72%다. 대응 일치율 차이(First-Divergence − Full-Answer)는 슬롯 가중 +64.15%p/+44.83%p, 문항 가중 **+58.62%p [40.00, 76.67] / +41.94%p [21.88, 61.29]**다. 각 지표의 개별 전체 표본 평균을 빼는 대신 공통 유효 표본의 대응 차이로 계산했다.

First-Divergence는 이 표본의 비Tie 응답에서 높은 일치를 보인다. +7의 [100, 100] 구간은 관측된 유효 문항이 모두 일치해서 생긴 경험적 Bootstrap 결과이며 모집단에서 완벽한 일치를 보장하지 않는다. Tie 문항은 포함되지 않고, 이 진단값은 teacher-forced 후보 경로에서 계산했으므로 실제 generate() logits를 직접 관측한 인과 증거가 아니다. 기존 Full-Answer 일치율 28/64 및 36/64는 재현되었다. 수치를 연구 결론에 맞춰 조정하지 않았다.

## 테스트 및 재현

추가 테스트 10개 통과. 정확한 0/−0/작은 비영 값, Tie 분모, 반복 case draw, 문항 내 평균, 전부 Tie인 지표/추출, 실제 CSV 수치와 재현성, Full-Answer Tie, 입력 이상 16종, 계수 간 슬롯 불일치, CSV 형식/누락 파일, CLI 옵션, 덮어쓰기 방지, 두 번째 출력 실패 시 첫 번째 출력 정리를 검사했다. 이 실행에서는 관련 CPU 단위 테스트만 수행했으며 전체 저장소 테스트는 수행하지 않았다.

```bash
python3 -m unittest discover -s tests -p 'test_analyze_first_divergence_behavior.py' -v
python3 scripts/analyze_first_divergence_behavior.py \
  --output /tmp/mindscopex-fd-new/summary.json \
  --details /tmp/mindscopex-fd-new/slots.csv
```

출력은 이 보고서와 같은 디렉터리의 `summary.json`, `slots.csv`에 신규 저장했다. 재실행 시 비어 있는 출력 경로를 사용해야 한다.

입력 SHA-256:

- `results/analysis/protocol_b_behavioral_join_slots.csv`: `5b6d8d1ba7c36e932540800097627c0fbf65d3e22e6fd49b49cdb6abcc9a035d`
- `results/analysis/protocol_b_20261029/full_answer_scores.csv`: `56cef3b86da88109c747b608f51a331e09b2d9ad71c316d44ebe0348b6306459`
