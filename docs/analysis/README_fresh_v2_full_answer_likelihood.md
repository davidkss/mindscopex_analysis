# Growth Full-Answer Likelihood / Token Trajectory

## 실험 범위

Qwen/Qwen3.5-2B-Base, Layer 5, SAE Feature 30908, coefficients 0/+7/+8, token_position=all을 고정한다. 기존 experiments/inputs/fresh_v2_growth_controls_f30908.json의 **cases 33개 전체**를 선택한다. case_ids의 추가 대조군 27개만 선택하는 실험과 구분된다.

Inventory는 64개 평가 슬롯을 보존한다. correct_to_lure는 6문항/14슬롯, correct_to_correct는 20문항/34슬롯, lure_to_lure는 7문항/16슬롯이다. 새 문항이나 feature 탐색은 없다.

## 필수 입력 프로토콜

TOML의 [diagnostic].input_protocol은 필수다. 누락·알 수 없는 값은 모델 실행 전에 거부한다. 직접 진단 CLI와 분석 CLI도 --input-protocol을 명시해야 한다.

| 프로토콜 | Prompt | 후보 토큰화 | 비교 대상 |
| --- | --- | --- | --- |
| diagnostic_compatible (A) | 기존 _prepare_inputs의 chat template, system_prompt="", enable_thinking=False | answer.strip(), add_special_tokens=False | 기존 First-Divergence 진단 |
| behavior_compatible (B) | text_tokenizer(case.prompt, return_tensors="pt"), template 없음 | 원본 case.correct_answer/case.lure_answer의 선행 공백 보존, add_special_tokens=False | research._binary_choice_completion()의 binary_choice 입력 |

A 설정 파일은 diagnose_fresh_v2_growth_full_answer_f30908_2b.toml로 유지한다. B는 diagnose_fresh_v2_growth_full_answer_behavior_f30908_2b.toml이며 run.name도 다르다. Inventory는 변경하지 않는다.

B의 참조 구현은 src/mindscopex_analysis/research.py의 _binary_choice_completion()이다. 현재 함수 시작은 721행이며 후보 구성·토큰화는 741–744행, raw prompt 호출은 751행 부근, 선택 규칙은 allowed_tokens()에 있다. 두 후보의 prefix trie 내에서 greedy 선택하고 완성 후보 뒤에 EOS를 허용한다. 함수 마지막의 answer.strip()은 반환 문자열 처리이며 후보 토큰화 전에 적용되는 것이 아니다.

CPU 테스트는 실제 _binary_choice_completion()을 호출하여 generate()에 전달된 prompt IDs를 캡처하고, 두 후보의 전체 경로와 EOS가 실제 prefix callback에서 허용되는지 확인한다. 33문항 × 두 후보 모두 신규 B 준비 결과와 같은 Token IDs인지 대조한다. 테스트 tokenizer는 선행 공백을 별도 ID로 표현하므로 공백 손실도 검출된다. 실제 Qwen tokenizer/weights를 다운로드해 실행한 테스트로 오해하면 안 된다.

추가로 B의 prepare_case()는 실행에 사용되는 tokenizer로 실제 참조 함수를 CPU 캡처 모델에 통과시켜 Token IDs를 재검증한다. 이 캡처 모델은 forward나 실제 model.generate를 실행하지 않는다. 참조 코드가 향후 변경되어 prompt·후보·선택 규칙이 달라지면 호환성 오류로 중단한다.

두 프로토콜의 margin은 **서로 다른 입력 조건**이다. 같은 조건의 반복 측정으로 합치지 않는다. CSV 행과 manifest의 프로토콜이 섞이거나 분석 CLI의 요청과 다르면 거부한다. 프로토콜이 없는 구형 결과에는 A를 자동 추정하지 않으며 기존 파일을 수정하지 않는다.

## Prompt / Answer 경계와 점수 정의

1. 기존 instruct_lure_case로 prompt를 구성한다.
2. A는 기존 _prepare_inputs(..., system_prompt="", enable_thinking=False, use_chat_template=True)를 사용한다. B는 행동 실험과 같은 raw tokenizer 호출을 사용한다.
3. A는 answer.strip(), B는 원본 답안 문자열을 add_special_tokens=False로 단독 토큰화한다.
4. 평가 경로는 **준비된 prompt IDs + 단독 후보 IDs**로 고정한다.
5. 두 프로토콜 모두 실제 준비된 prompt IDs를 special token을 보존해 decode한 뒤, 재토큰화가 같은지 확인한다. 이 rendered prompt와 answer를 이어 붙여 토큰화한 결과를 명시적 경로와 비교한다. 경계 merge, 후보 suffix 차이 및 prompt roundtrip 실패를 manifest에 기록한다. 경계 차이를 발견해도 평가 후보를 바꾸지 않는다. roundtrip이 실패하면 joint-text 비교는 확정적 경계 검증이 아니다.

후보마다 prompt + candidate[:-1]를 한 번 forward한다. 위치 prompt_length-1+i의 FP32 logits에 전체 vocabulary log_softmax를 적용하여 후보 token_i를 gather한다.

~~~text
S(candidate) = sum_i log P(token_i | prompt, candidate[:i])
Full Margin = S(lure) - S(correct)
~~~

sampling, generate, 후보 집합 내 softmax 재정규화는 사용하지 않는다. torch.inference_mode, model.eval, use_cache=False를 사용한다. baseline은 hook 없이, +7/+8은 기존 SAE 로딩과 make_feature_steering_hook(..., [30908], coefficient=c, token_position="all")로 평가한다. hook은 실패 시에도 제거한다.

공통 접두의 token IDs와 first divergence 위치는 두 후보가 공유한다. 각 후보는 자기 경로를 별도로 teacher force하므로 분기 이후의 실제 후보 prefix가 다음 토큰의 조건이 된다. EOS 미포함 상태에서 한 후보가 다른 후보의 strict prefix이면 두 token으로 정의되는 branch margin은 비워 두되 full score는 계산한다.

B에는 raw prompt+원본 후보 문자열을 한 번에 토큰화한 결과와 명시적 prompt IDs+후보 IDs의 비교도 별도로 기록한다. 경계 merge가 있어도 행동 생성이 사용하는 명시적 후보 경로를 유지한다.

## EOS 정책과 해석

두 프로토콜 모두 기본 include_eos=false: 후보 문자 토큰만 계산한다. 점수는 답안 prefix의 likelihood이며 “이 답안으로 종료할 확률”이 아니다.

EOS 포함 실험은 **새 run.name**으로 TOML을 복사하고 include_eos=true로 설정한다. tokenizer EOS가 여러 개이면 [diagnostic].eos_token_id를 명시해야 한다. 설정한 ID는 tokenizer EOS 집합에 속해야 한다. 단일 EOS는 자동 선택한다.

EOS 포함 시 scored token length는 EOS를 포함하고 answer token length는 제외한다. 평균 token log probability의 분모는 scored token length다. EOS를 예측하는 조건에는 전체 답안 토큰이 포함된다. 두 정책은 manifest/CSV에 기록되며 혼합 분석은 거부한다.

Full score 합계와 token 평균을 함께 저장한다. 길이가 다른 후보에서는 합계와 길이 정규화 점수가 다른 질문에 답한다. likelihood 우위가 greedy/constrained 생성 label이나 CRT 추론 개선을 직접 의미하지 않는다.

## Colab CLI

~~~bash
bash experiments/run_colab.sh \
  experiments/configs/diagnose_fresh_v2_growth_full_answer_f30908_2b.toml --dry-run

bash experiments/run_colab.sh \
  experiments/configs/diagnose_fresh_v2_growth_full_answer_f30908_2b.toml
~~~

B 설정은 다음 명령을 사용한다.

~~~bash
bash experiments/run_colab.sh \
  experiments/configs/diagnose_fresh_v2_growth_full_answer_behavior_f30908_2b.toml --dry-run

bash experiments/run_colab.sh \
  experiments/configs/diagnose_fresh_v2_growth_full_answer_behavior_f30908_2b.toml
~~~

A100, source archive 업로드와 ZIP 다운로드는 기존 CLI를 사용한다. Inventory JSON은 experiments/inputs/에 있으므로 results/ archive 제외와 무관하게 전달된다. TOML의 SHA-256과 실제 Inventory를 대조한다. 신규 Job은 원격/로컬 run directory 및 ZIP 덮어쓰기를 거부한다.

기본 예상 산출물:

~~~text
results/runs/<timestamp>_diagnose_fresh_v2_growth_full_answer_f30908_2b/
  artifacts.zip
  artifacts/
    job.log
    config.toml
    job_manifest.json
    likelihood/
      full_answer_scores.csv
      token_trajectory.csv
      manifest.json
      inventory.json
  colab_log.md
  colab_log.jsonl
~~~

B run 폴더는 diagnose_fresh_v2_growth_full_answer_behavior_f30908_2b 이름을 사용하며 A와 결과를 공유하지 않는다.

Manifest schema 2에는 input_protocol, prompt Token IDs 및 SHA-256, 후보 Token IDs 및 SHA-256, EOS 제외 answer-only ID 해시, EOS 정책, 모델 ID·로딩 설정, SAE repo/layer/feature/계수/위치, 참조 코드 해시를 기록한다. 해시는 UTF-8 JSON 정수 배열(separators=(',', ':'))의 SHA-256이다. 분석 시 CSV와 manifest의 실제 ID 배열·해시를 대조한다.

scores는 33×3=99행, 후보 경로는 33×3×2=198개다. trajectory 행 수는 실제 후보 토큰 수와 EOS 정책에 따라 달라진다. token_index와 first_divergence_position은 1-based다. trajectory는 token ID/text, logprob, cumulative logprob, shared_prefix, is_eos를 기록한다. manifest에는 prompt IDs와 경계 감사, Inventory 및 CSV SHA-256, 성공/실패 상태를 기록한다.

## CPU 분석

다운로드 후 <timestamp>를 실제 폴더명으로 바꾼다. 아래 branch-margin 경로는 구현 시 실제 존재를 확인한 추가 27개 Growth 결과다. 최초 6개는 Inventory에 저장된 측정값을 재사용한다.

~~~bash
.venv/bin/python scripts/analyze_fresh_v2_full_answer_likelihood.py \
  --input-protocol diagnostic_compatible \
  --run-dir results/runs/<timestamp>_diagnose_fresh_v2_growth_full_answer_f30908_2b/artifacts/likelihood \
  --branch-margin results/runs/20261023-154413_diagnose_fresh_v2_growth_controls_f30908_2b/artifacts/diagnostics/fresh_v2_growth_controls_f30908_branch_margin.csv \
  --output-dir results/analysis/full_answer_<timestamp> \
  --bootstrap 2000 --seed 0
~~~

B 분석에서는 --input-protocol behavior_compatible, B run 경로, 별도의 --output-dir를 지정한다. 같은 출력 파일을 두 프로토콜이 덮어쓰는 것은 금지한다.

--branch-margin은 여러 번 지정할 수 있다. 생략하면 기존 6문항에 대해서만 이전 branch 수치와 비교하며 나머지를 미측정으로 명시한다. 중복 입력은 같은 수치/branch metadata인지 검사한다. B에서는 기존 A 방식의 branch margin을 참고 열로만 남기고 cross_protocol_not_comparable을 표시한다. 토큰 ID가 우연히 같더라도 동일 조건으로 간주하거나 수치 동일성 검사를 하지 않는다. 새 full-answer 실행 자체가 모든 문항의 trajectory 기반 branch margin도 계산하므로 full score와 branch-only 수치의 차이를 직접 볼 수 있다.

출력:

- fresh_v2_growth_full_answer_behavior_join.csv: 64슬롯, baseline/+7/+8 누적 점수·평균·길이·Full Margin과 변화, 기존 branch margin·토큰 호환 여부·수치 오차.
- fresh_v2_growth_full_answer_summary.json: 그룹별 고유 문항 통계, 슬롯 가중 기술통계, Seed별 슬롯 기술통계, 경계 감사 집계, branch coverage 및 Case ID bootstrap.

분석 전에 99개 case/coefficient와 전체 token trajectory를 검증한다. 중복/누락, 잘못된 token index, 후보 길이·EOS·공통 접두 불일치, 합계/누적/평균/차이값 오류, 비유한 수치, CSV 해시 불일치, 기존 출력 덮어쓰기를 거부한다.

## 기존 First-Divergence와의 차이

기존 margin은 동일한 prompt+공통 접두에서 두 분기 token의 **logit 차이**를 측정한다. Full Margin은 후보별 모든 token의 **조건부 logprob 합계 차이**이며 후보 경로와 길이에 의존한다.

공통 context에서 두 logprob의 차이는 logit 차이와 수학적으로 같다. 다만 각 전체 후보의 별도 forward에서 추출한 branch 값은 길이·kernel·정밀도 등의 수치 차이가 생길 수 있으므로 기존 branch CSV와의 최대 오차를 보고한다. 비교 전에 branch position과 correct/lure token IDs를 확인한다. 토큰 호환 자체가 이전 실행의 prompt나 checkpoint revision까지 증명하지는 않는다.

같은 문항의 margin을 Seed마다 반복 사용하므로 슬롯은 독립 margin 관측이 아니다. 고유 문항 통계와 슬롯 기술통계를 구분한다. Bootstrap은 Case ID 합집합을 공동 재표집하여 그룹 중복을 유지하며, Full Margin +8−+7의 고유 문항 평균과 그룹 간 평균 차이에 대한 탐색적 95% percentile 구간이다. 소표본·선택된 문항·그룹 중복을 고려해야 하고 인과적 검정이 아니다.

## CPU 테스트

~~~bash
MPLCONFIGDIR=/tmp/mindscopex-mpl .venv/bin/python \
  -m unittest discover -s tests -p 'test_full_answer*.py'
~~~

테스트의 모델과 tokenizer는 CPU stub이다. 합성 데이터는 임시 폴더에만 쓰며 연구 결과와 혼합하지 않는다. 구현 단계에서는 실제 GPU 진단 및 실제 full-answer 연구 통계를 생성하지 않는다.
