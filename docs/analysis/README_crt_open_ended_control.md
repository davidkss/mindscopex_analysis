# Open-ended CRT Steering Control

이번 작업은 코드 구현과 CPU 테스트만 수행했다. 모델 다운로드·실제 추론·GPU·Colab 실행,
연구 결과 생성, Git commit/push는 수행하지 않았다. 기존 결과와 mindscopex_inventory.py는 변경하지 않았다.

## 기존 경로 확인과 재사용

- `research.py::_greedy_completion`은 기존 `free` 모드의 Base 모델 텍스트 완성 경로다.
  동일 함수를 재사용하며 `return_details=True`일 때만 새 모드의 원문·토큰·종료 정보를 반환한다.
  기존 free·binary_choice·binary_choice_letter의 기본 동작은 유지한다.
- `research.py::steer_generation_labels`의 baseline 생성, SAE decoder 방향 forward hook,
  finally hook 해제, 모델/SAE 로딩을 재사용한다. 별도의 모델 추론 구현은 추가하지 않았다.
- `research_experiments.py::_load_splits`와 `family_balanced_subset`을 재사용한다.
  새 설정은 기존 Choice Letter와 같은 모델·데이터·feature 설정을 사용하며 Seed별 40개
  LureCase 전체 정의가 동일함을 CPU 회귀 테스트로 확인한다.
- 기존 `choice_order.py`를 호출하지 않고 원본 LureCase.prompt를 그대로 사용한다.
  원본 질문 뒤 `Answer:`와 기존 답값·단위 출력 지시문을 유지하고 A/B나 후보값을 추가하지 않는다.
  문제 본문에 계산에 필요한 수치가 포함되는 것은 정상이며 이를 삭제하지 않는다.
- 기존 `generation.py::classify_lure_answer`는 답의 부분 문자열 언급만으로 일치할 수 있다.
  새 점수기는 기존 Unicode/공백 정규화를 재사용하되, 전체 답변의 단일 수치·단위 형태를
  검증한 후 Decimal 값과 단위를 비교한다. 기존 분류기는 변경하지 않는다.

## 실험 설정과 생성

모드: `open_ended_crt`. Qwen/Qwen3.5-2B-Base, SAE layer 5, feature 30908,
crt_fresh_v2, Seeds 1~5, coefficients [0, -7, -8], token_position all.
기본 Seed 1~5 설정에서는 max_new_tokens=16, instruction=true, do_sample=False를 유지한다.
별도 nothink64 Seed 1 검증 설정은 아래 수정 절을 따른다.
기존과 같은 train_frac=.6과 split_seed, 유형별 round-robin 선정으로 40문항을 사용한다.

이 모드는 A/B 선택지나 정답/Lure 후보를 프롬프트에 추가하지 않으며, prefix_allowed_tokens_fn,
후보 trie, lexical constraints를 사용하지 않는다. 모델은 후보 밖의 답도 자유롭게 생성할 수 있다.
EOS ID는 tokenizer에서 얻어 명시적으로 전달하고 강제 BOS/EOS 토큰은 비활성화한다.
따라서 토큰 예산 끝에 강제로 붙은 EOS를 정상 종료로 오인하지 않는다.
기본 경로는 tokenizer 또는 processor의 내부 tokenizer를 사용하며 chat template을 적용하지 않는다.
thinking_mode=disabled 검증 경로는 아래와 같이 checkpoint의 공식 template을 사용한다.

각 응답에 다음을 저장한다.

- 원본 case_id, family, 실제 prompt
- answer: special token을 제외한 디코딩 텍스트 (공백을 임의로 제거하지 않음)
- raw_text: special token을 포함한 원본 디코딩 텍스트
- generated_token_ids/count, max_new_tokens, eos_token_ids, eos_reached
- termination_reason: eos / max_new_tokens / other_stop
- label, classification: 정규화된 수치·차원·단위, 참조 답, 판정 근거와 scorer_version

실제 토큰 ID도 보존하므로 디코딩 문자열에서 보이지 않는 토큰을 확인할 수 있다.
EOS가 없으면 조기 종료와 예산 소진을 구분하고 보수적으로 invalid 처리한다.
모든 조건에 같은 예산·지시문·파서가 적용된다.

Seed당 40 × 3 = 120개 steered 관측과 기존 evaluator 방식의 baseline 재생성 120개,
총 240회 생성이다. Seed 1~5 총 1,200회 생성이며 독립 문항 1,200개라는 뜻이 아니다.
Coefficient 0의 steered 응답은 baseline과 완전히 같아야 한다.

## 수치 분류 규칙

| 분류 | 규칙 |
|---|---|
| correct | 완성된 단일 수치 답변이 단위 환산 후 정답과 정확히 동일 |
| lure | 동일 방식으로 Lure와 정확히 동일 |
| other | 유효한 단일 수치·인식 가능한 단위이지만 두 참조와 불일치 |
| invalid | 빈 답, 복수 답, 설명·식·미지원 단위·판정 불가 형식 또는 미완성 생성 |

- NFKC, 공백·대소문자 정규화와 Decimal 기반 정확 비교를 사용한다.
- `$6`, `6.00 USD`, `600 cents`, `600¢`는 같은 금액으로 처리한다.
- 초·분·시간·일·주의 정확한 환산과 단수/복수, `19th day`·`day 19`를 지원한다.
- 단위 없는 숫자는 원본 질문이 요청한 단위(참조 정답 단위)로 해석한다. 이 가정은 판정 기록에 남는다.
- `Answer:` 또는 `The answer is` 뒤 하나의 답, 마지막 마침표를 허용한다.
- 여러 숫자·설명·Markdown·과학 표기·철자 숫자·미지원 단위는 추측해서 정답 처리하지 않는다.
- 정답과 Lure가 수치·단위 환산 후 같으면 모델 생성 전에 오류로 중단한다.

예컨대 `not $6`, `$6 or $12`, 계산 과정이 붙은 답은 정답 언급이 있어도 invalid다.
모든 invalid 및 other 응답의 원문을 보존하므로 점수기 오류를 별도로 검토할 수 있다.
엄격한 문법은 의미상 맞지만 형식이 다른 답을 놓칠 수 있어 수동 audit이 필요하다.

## 안전성과 provenance

기존 Letter용 디렉터리 인계 모듈에 모드별 상태 파일·환경 변수·배타적 claim 이름을 추가했다.
Letter의 기본 동작은 유지한다. Open-ended는 `open_ended_run_status.json`을 사용한다.
미존재/빈 디렉터리 또는 현재 bootstrap의 검증된 prepared 상태만 허용한다.
기존 결과·로그만 남은 디렉터리·불명확한 파일·심볼릭 링크·기존 ZIP은 자동 덮어쓰지 않는다.

Bootstrap은 새 모드에도 기존 성공/실패 패키징을 적용한다. 상태는
prepared → running → succeeded/failed이고 종료 코드와 함께 ZIP에 들어간다.
강제 종료되거나 로컬 직접 실행에서 예외가 나면 running이 남을 수 있으며 성공으로 해석하지 않는다.

실행 디렉터리에 config.toml, manifest.json (모델/SAE/Seed/Git commit/실행 환경),
behavioral/generations.json, behavioral/provenance.json을 저장한다.
Provenance에는 결과 SHA-256, 설정 SHA-256, 모델 ID, Seed, layer/feature, Git commit을 기록한다.
Launcher의 소스 ZIP과 launcher_manifest도 기존 구조로 보존된다.

분석은 성공 상태·완료 manifest, 결과와 설정 해시, 실험 식별 정보, 공통 토큰 예산,
세 계수, 40개 문항, 원본 정의, prompt, baseline 일관성, coefficient 0,
종료 정보와 저장된 분류의 재계산 결과를 검증한다. 실패/미완성/불일치는 중단한다.

## Seed 1 Colab A100 실행 명령

아래 명령은 이번 작업에서 실행하지 않았다. 깨끗한 새 원격 세션에서 최신 소스로 실행한다.
기존 실패 디렉터리나 결과는 보존한다.

```bash
bash experiments/run_colab.sh experiments/configs/study_open_ended_crt_f30908_seed1_2b.toml \
  --session mindscopex-open-ended-seed1 --gpu A100 --exec-timeout 14400
```

전체 Seed 실행:

```bash
bash experiments/run_colab.sh experiments/suites/crt_open_ended_control.toml \
  --session mindscopex-open-ended --gpu A100 --exec-timeout 14400
```

## 실제 결과 회수 후 통합 분석

Open-ended와 기존 두 Choice Order 모드의 Seed 1~5 결과가 모두 있는 경우:

```bash
.venv/bin/python scripts/analyze_crt_open_ended.py \
  --input-root results/runs \
  --objective-input-root results/runs \
  --output results/analysis/crt_open_ended_control
```

주관식만 분석하려면 `--objective-input-root`를 생략한다. 비교를 요청했는데 객관식 입력이 없거나
중복이면 중단하며 해당 Seed를 조용히 제외하지 않는다. `--seeds`로 일부 Seed만 명시할 수도 있다.
기존 출력 디렉터리가 존재하면 중단한다. 이번 작업에서는 연구 결과 디렉터리를 생성하지 않았다.

| 산출물 | 내용 |
|---|---|
| seed_summary.csv | Seed·계수별 4개 범주 수/비율, pp 변화, 16개 전환 |
| family_summary.csv | Difference/Rate/Growth별 같은 지표 |
| coefficient_summary.csv | 전체 Seed-문항 관측치 기준 계수별 집계 |
| seed_variability.csv | 정확도·Lure pp 변화의 Seed 평균·표본 SD·최소·최대·효과 방향별 Seed 수 |
| paired_case_transitions.csv | 모든 문항별 baseline/steered 응답·원문·판정 이유·종료 사유·전환 |
| case_overlap.csv | 고유 case_id별 Seed 출현 현황 |
| objective_case_comparison.csv | 비교 요청 시 두 객관식 모드와의 문항별 의미 라벨 대응 |
| objective_comparison_summary.csv | 비교 요청 시 Seed·계수·순서·유형별 일치율과 정확도·pp 차이 |
| README.md | 입력 해시·검증·방법·실제 결과·중복·한계·재실행 명령 |

모든 비율은 invalid를 포함한 전체 그룹 문항 수를 분모로 사용한다. Invalid를 제외하여 정답률을
부풀리지 않는다. 비율은 [0,1], 변화는 percentage point다. 4×4 전환 전체를 저장하므로
Other/Invalid로의 이동과 해당 범주에서의 회복도 보존한다. Net 정답 변화는 L→C−C→L만으로
계산하지 않는다. Seed SD는 ddof=1이며 Seed가 하나면 정의되지 않아 빈 CSV 셀로 기록한다.

객관식 비교는 seed/coefficient/choice_order/case_id로 대응하며 원본 case 정의도 비교한다.
주관식 관측을 두 선택지 순서와 각각 연결하지만 이를 새로운 독립 관측으로 세지 않는다.
Seed 간 중복 문항도 독립 표본으로 취급하지 않는다.

## 검증과 변경 파일

```bash
MPLCONFIGDIR=/tmp/mindscopex-mpl .venv/bin/python -m unittest discover -s tests -v
git diff --check
```

CPU 스텁과 임시 합성 fixture로 비제약 생성, 원문/종료 저장, 4개 분류·단위 정규화,
문항 집합·baseline·계수·분류 불일치 검출, 16개 전환, Seed 중복,
실제 job lifecycle과 provenance, 해시 검증·덮어쓰기 방지,
5개 Seed 통합 및 두 객관식 모드 비교, bootstrap 성공/실패 ZIP을 검증한다.
기존 회귀 테스트도 함께 실행했다. 초기 구현 검증은 전체 unittest 501개 실행, 실패 0, skip 4였다.
아래 Thinking 수정 후 최종 검증은 507개 실행, 실패 0, skip 4다.
`git diff --check` 통과. 합성 데이터는 연구 결과로 저장하지 않는다.

신규: `src/mindscopex_analysis/open_ended_crt.py`, `scripts/analyze_crt_open_ended.py`,
`tests/test_crt_open_ended.py`, 이 문서, `study_open_ended_crt_f30908_seed{1..5}_2b.toml`,
`experiments/suites/crt_open_ended_control.toml`.

수정: `src/mindscopex_analysis/research.py`, `experiments/jobs/research_experiments.py`,
`experiments/runners/letter_run_directory.py`, `experiments/runners/remote_bootstrap.py`.
작업 시작 시 있던 scripts/analyze_crt_choice_order.py 변경과 tests/test_letter_run_directory.py는 건드리지 않았다.

## 연구 해석 한계

객관식과 비교할 때 선택지 제거·디코딩 제약 제거·출력 지시 조건의 차이가 함께 존재한다.
차이를 순수한 reasoning 또는 출력 형식 효과로 단정하지 않는다. 단위 생략 해석과 엄격한
문법 분류의 오류 가능성, 짧은 토큰 예산의 truncation, 5개 Seed와 중복 문항을 함께 보고한다.
음수 steering은 완전한 ablation과 다르다. 실제 결과가 나오기 전 정확도나 효과를 추정하지 않는다.
다음 단계는 실제 실험 회수 후 invalid/other 원문과 분류기 오류를 먼저 audit하고,
유형별 생성 변화·객관식과의 불일치를 확인하는 것이다.


## Seed 1 Thinking / 64-token 검증 수정

공식 자료 확인일: 2026-10-22.

- [Base 공식 tokenizer 설정](https://huggingface.co/Qwen/Qwen3.5-2B-Base/blob/main/tokenizer_config.json)
  에 포함된 chat_template은 enable_thinking 분기를 지원한다. False이면 assistant 입력 끝에
  빈 think 블록을 닫은 접두사를 렌더링한다. 단순히 generate에 같은 이름의 인자를 넣는 방식이 아니다.
- [Base 모델 카드](https://huggingface.co/Qwen/Qwen3.5-2B-Base/blob/main/README.md)는
  사전학습 모델이며 직접 대화보다는 연구·개발 용도를 설명한다. 공식 템플릿의 제어 입력 지원은
  Base가 반드시 비사고 동작을 따른다는 보장이 아니다.

현재 로더는 Qwen3.5에서 AutoProcessor를 반환하고 기존 open-ended 경로는 내부 text tokenizer로
원본 문자열을 바로 인코딩했다. 공식 chat_template을 적용하지 않았으므로 Thinking 비활성화용
접두사가 입력에 없었다. 보고된 16토큰 truncation과 강한 steering 반복을 scoring 오류로 보아
정답 처리하지 않는다. 기존 분류 규칙과 미완성 응답 invalid 처리는 유지한다.

새 옵션은 open_ended_crt 전용 `thinking_mode="disabled"`다. 옵션 생략 시 기존 default raw
completion을 유지한다. 기존 5개 16-token 설정은 변경하지 않았다. 새 검증 설정은
`experiments/configs/study_open_ended_crt_nothink64_f30908_seed1_2b.toml`이며
run 이름도 `study_open_ended_crt_nothink64_f30908_seed1_2b`로 구분한다.

Disabled 경로는 로딩된 checkpoint 자체의 template을 찾아
apply_chat_template(tokenize=False, add_generation_prompt=True, enable_thinking=False)를 사용한다.
True 렌더링과 비교해 분기가 실제 동작하고 False 렌더링이 공식 비사고용 접두사로 끝나는지 검증한다.
다음 접두사는 모델이 생성한 답변이 아니라 입력의 일부다.

```text
<|im_start|>assistant
<think>

</think>

```

이어 내부 tokenizer로 add_special_tokens=False 인코딩하고, 실제 input_ids의 디코딩 결과에도
접두사가 남는지 확인한다. Template 부재·무시된 옵션·손실된 토큰화는 생성 전에 오류로 중단한다.
별도 template을 임의로 만들거나 /no_think를 덧붙이는 대체 경로는 없다.
정답·Lure 후보와 선택지는 추가하지 않으며 후보 제약 decoder도 사용하지 않는다.

원본 prompt와 렌더링된 입력을 따로 보존한다. prompt_rendering에는 요청 thinking_mode,
enable_thinking, template 적용/검증 여부, template SHA-256, tokenizer 이름, template 소유 객체,
실제 rendered_prompt, input_token_ids, input_decoded 및 입력 토큰 SHA-256을 저장한다.
generation_settings에는 실제 전달한 max_new_tokens, do_sample, EOS 목록, pad,
forced BOS/EOS 설정을 기록한다. baseline/steered의 rendering과 생성 설정이 다르면 검증 실패다.
계수 간 baseline 일관성과 coefficient 0 동일성도 기존대로 검증한다.

새 chat 경로에서는 tokenizer EOS에 등록된 `<|im_end|>` ID도 추가한다. Base tokenizer의 EOS는
endoftext이므로 assistant turn 종료를 놓쳐 불필요하게 계속 생성하지 않도록 한다.
추가 토큰은 단일 토큰 인코딩 및 역디코딩을 확인하며 실제 ID와 stop_token_policy를 기록한다.
기존 raw·free·객관식 모드의 EOS 설정은 변경하지 않는다.

이 변경은 입력을 chat 형태로 렌더링하고 생성 예산도 16→64로 늘린 별도 검증 조건이다.
기존 16-token 결과와 차이가 나더라도 Thinking 설정만의 효과라고 단정할 수 없다.
특히 강한 steering의 반복 현상은 Thinking 옵션으로 해결된다고 가정하지 않는다.

### 반복 및 비정상 응답 진단

분류와 독립적인 diagnostics 필드를 추가했다. 생성된 원문·토큰만 검사하며 입력에 넣은 빈
think 블록은 thinking_tags 집계에 포함하지 않는다. 진단은 버전이 있는 기술적 heuristic이다.

- thinking_tags: 생성 원문에 think 여닫기 태그가 있음
- symbol_heavy: 공백 제외 8자 이상이고 비문자·비숫자 비율이 60% 이상
- repeated_character: 공백 제외 같은 문자가 8회 이상 연속
- repeated_token: EOS를 제외한 같은 토큰 ID가 8회 이상 연속
- repeated_phrase: 1~8개 단어의 같은 구절이 3회 연속 (구절 2자 이상)
- abnormal_response: 위 진단 중 하나 이상

이 플래그는 서로 겹칠 수 있으며 의미적 정오답 판정을 대체하지 않는다.
정상적인 답도 heuristic에 걸릴 수 있으므로 원문과 함께 해석한다.
분석의 response_diagnostics_summary.csv는 Seed·계수·유형·baseline/steered별 진단 수/비율과
invalid 및 non-EOS 개수를 기록한다. 문항별 CSV에도 각 플래그와 진단 상세를 보존한다.
이전 결과에 diagnostics가 없으면 저장된 원문·토큰에서 계산할 수 있다.

### Seed 1 재실행 및 분석 명령 (아직 실행하지 않음)

```bash
bash experiments/run_colab.sh \
  experiments/configs/study_open_ended_crt_nothink64_f30908_seed1_2b.toml \
  --session mindscopex-open-ended-nothink64-seed1 --gpu A100 --exec-timeout 14400
```

새 결과를 회수한 다음 전용 run prefix와 새 출력 경로를 지정한다.

```bash
.venv/bin/python scripts/analyze_crt_open_ended.py \
  --input-root results/runs --seeds 1 \
  --run-prefix study_open_ended_crt_nothink64 \
  --output results/analysis/crt_open_ended_nothink64_seed1
```

분석은 요청한 prefix만 검색하며 기존 16-token 결과와 섞지 않는다. config/result/provenance의
thinking_mode와 Seed 간 공통 budget/mode를 검증한다. 원하면 기존 객관식 비교 옵션을 함께 지정한다.
기존 결과를 삭제하거나 재사용하지 않으며 새 출력 디렉터리도 존재하면 중단한다.

수정: research.py, open_ended_crt.py, research_experiments.py,
scripts/analyze_crt_open_ended.py, tests/test_crt_open_ended.py, 이 문서.
신규: nothink64 Seed 1 TOML 한 개. 작업 시작 시 있던 다른 파일의 변경은 건드리지 않았다.

최종 CPU unittest 507개 실행, 실패 0, skip 4. git diff --check 통과.
실제 모델·tokenizer checkpoint는 로딩하지 않고, Jinja 템플릿 분기와 CPU tokenizer/generator
스텁으로 입력 반영, processor 래퍼, 미지원/무시된 옵션, 토큰화 손실, 동일 조건, chat EOS,
64-token truncation, 진단 집계, 새 설정→job→저장→분석까지 검증했다.
실제 모델의 비사고 행동과 반복 완화 여부는 검증하지 않았으며 Colab/GPU/모델 추론은 실행하지 않았다.
기존 결과와 mindscopex_inventory.py를 변경하지 않았고 commit/push도 수행하지 않았다.
