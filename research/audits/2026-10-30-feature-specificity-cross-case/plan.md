# Phase 5-D-1 — Feature Specificity & Cross-Case Validation

## 상태와 연구 질문

시작 로컬/원격 main=`947ecb130b5b4d16a9465b7aaaad8bb6af7c2d63`, clean. 마지막 완료는5-C-12의 A100 최소 파일럿 감사다. 관련 Journal19~21, 두 Audit, 기존 생성·steering·모델/SAE loader와 데이터·테스트를 확인했다. Growth019의30908+8은 동률/Lure, +8.5는 양의 Margin/Lure, 12207+8도 양의 Margin/Lure였다. 정확한 revision·TF/Actual 차이 원인·실제 HF 동적 선택 경로·specificity는 미검증이다.

이번 산출물은 **CPU 설계/Mock 검증 및 Dry-run 준비 완료**이며 **GPU 실행 준비 완료가 아니다**. 정확한 model/tokenizer/SAE commit 선정과 실제 loader pinning 검증이 해결될 때까지 `execution_ready=false`이며 실행은 로딩/할당 전에 실패한다. 이번에 GPU·Colab 세션·모델/패키지 다운로드는 하지 않았다.

- RQ1:30908 효과가 Growth019 밖의 사전 선정 문항에서도 나타나는가?
- RQ2:30908과12207의 변화량/응답이 구별되는가? 12207은 효과가 관측된 비교 Feature이며 중립군이 아니다.
- RQ3:7/8/8.5에서 Margin과 응답 전이가 어떻게 변하는가? 양의 Margin/전환/단조성을 강제하지 않는다.

## 입력 대조와 후보 적합성

`src/mindscopex_analysis/data/crt_fresh_v2.json`, 기존 `instruct_lure_case`, Protocol B `manifest.json` 및 `full_answer_scores.csv`, `protocol_b_behavioral_join_slots.csv`를 읽어 대조했다. 정확한 prepared Raw Prompt 전체와 Prompt IDs/SHA·후보·과거 Baseline 슬롯은 신규 `inputs_verified.json`에 기록했다. tokenizer를 새로 실행한 확인과 구분한다. 새로 고정할 revision의 tokenizer가 다르면 생성 전에 다시 검증하고 중단한다.

| Case | Correct / Lure (앞 공백 보존) | Correct IDs | Lure IDs | 분기 Correct/Lure | 과거 Baseline 참조 |
|---|---|---|---|---|---|
| Growth019 | ` 55 days` / ` 28 days` |[220,20,20,2756]|[220,17,23,2756]|20/17| seed2,4 Correct |
| Growth004 | ` 25 days` / ` 13 days` |[220,17,20,2756]|[220,16,18,2756]|17/16| seed1 Correct |
| Growth016 | ` 49 days` / ` 25 days` |[220,19,24,2756]|[220,17,20,2756]|19/17| seed1,4 Correct |

Prompt Token SHA:

-019: `145bbd68f373722e8ae38ae183fab16f0ba13c529f315fb824ecf1e68fa45ea2`
-004: `774e80e40a78feee7a6b5b3adb0ba21b71f98bd4c29bbaba09c564bfae836371`
-016: `e9a8a056350b1a89d845a988f34d2c6a7d50d40ae1c6df2fc96b95511c8299a6`

세 후보쌍 모두 각각4tokens·공통 prefix[220]/k=1, 후보 prefix collision/EOS 포함 없음이다. 따라서 기존 token-first-divergence·완성/EOS 생성 구조와 저장 입력이 호환된다. 004/016을 **입력상 호환되는 잠정 대상**으로 선정하나 runtime tokenizer/revision 미검증 상태에서 최종 실행 확정으로 간주하지 않는다. 세 문항은 전부 Growth 유형이고 대표/무작위 CRT 표본이 아니다. 004/016은5-C-9에서 사전 제안됐고 과거 결과가 이미 알려졌으므로 새 held-out 검증이라 부르지 않는다.

세 문항 모두 Lure 분기 ID가 Correct보다 작다. 따라서 문항 확장만으로 token-ID tie 선택의 영향을 제거하지 못하며, tie와 양의 Margin을 계속 구분한다.

TF Baseline First/Full 참조:019=-1.3750000596046448/-1.4600286781787872,004=-1.9999999403953552/-2.7797634452581406,016=-2.5000000298023224/-3.3428827337920666. 이는 과거 Teacher-forced 점수이며 이번 Actual Raw가 아니다. 019 외 Actual Raw Baseline은 이번 준비에서 새로 측정하지 않았다. 과거004의30908+7/+8은 Correct,016은 Lure였으며 이는 입력 Manifest의 역사적 슬롯 참조다. seed를 신규 반복 수로 계산하지 않는다.

## 조건·단위·예산

정식 Manifest=`experiments/inputs/feature_specificity_cross_case.json` (schema1), 결과 schema3, study_id=`feature-specificity-cross-case-v1`. 모델 Qwen/Qwen3.5-2B-Base, SAE Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50, Layer5/all 고정.

각 문항에 공통 무Hook0,30908+7/+8/+8.5,12207+7/+8/+8.5의7조건. block-1과 block-2는 별도 진단 Python 프로세스/모델 로딩을 계획한다. 각 블록21관측·42 generate, 합계 **3×7×2=42관측·84 generate**다. 계측/비계측 한 쌍은 동일 조건 내 동등성 검사다. 과거 seed1/2/4, 계측 쌍은 신규 독립 관측이 아니다. 두 블록은 순서를 동일하게 고정했으므로 순서/동일 VM 영향이 가능하며 독립성·통계적 유의성을 주장하지 않는다.

84는 CPU 잠정 예산이다. 추가 Feature, random direction, 별도 norm forward, 재시도·추가 진단은 포함되지 않는다. 이번 norm observer 자체는 추가 forward/generate를 요구하지 않는다. 모델 로딩/전송/할당·추가 비용의 상한은 아니며 기존234 generate 계획과 별개다. 실패는 결과/로그를 보존하고 중단한다.

## 구현 경로·측정·검증

새 helper `experiments/feature_specificity_cross_case.py`의 build_plan/require_ready/validate_result/compare_blocks가 계획·입력·결과·블록 비교를 담당한다. `--cross-case-block` 및 `[cross_case]` config로만 활성화한다. 기존 legacy CLI 0/7/8와 Growth019 pilot 스키마2는 유지한다. 공통 후보/EOS/동등성 검증은 기존 pilot의 validate_capture로 추출해 재사용한다.

기존 `trace.run_model` 측정 반복문→capture_case→RecordingModel→production `_binary_choice_completion`→`model.generate`를 그대로 사용한다. 새 생성 루프/새 Prompt/tokenization 규칙은 없다. Hook 제거 finally 및0의 Hook 미등록을 유지한다. 기존 Rate027·Behavioral 생성 코드와 과거 설정은 변경하지 않았다. 새출력도 exclusive-create/allow_nan=False이며 실패 JSON·Manifest·로그 보존과 launcher 회수를 유지한다. 신규 입력 protocol manifest도 source archive에 명시적으로 포함한다.

조건별 JSON은 case/feature/α/block ID, 입력 SHA/Prompt/후보 IDs, 실제 전체 생성 IDs·응답/label, 각 step 허용/선택 ID·Raw/Processed 값/dtype, 최초 분기 Margin(Lure−Correct), raw tie, 분기 선택 ID, EOS/후보 유효성, 동등성, Raw/Processed baseline delta를 구분한다. TF/Full은 같은 Feature/계수의 과거 참조가 있을 때만 별도 표시한다. 12207과+8.5에 다른 조건의 TF를 전용하지 않는다. 환경/GPU/dtype·source SHA 및 revision 요청/증거는 기존 metadata와 함께 저장한다.

### Activation Norm

`make_feature_steering_hook(...,diagnostics=records)`의 기존 관측을 계측 호출에만 연결한다. 실제 반환할 edited와 원래 hidden을 같은 개입 지점에서 읽으며 별도 수정/정규화는 하지 않는다. 비계측 쌍은 같은 vector edit을 diagnostics 없이 적용해 Token/응답 동등성을 검사한다.

- unit_decoder_direction_norm: 계수 적용 전 해당 SAE decoder 방향의 L2 norm.
- requested_vector_norm: 계수가 SAE dtype에서 적용된 방향의 norm.
- cast_vector_norm: 실제 residual device/dtype로 변환된 개입 벡터 norm.
- residual_before/after_*: 개입 전후 각 token의 residual L2 norm 통계.
- actual_delta_*: **float(after)−float(before)**의 token별 L2 norm 통계. BF16 등 실제 edit 반올림이 반영된다.
- delta_residual_ratio_*: norm(delta)/norm(before), before>0·finite에서만 정의.

forward_index/affected_token_count 및 비유한/zero 수를 기록한다. prefill과 decode의 token 수가 다르므로 mean들을 가중치 없이 평균해 동일 측정처럼 쓰지 않는다. forward record와 분기 step의 정확한 동적 대응은 추가 검증 사항이다. Baseline은 진짜 무Hook을 유지하여 activation_norms={status:no_hook,records:[]}다. Baseline residual norm은 not measured이며0으로 위조하지 않는다. 같은 α가 같은 actual delta가 아니며 norm 보조 분석만으로 의미적 specificity가 입증되지 않는다.

## Revision pinning — 실행 차단 항목

조사 결과 기존 text loader는 revision이 kwargs로 model에만 도달하고 AutoConfig/AutoProcessor/AutoTokenizer에는 전달되지 않았다. SAE loader에는 revision 인자가 없었다. 이번에 opt-in revision/tokenizer_revision을 config·model·processor/tokenizer까지 전달하고, SAE revision은 hf_hub_download에 전달하도록 보완했다. 인자를 주지 않은 기존 호출에는 revision keyword 자체를 추가하지 않는다. Mock/AST로 전달과 기본 호환성만 검증했다.

Manifest의 model/tokenizer/SAE revision은 현재 모두 null, pinning_validation=unverified다. require_ready는 **40자리 immutable SHA 3개와 실제 검증 완료 상태**를 요구한다. Launcher는 할당 전에, trace는 Torch import 전에 검사한다. CPU Mock 테스트의 준비상태 우회는 테스트 내부에만 있다. `execution_ready` 표시를 바꾸거나 SHA 문자열만 채워 실제 검증 완료를 대체하지 않는다.

해결 절차: 권위 있는 저장소 metadata로 앞으로 사용할 세 정확한 commit을 정하고, 모델 config/weights·processor/tokenizer·SAE가 해당 snapshot에 묶이는지 실제 설치 환경의 loader 경로/로컬 캐시로 검증한다. checkpoint 파일 SHA/캐시 snapshot 및 tokenizer 입력 정합성을 기록한 뒤 Manifest의 증거와 상태를 별도 변경·리뷰한다. 현재 로컬 Torch/HF가 없어 실제 로딩 검증을 하지 않았다. 캐시 힌트로 과거 revision을 소급 확정하지 않는다. 이 blocker 해결과 명시적 GPU 승인 전 실행하지 않는다.

## 추가 대조군 제안 — 실행 목록에 추가하지 않음

9월 기록의 near-zero12207 분류는 현재 파일럿의 양의 Margin과 조건/시점이 달라 중립성 근거로 재사용하지 않는다.10월19~21 Journal에는7438도 행동 변화가 기록돼 있어 역시 중립 Feature라고 가정할 수 없다. 7438은 비교 후보로 검토할 수 있으나 이번 Manifest에는 추가하지 않았다.

추가 Feature는 평가3문항 결과를 보고 고르지 말고, 별도 discovery 데이터의 활성 빈도·decoder norm·동일 layer 등을 사전 정한 bin/seed 규칙으로 매칭해 선정한 뒤 목록을 고정할 것을 제안한다. 그 정보가 미확인인 지금 ID를 임의 확정하지 않는다. 기존 `research.random_direction_margin_deltas`/`random_direction_null_for_feature`는 norm-matched **Teacher-forced ablation** 도구다. Actual generation control로 바로 실행한 것으로 취급하지 않는다. 향후 generator seed·방향 벡터·norm·개수/부호·새 예산을 사전 등록하고 별도 adapter 검증이 필요하다.

## CPU 실행 기록 및 향후 명령

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bash experiments/run_colab.sh \
 experiments/suites/cross_case_specificity_2b.toml --dry-run \
 --session mindscopex-cross-case-specificity --gpu A100
```

두 신규 `results/runs/20261030-152104_cross_case_specificity_block-{1,2}_2b/`에 파일 준비만 완료했다. 실제 세션/원격/모델은 호출하지 않았다. Dry-run archive는 그 시점의 미커밋 소스를 포함한 준비본이다. 실제 승인 후에는 새 경로/Archive로 준비한다.

Plan-only: `python3 experiments/trace_fresh_v2_first_divergence.py --plan --cross-case-block block-1 --output <신규경로.json>` (block-2도 동일). 향후 두 실제 결과 비교는 helper.compare_blocks({'block-1':..., 'block-2':...})로 수행하고 기존 write_new로 신규 JSON에 저장할 수 있다.

아래는 blocker 해결·사용자 GPU 승인 **이후에만** 사용할 명령 형태이며 현재는 revision 오류로 할당 전에 중단된다.

```bash
bash experiments/run_colab.sh experiments/suites/cross_case_specificity_2b.toml \
 --session mindscopex-cross-case-specificity --gpu A100 --exec-timeout 900
```

WSL Bash에서 Jupyter `!git`/`%cd`를 쓰지 않는다.2job×900초 대기 예산은 과금 상한이 아니다.

검증: 기존5개 unittest suite79통과, 신규12개 중 **11통과·1skip**, 총 **90통과·실패/오류0·1skip**. skip은 Torch/HF 미설치로 실제 Tensor Hook norm 검사 미실행이다. Mock/AST는 production helper 경로·21조건/42호출·feature/8.5·무Hook·Hook 제거·EOS/후보/SHA·Margin 산술·결과 스키마·norm 산술·revision 전달/차단·실패 보존/덮어쓰기·두 블록 비교를 검사했다. 실제 GPU Hook 검증 완료로 해석하지 않는다.

기존 `test_launch_colab.py`는 pytest 형식이라 unittest discover가0개를 수집했다. pytest도 미설치였다. 설치하지 않고 runpy로 그3개 무인자 테스트 함수를 직접 실행해 모두 통과했으며 pytest suite 통과로 표현하지 않는다. 실제 models/qwen_scope 전체 Tensor 의존 suite는 미실행이며 관련 loader/hook 함수 AST Mock 및 선택적 Tensor 테스트와 구분한다.

기존 A100 파일럿 두 actual JSON을 리팩터링한 공통 검증으로 메모리 재비교했고 기존 비교본과 완전히 일치했다.

py_compile(변경 Python8파일) 및 git diff --check 통과. 기존 Run/Audit 원본과 Journal1~21절은 해시/바이트 접두로 보존한다. GPU 연구 결과는 새로 생성하지 않았다.
