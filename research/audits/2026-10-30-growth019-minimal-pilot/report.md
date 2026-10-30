# Phase 5-C-11/12 — Growth019 A100 최소 파일럿 검증

## Executive Summary

원본 A100 Artifact 두 블록을 CPU로 검증했다. Growth019에서 Feature30908 α8은 Raw/Processed 동률에서 Lure, α8.5는 양의 Raw/Processed Margin(+0.125)에서 Lure였다. Feature12207 α8도 양의 Raw(+0.625) 및 Lure였다. 두 블록의 모든 조건에서 Margin·생성 Token ID·응답이 일치하고 EOS/계측 동등성이 통과했다. 기존 비교 JSON도 원본으로 재생성한 결과와 구조·값 전체가 일치했다.

이는 한 문항·동일 VM을 재사용한 두 실행 블록의 반복 일치다. Feature30908 고유성, 문항 간 일반화, 통계적 독립성/유의성 또는 HF tie-breaking 원인 확정의 증거로 해석하지 않는다. 이번 감사에서 GPU·Colab 세션 생성·다운로드는 수행하지 않았다.

## 기준 및 실험 조건

감사 시작 로컬 HEAD 및 원격 main은 `17f8c40a4ff538d4d77b6abaea1dea95ea62adcf`, 작업 트리는 clean이었다. 원격은 GitHub ref 조회로 확인했다. 운영 규칙 전체, README 두 개, 최신 Journal18~20절 및 관련 이력, 최소 파일럿 계획·동률 보고서, pilot helper/Manifest·Job·테스트를 확인했다. Journal의 마지막 완료 단계는5-C-10 CPU 준비였고,5-C-11 실행은 아래 원본으로 이번에 검증했다.

모델 `Qwen/Qwen3.5-2B-Base`, SAE `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50`, Case `crt_fresh_v2_growth_019`, Layer5, token_position=all. 각 블록은 공통 무Hook0→30908+7→30908+8→30908+8.5→12207+8 순서다. 두 블록/10관측 및 각 관측의 계측/비계측 비교를 합쳐20 generate다. 과거 seed2/4와 동등성 쌍을 신규 독립 반복으로 합산하지 않는다. 계수0은 기존 intervention 코드에서 Hook 등록 전에 return하는 no-hook 경로이며, Artifact metadata만으로 per-call Hook 실행 trace를 직접 관측했다고 주장하지 않는다.

## 원본 Artifact 및 검증

| 블록 | Run 경로 | 진단 시작~종료(UTC) |
|---|---|---|
| 1 | `results/runs/20261030-145746_pilot_growth019_block-1_2b/` | 05:58:25.525569~05:59:08.821765 |
| 2 | `results/runs/20261030-145912_pilot_growth019_block-2_2b/` | 05:59:45.699208~05:59:57.427069 |

각 Run의 `artifacts/{actual_first_divergence.json,job_manifest.json,plan.json,config.toml,diagnostic.log,job.log}`, 루트 `launcher_manifest.json`, `source.zip`, `artifacts.zip`, config/remote bootstrap 및 `colab_log.{md,jsonl}`을 확인했다. 두 Run 각13파일의 경로·크기·SHA는 신규 `verification.json`에 보존했다. 입력 비교본은 `results/growth019_minimal_pilot_comparison_20261030.json`이다.

- Job status=succeeded, equivalence_validation=passed, 진단 status=captured, 블록/Run ID·5조건 및 순서·Feature/계수·Layer/all·모델/SAE ID가 Manifest와 일치했다.
- 두 ZIP 모두 CRC 검사를 통과했다. artifacts.zip의7개 멤버(숨김 claim 포함)는 회수 디렉터리의 바이트와 동일했다.
- source.zip의 핵심6소스/Manifest 파일은 결과의 source_hashes 및 현 로컬 파일과 동일했다. 과거 Behavioral/TF/Audit 입력3파일도 ZIP 바이트·저장 SHA·현 계획 SHA가 일치했다. 이는 대조한 파일의 일치이지 전체 weights/환경 동일성의 증명이 아니다.
- Prompt Token SHA `145bbd68f373722e8ae38ae183fab16f0ba13c529f315fb824ecf1e68fa45ea2`, P=109가10관측 모두 동일했다. Prompt 문자열 SHA도 같았다. Correct `[220,20,20,2756]`, Lure `[220,17,23,2756]`는 과거 입력과 같았다.
- 공통 접두 `[220]`, k=1; 실제 step1 허용 ID `[17,20]`. 각 실제 prefix에서 후보 trie 허용 set을 별도 계산하고 모든 step의 선택/sequence 정렬과 대조했다. 모든 관측이 분기에 도달했다.
- 5개 생성 step, 후보 완성 후 마지막 EOS248044, candidate completion/behavior valid가 모두 확인됐다. Equivalence compared 및 세 True를 실제 비계측 Token ID/응답과 대조했다.
- 분기 step의 저장 두 raw/processed 값을 직접 빼서 각 저장 Margin을 정확히 재검증했다. 이 독립 산술 검증은 기존 비교기의 출력 복제와 구분된다.

콜랩 로그에는 두 ZIP download와 remote artifact 경로가 있다. 두 Launcher는 같은 session 이름 `mindscopex-growth019-minimal-pilot`, keep_session=false다. 사용자는 동일 VM 재사용 및 세션 종료를 확인했다고 보고했다. 회수 Artifact에는 고유 VM ID/종료 상태의 별도 증명이 없으므로 동일 session 이름만으로 VM identity/종료를 독립 확증하지 않는다.

## 조건별 결과

Margin=Lure−Correct. 아래 Processed 값은 표시만 반올림했으며 JSON에 원래 수치를 저장했다.

| 조건 | Raw B1 | Raw B2 | Processed B1/B2 | 선택 ID B1/B2 | 응답 B1/B2 | Label |
|---|---:|---:|---:|---|---|---|
| baseline | -1.375 | -1.375 | -1.3749999850988388 | 20/20 | 55 days / 55 days | correct |
| f30908-c7 | -0.375 | -0.375 | -0.375 | 20/20 | 55 days / 55 days | correct |
| f30908-c8 | 0 | 0 | 0 | 17/17 | 28 days / 28 days | lure |
| f30908-c8p5 | +0.125 | +0.125 | +0.125 | 17/17 | 28 days / 28 days | lure |
| f12207-c8 | +0.625 | +0.625 | +0.6250000298023224 | 17/17 | 28 days / 28 days | lure |

Correct 전체 생성은 `[220,20,20,2756,248044]`, Lure는 `[220,17,23,2756,248044]`다. baseline 대비 Raw 변화는 +7:+1.0, +8:+1.375, +8.5:+1.5, 12207+8:+2.0이다. Raw/Processed를 동일 측정으로 합치지 않는다. baseline과12207에서 작은 subtraction 차이는 원본 그대로 보존하고 근사 tie로 처리하지 않았다.

30908+8의 Lure/Correct raw는 모두16.375, Processed는 모두-0.6931471824645996이다. 동률에서17이 선택됐으나 Lure Logit 우위가 아니다. 반면 +8.5 raw는 Lure16.125/Correct16.0, processed는 -0.6325989961624146/-0.7575989961624146으로 Lure 우위다. 사후 기대값을 강제한 결과가 아니다.

12207+8 raw는17.875/17.25로 Lure 우위다. 따라서12207은 중립 대조군이 아니며30908의 효과가 고유하다는 주장을 지지하지 않는다. 동일 α는 SAE feature별 기저 활성·방향 norm·개입 activation 변화량의 동등성을 뜻하지 않는다. 실제 activation 변화량을 이번 Artifact에서 별도 측정하지 않았다.

## Teacher-forced 역사적 참조와 Actual

| 조건 | 역사적 TF First | Actual Raw | Actual−TF | 역사적 Full Answer |
|---|---:|---:|---:|---:|
| baseline | -1.3750000596046448 | -1.375 | +0.0000000596046448 | -1.4600286781787872 |
| 30908+7 | -0.5 | -0.375 | +0.125 | +1.573784351348877 |
| 30908+8 | -0.125 | 0 | +0.125 | +0.7332298755645752 |
| 30908+8.5 | unavailable | +0.125 | unavailable | unavailable |
| 12207+8 | unavailable | +0.625 | unavailable | unavailable |

TF/Full은 기존 Protocol B score_reference이지 이번 파일럿에서 새로 실행한 측정이 아니다. 새 계수·Feature에 baseline 입력 참조의 TF 점수를 전용하지 않았다. +7 Full이 양수여도 실제 greedy는 Correct다. TF의 전체 후보 forward/use_cache=False와 실제 generation의 기록된 use_cache=True·길이/정밀도 경로 차이는 이전 보고서의 사실/가설 구분을 유지한다. +0.125 차이의 원인은 여전히 unverified다.

## 실행 환경과 추가로 확보한 metadata

두 블록의 기록된 environment dict와 SAE metadata 전체는 정확히 일치했다.

| 항목 | 확인 결과 |
|---|---|
| 실행 Git | 양쪽17f8c40a4ff538d4d77b6abaea1dea95ea62adcf, git_status 빈 문자열, source=archive |
| Python / Torch / Transformers |3.13.16 / 2.11.0+cu130 / 5.20.2 |
| Tokenizers / Accelerate |0.23.3 / 1.15.0 |
| GPU / CUDA / cuDNN | NVIDIA A100-SXM4-40GB capability[8,0] / 13.0 / 92700 |
| 모델/SAE dtype | requested bfloat16; model/text/parameters·SAE=torch.bfloat16, CUDA |
| 저장 Raw/Processed tensor dtype | **모든 step에서 torch.float32 / torch.float32** |
| 실제 greedy kwargs | do_sample=false, renormalize_logits=true, max_new_tokens=16, EOS=[248044], pad248044 |
| Generation config | 기록 dict 동일, use_cache=true; 실제 cache tensor/forward shape은 미기록 |
| eval/Hook 사전 상태 | model_eval=true, model_training=false, 기존 forward hooks={}, diagnostic_registered_hooks=0 |
| 결정성 설정 | deterministic_algorithms=false, cudnn_deterministic=false; 비트 재현 보증 아님 |
| 커널 | 로그의 causal-conv1d/flash-linear-attention 미설치 reference fallback; 실제 모든 dispatch 미검증 |
| 모델/Tokenizer/SAE exact revision | null/not recorded, exact loaded state unverified |

캐시 config snapshot hint `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`는 weights/tokenizer revision 증거가 아니다. FP32 저장 dtype은 **BF16 모델 forward를 FP32로 재실행했다는 뜻이 아니며** BF16 수치 정보의 복원을 의미하지 않는다. 과거 Run의 미기록 dtype을 이번 값으로 소급 확정하지 않는다.

generation_source_evidence는 설치된 `transformers.generation.utils`의 generate/_sample/_get_logits_processor 소스 위치, SHA `7944240ff1ee27a7b3acc16571b7a480cc14f16e7222aeb1a79d289ebc96e3ea`, 관련 행을 기록한다. `_sample` 발췌에 `next_token_scores = logits_processor(input_ids, next_token_logits)` 및 `next_tokens = torch.argmax(next_token_scores, dim=-1)`가 있다. 이는 설치된 메서드의 소스 introspection 기록이며 그 메서드가 실제 선택 경로였다는 동적 증거 또는 실제 processor 목록·순서의 trace가 아니다. 완전한 HF 소스를 ZIP에서 새로 검증한 것도 아니다. +8 낮은 ID 선택은 argmax 가설과 부합하지만 원인 확정은 하지 않는다.

## 재현 및 CPU 검증

기존 비교기를 원본에 재실행했다. 원래 비교본은 수정하지 않았다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -m experiments.growth019_minimal_pilot \
 --block-1 results/runs/20261030-145746_pilot_growth019_block-1_2b/artifacts/actual_first_divergence.json \
 --block-2 results/runs/20261030-145912_pilot_growth019_block-2_2b/artifacts/actual_first_divergence.json \
 --output research/audits/2026-10-30-growth019-minimal-pilot/recomputed_comparison.json
```

재실행은 위 output을 새 경로로 바꾼다. 신규 `recomputed_comparison.json`은 원본을 입력으로 검증·재계산한 결과다. 신규 `verification.json`은 별도 stdlib 검사의 ZIP CRC/바이트·SHA·trie·분기 산술 및 원본 경로/해시/환경/점수 근거를 담는다. 새 모델 관측이 아니다. 누락/불일치는 발견되지 않았으며 미기록 revision까지 성공으로 승격하지 않는다.

관련5개 suite를 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 환경에서 `python3 -m unittest discover -s tests -p '<pattern>' -v`와 같은 discover 대상 범위로 실행했다(실제 실행은 Python unittest loader/runner에서 패턴별 집계). adapter21, trace35, 기존 비교7, precision6, pilot10: **79통과·실패/오류0·skip0**. 새 production 코드/테스트는 불필요하여 추가하지 않았다. 로컬 Torch/Transformers는 미설치이며 설치하지 않았다. 실제 모델 라이브러리 기반 검사를 새로 수행했다고 쓰지 않는다.

`python3 -m py_compile experiments/jobs/first_divergence_trace.py experiments/growth019_minimal_pilot.py experiments/trace_fresh_v2_first_divergence.py`와 `git diff --check`를 통과했다. 작업 전 스냅샷의 기존 Run/Audit/비교 JSON 및 Journal722파일 중 Journal 이외721파일의 SHA가 그대로이며 Journal1~20절은 바이트 접두 보존을 확인한다. 기존 A10042파일도 이 보존 대상에 포함된다.

## 논문에 사용할 수 있는 제한적 결론 및 다음 단계

“Growth019 단일 문항에서 Layer5의 Feature30908 개입은 α7에서 Correct 응답, α8에서 저장 Raw/Processed 동률과 Lure 응답, α8.5에서 양의 분기 Margin과 Lure 응답을 보였다. 같은 VM을 재사용한 두 실행 블록에서 이 결과가 일치했다. Feature12207 α8에서도 Lure 우위가 관측되어, 효과의 Feature30908 specificity는 입증되지 않았다.”

이 문장을 일반 reasoning 기능의 발견·통계적 유의성·Feature 고유성의 주장으로 확장하지 않는다. 계측 동등성은 동일 조건 내 instrumentation 검증이며 no-hook 대비 효과를 뜻하지 않는다.

다음 우선순위는 정확한 revision/checkpoint 해시 및 실제 forward/cache·processor trace 기록 설계, 사전 선정 추가 문항과 여러 대조 Feature/방향, 활성화 변화 norm을 고려한 대조 조건이다. 필요하면 좁은 계수 구간/실제 FP32 forward 비교를 별도로 설계한다. 고정 순서·동일 VM 영향을 줄이는 block 설계와 사전 성공 기준/예산이 필요하다. 새 GPU 실험과 기존234 generate 확장 계획은 별도 사용자 승인 전 실행하지 않는다.
