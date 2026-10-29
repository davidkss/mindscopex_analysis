# MindScopeX — Thinking ON/OFF 비교 실험 준비 및 실행 게이트

**작성:** 2026-10-29  
**상태:** 기존 코드·문서 정적 검토 완료. Thinking ON 지원 구현/CPU 테스트 및 GPU 실행은 미완료.

## 1. 기존 코드에서 확인한 사항

- `src/mindscopex_analysis/research.py::_binary_choice_completion`은 Base 모델에서 Correct/Lure 두 후보의 prefix-constrained greedy decoding을 사용한다. `enable_thinking`을 모델의 generate 인자로 전달하지 않으며 ON/OFF 실험이 아니다.
- `src/mindscopex_analysis/research.py::_greedy_completion`은 `thinking_mode`를 인자로 받는다. `open_ended_crt`의 `disabled` 경로는 모델 checkpoint의 공식 Chat Template에 `enable_thinking=False`를 전달하여 비사고용 입력 prefix를 구성한다.
- `src/mindscopex_analysis/open_ended_crt.py::evaluate`는 현재 `thinking_mode in ('default', 'disabled')`만 허용한다. **`default`는 명시적 Thinking ON이 아니다.**
- `docs/analysis/README_crt_open_ended_control.md`에 Thinking OFF/64-token Seed 1 실행 설정과 명령이 이미 기록되어 있다.
- 기존 Protocol A/B의 Likelihood 결과는 ON/OFF 비교가 아니며, 개입 계수 0/+7/+8의 비교이다.

## 2. 비교 실험의 연구 질문

동일한 CRT 문항과 Qwen3.5-2B-Base/SAE 조건에서, 공식 Chat Template로 구성한 **명시적 Thinking ON과 OFF**의 생성 답변·정답률·Lure 비율·무효 응답 비율은 어떻게 다른가? 그리고 Layer 5 Feature 30908 개입 효과는 두 모드에서 달라지는가?

**주의:** 공식 Chat Template 지원과 실제 모델의 Thinking 출력 준수는 다른 문제다. Base 모델이 Thinking ON을 안정적으로 생성한다고 가정하지 않는다.

## 3. 권장 통제 설계

| 변수 | 통제/실험 |
|---|---|
| 모델/SAE | 동일 체크포인트 revision, Layer 5, Feature 30908 |
| 문항 | 동일 Fresh v2 문항 ID, 동일 원문 및 평가 분류기 |
| Thinking | `enabled` 대 `disabled` (둘 다 공식 Chat Template 사용) |
| 토큰 예산 | 동일 `max_new_tokens` (예: 64, 필요 시 별도 128-token 민감도 분석) |
| 디코딩 | `do_sample=False`, 동일 EOS/pad 정책 |
| 개입 | 동일 coefficient 집합, 최소 0 및 사전 등록한 한 개의 개입값 |
| 개입 위치 | `token_position=all` 고정 |
| 출력 | 자유 생성 `open_ended_crt`; 정답/Lure 후보를 프롬프트에 추가하지 않음 |
| 분석 단위 | 고유 문항 ID; Seed 반복은 문항 단위로 군집화 |

**단계별 진행:**

1. `open_ended_crt.py`와 `research.py`의 Thinking ON 경로를 구현하기 전에 현재 tokenizer의 `apply_chat_template(enable_thinking=True)` 출력이 OFF와 구분되는지 CPU 검증한다. 실제 토큰 ID/렌더링/템플릿 해시를 기록한다.
2. ON 모드를 구현한다면 `evaluate`의 허용값, 출력 provenance, 분석 스크립트의 검증 조건과 테스트를 함께 변경한다. 기존 `default`의 의미는 변경하지 않는다.
3. 먼저 Seed 1 소규모 smoke 테스트를 수행하여 Thinking 태그, 종료 원인, 64-token truncation, 반복·비정상 응답을 검토한다.
4. Smoke가 통과하면 동일 문항의 ON/OFF 대응 실험을 실행한다. 생성 텍스트·토큰·설정·manifest를 보존한다.
5. 정확도·Lure·Other·Invalid 비율과 응답 전이 및 개입 상호작용을 비교한다. 문항 단위 bootstrap을 사용하며, ON/OFF만의 효과라고 단정하지 않는다.

## 4. 현재 사용 가능한 OFF 경로

기존 문서에 기록된 검증 명령(실행 여부는 확인되지 않음):

```bash
bash experiments/run_colab.sh \
  experiments/configs/study_open_ended_crt_nothink64_f30908_seed1_2b.toml \
  --session mindscopex-open-ended-nothink64-seed1 --gpu A100 --exec-timeout 14400
```

이 명령은 OFF 단독 실험이며 ON/OFF 비교를 완성하지 않는다. 새로운 실행 전에 최신 소스/설정과 동일한 실행 디렉터리 존재 여부를 확인한다.

## 5. 현재 결론과 완료 기준

- **완료:** Thinking ON/OFF를 기존 Protocol A/B 결과로 오인하지 않도록 코드 경로와 현재 지원 범위를 확인.
- **미완료:** Thinking ON 구현, 공식 template 검증, CPU 회귀 테스트, 동일 조건 GPU 실험, 대응 통계 검증.
- **실험 성공 기준:** ON/OFF 입력 차이가 명확히 기록되고, 다른 실험 조건이 통제되며, 결과의 실제 생성 원문과 종료 이유가 모두 보존되어야 한다.
