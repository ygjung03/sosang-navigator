# 소상공인 나침반 (sosang-navigator)

서울 소상공인에게 **정책 조합**을 추천한다. 정책 목록을 나열하는 기존 서비스와 달리,
36개월 현금흐름 시뮬레이션을 공통 통화로 삼아 융자·보조금·채무조정을 함께 계산한다.

- **대회**: 2026 금융 AI Challenge (금융보안원) · **마감 9월 7일**
- **개발**: 하늘(ygjung03) 1인 · private repo
- **우선순위**: 수상 순위보다 **완성**. 신용보증기금·기술보증기금·기업은행·산업은행
  AI/디지털 직군 포트폴리오로도 쓴다.

---

## 무엇을 만드는가

```
POS 파일 업로드
  ↓
[코드] 국면 판정          정상 / 일시적자금난 / 개선여지 / 회복불가
  ↓
[코드] 자격 필터          국면·연체·매출·신용·업력 등으로 후보를 거른다
  ↓
[코드] 조합 생성          융자 1 · 채무조정 1 · 보조금 2
  ↓
[코드] 36개월 시뮬레이션   조합마다 현금흐름을 계산한다
  ↓
[코드] 2축 + 함정 판정     생존 개월수 · 종료 시점 부채 / 함정 4종
  ↓
[코드] 비지배 정렬        파레토 레이어링으로 후보 그룹 3~8개
  ↓
[LLM] 최종 3개 선정       위험을 감안해 고르고 사용자 말로 설명한다
```

**역할 분담이 핵심 원칙이다.**
- **코드** — 계산과 하드 제약. 근거가 있는 것만 판정한다.
- **LLM** — 대화, 최종 3개 선정, 설명. 계산하지 않는다.

---

## 파일

```
core/
  pos.py           POS 파서. 일별 매출 → 월별 집계 · 비교 단위별 증감률
  synthetic_pos.py 합성 POS 생성기 (테스트용)
  phase.py         국면 판정 (지표 4개 × 0~2점 + 하드 규칙 3개)
  loader.py        policies_all.json 로더. 정책 → 후보 2계층
  combos.py        자격 필터 · 조합 생성 · 금액 역산
  cashflow.py      36개월 현금흐름 시뮬레이션
  forecast.py      매출 예측. Prophet·ETS·선형회귀를 로그 공간에서 적합해 월별 중앙값
  metrics.py       함정 4종 판정 (상환절벽 · 회복불가 신규융자 등)
  optimizer.py     비지배 정렬 · 후보 선정 · LLM 프롬프트 생성

tests/             각 모듈마다 test_*.py. `python -m tests.test_cashflow` 로 실행
                   cases_phase.py 는 국면 검증용 케이스 33개 (테스트 아님)

data/policies_all.json   정책 데이터. 12개 정책 · 25개 후보
planning.md              기획서 (1,500줄). 설계 근거와 기각한 대안
inquiries.md             기관 문의 관리. 해결 37건 · 미해결 4건
```

**미구현** — 없다. `agent.py`·`app.py`·`forecast.py` 모두 붙었다.

---

## 파이프라인 호출 순서

```python
pool    = eligible_candidates(db, profile, ctx)
amounts = default_loan_amounts(db, pool, profile, target_months)
amounts.update(fixed_amounts(profile, pool))   # 대환은 갈아탈 금액이 따로 정해진다
amounts.update(grant_amounts(db, pool, ctx))   # 점포철거비는 면적 비례
combos  = build_combos(db, pool, amounts)
combos  = rebalance_to_months(combos, profile, target_months)   # 조합 단위 재역산
evals   = evaluate(profile, combos)
evals, satisfied = filter_by_goal(evals, target_months=target_months)
reports = detect_all(profile, evals, ctx.phase)
groups  = select(evals, reports)
text    = to_prompt_text(
    groups, reports,
    baseline_survival=..., goal_months=..., goal_satisfied=satisfied,
    extras=extra_candidates(pool), context=ctx,
)
```

**입력 경로가 셋이다.**

| 입력 | 함수 | 하는 일 |
|---|---|---|
| 목표 개월수만 | `rebalance_to_months()` | 목표를 채우는 최소 금액을 역산 |
| 금액만 | `rebalance_to_inflow()` | 보조금을 먼저 세고 부족분을 융자로 |
| 둘 다 | `resolve_by_both()` | 금액 기준으로 만들고, 목표 미달이면 개월수 기준으로 |

---

## 설계 원칙

**임의 상수를 쓰지 않는다.** 경계값은 제도 조문·공식 통계·학술 근거에서 가져온다.
근거가 없으면 판정하지 않고, 판정하지 않는 이유와 그 대가를 기록한다.

**판정에 필요한 값이 없으면 자격을 단정하지 않는다.**
```
면적 모름         → 점포철거비 제외
업력 모름         → 긴급경영안정 직접대출 제외
사업 개시일 모름   → 새출발기금·경영안정 바우처 제외
자격 요건 모름     → '아니오'와 똑같이 제외
```
예외는 "지금 하면 되는 일"뿐이다(신용관리 교육 이수 → `_확인필요_자격_지금가능`).

**두 축은 생존 개월수와 종료 시점 부채다.** 가중 점수나 단일 축 순위를 쓰지 않고
비지배 정렬로 둘을 균등하게 다룬다. 부채 축 후보 4종을 케이스 12개 × 입력 경로 3종으로
실측 비교해 종료부채를 택했다(planning.md 참조).

**JSON의 `_` 필드는 근거 기록이다.** 런타임에 읽지 않는다. 판단이 뒤집힐 때 다시
찾지 않기 위해, 그리고 심사에서 "왜 이렇게 했나"에 답하기 위해 남긴다.

---

## 작업 규칙

**파일을 고치기 전에 허락을 받는다.** 고친 뒤에는 바로 보여준다.

**모르면 모른다고 한다.** 확실한 것만 말하고, 추측은 추측이라고 밝힌다.
근거 없이 "확인했다"고 쓰지 않는다 — 실제로 그런 문장이 여러 번 나왔고 매번 정정했다.

**설계 결정에는 근거와 기각한 대안을 함께 적는다.** 코드 docstring이나 JSON `_` 필드에.

**코드 규칙**
- 한국어 주석 + 영어 식별자
- ASCII 전용 파일명 (크로스플랫폼 Git 호환)
- 스톡 변수는 `_at` (길이 horizon+1), 플로우 변수는 `_during` (길이 horizon)

**변경 후 테스트를 돌린다.** `python -m tests.test_<모듈>` — 7개가 있다.

한국어 Windows 에서는 `PYTHONUTF8=1` 을 붙여야 한글 출력이 안 깨진다. **`test_forecast`
하나만 예외로 붙이지 않는다.**

```
PYTHONUTF8=1 python -m tests.test_metrics     그 외 전부
python -m tests.test_forecast                 예측만 (PYTHONUTF8 없이)
```

`PYTHONUTF8=1` 이면 이 기계에서 Prophet 초기화가 실패한다. `cmdstanpy` 가
`where.exe tbb.dll` 을 돌려 그 출력을 읽는데, 한국어 Windows 는 cp949 로 내보내고
`PYTHONUTF8=1` 은 UTF-8 로 읽게 만들어 `UnicodeDecodeError` 가 난다. Prophet 이
그 예외를 삼켜서 표면에는 `AttributeError: 'Prophet' object has no attribute
'stan_backend'` 로 나온다 — **원인과 메시지가 전혀 다르다.** Linux(배포 환경)는
로케일이 UTF-8 이라 해당 없다.

앙상블은 모델 하나가 실패해도 남은 둘로 값을 낸다. 그래서 이 문제가 있어도 나머지
검사는 전부 통과하고 조용히 두 모델만 돈다. `test_prophet_actually_runs_here` 가
그것만 따로 확인한다.

같은 이유로 **`test_app` 은 `forecast.ENABLED = False` 로 두고 배선만 본다.** 그 파일은
`PYTHONUTF8=1` 로 도는데, 그러면 두 모델짜리 앙상블을 검사하게 되어 배포(세 모델)와
다른 것을 보게 된다. 앱 경로가 앙상블에 닿는지는 `test_forecast` 의
`test_app_path_uses_the_ensemble` 이 세 모델이 다 도는 환경에서 확인한다.

---

## 다음 작업 (우선순위)

### 1. `app.py` — Streamlit UI

**반드시 반영할 것**

**(a) 조건부 추가 질문 — 2단계 흐름**
```
1차 입력 → 국면 판정 → 자격 필터 → 추가 질문 → 재계산
```
지금은 한 번에 다 받는 구조를 전제한다. 해당 정책이 후보에 오를 때만 묻는다.
```
business_started    긴급경영안정(연매출 1억 400만 미만일 때) · 새출발기금(회복불가일 때)
                    · 경영안정 바우처
재산가액·담보대출     새출발기금 감면액 (현재 미반영, 감면액 과대평가)
기존 보증 잔액        서울시 융자 한도 차감 (현재 미반영)
과거 정책자금 횟수     소진공 5년 3회 제한 (현재 미판정)
```

**(b) `매출_요건_택1대체` 플래그는 유지한다** — 2026-08-24 정정
서울시 긴급자영업자금의 택1요건 네 갈래 중 매출만 우리가 판정한다. 예전에는 "2단계
흐름을 넣으면 자격 필터가 직접 판정하니 이 플래그를 없앤다"고 적어 두었는데, 그 흐름을
실제로 만든 뒤 확인하니 **없앨 수 없다.**

```
후보에 올려두고   →   물어보고    →   답에 따라 제외
(이 플래그)          (⑥ 단계)        (excluded_by_confirm)
```

플래그가 **첫 단계**를 담당한다. 자격 필터에서 매출 미달을 걸러내면 그 후보가 풀에서
사라져 **물어볼 대상 자체가 없어진다.** 판정을 앞당기는 게 아니라 미루는 것이 이
플래그의 일이다. (`combos._passes_revenue_requirement` 주석 참조)

**(c) 최종 3개에 대해서만 호출할 것**
```python
optimizer.apply_links(combo.items)      # 신청링크·문의처
optimizer.prerequisites(combo.items)    # 신청 전에 해야 할 일 (교육 이수 등)
```
프롬프트에는 넣지 않는다 — 순위 판단에 도움이 되지 않고 후보 그룹 5~8개에 모두
붙으면 프롬프트만 길어진다.

**(d) 목표 미달 배너**
`filter_by_goal` 이 `satisfied=False` 면 화면에 표시한다.
"목표 N개월을 채우는 조합이 없습니다. 아래는 차선입니다."
LLM 안내(`goal_satisfied` 인자)와 화면 배너를 **둘 다** 한다.

### 2. `agent.py` — LLM 연결
프롬프트는 `to_prompt_text()` 가 이미 만든다(1,000~2,800자). 호출·응답 파싱과
`_확인필요_자격` 재질문 흐름만 구현하면 된다.

### 3. `forecast.py` — 매출 예측
Prophet·ETS·선형회귀 앙상블의 중앙값. 현재 `trend_revenue()` 로 대체 중이라
없어도 전부 돌아간다. 인터페이스가 같아 마지막에 갈아끼워도 된다.
**학습된 모델을 배포하는 방식이 아니다** — 요청마다 그 사용자의 POS에 적합한다.

### 4. 나중에
- `TODO(adr)` 마커 대상 docstring의 설계 근거를 `docs/adr/` 로 이전
- JSON `_` 필드도 같이 정리 (파일 129KB 중 상당 부분)
- 기업마당 API (MVP 범위 밖)

---

## 알려진 한계

**계산이 실제와 다를 수 있는 것**
```
새출발기금 감면액    "신용대출 중 재산가액을 뺀 순부채"인데 전액 기준으로 계산
                   → 과대평가. `_계산전제` 로 LLM에 알린다
서울시 융자 한도     기보증잔액 차감 미반영 → 과대평가
상환연장 기간       7년이 누적 상한인데 과거 이력을 안 받아 항상 84개월 적용
파산 이후          잔고 0 고정. 연체이자를 반영하지 않아 부채가 과소평가
변동금리           소진공 자금은 분기마다 바뀌는데 현재 금리로 고정 계산
                   → 안내 문구로 알린다
```

**순서를 표현하지 못한다**
조합 안의 모든 항목이 t=0에 동시 신청된 것으로 계산한다. 현실에는 신청·심사·실행의
시차가 있고 순서가 자격을 가르는 경우가 있다(`_순차경로` · A-46 참조).

**미해결 문의 4건** — inquiries.md 참조. A-46(상환연장 ↔ 융자 순서)만 조합에
실질 영향이 있다.

---

## 참고

- 소진공 공고 제2026-478호 / FAQ (`ols.semas.or.kr`)
- 서울신용보증재단 보증대상·FAQ / 서울시 민생노동국 소상공인정책과
- 새출발기금 FAQ (`newstartfund.or.kr`) / 신용회복지원협약
- 채권은행의 기업신용위험 상시평가 운영협약 (2024.3.22)
- 중소기업 신속 금융지원 프로그램 공동운영 모범규준 (2024.6.25)
- Pindado, Rodrigues & De la Torre (2008), *Estimating financial distress likelihood*
