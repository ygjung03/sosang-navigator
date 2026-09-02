"""
소상공인 나침반 — Streamlit 대화형 UI.

파이프라인에서 이 파일의 자리:

    [사용자 입력]  POS 업로드(정밀) 또는 대략값 입력(간편)
        │
        │  이 파일이 core/ 모듈을 순서대로 호출한다.
        │  계산은 전부 core/ 가 하고, 여기서는 입력을 모으고 결과를 그린다.
        ▼
    pos → phase → combos → cashflow → metrics → optimizer
        │
        ▼
    후보 그룹 3~8개 → [LLM 최종 3개 선정]  (agent.select_final_three)
        ▼
    화면: 국면 진단 · 추천 카드 · 현금흐름 그래프 · 융자 슬라이더


한 번에 한 단계
---------------
사이드바 폼이 아니라 **한 화면에 한 단계**다. 위젯(number_input·selectbox·radio)으로
답하고 "◀ 이전 / 다음 ▶" 으로 오간다. 답은 `ss.answers` 에 남아 되돌아가도 그대로
있고, 고치면 아래가 Streamlit 재실행으로 자동 재계산된다.

    진행 상태는 `st.session_state.step`(정수) 하나로 관리한다.
    `_step_after` · `_step_before` 가 물을 것이 없는 단계를 건너뛴다.

질문은 관련된 것끼리 묶어 묻는다. 하나씩 쪼개면 답할 횟수가 너무 많아진다.

    ① 기본 상황(지역·재창업·폐업·대출포함)      S_SITUATION
    ② 매출 자료(POS 업로드/데모/없어요)         S_REVENUE
    ③ 월 비용(고정비·변동비·현금)               S_COST
    ④ 부채 규모(총잔액·월상환·금리)             S_DEBT
    ⑤ 연체·신용(연체일수·체납·채무조정·고금리)   S_DEBT2
    → 진단 (번호 없음. 우리가 내놓는 화면이다)   S_DIAGNOSIS
    ⑥ 추가 확인 — needed_conditional_fields() 가 돌려준 것 + 자격 확인 질문
    ⑦ 조건 설정(희망 금액·목표 개월수)          S_GOALS
    → 결과 (번호 없음)                        S_RESULTS

**입력 단계는 일곱, 우리가 내놓는 화면은 둘이다.** 진단·결과에 번호를 붙이지 않는
이유는 사용자가 답하는 자리가 아니기 때문이다(`OUTPUT_STEPS`).

자유 텍스트 파싱은 없다. `st.chat_input` 도 쓰지 않는다 — 위젯으로만 답한다.


⚠ 이 파일에서는 상수 설명을 `#` 주석으로 쓴다
----------------------------------------------
app.py 는 Streamlit 이 직접 실행하는 스크립트라, 홀로 놓인 문자열 리터럴을 magic 이
`st.write` 로 바꿔 **화면에 그대로 출력한다.** 상수 아래에 파이썬 관례대로
`\"\"\"...\"\"\"` 를 붙이면 그것이 모든 단계 맨 위에 뜬다(실제로 `DEMO_END_DATE`
설명이 그렇게 떴다). `tests/test_app.test_no_bare_expressions_in_app` 이 검사한다.
`core/` 는 import 되는 모듈이라 해당 없다.


명명 규칙: 코드 식별자는 영어, 도메인 값은 한글. 주석은 한글.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import math
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import agent, forecast, future, optimizer, pos
from core.cashflow import (
    SIM_MONTHS,
    Applied,
    UserProfile,
    flat_revenue,
    month_display,
    simulate,
    trend_revenue,
)
from core.combos import (
    Combo,
    Evaluated,
    UserContext,
    build_combos,
    default_loan_amounts,
    eligible_candidates,
    evaluate,
    extra_candidates,
    filter_by_goal,
    fixed_amounts,
    grant_amounts,
    rebalance_to_inflow,
    rebalance_to_months,
    resolve_by_both,
    snap_amount,
)
from core.loader import PolicyDB, load
from core.metrics import TrapKind, detect_all, detect_traps
from core.optimizer import (CandidateGroup, _dominates, apply_links, prerequisites,
                            select, to_prompt_text)
from core.phase import (
    DROP_SEVERE,
    DROP_WARN,
    PHASE_ORDER,
    SCORE_IMPROVABLE,
    SCORE_TEMPORARY,
    PhaseResult,
    describe,
    judge,
    neighbor_phases,
    score_icr,
    score_operating_profit,
    score_overdue,
    score_revenue_drop,
)

# ══════════════════════════════════════════════════════════
# 상수 · 헬퍼
# ══════════════════════════════════════════════════════════

MAN = 10_000  # 만원
DEFAULT_TARGET_MONTHS = 6  # 목표 생존 개월수 기본값 (기획서 5.4)

# 경영안정 바우처 연매출 상한 = 1억 400만원. 조건부 질문 트리거에 쓴다.
VOUCHER_REVENUE_CAP = 104_000_000

# 대화 단계 인덱스. 한 번에 한 단계만 보이고 ◀ 이전 / 다음 ▶ 으로 이동한다.
# 부채는 입력이 많아(총잔액·월상환·금리 / 연체·체납·고금리·직접대출·신용) 두 단계로 나눈다.
S_SITUATION = 0
S_REVENUE = 1
S_COST = 2
S_DEBT = 3        # 부채 규모 (총잔액·월상환·금리)
S_DEBT2 = 4       # 연체·신용 (연체일수·체납·고금리·직접대출잔액·신용점수)
S_DIAGNOSIS = 5
S_CONDITIONAL = 6
S_GOALS = 7
S_RESULTS = 8
S_DONE = 9        # 완료 화면. 단계가 아니라 '끝'이라 진행바에 칸을 더하지 않는다.


def won(man: float) -> int:
    """만원 단위 입력을 원으로."""
    return int(round(man)) * MAN


def fmt_man(amount: float) -> str:
    """원 → '3,000만원' 문자열."""
    return f"{amount / MAN:,.0f}만원"


def months_label(months: int) -> str:
    """생존 개월 표기 — 상한(36)에 걸리면 '36개월 이상', 아니면 'N개월'.

    36은 시뮬레이션 창의 상한이라, 끝까지 버틴 조합은 '36개월'이 아니라
    '36개월 이상'이 정확하다(진단·결과·카드 제목을 이걸로 통일한다).
    """
    return f"{SIM_MONTHS}개월 이상" if months >= SIM_MONTHS else f"{months}개월"


# 문자열 속 '원' 단위 금액(콤마 포함 4자리 이상)을 만원 단위로 바꾼다. 화면 표기 전용이며
# 계산에는 쓰지 않는다. cashflow가 만든 notes 가 원 단위라 표시 직전에 변환한다.
# 비율(3.00%)·개월수(60개월)는 자릿수가 짧거나 %라 걸리지 않는다.
# (?:\s*원) 을 통째로 선택적으로 둔다 — '원'이 없으면 뒤 공백도 먹지 않아야
# '9,500만원 -> 2,400만원' 처럼 -> 앞뒤 공백이 유지된다.
_MONEY_IN_TEXT = re.compile(r"([\d,]{4,})(?:\s*원)?")


def money_to_man(text: str) -> str:
    """'500,000,000원 @ 3.00%' → '50,000만원 @ 3.00%'. 소수점은 노출하지 않는다."""
    def repl(m: "re.Match") -> str:
        won_amount = int(m.group(1).replace(",", ""))
        return f"{won_amount // MAN:,}만원"

    return _MONEY_IN_TEXT.sub(repl, text)


# ══════════════════════════════════════════════════════════
# 데이터 로딩 (캐시)
# ══════════════════════════════════════════════════════════


@st.cache_resource
def load_db() -> PolicyDB:
    """정책 DB는 한 번만 읽어 재사용한다."""
    return load()


@st.cache_data(show_spinner=False)
def summarize_uploaded(raw_bytes: bytes, name: str) -> pos.PosSummary:
    """업로드한 POS 파일에서 지표를 뽑는다. 파일 내용으로 캐시한다."""
    buffer = io.BytesIO(raw_bytes)
    buffer.name = name  # 확장자 판별용
    return pos.summarize(pos.load(buffer))


# 데모 POS 의 마지막 날. 고정한다.
#
# ⚠ 이 설명은 반드시 `#` 주석이어야 한다. app.py 는 Streamlit 이 직접 실행하는
#   스크립트라, 홀로 놓인 문자열 리터럴을 magic 이 st.write 로 바꿔 **화면에 그대로
#   출력한다.** 한때 이 자리에 """..."""" 로 적혀 있었고, 모든 단계 맨 위에 개발
#   메모가 떴다. 모듈 상수 아래 설명을 붙일 때는 이 파일에서만 규칙이 다르다.
#
# synthetic_pos.generate(end_date=None) 은 "전월 말일"을 쓴다. 그대로 두면 달이
# 바뀔 때마다 12개월 창이 통째로 밀리고, 그 안의 계절 구간·변곡점이 달라져
# baseline 이 움직인다. 데모는 재현 가능해야 한다 — 재무값을 프리셋으로 고정한
# 것과 같은 이유다.
#
# 2026-09-01 에 실제로 깨졌다. 창이 2025-08~2026-07 에서 2025-09~2026-08 로
# 밀리면서 C 의 baseline 이 6 -> 5 개월이 됐고, 새출발기금 확정 시점(6개월)에 못
# 미쳐 감면이 통째로 사라졌다. test_debt_relief_reaches_confirmation 이 그것을
# 잡았다 — 테스트가 제 일을 한 것이다.
#
# 날짜는 새로 고른 값이 아니다. tests/test_pos.py 와 tests/test_synthetic_pos.py 가
# 재현성 때문에 이미 2026-07-31 로 고정해 두었고, 데모만 안 하고 있었다.
#
# 실제 사용자에게는 영향이 없다. 자기 POS 파일을 올리므로 이 경로를 안 탄다.
DEMO_END_DATE = datetime.date(2026, 7, 31)


@st.cache_data(show_spinner=False)
def summarize_scenario(key: str, months: int) -> pos.PosSummary:
    """데모 시나리오(A/B/C)의 합성 POS에서 지표를 뽑는다."""
    from data.synthetic_pos import generate

    return pos.summarize(
        pos.load(generate(key, months=months, end_date=DEMO_END_DATE))
    )


# ══════════════════════════════════════════════════════════
# LLM 연결 (core/agent.py)
# ══════════════════════════════════════════════════════════


_FALLBACK_NOTE = "아래는 AI 추천 없이 시뮬레이션으로 계산된 결과입니다."

_FAILURE_TEXT = {
    # (무슨 일이 있었는지, 재시도 버튼을 띄울지)
    #
    # **폴백 안내(_FALLBACK_NOTE)는 여기 넣지 않는다.** 배너가 줄을 바꿔 붙인다 —
    # 한 덩어리로 이어 붙이면 두 문장이 한 문단이 되어 뒤쪽이 안 읽힌다.
    #
    # 문구는 **일어난 일을 그대로** 쓴다. "응답을 이해하지 못했습니다" 같은 표현은
    # 우리 쪽 결함처럼 읽히는데, 실제로는 형식이 안 맞거나 근거 없는 값이 섞여
    # **검증이 걸러낸 것**이다. 그건 결함이 아니라 안전장치가 작동한 것이다.
    #
    # 키가 없으면 버튼만 빼고 안내는 남긴다 — 지금 눌러 봐야 같은 실패가 나지만,
    # 키를 설정하면 재시도가 통하므로 "무엇을 하고 나서 다시 시도하라"를 알려야 한다.
    agent.Failure.NO_KEY: (
        "AI 키가 설정되지 않았습니다. "
        "**ANTHROPIC_API_KEY 를 설정한 뒤 다시 시도해 주세요.**", False,
    ),
    agent.Failure.API_ERROR: (
        "AI 연결에 실패했습니다. **다시 시도**하면 AI 추천을 받을 수 있습니다.", True,
    ),
    agent.Failure.TIMEOUT: (
        "AI 응답이 지연됐습니다. **다시 시도**해 주세요.", True,
    ),
    agent.Failure.PARSE: (
        "AI 응답이 정해진 형식으로 오지 않아 사용하지 않았습니다. "
        "**다시 시도**해 주세요.", True,
    ),
    agent.Failure.VERIFY: (
        "AI 응답에 **계산 결과에 없는 숫자**가 섞여 있어 사용하지 않았습니다. "
        "**다시 시도**해 주세요.", True,
    ),
}


def _render_advice_banner(advice: agent.Advice | None) -> None:
    """LLM 실패를 **눈에 띄게** 알린다.

    조용히 폴백하면 시연에서 심사위원은 그게 원래 화면인 줄 안다. 실패 이유를 구분해
    보여줘야 무엇이 문제인지 바로 안다.
    성공했을 때는 재시도 버튼을 띄우지 않는다 — 같은 입력에 다른 답이 나오면 신뢰를
    잃는다. 실패했을 때의 재시도는 '다른 답'이 아니라 '첫 답'이라 괜찮다.

    두 문장을 문단으로 나눈다 — 한 덩어리면 뒤 문장이 안 읽힌다.
    옛 셋째 줄("아래 순서는 AI 판단이 아니라 계산 순서입니다")은 지웠다. 앞 두 문장이
    이미 다 말한다.
    `advice_banner` 컨테이너로 감싸는 것은 테두리 CSS 를 이 배너에만 걸기 위해서다 —
    같은 st.warning 인 함정 경고까지 굵어지면 위계가 사라진다.
    """
    if advice is None or advice.ok:
        return
    text, retryable = _FAILURE_TEXT.get(
        advice.failure, ("AI 추천을 받지 못했습니다. **다시 시도**해 주세요.", True),
    )
    with st.container(key="advice_banner"):
        st.warning(f"⚠ {text}\n\n{_FALLBACK_NOTE}")
        if retryable and st.button("다시 시도", key="retry_advice"):
            st.session_state.pop("_advice", None)  # 캐시를 비우고 재호출
            st.rerun()


_LOADING_HTML = """
<style>
@keyframes sn-flip { 0%,45% {transform:rotate(0deg);} 55%,100% {transform:rotate(180deg);} }
@keyframes sn-sand { 0% {height:0;} 100% {height:11px;} }
.sn-load {
  display:flex; align-items:center; justify-content:center; gap:16px;
  background:#EAF3FC; border:1px solid #b6d4f2; border-radius:8px;
  padding:26px 20px; margin:10px 0;
}
.sn-glass { animation: sn-flip 2.4s ease-in-out infinite; transform-origin:50% 50%; }
.sn-glass .sn-fill { animation: sn-sand 1.2s linear infinite; }
.sn-text { font-size:1.15rem; font-weight:700; color:#1F2933; line-height:1.5; }
.sn-sub  { font-size:0.9rem; font-weight:400; color:#4a5568; }
</style>
<div class="sn-load">
  <svg class="sn-glass" width="40" height="52" viewBox="0 0 30 40" aria-hidden="true">
    <path d="M5 3 h20 M5 37 h20" stroke="#2b6cb0" stroke-width="3" stroke-linecap="round"/>
    <path d="M7 4 L7 12 Q7 17 15 20 Q23 17 23 12 L23 4 Z" fill="#2b6cb0" opacity="0.75"/>
    <path d="M7 36 L7 28 Q7 23 15 20 Q23 23 23 28 L23 36 Z" fill="none"
          stroke="#2b6cb0" stroke-width="1.5"/>
    <rect class="sn-fill" x="9" y="25" width="12" height="11" fill="#2b6cb0" opacity="0.55"/>
  </svg>
  <div class="sn-text">AI가 최종 3개를 고르는 중입니다<br>
    <span class="sn-sub">1분 정도 소요됩니다. 창을 닫지 마세요.</span></div>
</div>
"""


def get_advice(prompt_text: str, labels: list[str]) -> agent.Advice:
    """LLM 최종 선정. **같은 프롬프트면 한 번만 호출**하고 세션에 캐시한다.

    슬라이더는 카드 안의 what-if 라 순위를 다시 뽑지 않는다 — 재호출하지 않는다.
    입력을 고쳐 프롬프트가 달라지면 자동으로 다시 부른다.
    실패했을 때만 '다시 시도' 버튼이 이 캐시를 비운다 — 성공한 답을 다시 부르면
    같은 입력에 다른 답이 나와 신뢰를 잃는다.

    **키에 시스템 프롬프트와 모델명도 넣는다.** 사용자 프롬프트만 해싱하면 시스템
    프롬프트를 고쳐도 키가 그대로라 옛 응답이 재사용된다(실제로 '3개를 채워라' 지시가
    반영되지 않았다). 모델을 바꿀 때도 마찬가지다.
    """
    key = hashlib.sha256(
        "\x00".join((prompt_text, agent.SYSTEM_PROMPT, agent.MODEL)).encode("utf-8")
    ).hexdigest()[:16]
    # 실측 25~28초(데모 A·B·C 각 1회, 2026-09-02). 화면에는 여유를 둬 '1분'이라 쓴다
    # — 짧게 적었다가 넘기면 멈춘 줄 안다. 기본 st.spinner 는 작은 글씨 한 줄이라
    # 같은 오해를 부른다.
    # placeholder 를 쓰는 이유 — st.spinner 는 문구와 크기를 못 바꾼다. 직접 그리고
    # 끝나면 비운다.
    #
    # ⚠ 자리는 **캐시가 맞을 때도** 잡는다. 캐시 적중일 때만 st.empty() 를 건너뛰면
    #   그 실행에서만 요소가 하나 적어져 뒤따르는 결과 카드가 한 칸씩 앞으로 밀린다.
    #   Streamlit 은 요소를 위치로 잇기 때문에 접힘 상자가 새로 그려지며 expanded
    #   기본값으로 돌아간다 — 슬라이더를 **처음** 움직였을 때만 카드가 접히던 원인이다.
    slot = st.empty()
    cached = st.session_state.get("_advice")
    if cached and cached[0] == key:
        return cached[1]
    slot.markdown(_LOADING_HTML, unsafe_allow_html=True)
    try:
        advice = agent.select_final_three(prompt_text, labels)
    finally:
        slot.empty()
    st.session_state["_advice"] = (key, advice)
    return advice


# order_by_advice() 는 지웠다. 그룹 단위로 재배열하면서 `g.members[0].label` 로 조회해,
# LLM 이 대표 아닌 멤버를 고르면 그 pick 이 갈 자리가 없어 통째로 사라졌다.
# render_results 가 picks 를 돌며 label 로 조합을 직접 찾는다.


# ══════════════════════════════════════════════════════════
# 조건부 질문 판정
# ══════════════════════════════════════════════════════════


def needed_conditional_fields(phase: str, annual_revenue: int | None, include_loans: bool) -> list[str]:
    """후보에 오를 정책이 요구하는 추가 질문만 고른다 (기획서 5.2.5 ④단계).

    처음부터 다 묻지 않고, 국면·연매출로 그 정책이 후보에 오를 수 있을 때만 묻는다.
    """
    fields: list[str] = []
    # 사업 개시 연월:
    #   새출발기금(회복불가) · 긴급경영안정 직접대출/경영안정 바우처(연매출 1억 400만 미만)
    if phase == "회복불가" or (
        annual_revenue is not None and annual_revenue < VOUCHER_REVENUE_CAP
    ):
        fields.append("business_started")
    # 점포 전용면적: 점포철거비(폐업 확정 전제 → 회복불가)
    if phase == "회복불가":
        fields.append("store_area_pyeong")
    # 과거 정책자금 횟수: 소진공 융자 5년 3회 제한
    if include_loans:
        fields.append("policy_loan_count_5y")
    return fields


# ══════════════════════════════════════════════════════════
# 프로필·컨텍스트 조립
# ══════════════════════════════════════════════════════════


def build_monthly_revenue(precise: bool, summary: pos.PosSummary | None, rough_revenue: int) -> list[int]:
    """36개월 예상 매출선.

        정밀   POS 월별 매출로 앙상블 예측(forecast.py)
        간편   현 수준 유지(추세 없음)

    앙상블이 None 을 주면(꺼져 있음·데이터 12개월 미만·세 모델 모두 실패) 추세선으로
    돌아간다. 추세선은 3개월 증감률을 한 달치 비율로 바꿔 36개월 등비로 연장하는
    것이라 최근 구간이 가파르면 폭주한다 — 급락 후 반등한 계열에서 3개월 증감률이
    +35.1% 라 3년 뒤가 현재의 33배로 나왔다. fallback 으로만 두고 평소에는 앙상블을 쓴다.
    """
    line, _, _ = revenue_line(precise, summary, rough_revenue)
    return line


def revenue_line(
    precise: bool, summary: pos.PosSummary | None, rough_revenue: int
) -> tuple[list[int], bool, bool]:
    """`build_monthly_revenue` 와 같되 **플래그 둘을 함께** 돌려준다.

        from_ensemble   앙상블 예측선인가, 추세선 폴백인가
        all_models      그 앙상블에 세 모델이 다 들어갔는가

    미래 자격은 앙상블 예측선 위에서만 말할 수 있다. 추세선 폴백이면 그 기능을 꺼야
    하는데, 나중에 매출선만 보고 되짚으려면 앙상블을 다시 부르거나(한 번에 287ms)
    값을 비교해야 한다. **만든 자리에서 플래그를 세워 넘긴다.**

    `all_models` 는 **미래 자격을 가르지 않는다.** 모델 하나가 빠져도 남은 것으로 나온
    예측선은 추세선보다 낫고, 그때 미래 카드를 없애면 문제를 알리는 대신 기능만
    사라진다. 이 플래그는 **두 모델짜리로 도는 것을 모르고 지나가지 않으려고** 둔다 —
    prophet 이 빠진 채 한동안 그렇게 돌았다(2026-09-01). `test_forecast` 가 검사한다.
    """
    if precise and summary is not None:
        predicted, models = forecast.ensemble_detail(summary.monthly_revenue)
        if predicted is not None:
            return predicted, True, len(models) == len(forecast.MODEL_NAMES)
        current = max(summary.recent_3m_avg, 1)
        # 3개월 증감률을 월율로: (1+change_3m)^(1/3) - 1
        monthly_change = (1 + summary.revenue_change_3m) ** (1 / 3) - 1
        return trend_revenue(current, monthly_change), False, False
    return flat_revenue(max(rough_revenue, 1)), False, False


def build_profile(finance: dict, monthly_revenue: list[int], current_revenue: int) -> UserProfile:
    """UserProfile 조립. 변동비 금액을 매출 대비 비율로 환산한다."""
    denom = max(current_revenue, 1)
    variable_ratio = min(max(finance["variable_cost"] / denom, 0.0), 0.95)

    high_rate = finance["high_rate_debt"]
    if high_rate is not None and high_rate > finance["debt"]:
        high_rate = finance["debt"]  # 고금리 부분이 총 부채를 넘을 수 없다

    return UserProfile(
        cash_on_hand=finance["cash"],
        monthly_revenue=monthly_revenue,
        fixed_cost=finance["fixed_cost"],
        variable_ratio=variable_ratio,
        existing_debt=finance["debt"],
        existing_debt_payment=finance["debt_payment"],
        existing_debt_rate=finance["debt_rate"],
        high_rate_debt=high_rate,
        overdue_days=finance["overdue_days"],
        policy_loan_balance=finance["policy_loan_balance"],
    )


def build_context(
    phase: str,
    elig: dict,
    conditional: dict,
    summary: pos.PosSummary | None,
    annual_revenue: int | None,
) -> UserContext:
    """UserContext 조립."""
    drop_by_window: dict[str, float] = {}
    drop_rate = 0.0
    growth_rate = 0.0
    if summary is not None:
        # revenue_change_by_window: 증감률(-0.15=15%감소). 부호를 뒤집어 '감소를 양수로'.
        drop_by_window = {
            name: -change
            for name, change in summary.revenue_change_by_window.items()
            if change is not None
        }
        # 국면·자격용 감소율은 분기·반기 중 큰 값
        drop_rate = summary.drop_rate(("분기", "반기"))
        # 재도전 도약형처럼 '증가'가 요건인 경우에 쓴다. 증가했을 때만 양수.
        growth_rate = summary.revenue_change_3m if summary.revenue_change_3m > 0 else 0.0

    return UserContext(
        phase=phase,
        region=elig["region"],
        credit_score=elig["credit_score"],
        restart_status=elig["restart_status"],
        is_closed=elig["is_closed"],
        tax_delinquent=elig["tax_delinquent"],
        in_debt_restructuring=elig["in_debt_restructuring"],
        store_area_pyeong=conditional["store_area_pyeong"],
        business_started=conditional["business_started"],
        revenue_drop_by_window=drop_by_window,
        revenue_drop_rate=drop_rate,
        revenue_growth_rate=growth_rate,
        annual_revenue=annual_revenue,
        include_loans=elig["include_loans"],
        policy_loan_count_5y=conditional["policy_loan_count_5y"],
    )


def _profit_and_interest(profile: UserProfile, current_revenue: int) -> tuple[float, float]:
    """월 영업이익과 월 이자비용. **두 모드가 이 한 곳을 함께 쓴다.**

    정밀 모드(`_phase_inputs`)와 간편 모드(`simple_mode_phases`)가 같은 식을 쓰는데,
    각자 계산하면 한쪽만 고쳤을 때 두 모드가 조용히 갈린다.
    """
    operating_profit = current_revenue * (1 - profile.variable_ratio) - profile.fixed_cost
    interest_cost = profile.existing_debt * (profile.existing_debt_rate or 0.0) / 12
    return operating_profit, interest_cost


def _phase_inputs(
    profile: UserProfile, summary: pos.PosSummary, current_revenue: int
) -> tuple[float, float, float, int]:
    """국면 지표의 입력값 (월 영업이익 · 월 이자비용 · 매출감소율 · 그 감소율의 비교 개월 수).

    감소율은 분기·반기 중 큰 쪽이라 **실제 채택된 단위가 그때그때 다르다.** 개월 수를
    함께 넘겨야 근거 문장이 "최근 6개월 …"처럼 실제 기간을 말한다(반기 값을 3개월이라
    쓰면 매출 요약 바의 3개월 수치와 어긋나 보인다).
    """
    operating_profit, interest_cost = _profit_and_interest(profile, current_revenue)
    drop_rate, window = summary.drop_rate_detail(("분기", "반기"))
    # 단위가 안 잡히면(데이터 부족) 감소율이 0.0 이라 0점 → 문장이 없다. 기본 3을 준다.
    window_months = pos.COMPARE_WINDOWS.get(window or "", 3)
    return operating_profit, interest_cost, drop_rate, window_months


def judge_phase(
    profile: UserProfile, summary: pos.PosSummary, current_revenue: int, is_closed: bool
) -> PhaseResult:
    """지표 4종 + 하드 규칙으로 국면을 판정한다."""
    operating_profit, interest_cost, drop_rate, drop_months = _phase_inputs(
        profile, summary, current_revenue
    )
    buffer_months = simulate(profile, []).survival_months
    return judge(
        operating_profit=operating_profit,
        interest_cost=interest_cost,
        revenue_drop_rate=drop_rate,
        revenue_drop_window_months=drop_months,
        overdue_days=profile.overdue_days,
        is_closed=is_closed,  # 하드 규칙 ①: 폐업·휴업 확정이면 회복불가
        buffer_months=buffer_months,
    )


def simple_mode_phases(
    profile: UserProfile, current_revenue: int, is_closed: bool
) -> tuple[list[str], list[tuple[str, int, str | None]], int]:
    """간편 모드에서 **가능한 국면들**과 잴 수 있는 지표 3종, 무지원 생존 개월수.

    POS 가 없어도 ③④⑤ 단계에서 받은 값으로 지표 4종 중 **셋이 계산된다** —
    ICR(부채·금리·영업이익) · 연체(연체 일수) · 영업이익(매출 대략값·비용).
    못 재는 것은 매출 감소율 하나뿐이다. 그것을 버리고 사용자 선택만 쓰는 것은
    있는 근거를 안 쓰는 것이다.

    **그렇다고 셋으로 총점을 내 국면을 확정할 수는 없다.** 국면 경계(2점·5점)가
    8점 만점 기준이라, 넷 중 셋으로 낸 합계에 그 경계를 그대로 쓰면 근거가 없다.
    매출감소를 0점으로 채우면 항상 낙관 쪽으로 기운다.

    그래서 **못 잰 지표가 낼 수 있는 점수 범위만큼만 열어 둔다.** 매출감소는
    0·1·2점 중 하나이므로, 그 셋을 각각 넣어 `judge()` 를 세 번 부르고 나온
    국면들을 후보로 삼는다. 임의로 좁히는 것이 아니라 계산으로 나온다.

    **`judge()` 를 그대로 부르는 것이 중요하다.** 하드 규칙 셋(폐업·휴업 확정 /
    총점 0인데 36개월 미생존 / 한 지표라도 2점)이 정밀 모드와 똑같이 적용된다.
    범위를 우리가 다시 계산하면 그 규칙을 재구현하게 되고 두 모드가 갈린다.

    후보가 하나로 좁혀지면 물을 것이 없다 — 호출부가 그대로 확정한다.

    Returns:
        (가능한 국면들, 잴 수 있는 지표 3종의 행). 국면은 `PHASE_ORDER` 순이다.
    """
    operating_profit, interest_cost = _profit_and_interest(profile, current_revenue)
    buffer_months = simulate(profile, []).survival_months

    # 매출감소가 0·1·2점이 되는 감소율. 경계값은 phase.py 의 DROP_WARN·DROP_SEVERE 를
    # 그대로 가져오고, 점수 매김은 judge() 에 맡긴다 — 여기서 숫자를 다시 쓰면
    # 경계가 바뀔 때 두 모드가 갈린다. 0.0 은 DROP_WARN 미만이면 되는 자리다.
    rates = (0.0, DROP_WARN, DROP_SEVERE)
    phases = {
        judge(
            operating_profit=operating_profit,
            interest_cost=interest_cost,
            revenue_drop_rate=rate,
            overdue_days=profile.overdue_days,
            is_closed=is_closed,
            buffer_months=buffer_months,
        ).phase
        for rate in rates
    }
    rows = [
        ("이자보상배율", *score_icr(operating_profit, interest_cost)),
        ("연체", *score_overdue(profile.overdue_days)),
        ("영업이익", *score_operating_profit(operating_profit)),
    ]
    return sorted(phases, key=PHASE_ORDER.get), rows


def exclusion_note(
    phases: list[str], rows: list[tuple[str, int, str | None]], is_closed: bool = False
) -> str:
    """어느 국면이 왜 빠졌는지 한 줄. "좁혀집니다"의 근거다.

    **규칙을 다시 구현하지 않는다.** 어느 지표가 어느 국면을 배제했는지는
    `judge()` 안의 점수·하드 규칙이 얽혀 결정되므로 밖에서 되짚으면 두 모드가
    갈릴 위험이 있다. 대신 **측정된 사실과 방향만** 말한다.

        빠진 국면이 남은 것보다 나쁜 쪽   -> 0점인 지표를 든다("위험 신호가 없어")
        빠진 국면이 남은 것보다 가벼운 쪽 -> 1점 이상인 지표를 든다("위험 신호가 있어")

    방향은 `PHASE_ORDER` 로 정한다 — 국면 순서는 제도에서 온 참인 값이다.
    지표 점수도 계산된 값이라, 둘을 잇는 문장은 우리가 지어내는 것이 아니다.
    """
    excluded = [p for p in PHASE_ORDER if p not in phases]
    if not excluded:
        return ""
    # 하드 규칙 ①이 걸렸으면 **지표가 아니라 그 규칙이 정한 것**이다. 지표 탓으로 쓰면
    # 사실과 다르다 — 같은 지표로 폐업이 아니면 다른 국면도 가능하다.
    # 문구는 정밀 모드의 `hard_rule_sentence` 와 맞춘다.
    if is_closed:
        return f"폐업·휴업으로 인해 {phases[0]} 국면으로 판정되었습니다."
    ranks = [PHASE_ORDER[p] for p in phases]
    worse = [p for p in excluded if PHASE_ORDER[p] > max(ranks)]
    lighter = [p for p in excluded if PHASE_ORDER[p] < min(ranks)]

    def names(pick) -> str:
        return "·".join(n for n, s, _ in rows if pick(s))

    def quoted(group: list[str]) -> str:
        """'정상'·'일시적자금난' 처럼. 따옴표를 국면마다 씌워야 경계가 보인다."""
        return "·".join(f"'{p}'" for p in group)

    parts = []
    if worse:
        clean = names(lambda s: s == 0)
        parts.append(
            f"{clean}에 위험 신호가 없어 {quoted(worse)}{_eun_neun(worse[-1])} 제외했습니다."
            if clean else f"측정된 지표로는 {quoted(worse)}까지 가지 않습니다."
        )
    if lighter:
        risky = names(lambda s: s >= 1)
        parts.append(
            f"{risky}에 위험 신호가 있어 {quoted(lighter)}{_eun_neun(lighter[-1])} 제외했습니다."
            if risky else f"측정된 지표로는 {quoted(lighter)}에 해당하지 않습니다."
        )
    return " ".join(parts)


def _eun_neun(word: str) -> str:
    """'은' / '는' 을 고른다. 앞 글자에 받침이 있으면 '은'.

    국면 이름이 상황마다 달라 조사를 붙박이로 쓸 수 없다 — '정상은'인데 '회복불가는'이다.
    한글 음절은 0xAC00 부터 28개씩 묶여 있고, 그 나머지가 0이면 받침이 없다.
    """
    last = word.strip("'")[-1]
    if not ("가" <= last <= "힣"):
        return "는"  # 한글이 아니면 판단하지 않는다
    return "는" if (ord(last) - 0xAC00) % 28 == 0 else "은"


def phase_indicator_rows(
    profile: UserProfile, summary: pos.PosSummary, current_revenue: int
) -> list[tuple[str, int, str | None]]:
    """4개 지표의 (이름, 점수, 사유문장). 진단 화면에서 0점짜리도 전부 보여주려고 쓴다.

    PhaseResult.reasons 는 문장이 붙은 지표만 담고(그마저 ICR 2점은 문장이 None),
    지표별로 매핑되지 않는다. 그래서 판정과 **같은 입력**으로 score_* 를 다시 불러
    지표별 (점수, 문장)을 얻는다. 계산을 바꾸는 게 아니라 표시용으로 재사용하는 것이다.
    """
    operating_profit, interest_cost, drop_rate, drop_months = _phase_inputs(
        profile, summary, current_revenue
    )
    return [
        ("ICR (이자 대비 영업이익)", *score_icr(operating_profit, interest_cost)),
        ("매출 감소", *score_revenue_drop(drop_rate, drop_months)),
        ("연체", *score_overdue(profile.overdue_days)),
        ("영업이익", *score_operating_profit(operating_profit)),
    ]


# ══════════════════════════════════════════════════════════
# 파이프라인 실행
# ══════════════════════════════════════════════════════════


CONFIRM_YES, CONFIRM_NO, CONFIRM_UNSURE = "예", "아니오", "잘 모르겠어요"


def stable_key(prefix: str, text: str) -> str:
    """내용으로 정해지는 위젯 key.

    프로세스마다 값이 바뀌는 것(파이썬 `hash()` 는 PYTHONHASHSEED 로 랜덤화된다)이나
    **순서에 의존하는 것**(카드 index)을 key 로 쓰면, 앱을 다시 띄우거나 순서가 바뀔 때
    저장된 답이 엉뚱한 대상에 붙는다. 시연 중에 터지는 종류라 내용 해시로 고정한다.
    """
    return f"{prefix}_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def confirm_key(question: str) -> str:
    """자격 확인 질문의 위젯 key. 질문 문구가 같으면 항상 같은 key 다."""
    return stable_key("confirm", question)


def confirm_key_of(db: PolicyDB, candidate_id: str) -> str | None:
    """후보 id 로 그 후보의 자격 확인 질문 key 를 찾는다.

    데모 프리셋이 **후보 id 로** 답을 지정하고 여기서 key 로 옮긴다. 프리셋에 해시
    문자열을 박아 두면 정책 데이터의 질문 문구가 바뀌었을 때 조용히 어긋난다.
    """
    for cand in db.candidates:
        if cand.candidate_id != candidate_id:
            continue
        for field_name in ("_확인필요_자격", "_확인필요_자격_매출미달시"):
            if text := cand.get(field_name):
                return confirm_key(text)
    return None


_CIRCLED = "①②③④⑤⑥⑦⑧⑨"
_LEAD_END = re.compile(r"[.?]\s*(?=[" + _CIRCLED + r"])")


def confirm_question_lines(question: str) -> list[str]:
    """자격 확인 질문을 화면에 그릴 줄들로 쪼갠다. JSON 원문은 건드리지 않는다.

    원문은 한 문단에 '앞말 + ①②③' 이 이어 붙어 있어 읽히지 않는다. 앞말은 버리고
    (화면이 '해당사항이 있는지 체크해주세요.' 를 대신 쓴다) 번호마다 줄을 나눈다.

    ①의 나열은 15개가 넘어 한 줄로도 길다. 굵은 표제와 나열을 갈라 두 줄로 만들고,
    나열은 가운뎃점을 쉼표로 바꿔 눈이 끊어 읽게 한다.
    """
    body = question.strip()
    if head := _LEAD_END.search(body):          # '…해당하시나요?' 같은 앞말 제거
        body = body[head.end():].strip()
    if not any(c in body for c in _CIRCLED):    # 번호가 없으면 한 줄짜리 질문이다
        return [body]

    lines: list[str] = []
    parts = re.split(r"(?=[" + _CIRCLED + r"])", body)
    for part in (p.strip() for p in parts):
        if not part:
            continue
        # '① **표제** — 나열' 을 표제 줄과 나열 줄로 가른다
        if " — " in part and part.count("·") >= 3:
            label, rest = part.split(" — ", 1)
            lines.append(f"- {label}")
            lines.append(f"    {rest.replace('·', ', ')}")
        else:
            lines.append(f"- {part}")
    return lines


def confirm_questions(pool, ctx=None) -> list[dict]:
    """후보에 붙은 **사용자 확인 질문**. 같은 문구는 하나로 묶는다.

    질문 문구가 조합과 무관하게 완결돼 있어(8개 후보 전부 확인) 코드가 그대로 띄운다.
    LLM 이 되물으면 호출이 2회가 되고, LLM 이 실패하면 자격 확인까지 사라진다.

    두 종류를 함께 모은다.
        `_확인필요_자격`            우리가 판정할 수 없는 사실(교육 이수·선정 이력 등)
        `_확인필요_자격_매출미달시`   택1요건 중 우리가 판정하는 갈래(매출)를 못 넘겼을 때
    """
    found: dict[str, dict] = {}
    for cand in pool:
        for key in ("_확인필요_자격", "_확인필요_자격_매출미달시"):
            text = cand.get(key)
            if not text:
                continue
            if key.endswith("매출미달시") and optimizer._meets_revenue_rule(cand, None):
                continue  # 매출 요건을 이미 충족했으면 물을 필요가 없다
            entry = found.setdefault(
                text,
                {
                    "key": confirm_key(text),
                    "question": text,
                    "ids": [],
                    # '지금 하면 되는 일'(교육 이수 등)은 '아니오'여도 제외하지 않는다.
                    "now_possible": bool(cand.get("_확인필요_자격_지금가능")),
                    **_confirm_options(cand, key, ctx),
                },
            )
            entry["ids"].append(cand.candidate_id)
            entry["now_possible"] = entry["now_possible"] or bool(
                cand.get("_확인필요_자격_지금가능")
            )
    return list(found.values())


def _confirm_options(cand, field: str, ctx) -> dict:
    """확인 질문의 선택지와 **통과로 치는 답**.

    기본은 예/아니오/모름이고 '예'만 통과다. 택1요건처럼 갈래가 여럿이면 정책
    데이터의 `_확인필요_자격_선택지` 를 그대로 쓴다 — 예/아니오로 받으면 **어느
    갈래인지 알 수 없다.** 제외로 치는 답(`_확인필요_자격_제외선택지`, 보통 '해당
    없음')을 빼고 나머지가 통과이며, 답하지 않은 것도 제외다(기존 규칙과 같다).

    ⑤에서 채무조정 이행 중이 **아니라고** 답했으면 그 갈래를 선택지에서 뺀다.
    방금 아니라고 답한 것을 다시 물으면 어긋난 답이 생길 자리만 만든다.
    반대로 '예'라고 답했어도 그 갈래를 지우지는 않는다 — 그쪽은 회차 조건까지
    요구해서 ⑤의 답보다 좁다(`UserContext.in_debt_restructuring` 참조).
    """
    options = cand.get(f"{field}_선택지")
    if not options:
        return {"options": (CONFIRM_YES, CONFIRM_NO, CONFIRM_UNSURE),
                "pass_answers": frozenset({CONFIRM_YES})}
    excluded = cand.get(f"{field}_제외선택지")
    if ctx is not None and not getattr(ctx, "in_debt_restructuring", False):
        options = [o for o in options if "채무조정 이행 중" not in o]
    return {"options": tuple(options),
            "pass_answers": frozenset(o for o in options if o != excluded)}


def excluded_by_confirm(questions: list[dict], answers: dict) -> set[str]:
    """확인 질문의 답으로 **제외되는 후보 id**.

    기획서 5.2.5 ⑥ — **'아니오'와 '잘 모르겠어요'는 똑같이 제외한다.** 확인하지 못한
    자격을 있다고 가정하면 신청했다가 거절당한다. '모름'을 통과시키면 안 된다.
    답하지 않은 질문도 아직 확인되지 않은 것이라 같이 제외한다.

    예외는 `_확인필요_자격_지금가능: true` — 지금 못 했어도 신청 전에 할 수 있으므로
    남기고, `optimizer.prerequisites()` 가 '신청 전에 이수하셔야 합니다'로 안내한다.
    """
    dropped: set[str] = set()
    for item in questions:
        if item["now_possible"]:
            continue
        # 통과로 치는 답은 질문마다 다르다. 예/아니오 질문은 '예' 하나뿐이고,
        # 택1 객관식은 '해당 없음'을 뺀 나머지가 전부 통과다(_confirm_options).
        if answers.get(item["key"]) not in item.get("pass_answers", {CONFIRM_YES}):
            dropped.update(item["ids"])
    return dropped


def run_pipeline(
    db: PolicyDB,
    profile: UserProfile,
    ctx: UserContext,
    target_inflow: int | None,
    target_months: int | None,
    drop_ids: set[str] | None = None,
):
    """CLAUDE.md의 파이프라인을 입력 경로에 맞게 호출한다.

    Args:
        drop_ids: 확인 질문에 '아니오/모름'으로 답해 제외된 후보 id.

    Returns:
        (groups, reports, evals, extras, baseline_survival, goal_satisfied, resolve_info)
    """
    pool = eligible_candidates(db, profile, ctx)
    if drop_ids:
        pool = [c for c in pool if c.candidate_id not in drop_ids]
    baseline_survival = simulate(profile, []).survival_months
    resolve_info = None

    if not pool:
        return [], {}, [], [], baseline_survival, True, None

    if target_inflow is not None and target_months is not None:
        # 둘 다 입력 — 금액 기준으로 만들고 목표 미달이면 개월수 기준으로
        combos, satisfied, need_amount, best_months = resolve_by_both(
            db, pool, profile, ctx, target_inflow, target_months
        )
        resolve_info = dict(satisfied=satisfied, need_amount=need_amount, best_months=best_months)
        evals = evaluate(profile, combos)
        evals, goal_satisfied = filter_by_goal(evals, target_months=target_months)

    elif target_inflow is not None:
        # 금액만 — 보조금 먼저, 부족분을 융자로
        grants = grant_amounts(db, pool, ctx)
        zero = {c.candidate_id: 0 for c in pool if c.is_loan}
        base = build_combos(db, pool, {**zero, **fixed_amounts(profile, pool), **grants})
        combos = rebalance_to_inflow(base, target_inflow)
        evals = evaluate(profile, combos)
        evals, goal_satisfied = filter_by_goal(evals, target_inflow=target_inflow)

    else:
        # 목표 개월수만(또는 미지정 시 기본 6) — 필요액 역산
        months = target_months or DEFAULT_TARGET_MONTHS
        amounts = default_loan_amounts(db, pool, profile, months)
        amounts.update(fixed_amounts(profile, pool))
        amounts.update(grant_amounts(db, pool, ctx))
        combos = build_combos(db, pool, amounts)
        combos = rebalance_to_months(combos, profile, months)
        evals = evaluate(profile, combos)
        evals, goal_satisfied = filter_by_goal(evals, target_months=months)

    reports = detect_all(profile, evals, ctx.phase)
    groups = select(evals, reports)
    return groups, reports, evals, extra_candidates(pool), baseline_survival, goal_satisfied, resolve_info


# ══════════════════════════════════════════════════════════
# 미래 자격 파이프라인 (기획서 5.2.3)
# ══════════════════════════════════════════════════════════


def run_future_pipeline(
    db: PolicyDB,
    profile: UserProfile,
    ctx: UserContext,
    summary: pos.PosSummary | None,
    monthly_revenue: list[int],
    shown_groups: list[CandidateGroup],
    target_months: int | None = None,
    drop_ids: set[str] | None = None,
    confirm_answers: dict[str, str] | None = None,
    from_ensemble: bool = False,
):
    """미래 자격 조합. **본 결과와 완전히 분리된 두 번째 파이프라인이다.**

    Returns:
        (groups, reports, months) — 레이어 1 그룹 · 함정 리포트 · {후보id: N개월}
        미래 후보가 없거나 공개 조건을 못 넘으면 ([], {}, {}).

    본 결과의 파레토·함정·순위를 건드리지 않는 것이 핵심이다. 미래 후보를 현재
    후보와 같은 풀에 넣으면 순위가 흔들리는데, 그건 "지금 신청할 수 있는 것들"의
    비교를 "몇 달 뒤에나 되는 것"이 흔드는 셈이라 맞지 않는다.

    Args:
        shown_groups: **화면에 그려진 카드 셋**(LLM 이 고른 것). 파레토 순서의 앞
            셋이 아니다 — `render_results` 가 `advice.picks` 로 그리므로 둘이 다를
            수 있고, 사용자가 보고 있는 것과 비교해야 "기다릴 값어치"가 성립한다.

    **공개 조건** — 그 카드 셋 **모두에게** 두 축에서 지배당하면 보여주지 않는다.
    기다리는 비용이 있는데 얻는 게 없으면 안내가 아니라 소음이다. 판정은 이미 있는
    `optimizer._dominates` 를 그대로 쓴다 — 새 기준을 만들지 않는다.

    간편 모드(POS 없음)에서는 `summary` 가 None 이라 곧바로 빈 결과를 준다.

    **`from_ensemble` 이 False 면 아무것도 내지 않는다.** 미래 자격은 앙상블 예측선
    위에서만 말할 수 있다. POS 가 12개월 미만이면 `forecast.MIN_MONTHS` 에 걸려
    추세선으로 돌아가는데(정밀 모드는 그대로 유지된다), 추세선은 최근 3개월 증감률을
    36개월 등비로 미는 것이라 반등 계열에서 3년 뒤 33배를 그린다(실측). 그런 선으로
    "N개월 뒤 자격이 생깁니다"를 말하면 "예측선대로 움직이면"이라는 단서로도 가려지지
    않는다. 기본값을 False 로 두어 **호출부가 앙상블이었다고 밝힐 때만** 계산한다.
    """
    if summary is None or not monthly_revenue or not from_ensemble:
        return [], {}, {}

    past = [float(v) for v in summary.monthly_revenue.values]
    months_by_id = future.future_candidates(
        db, profile, ctx, past, monthly_revenue, ctx.is_closed
    )
    if drop_ids:
        months_by_id = {k: v for k, v in months_by_id.items() if k not in drop_ids}
    months_by_id = _drop_unconfirmed_future(db, months_by_id, confirm_answers, ctx)
    if not months_by_id:
        return [], {}, {}

    by_id = {c.candidate_id: c for c in db.candidates}
    # 미래 후보는 그 시점으로 미룬 사본을 쓴다. 함께 묶이는 현재 후보는 그대로다 —
    # 지금 신청할 수 있으니 자기 시점을 지킨다.
    pool = [future.delayed(by_id[cid], n) for cid, n in months_by_id.items()]

    # 조합에 함께 넣을 현재 후보. 융자는 뺀다 — 조합당 융자가 1개라 현재 융자를
    # 넣으면 미래 후보가 빠진 조합(= 본 결과와 같은 것)이 다시 만들어진다.
    now_pool = eligible_candidates(db, profile, ctx)
    if drop_ids:
        now_pool = [c for c in now_pool if c.candidate_id not in drop_ids]
    pool += [c for c in now_pool if not c.is_loan and not c.is_extra]

    latest = max(months_by_id.values())
    pool = _with_extension_timings(pool, latest)

    # 목표는 본 결과와 같은 값을 쓴다. 세션 상태를 읽지 않는다 — 그러면 테스트나
    # 다른 호출부에서 조용히 기본값(6)으로 떨어져 융자 필요액이 0이 된다.
    amounts = default_loan_amounts(
        db, pool, profile, target_months or DEFAULT_TARGET_MONTHS)
    amounts.update(fixed_amounts(profile, pool))
    amounts.update(grant_amounts(db, pool, ctx))
    combos = build_combos(db, pool, amounts)

    future_ids = set(months_by_id)
    combos = [c for c in combos
              if any(i.candidate.candidate_id in future_ids for i in c.items)]
    if not combos:
        return [], {}, {}

    evals = evaluate(profile, combos)
    reports = detect_all(profile, evals, ctx.phase)
    groups = [g for g in select(evals, reports) if g.layer == 1]
    groups = [g for g in groups if _worth_waiting_for(g, shown_groups)]
    return groups, reports, months_by_id


def _drop_unconfirmed_future(
    db: PolicyDB, months_by_id: dict[str, int], answers: dict[str, str] | None,
    ctx: UserContext | None = None,
) -> dict[str, int]:
    """자격 확인 질문에 '예'를 못 받은 미래 후보를 뺀다.

    **순서 때문에 생기는 구멍을 막는다.** ⑥ 추가 확인은 **본 결과 후보**를 보고 질문을
    만드는데, 미래 후보는 결과 화면에서 처음 계산된다. 미래 후보에만 붙은 질문은
    물어볼 기회가 없어 답이 없는 채로 통과한다.

    `_확인필요_자격` 은 "판정할 수 없어 통과시켜 둔 것"이지 "자격이 있다"가 아니다.
    답을 못 받았으면 **'아니오'·'모름'과 똑같이 제외한다**(기획서 5.2.5 ⑥). 판정은
    `excluded_by_confirm` 을 그대로 쓴다 — `_확인필요_자격_지금가능: true` 예외
    (신청 전에 할 수 있는 일)도 함께 따라온다.

    지금 데모에서는 걸리는 후보가 없다. A 의 미래 후보(긴급경영안정자금)에는 확인
    질문이 없기 때문이다. 다만 재도전특별자금 세 후보에는 모두 붙어 있어, **회복불가
    사용자에게는 실제로 열리는 구멍이다.**
    """
    if not months_by_id:
        return months_by_id
    by_id = {c.candidate_id: c for c in db.candidates}
    cands = [by_id[cid] for cid in months_by_id if cid in by_id]
    blocked = excluded_by_confirm(confirm_questions(cands, ctx), answers or {})
    return {cid: n for cid, n in months_by_id.items() if cid not in blocked}


def _with_extension_timings(pool, latest_month: int):
    """상환연장이 풀에 있으면 **적용 시점을 여럿** 만들어 넣는다.

    실측에서 상환연장은 시점에 따라 두 축이 갈렸다(조건 108개 중 25개). 생존이 짧으면
    늦춘 시점에 도달하지 못해 "연장을 안 한 것"과 같아지는데, 그때 부채가 오히려 작게
    잡히기도 한다. 어느 시점이 나은지 정량으로 못 고르므로 모두 후보로 넣고 두 축이
    순위를 매기게 한다(기획서 5.5 "정량으로 고를 수 없으면 고르지 않는다").

    범위는 **가장 이른 시점 ~ 미래 융자 시점**이다. 임의 상한이 아니라 조합 안의
    사건이 정한다 — 일반 조합에서는 상환연장을 지금 신청하는 것으로 고정하고 있고,
    늦출 이유가 생기는 것은 미래 융자가 t=N 에 오기 때문이라 범위도 거기까지다.
    """
    out, extras = [], []
    for cand in pool:
        out.append(cand)
        if cand.effect != "상환연장":
            continue
        base = int(cand.get("적용시점_개월", 0) or 0)
        for extra in range(1, latest_month + 1):
            extras.append(future.delayed(cand, extra))
    return out + extras


def _months_until(member: Evaluated, months_by_id: dict[str, int]) -> int:
    """이 조합을 신청할 수 있게 되는 시점. 조합 안 미래 후보들의 **가장 늦은** 달이다.

    조합에 미래 후보가 둘 이상이면 전부 자격이 생겨야 그 조합을 신청할 수 있다.
    미래 후보가 없으면(있을 수 없지만) 0 을 준다.
    """
    months = [months_by_id[i.candidate.candidate_id]
              for i in member.combo.items
              if i.candidate.candidate_id in months_by_id]
    return max(months) if months else 0


def _worth_waiting_for(group: CandidateGroup, shown_groups) -> bool:
    """**화면에 그려진 카드 셋 모두에게** 지배당하면 보여주지 않는다.

    비교 대상은 파레토 순서의 앞 셋이 아니라 **LLM 이 고른 셋**이다. LLM 은 그룹
    5개 중 앞 셋이 아닌 것을 고를 수 있어(`render_results` 의 `advice.picks`) 둘이
    다르다. 사용자가 보고 있는 것과 비교해야 "기다릴 값어치가 있나"가 성립한다.

    하나라도 안 지배당하면 남긴다 — 두 축 중 하나에서라도 나은 데가 있다는 뜻이다.
    "2개월 기다렸는데 지금보다 나쁘다"를 보여줄 이유가 없다.
    """
    shown = list(shown_groups)[:3]
    if not shown:
        return True
    return not all(_dominates(g, group) for g in shown)


# ══════════════════════════════════════════════════════════
# 결과 렌더링 (카드·그래프·슬라이더) — UI 레이어와 무관하게 재사용
# ══════════════════════════════════════════════════════════


def _cliff_marker(evaluated: Evaluated, report) -> tuple[int, str] | None:
    """그래프에 표시할 절벽(급락) 시점과 주석. **함정 판정 결과에서만** 가져온다.

    그래프 기울기를 보고 새로 찾지 않는다 — metrics.py 가 이미 판정한 CLIFF 함정의
    시점을 그대로 쓴다. 함정이 없으면 None(=아무것도 그리지 않는다).
    """
    if report is None or not report.traps:
        return None
    cliff = next((t for t in report.traps if t.kind == TrapKind.CLIFF and t.at), None)
    if cliff is None:
        return None
    at = cliff.at
    # **금액을 쓰지 않는다.** 막대 높이가 이미 "여기서 꺾인다"를 보여주고, 정확한 금액은
    # 바로 아래 함정 경고가 시점과 함께 말한다. 여기서 원금만 떼어 적으면(예전 동작)
    # 경고문의 총액과 어긋나 보인다 — 같은 회차의 다른 구성인데 라벨이 그걸 안 밝혔다.
    is_bullet = any(
        it.candidate.is_loan and it.candidate.get("상환방식") == "만기일시"
        for it in evaluated.combo.items
    )
    label = "만기 일시상환" if is_bullet else "상환액 증가"
    return at, f"{at}개월차부터 {label}"


def render_flow_chart(
    evaluated: Evaluated, key: str, report=None, baseline_survival: int | None = None
) -> None:
    """36개월 현금흐름 **막대**그래프. 잔고(만원)만 그린다.

    - 꺾은선 대신 월별 막대라 24개월차 절벽 같은 급락이 눈에 잘 들어온다.
    - x축 0~horizon 고정, y축 하한 0(balance_at 은 음수가 없다). shortfall_at 은
      역산용 가상 궤적이라 그리지 않는다(오해 방지).
    - 값 라벨은 시작·최고점 2곳만(37개 전부 붙이면 안 읽힌다).

    **세 시점을 성격에 따라 다르게 표시한다.** 세로선을 셋 그으면 복잡하다.
        결과   소진 시점        빨강 세로 점선 — 가장 눈에 띄게
        기준   baseline 소진    회색 세로 점선(옅게) — 비교 기준이라 튀면 안 된다
        원인   원금 상환 시작   **세로선 없이** 막대 색이 바뀌는 것으로 보이고,
                               그 자리에만 작은 ▼ 마커를 얹는다
    라벨 셋은 위쪽에 두되 세로 위치를 나눠 서로·x축과 겹치지 않게 한다.
    금액은 쓰지 않는다 — 정확한 값은 아래 함정 경고가 시점과 함께 말한다.
    """
    result = evaluated.result
    months = list(range(len(result.balance_at)))  # 0 .. horizon
    balance = [b / MAN for b in result.balance_at]

    marker = _cliff_marker(evaluated, report)
    cliff_at = marker[0] if marker is not None else None

    # 막대는 **전부 같은 파란색**이다. 상환 시작 이후를 색이나 빗금으로 가르지 않는다 —
    # 옅은 색은 가장 위험한 구간을 흐리게 만들었고, 빗금은 산만했다. 그 구간이 어디부터인지는
    # ▼ 마커와 라벨이 말한다(잔고라는 같은 값을 같은 색으로 두는 게 읽기 쉽다).
    bar_color = "#2b6cb0"

    # 값 라벨: 시작(t=0)·최고점 2곳만. 절벽 달은 ▼ 마커에 자리를 내준다 —
    # 숫자가 있으면 마커가 그 위로 밀려 어느 막대를 가리키는지 안 보인다.
    peak_idx = max(range(len(balance)), key=lambda i: balance[i])
    label_idx = {0, peak_idx}
    text = [f"{balance[i]:,.0f}" if i in label_idx else "" for i in months]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=months, y=balance,
            marker_color=bar_color,
            text=text, textposition="outside", cliponaxis=False,
            textfont=dict(size=14),  # 값 라벨을 본문 크기로 키운다
            hovertemplate="%{x}개월차 · %{y:,.0f}만원<extra></extra>",  # 툴팁도 만원 정수
        )
    )
    y_top = max(balance) if any(b > 0 for b in balance) else 1
    fig.update_xaxes(range=[-0.5, SIM_MONTHS + 0.5], dtick=6, title_text="개월차")
    # (만원)은 우상단 '단위: 만원'이 대신한다. SI 접두어(40k) 없이 40,000 로.
    fig.update_yaxes(
        # 두 줄짜리 라벨(지원 없었다면 N개월 · N개월차부터 상환액 증가)이 들어갈
        # 위 여백을 확보한다. 좁으면 라벨이 값 라벨·제목과 겹친다.
        range=[0, y_top * 1.46], title_text="현금잔고",
        tickformat=",d", exponentformat="none",
    )

    # ── 기준: 지원이 없었다면 언제 끝났는지 (옅은 회색 점선) ──
    # 결과(소진)와 같은 자리면 그리지 않는다 — 겹쳐서 둘 다 안 읽힌다.
    if (
        baseline_survival is not None
        and baseline_survival > 0
        and baseline_survival != result.survival_months
    ):
        fig.add_vline(
            x=baseline_survival, line=dict(color="#a0aec0", width=1.5, dash="dot")
        )
        fig.add_annotation(
            x=baseline_survival, y=y_top * 1.30, yref="y",
            text=f"지원 없었다면<br>{baseline_survival}개월",  # 두 줄로 — 옆 라벨과 붙지 않게
            showarrow=False, font=dict(size=11, color="#718096"), align="center",
            xanchor="left" if baseline_survival < SIM_MONTHS * 0.7 else "right",
        )

    # ── 결과: 실제 소진 시점 (빨강 점선). 끝까지 버티면 표시하지 않는다 ──
    if not result.survives_full_term:
        fig.add_vline(
            x=result.survival_months, line=dict(color="crimson", width=2, dash="dash")
        )
        fig.add_annotation(
            x=result.survival_months, y=y_top * 1.07, yref="y",
            text=f"{result.survival_months}개월차 소진",
            showarrow=False, font=dict(size=12, color="crimson"),
            xanchor="left" if result.survival_months < SIM_MONTHS * 0.7 else "right",
        )

    # ── 원인: 원금 상환 시작 — 세로선 없이 막대 색 변화 + 그 자리에만 ▼ 마커 ──
    # 세로선 셋은 과하다. 이 시점부터 막대가 옅어지므로(위 bar_colors) 꺾임은 이미
    # 보이고, 마커는 "어디서부터인지"만 짚는다.
    if marker is not None:
        at, note = marker
        bar_top = balance[at] if at < len(balance) else 0
        # 라벨 → ▼ → 막대가 세로로 이어지게 붙인다. 띄우면 어느 막대를 가리키는지
        # 안 보인다. 절벽 달이 최고점과 겹쳐 값 라벨이 있을 때만 그만큼 비켜 준다.
        tri_shift = 30 if at in label_idx else 12
        fig.add_annotation(
            x=at, y=bar_top, yref="y",
            text="▼", showarrow=False, yshift=tri_shift,
            font=dict(size=22, color="#dd6b20"),
        )
        # 두 줄로 쪼개 가로 폭을 줄인다 — 한 줄이면 baseline 라벨과 겹친다.
        fig.add_annotation(
            x=at, y=bar_top, yref="y", text=note.replace("부터 ", "부터<br>"),
            showarrow=False, yshift=tri_shift + 30, align="center",
            font=dict(size=11, color="#dd6b20"),
        )

    # 우측 상단 구석에 단위 표기
    fig.add_annotation(
        xref="paper", yref="paper", x=1, y=1.14, showarrow=False,
        text="단위: 만원", font=dict(size=11, color="#888"), xanchor="right",
    )
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=54, b=10), showlegend=False, bargap=0.15
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_support_detail(evaluated: Evaluated) -> None:
    """지원 내역 — '무엇을 얼마나 받는지'. 둥근 테두리 박스로 감싼다.

    simulate가 만든 문장(result.notes)은 원 단위라, 표시 직전에 money_to_man 으로
    만원 단위로 바꾼다(계산은 그대로 원 단위, 화면 표기만 변환).

    '현금 유입'과 '첫 달 상환 부담'은 두지 않는다 — 유입은 아래 카드의 보조금(확정)·
    융자(조정)이 이미 나눠 보여주고, 첫 달 상환액은 상환 스케줄 표에 있다.
    감면액은 남긴다. 네 칸 어디에도 없는 값이다.
    """
    result = evaluated.result
    with st.container(border=True):
        st.markdown("**지원 내역**")
        for note in result.notes:
            st.markdown(f"- {money_to_man(note)}")
        if result.forgiven > 0:
            st.caption(f"부채 감소 {fmt_man(result.forgiven)} (추정치)")


def render_traps(report) -> None:
    """함정 경고.

    예전에는 아래에 '월 상환액 N만원 / 월 영업이익 …' 캡션을 함께 뒀는데 **뺐다.**
    그 값은 payment_now(=1회차)라 **거치 중이라 상환액이 가장 낮은 시점**이다.
    부담률을 그 시점으로 계산하면 실제보다 안전해 보이게 만든다 — 부담이 커지는 것은
    거치가 끝난 뒤다. 게다가 화면에 상환액이 시점을 안 밝힌 채 여러 번 나와, 같은 값을
    다르게 말하는 것처럼 읽혔다. 상환액은 **시점을 밝히는** 함정 경고와 상환 스케줄 표가
    맡는다.
    """
    for trap in report.traps:
        st.warning(f"⚠ {trap.message}")


def adjustable_loan(member: Evaluated):
    """이 조합에서 **슬라이더로 조정할 수 있는** 신규유입 융자. 없으면 None.

    조정 대상은 신규유입 융자 하나뿐이다(대환은 갈아탈 금액이 정해져 있다).
    한도와 하한이 같으면 움직일 폭이 없어 조정 대상이 아니다.
    """
    item = next(
        (
            it
            for it in member.combo.items
            if it.candidate.is_loan and it.candidate.effect == "신규유입"
        ),
        None,
    )
    if item is None:
        return None
    cand = item.candidate
    if int(cand.get("금액_한도") or 0) <= int(cand.get("금액_하한") or 0):
        return None
    return item


def restore_loan_amount(widget_key: str, default_man: int) -> None:
    """[↺ 추천 금액으로] 콜백 — 세션에 추천 금액을 **대입**한다.

    ⚠ `st.session_state.pop(widget_key)` 로 지우면 안 된다. 파이썬 쪽 계산은 추천
    금액으로 돌아가지만 브라우저의 눈금은 굴린 자리에 그대로 남아 화면에 두 값이
    다르게 뜬다. 슬라이더 proto 의 set_value 가 "눈금을 옮겨라"를 전달하는데,
    지우면 False 이고 대입해야 True 가 된다.

    ⚠ st.rerun() 도 쓰지 마라. 스크립트를 그 자리에서 끊으면 아래 카드들의 슬라이더가
    그 실행에 등록되지 않아 Streamlit 이 그 상태를 지운다(화면 전환 때 기대는 purge).
    카드 하나를 되돌렸을 뿐인데 아래 카드들이 전부 기본값으로 돌아간다.

    콜백은 위젯이 만들어지기 전에 도므로 위젯 key 에 대입해도 예외가 나지 않는다.
    """
    st.session_state[widget_key] = default_man


def render_loan_slider(member: Evaluated, profile: UserProfile, key: str) -> Evaluated:
    """융자 슬라이더(있을 때만) + **결과 카드 4칸(항상)**.

    슬라이더가 없다고 카드까지 사라지면 안 된다 — 두 축(생존·종료부채)은 모든 조합에
    있고, 유입 두 칸도 '없음'으로 적으면 되기 때문이다. 예전에는 조정할 융자가 없으면
    함수가 곧장 반환해 이 블록이 통째로 비었다(C 처럼 융자가 배제된 국면에서 전부).
    """
    loan_item = adjustable_loan(member)
    amount = 0
    if loan_item is not None:
        cand = loan_item.candidate
        ceiling = int(cand.get("금액_한도") or 0)
        floor = int(cand.get("금액_하한") or 0)
        # 신청 단위는 재원마다 다르다(소진공 100만·서울시 만). 후보 데이터에 이미 있다.
        unit = int(cand.get("신청금액_단위") or 1_000_000)
        # 슬라이더는 만원 단위로 보여주고, step 도 신청 단위(만원)로 잡는다.
        min_man, max_man = floor // MAN, ceiling // MAN
        step_man = max(1, unit // MAN)
        default_man = min(max(loan_item.effective_amount // MAN, min_man), max_man)
        # 라벨과 버튼을 한 줄에 놓고 슬라이더는 그 아래 100% 폭으로 둔다(폭이 곧 조작
        # 정밀도다). 라벨을 직접 그리므로 native 위젯 라벨과 미세하게 다를 수 있는데,
        # 하늘색 상자 안에 다른 위젯이 없어 나란히 비교될 일이 없다.
        head, tail = st.columns([0.7, 0.3])
        with head:
            st.markdown(
                "<div style='font-size:0.875rem;padding-top:8px;'>"
                f"융자 금액 조정 — {cand.policy_name}</div>",
                unsafe_allow_html=True,
            )
        with tail:
            st.button(
                "↺ 추천 금액으로", key=f"reset_{key}", width="stretch",
                on_click=restore_loan_amount, args=(key, int(default_man)),
            )
        # 세션에 값이 있으면 value= 를 넘기지 않는다. 둘을 같이 주면 Streamlit 이
        # "default value 로 만들었는데 Session State API 로도 값을 넣었다"고 경고하며
        # 스택까지 찍는다. 되돌리기가 세션에 대입하는 방식이라 굴릴 때마다 나온다.
        # value 만 갈리고 min·max·step·format 은 두 경로에서 같다.
        initial = {} if key in st.session_state else {"value": int(default_man)}
        picked_man = st.slider(  # transient-widget: 결과 카드 what-if. 단계 답이 아니라
            # 되돌아가면 초기화되는 게 맞다(ss.answers 를 안 탄다).
            f"융자 금액 조정 — {cand.policy_name}",  # 위에 직접 그려 화면에서는 감춘다
            min_value=min_man,
            max_value=max_man,
            step=step_man,
            format="%d만원",
            key=key,
            label_visibility="collapsed",
            **initial,
        )
        # 슬라이더 값을 실제 신청 가능 금액으로 스냅한다(단위 올림·금액_하한 처리).
        # 같은 규칙을 UI에서 새로 구현하지 않고 combos.snap_amount 를 그대로 통과시킨다.
        amount = snap_amount(cand, picked_man * MAN)
    else:
        # 조정할 수 없는 융자(한도=하한)라도 금액 자체는 카드에 적는다.
        fixed = next(
            (
                it
                for it in member.combo.items
                if it.candidate.is_loan and it.candidate.effect == "신규유입"
            ),
            None,
        )
        amount = fixed.effective_amount if fixed is not None else 0
        st.caption("이 조합에는 금액을 조정할 융자가 없어 슬라이더를 두지 않았습니다.")

    grant_sum = sum(
        it.effective_amount
        for it in member.combo.items
        if it.candidate.effect == "신규유입" and not it.candidate.is_loan
    )

    if loan_item is None or amount == loan_item.effective_amount:
        shown = member
    else:
        # 이 조합만 다시 계산한다. 순위(select)는 다시 돌리지 않는다.
        new_items = tuple(
            Applied(it.candidate, amount) if it is loan_item else it for it in member.combo.items
        )
        shown = Evaluated(combo=Combo(items=new_items), result=simulate(profile, new_items))

    # 슬라이더 결과 — 왼쪽 1개(유입 내역) · 오른쪽 2개(생존 · 종료부채)를 위아래로.
    # 숫자가 먼저 바뀌고 그 근거로 아래 그래프가 잇는 순서라 슬라이더와 그래프 사이에 둔다.
    # '조정 후' 제목은 두지 않는다 — 하늘색 상자 자체가 "조정 결과"를 뜻해 중복이다.
    #
    # 결과를 **2x2 격자**로. 왼쪽 열이 유입(보조금·융자), 오른쪽 열이 결과(생존·부채)다.
    # '총 유입 내역' 제목과 '+' 는 두지 않는다 — 라벨이 이미 무엇인지 말하고, '+' 는
    # 카드 사이에 놓으면 붕 뜬다. 왼쪽/오른쪽의 뜻은 배치가 전한다.
    #
    # st.columns 대신 **CSS grid 하나**로 그린다. 열을 나누면 양쪽 높이를 맞추느라
    # min-height 를 손으로 맞춰야 했고(자연 높이가 그 값을 넘으면 한쪽만 길어져
    # 그래프에 닿았다), Streamlit 내부 구조에도 기대게 된다. 격자는 네 칸 크기가
    # 구조적으로 같고, 카드 행 전체가 한 요소라 아래 여백도 좌우가 자동으로 같다.
    box = (
        f"background:#fff;border-radius:{_CARD_RADIUS};padding:14px 16px;"
        "text-align:center;box-sizing:border-box;display:flex;flex-direction:column;"
        "align-items:center;justify-content:center;min-height:76px;"
    )
    head = "color:#4a5568;font-size:0.9rem;line-height:1.4;"
    val = "font-size:1.05rem;font-weight:700;color:#1F2933;line-height:1.45;"

    def cell(label: str, value: str) -> str:
        return (
            f"<div style='{box}'>"
            f"<div style='{head}margin-bottom:4px;'>{label}</div>"
            f"<div style='{val}'>{value}</div></div>"
        )

    # 네 칸은 **항상** 그린다. 오른쪽 둘은 두 축이라 모든 조합에 있고, 왼쪽 둘도 없으면
    # '없음'으로 적는다. 칸을 빼면 격자에 구멍이 나거나(자동 배치라 아래 칸이 위로 올라와)
    # 왼쪽=유입 / 오른쪽=결과 대응이 깨진다.
    # '0원'이라고 쓰지 않는다 — "0원을 받는다"로 읽혀 조정할 수 있는 것처럼 보인다.
    # 실제로는 그 조합에 그 정책이 들어 있지 않다는 뜻이다.
    grant_text = fmt_man(grant_sum) if grant_sum > 0 else "없음"
    loan_text = fmt_man(amount) if amount > 0 else "없음"
    # '(조정)'은 슬라이더가 있을 때만 붙인다 — 조정할 수 없는데 그렇게 적으면 어긋난다.
    loan_label = "융자(조정)" if loan_item is not None else "융자"
    # 카드 행과 그래프 사이 여백을 **카드 사이 간격과 같은 값**으로 둔다(한 상수를 쓴다).
    # 행 전체가 한 요소라 좌우가 자동으로 같다.
    gap_px = 10
    st.markdown(
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:{gap_px}px;"
        f"margin-bottom:{gap_px}px;'>"
        + cell("보조금(확정)", grant_text)
        + cell("생존", months_label(shown.result.survival_months))
        + cell(loan_label, loan_text)
        + cell("종료 시점 부채", fmt_man(shown.result.debt_at_exit))
        + "</div>",
        unsafe_allow_html=True,
    )
    return shown


def schedule_rows(result) -> list[dict]:
    """상환 스케줄 행들(만원). **기존 부채 / 신규 상환으로 나눠** 각각 원금·이자·잔액.

    나누는 이유: 합계만 보이면 "1년 거치인데 왜 1회차부터 원금이 나가지?"에 답이 없다.
    그 원금은 기존 부채분이고 신규 상환은 거치 중이라 0이다. 나눠 놓으면 신규 원금이
    0 → 76 으로 바뀌는 자리가 그대로 거치 절벽의 근거가 된다.
    metrics.find_cliff_at 이 같은 전환을 보고 절벽 시점을 찾는다 — 표와 판정이 한 근거를
    쓴다.

    원금과 이자를 합치지 않는다. 합치면 7.6 → 83.8 로 숫자만 커져 거치가 안 보인다.
    합계 열(옛 '상환원리금')도 두지 않는다 — 옆을 더하면 나오는 값이라 정보가 없고,
    총액은 함정 경고가 시점과 함께 말한다.

    반올림은 cashflow.month_display 한 곳에서만 한다(표와 문장이 같은 값을 쓰게).

    자르는 규칙 둘 중 **먼저 걸리는 쪽**:
      · 자금이 소진되면 그 시점까지만 — 잔고가 마른 뒤의 상환은 일어나지 않는 일이고,
        그래프는 이미 거기서 끊기는데 표만 계속되면 어긋나 보인다.
      · 끝에서부터 전부 0인 행은 잘라낸다(상환이 끝난 뒤). 거치 중 원금만 0인 중간
        행은 다른 열이 0이 아니라 남는다.
    """
    last = len(result.payment_during)
    if not result.survives_full_term:
        last = min(last, result.survival_months)
    rows = []
    for m in range(1, last + 1):
        cells = month_display(result, m)
        rows.append({
            "회차": m,
            "원금(기존)": cells.existing_principal,
            "이자(기존)": cells.existing_interest,
            "잔액(기존)": cells.existing_debt,
            "원금(신규)": cells.new_principal,
            "이자(신규)": cells.new_interest,
            "잔액(신규)": cells.new_debt,
        })
    money_cols = [c for c in (rows[0] if rows else {}) if c != "회차"]
    while rows and all(rows[-1][c] == 0 for c in money_cols):
        rows.pop()
    return rows


def render_schedule_table(result) -> None:
    """상환 스케줄을 **2단 헤더** HTML 표로. 기존/신규를 먼저 가르고 그 안에서 원금·이자·잔액.

    st.dataframe 은 2단 헤더를 못 그린다. 열 이름을 '원금(기존)'처럼 풀어 쓰면 7열이
    폭을 넘겨서, 상단에 그룹 헤더를 두고 아래는 '원금·이자·잔액'만 쓴다.
    그룹 사이에는 진한 세로 구분선을 넣되, **색·선만으로 뜻을 전하지 않게** 그룹 이름
    텍스트를 반드시 함께 둔다.
    """
    rows = schedule_rows(result)
    if not rows:
        st.caption("상환 일정이 없습니다.")
        return
    div = "border-left:2px solid #4a5568;"  # 기존/신규 그룹 경계
    thin = "1px solid #cbd5e0"
    # 헤더 글자는 전부 가운데 정렬(숫자 칸만 오른쪽 정렬)
    center = "text-align:center;"
    head = (
        "<thead>"
        f"<tr style='background:#eef1f5;'>"
        f"<th rowspan='2' style='border:{thin};padding:6px 8px;{center}'>회차</th>"
        # '신규 융자'가 아니라 **'신규 상환'** 이다. 이 열은 융자만이 아니라 조합 때문에
        # 생기거나 달라진 상환 전체를 담는다(대환·채무조정·이자조정·상환연장 포함).
        # 왼쪽은 '기존 상환'으로 대칭을 맞추지 않는다 — ④ 단계에서 사용자가 입력한
        # '기존 부채'와 같은 말이어야 연결이 끊기지 않는다.
        f"<th colspan='3' style='border:{thin};{div}padding:6px 8px;{center}'>기존 부채</th>"
        f"<th colspan='3' style='border:{thin};{div}padding:6px 8px;{center}'>신규 상환</th></tr>"
        f"<tr style='background:#f6f8fa;'>"
        + "".join(
            f"<th style='border:{thin};{div if i % 3 == 0 else ''}padding:5px 8px;{center}'>{name}</th>"
            for i, name in enumerate(("원금", "이자", "잔액", "원금", "이자", "잔액"))
        )
        + "</tr></thead>"
    )
    body = "".join(
        "<tr>"
        + f"<td style='border:{thin};padding:4px 8px;text-align:center;'>{r['회차']}</td>"
        + "".join(
            f"<td style='border:{thin};{div if i % 3 == 0 else ''}padding:4px 8px;"
            f"text-align:right;'>{r[key]:,}</td>"
            for i, key in enumerate(
                ("원금(기존)", "이자(기존)", "잔액(기존)", "원금(신규)", "이자(신규)", "잔액(신규)")
            )
        )
        + "</tr>"
        for r in rows
    )
    st.markdown(
        "<div style='max-height:320px;overflow:auto;'>"
        "<table style='width:100%;border-collapse:collapse;font-size:0.85rem;'>"
        + head + "<tbody>" + body + "</tbody></table></div>",
        unsafe_allow_html=True,
    )
    if not result.survives_full_term and len(rows) >= result.survival_months:
        st.caption(
            f"{result.survival_months}개월차에 자금이 소진되어 이후는 표시하지 않습니다."
        )


def render_group_card(
    rank: int | None,
    group: CandidateGroup,
    member: Evaluated,
    reports: dict,
    profile: UserProfile,
    ctx: UserContext,
    baseline_survival: int,
    expanded: bool,
    pick: agent.Pick | None = None,
    future_months: int | None = None,
) -> None:
    """조합 하나를 카드로 그린다.

    **그룹이 아니라 조합(member)을 받는다.** 그룹의 대표를 정해 그리던 옛 구조에서는
    LLM 이 대표 아닌 멤버를 고르면 그 pick 이 갈 자리가 없어 통째로 사라졌다.
    같은 좌표라도 상환 방식이 다르면 그래프·상환 스케줄·총 원리금·함정 판정이 갈리므로
    LLM 이 고른 그 조합을 그대로 그려야 한다.

    rank — LLM 이 고른 순위. 폴백이면 None 이고 제목이 '후보'가 된다. 폴백의 순서는
    계산 순서일 뿐이라 순위로 쓰지 않는다.

    future_months — 미래 자격 조합이면 **몇 개월 뒤 신청할 수 있는지**. 그때만
    ⏳ 제목과 조건부 안내를 붙이고 순위를 매기지 않는다(rank 는 None 으로 온다).

    위젯 key 도 `future_card` 네임스페이스로 가른다. label 이 겹칠 일은 지금 구조상
    없다 — `run_future_pipeline` 이 미래 후보가 든 조합만 남기고, 미래 후보는 정의상
    현재 자격이 없어 본 결과 조합에 들어갈 수 없다. 다만 그 보장이 두 단계 추론에
    기대고 있어, key 를 갈라 한 단계로 만든다.
    """
    trap_icon = " ⚠" if group.has_trap else ""
    # 제목: N위 │ 정책명(파랑) │ 성과.
    # st.expander 라벨의 마크다운 색상 지시자(:blue[…])는 Streamlit 1.62 문서상 지원되나
    # 렌더링은 브라우저에서 확인해야 한다. 안 먹으면 색 대신 다른 방식으로 바꾼다.
    perf = f"생존 {months_label(group.survival_months)} · 종료 시점 부채 {fmt_man(group.debt_at_exit)}"
    if future_months is not None:
        head = f"**⏳ {future_months}개월 뒤**"
    else:
        head = f"**{rank}위**" if rank is not None else "**후보**"
    title = f"{head}  │  :blue[{member.label}]{trap_icon}  │  {perf}"
    # key 를 카드 번호(idx)가 아니라 **조합 이름**으로 잡는다. LLM 이 순서를 바꾸면
    # (재시도 등) idx 기준 key 는 슬라이더 값이 엉뚱한 조합에 붙는다.
    card_id = stable_key(
        "future_card" if future_months is not None else "card", member.label
    )
    # 접힘 상자에도 key 를 준다. key 가 없으면 Streamlit 이 제목·위치로 정체성을
    # 만들어서, 앞쪽 요소가 하나 늘거나 줄면 상자가 새로 그려지며 펼침 상태가 풀린다.
    with st.expander(title, expanded=expanded, key=f"exp_{card_id}"):
        if future_months is not None:
            # 조건부로 쓴다. 두 가지를 밝혀야 정직하다 —
            #   ① 매출이 예측선대로 움직여야 성립한다
            #   ② 자격은 **그 시점의 진단**으로 계산했다(국면이 바뀔 수 있다)
            st.info(
                f"지금은 자격이 안 되지만, 매출이 예측선대로 움직이면 "
                f"**{future_months}개월 뒤** 신청 요건을 충족합니다. "
                f"그 시점의 진단 결과로 자격을 계산했습니다."
            )
            st.caption(
                "⚠ 예측이므로 실제와 다를 수 있습니다. 매출이 예측선과 달라지거나 "
                "그 사이 상황이 바뀌면 자격도 달라집니다. 신청 전에 다시 진단해 보세요."
            )
        # '레이어 N'(내부 용어) 을 지운 자리에 **LLM 이 고른 이유**가 들어간다.
        # axis_focus 는 앞에 배지로 — 어느 축을 보고 골랐는지가 한눈에 들어와야 한다.
        if pick is not None and pick.why:
            badge = (
                f"<span style='background:#e6effb;color:#2a5a92;border-radius:6px;"
                f"padding:2px 8px;font-size:0.82rem;margin-right:8px;'>"
                f"{pick.axis_focus} 중시</span>"
                if pick.axis_focus else ""
            )
            st.markdown(
                f"<div style='margin:2px 0 10px 0;color:#2d3748;'>{badge}{pick.why}</div>",
                unsafe_allow_html=True,
            )

        # '같은 성능의 다른 조합' 줄은 두지 않는다 — 아래 '추천에서 배제한 조합과 이유'가
        # 같은 내용을 이유와 함께 다루고, 카드 안에 긴 정책명이 두 줄씩 반복돼 읽기 어렵다.

        # 지원 전후 대비 — 이 서비스가 하는 일을 한 줄로. 카드마다 오른쪽 값이 다르다
        # (조합이 다르므로). 왼쪽은 무지원 기준이라 모든 카드에서 같다.
        # 막대그래프를 따로 넣지 않는다 — 바로 아래 현금흐름 그래프가 같은 얘기를 한다.
        # 진단 화면의 '진단 국면' 박스와 같은 형태(흰 바탕 + 짙은 파랑 테두리 + 같은 곡률).
        # 폭은 글자에 맞춘다(inline-block) — 카드 폭으로 늘리면 상자만 커 보인다.
        # 오른쪽 값만 굵게 하지 않는다. 한쪽만 진하면 그 숫자만 떠 보인다.
        st.markdown(
            "<div style='text-align:center;margin:2px 0 10px 0;'>"
            f"<span style='display:inline-block;border:2px solid #2b6cb0;"
            f"border-radius:{_CARD_RADIUS};padding:8px 16px;color:#1F2933;'>"
            f"지원 받기 전 생존 개월: {baseline_survival}"
            "&nbsp;&nbsp;→&nbsp;&nbsp;"
            f"지원 후 생존 개월: {group.survival_months}</span></div>",
            unsafe_allow_html=True,
        )

        render_support_detail(member)

        # ── 슬라이더로 다시 계산되는 영역 ──
        # 위쪽(대비 줄·지원 내역)은 **추천된 원래 조합** 기준이라 슬라이더를 움직여도
        # 바뀌지 않는다. 그 경계가 화면에 안 보이면 어디까지가 조정 결과인지 알 수 없어,
        # 이 안쪽만 연한 하늘색 상자로 묶는다(.st-key-adjust_*).
        # 신청처·자격 확인 문구는 상자 밖이다 — 슬라이더로 바뀌지 않는다.
        with st.container(key=f"adjust_{card_id}"):
            # 안내는 **모든 카드에** 같은 자리로. 카드를 펼치는 순서가 정해져 있지
            # 않아 첫 카드에만 두면 후보 3만 펼친 사람은 이 설명을 못 본다.
            if adjustable_loan(member) is not None:
                # 만져 보라고 만든 기능이라 권유하는 어투로 쓴다.
                # 뒷문장은 남긴다 — 슬라이더로 바뀌는 부분(하늘색 상자 안)과 그렇지
                # 않은 부분(위쪽 지원 전후 대비)을 명확히 한다.
                st.caption(
                    "슬라이더로 금액을 바꿔보세요. **하늘색 상자 안**이 다시 계산됩니다. "
                    "위쪽 지원 전후 대비는 원래 금액 기준입니다."
                )
            shown = render_loan_slider(member, profile, key=f"loan_{card_id}")

            # 함정 판정을 먼저 구한다(그래프의 절벽 마커가 이 리포트를 쓴다).
            # 슬라이더로 조정했으면 조정본으로 다시 판정한다.
            if shown is member:
                report = reports[member.label]
            else:
                report = detect_traps(profile, shown, ctx.phase)

            render_flow_chart(
                shown, key=f"chart_{card_id}", report=report,
                baseline_survival=baseline_survival,
            )

            # 함정 경고를 **스케줄 표보다 먼저** — 경고를 읽고 근거를 펼치는 순서가 자연스럽다
            render_traps(report)
            # LLM 이 푼 말은 원문 경고 **바로 아래**에 둔다. 숫자가 든 원문을 지우지
            # 않는 이유는 그게 근거이기 때문이고, 풀이는 그 옆에 있어야 이어 읽힌다.
            if pick is not None and pick.trap_note and report.traps:
                st.caption(f"↳ {pick.trap_note}")

            # 창 밖 만기 일시상환 — **함정 경고 바로 아래, 같은 하늘색 상자 안**에 둔다.
            # 예전에는 컨테이너 밖(8칸 들여쓰기)에 있어 상자 밖으로 튀어나왔다.
            # 본 카드에서는 드러나지 않았다 — 데모 아홉 장 중 만기 일시상환이 있는
            # 조합이 하나도 없었기 때문이다. 미래 카드(만기 30% 일시상환)에서 처음
            # 보였다. 같은 성격의 경고 둘이 다른 자리에 뜨면 안 된다.
            if shown.result.balloon_after_horizon > 0:
                st.warning(
                    f"⚠ {shown.result.balloon_due_at}개월차에 원금 "
                    f"{fmt_man(shown.result.balloon_after_horizon)}이 일시상환으로 남습니다 "
                    f"(36개월 시뮬레이션 밖)"
                )

            # 상환 스케줄 표 — 그래프와 같은 시뮬레이션 결과(shown)를 쓴다.
            # 회차 수는 표와 같은 함수에서 세어 제목과 표가 어긋나지 않게 한다
            # (슬라이더로 금액이 바뀌면 소진 시점이 움직여 표 길이도 따라 바뀐다).
            # 열 이름에서 '(월)' 을 뺐으므로 단위 표기에 '월'을 함께 밝힌다.
            rows = len(schedule_rows(shown.result))
            title = (
                f"상환 스케줄 보기 (총 {rows}회차 · 월 단위 금액, 만원)" if rows
                else "상환 스케줄 보기 (월 단위 금액, 만원)"
            )
            # 제목에 회차 수가 들어 있어 슬라이더로 금액을 바꾸면 제목이 바뀐다.
            # key 가 없으면 그때마다 상자가 새것으로 취급돼 펼쳐둔 것이 접힌다.
            with st.expander(title, key=f"sched_{card_id}"):
                # 옛 문구('대출잔액은 기존 부채와 신규 융자를 합친 값입니다')는 지웠다 —
                # 이제 잔액도 기존/신규로 나뉘어 있어 틀린 말이 된다.
                render_schedule_table(shown.result)

        # 자격 확인 필요 · 계산 전제 등 주의 문구 (optimizer가 프롬프트에 쓰는 것과 동일)
        for note in optimizer._caveats(member.combo.items, ctx):
            st.caption(f"※ {note}")

        # 신청 전 해야 할 일 · 신청처 — 카드에 오른 조합에 대해서만
        for line in prerequisites(member.combo.items):
            st.info(f"신청 전 확인: {line}")
        # LLM 이 덧붙인 준비 사항은 코드가 낸 선행요건 **바로 다음**에. 성격이 같다.
        if pick is not None and pick.before_apply:
            st.info(f"신청 전 확인: {pick.before_apply}")
        # 실행 순서는 신청처 바로 위에 — "어디에 신청하나" 직전에 "어떤 순서로"가 온다.
        # `_연계정보` 가 있는 조합에서만 채워지므로 지금은 거의 비어 있다(데이터 25개 중 1개).
        if pick is not None and pick.steps:
            st.markdown("**신청 순서**")
            for index, step in enumerate(pick.steps, start=1):
                st.markdown(f"{index}. {step}")
        links = apply_links(member.combo.items)
        if links:
            st.markdown("**신청처**")
            for line in links:
                st.markdown(f"- {line}")

        if group.dropped:
            st.caption(
                f"※ 같은 성능의 대안 {len(group.dropped)}개가 더 있으나 "
                "함정이 있거나 신청할 정책이 많아 제외했습니다."
            )


def render_results(
    groups: list[CandidateGroup],
    reports: dict,
    extras: list,
    profile: UserProfile,
    ctx: UserContext,
    baseline_survival: int,
    goal_satisfied: bool,
    target_months: int | None,
    resolve_info: dict | None,
    prompt_text: str,
    precise: bool,
    advice: agent.Advice | None = None,
    dropped_by_confirm: bool = False,
    future_groups: list[CandidateGroup] | None = None,
    future_reports: dict | None = None,
    future_months: dict[str, int] | None = None,
) -> None:
    """추천 조합 화면 — 카드·부가안내·심사용 프롬프트."""
    if not precise:
        st.info("간편 모드입니다. 매출 추세가 없어 모든 수치는 **입력값 기준 추정치**입니다.")

    _render_advice_banner(advice)

    # 두 축이 무엇인지 상단에 한 번 설명한다 (카드마다 축별 강점 라벨이 붙으므로).
    st.info(
        "핵심 평가 지표: **생존 개월 수**(지원을 받으면 몇 개월 버티는가), "
        "**종료 시점 부채**(그때 빚이 얼마 남는가)"
    )
    # 국면 판단은 카드가 아니라 목록 위 한 곳에 — 조합 하나가 아니라 전체에 대한 판단이다.
    if advice is not None and advice.ok and advice.phase_reading:
        st.markdown(
            f"<div style='{_CARD}padding:14px 16px;margin-bottom:10px;"
            "border-left:4px solid #2b6cb0;'>"
            f"<div style='font-weight:700;margin-bottom:4px;'>AI 판단</div>"
            f"<div style='color:#2d3748;'>{advice.phase_reading}</div></div>",
            unsafe_allow_html=True,
        )
    # 무지원 생존 개월수는 여기 한 줄로 두지 않는다 — 카드마다 '지원 전 → 지원 후'로
    # 나란히 보여야 정책이 얼마나 늘려주는지가 읽힌다(상단에 baseline 만 있으면
    # 비교 대상이 멀리 떨어져 있어 그 대비가 안 보인다).

    # 목표 미달 배너 (기획서 5.2.5 (d))
    if target_months is not None and not goal_satisfied:
        st.error(
            f"목표 {target_months}개월을 채우는 조합이 없습니다. 아래는 그중 나은 차선이며 "
            "모두 목표에 미달합니다. 목표를 낮추거나 고정비 절감을 함께 검토해 보세요."
        )

    # 금액·개월수 둘 다 입력한 경우의 안내
    if resolve_info is not None and not resolve_info["satisfied"]:
        need = resolve_info["need_amount"]
        best = resolve_info["best_months"]
        if need is not None:
            st.warning(
                f"입력하신 금액으로는 약 {best}개월까지입니다. "
                f"목표 {target_months}개월을 채우려면 약 {fmt_man(need)}이 필요합니다."
            )
        else:
            st.warning(
                f"입력하신 금액으로는 약 {best}개월까지이며, 목표 {target_months}개월은 "
                "정책 한도로도 채우기 어렵습니다."
            )

    # 후보가 하나도 없을 때. **원인에 따라 할 수 있는 일이 다르다.**
    # ⚠ "폐업 여부를 바꿔 보세요" 같은 안내는 쓰지 않는다 — 자격을 얻으려고 상황을
    #   다르게 적으라는 뜻으로 읽힌다. 잘못 입력한 값을 바로잡으라는 것과는 다르다.
    if not groups:
        if dropped_by_confirm:
            # '예'로 고치면 늘어난다고 **단정하지 않는다** — 재도전특별자금처럼 택1
            # 질문의 어떤 갈래(채무조정 이행 중)는 오히려 다른 정책을 배제한다.
            # 사실대로 답하라는 쪽으로 쓴다(inquiries.md 참조).
            st.error(
                "지금 답으로는 신청할 수 있는 조합이 없습니다. "
                "**확인되지 않은 자격은 없는 것으로 보고 제외했습니다.** "
                "⑥ 추가 확인에서 실제로 해당하시는 항목이 있다면 답을 고쳐 주세요."
            )
            st.caption("아래 ◀ 이전 버튼으로 돌아가 답을 고칠 수 있습니다.")
        else:
            st.error(
                "현재 조건으로 자격이 되는 정책이 없습니다. "
                "지역·업력·연체 일수 같은 요건이 맞지 않으면 신청 대상이 되지 않습니다."
            )
            st.caption(
                "답을 고쳐도 후보가 생기지 않는 경우입니다. 입력하신 값에 잘못 들어간 것이 "
                "없는지 확인해 보시고, 소상공인 지원기관에 상담하시면 여기에 담기지 않은 "
                "제도를 안내받을 수 있습니다."
            )
        return
    # 슬라이더 안내는 상단에 두지 않는다 — 그 상자 안에서만 쓰이는 설명이라, 첫 카드의
    # 하늘색 상자 안에 둔다(render_group_card). 상단 안내가 네 겹이 되면 카드까지
    # 내려가기 전에 지친다.

    # ── 카드는 **LLM 이 고른 조합**만 그린다 ──
    # 옛 구조는 그룹을 돌며 "이 그룹의 대표가 뽑혔나"를 물었다. 그러려면 그룹을 대표할
    # 이름을 정해야 하는데 members[0] 에는 근거가 없고(prune 뒤 조합 생성 순서상 첫째일
    # 뿐), LLM 이 대표 아닌 멤버를 고르면 그 pick 이 갈 자리가 없어 통째로 사라졌다.
    # 방향을 뒤집어 pick 의 label 로 조합을 직접 찾는다.
    by_label = {m.label: (g, m) for g in groups for m in g.members}
    if advice is not None and advice.ok and advice.picks:
        # 순위가 있는 셋만 그린다. 순위 없는 '후보' 카드를 섞으면 번호가 이어져 몇 위인지
        # 헷갈리고, 그 카드의 조합을 고를 근거도 없다.
        chosen = [
            (*by_label[p.label], p) for p in advice.picks if p.label in by_label
        ]
    else:
        # 폴백 — 상위 그룹 셋을 members[0] 으로 그린다.
        # members[0] 에 근거가 없다는 것을 알고 남긴다. 고른 주체가 없으니 대표를 정할
        # 수밖에 없고, 순위를 안 붙이므로(rank=None -> '후보') 잘못된 순위를 주장하지
        # 않는다. 같은 좌표의 다른 멤버를 골랐어야 할 근거도 없다.
        chosen = [(g, g.members[0], None) for g in groups[:3]]

    for rank, (group, member, pick) in enumerate(chosen, start=1):
        render_group_card(
            rank if pick is not None else None,
            group, member, reports, profile, ctx, baseline_survival,
            expanded=(rank == 1),  # 1위 기본 펼침
            pick=pick,
        )

    # ── 미래 자격 카드 — 순위 아래에 붙인다 ──
    # 공개 조건(화면에 그려진 카드 셋 모두에게 지배당하면 숨김)은 run_future_pipeline
    # 이 이미 걸렀다. 여기 온 것은 두 축 중 하나에서라도 나은 것들이다.
    # 비교 대상으로 **방금 그린 카드의 그룹**을 넘긴다 — LLM 픽이든 폴백이든
    # 사용자가 실제로 보고 있는 것과 비교해야 "기다릴 값어치"가 성립한다.
    if future_groups:
        st.markdown("---")
        st.markdown("**⏳ 지금은 안 되지만 곧 신청할 수 있는 지원**")
        for group in future_groups:
            member = group.members[0]
            render_group_card(
                None, group, member, future_reports or {}, profile, ctx,
                baseline_survival, expanded=False,
                future_months=_months_until(member, future_months or {}),
            )
        # 미래 카드 묶음의 끝에도 같은 실선을 둔다. 위쪽 실선만 있으면 그 아래 오는
        # 부가 추천·배제 목록이 미래 카드에 딸린 것처럼 읽힌다.
        st.markdown("---")

    # 부가 추천 (컨설팅·법률자문 등)
    if extras:
        st.markdown("**함께 안내할 것** — 순위엔 안 들지만 자격이 되는 지원")  # noqa: E501
        for cand in extras:
            note = cand.get("_순위제외근거") or cand.get("조건") or ""
            st.markdown(f"- {cand}" + (f" — {note[:80]}" if note else ""))

    # 추천에서 배제한 조합과 이유 — 자격은 되지만 뺀 것들(기획서 5.3).
    # 목록 맨 아래 접힘에 둔다. 궁금한 사람만 열어 보면 되는 정보라 카드를 밀어내면 안 된다.
    # 화면에는 3개만 뜨므로 여기 처음 보는 조합 이름이 나온다 — 접힘을 여는 사람은
    # "뭘 안 골랐나"가 궁금해서 여는 것이라 그게 맞다.
    if advice is not None and advice.ok and advice.excluded:
        with st.expander("추천에서 배제한 조합과 이유", key="exp_excluded"):
            for item in advice.excluded:
                st.markdown(f"- **{item.get('label', '')}** — {item.get('reason', '')}")

    # LLM 프롬프트·응답 (심사용)
    with st.expander("심사용: LLM 프롬프트 보기", key="exp_prompt"):
        if advice is not None and not advice.ok:
            st.error(f"실패 유형: {advice.failure} · {advice.detail}")
            if advice.bad_numbers:
                st.caption("검증에 걸린 숫자: " + ", ".join(advice.bad_numbers[:10]))
        if advice is not None and advice.usage:
            st.caption(
                f"토큰 — 입력 {advice.usage.get('input_tokens', 0):,} · "
                f"출력 {advice.usage.get('output_tokens', 0):,}"
            )
        st.text(prompt_text)


# ══════════════════════════════════════════════════════════
# 대화형 UI — 한 번에 한 단계 (◀ 이전 / 다음 ▶)
# ══════════════════════════════════════════════════════════
#
# 완료 단계를 접어도 접힌 바가 쌓여 스크롤이 생기므로, 한 화면에 한 단계만 보인다.
# 답은 전부 st.session_state 에 남으므로(위젯 key) ◀ 이전으로 돌아가 고치면
# 재실행으로 아래가 다시 계산된다. 진행은 ss.step(지금 보이는 단계) 하나로 관리한다.

# 세그먼트 진행바 라벨. 번호는 **사용자가 답하는 입력 단계에만** 붙인다.
# 진단·결과는 서비스가 내놓는 화면이라 번호가 없다(OUTPUT_STEPS).
STEP_TITLES = {
    S_SITUATION: "① 기본 상황",
    S_REVENUE: "② 매출 자료",
    S_COST: "③ 월 비용",
    S_DEBT: "④ 부채 규모",
    S_DEBT2: "⑤ 연체·신용",
    S_DIAGNOSIS: "진단",
    S_CONDITIONAL: "⑥ 추가 확인",
    S_GOALS: "⑦ 조건 설정",
    S_RESULTS: "결과",
}
# 번호 없는 '내놓는' 화면 — 세그먼트 채움 색·테두리를 입력 단계와 다르게 준다.
OUTPUT_STEPS = (S_DIAGNOSIS, S_RESULTS)
# 세그먼트 순서 (진단이 부채·연체 뒤, 추가 확인·조건 설정 앞)
SEGMENT_ORDER = (
    S_SITUATION, S_REVENUE, S_COST, S_DEBT, S_DEBT2,
    S_DIAGNOSIS, S_CONDITIONAL, S_GOALS, S_RESULTS,
)


def _has_step6(fields: list[str]) -> bool:
    """⑥ 단계에 물을 것이 있는지. 조건부 항목이나 자격 확인 질문 중 하나라도 있으면 뜬다."""
    return bool(fields) or bool(st.session_state.get("_confirm_questions"))


def _step_after(cur: int, fields: list[str]) -> int:
    """다음 단계. 물을 것이 없으면 ⑥ 추가 확인을 건너뛴다."""
    nxt = cur + 1
    if nxt == S_CONDITIONAL and not _has_step6(fields):
        nxt = S_GOALS
    return min(nxt, S_DONE)


def _step_before(cur: int, fields: list[str]) -> int:
    """이전 단계. 건너뛴 ⑥은 되돌아갈 때도 건너뛴다."""
    prv = cur - 1
    if prv == S_CONDITIONAL and not _has_step6(fields):
        prv = S_DIAGNOSIS
    return max(prv, S_SITUATION)


def _nav(cur: int, fields: list[str], ready: bool = True) -> None:
    """◀ 이전 / 다음 ▶. 화면 하단에 고정(sticky)하고 가운데에 배치한다.

    위젯이 많은 단계(⑤ 연체·신용)에서 버튼을 스크롤해야 보이는 문제를 없앤다.
    """
    ss = st.session_state
    nav = st.container(key="navbar")
    with nav:
        _, prev_col, next_col, _ = st.columns([2, 1, 1, 2])  # 양옆 여백 → 버튼이 가운데로
        with prev_col:
            if cur > S_SITUATION and st.button("◀ 이전", key=f"prev_{cur}", use_container_width=True):
                ss.step = _step_before(cur, fields)
                st.rerun()
        with next_col:
            if cur < S_DONE:
                nxt = _step_after(cur, fields)
                label = {S_RESULTS: "결과 보기 ▶", S_DONE: "완료 ▶"}.get(nxt, "다음 ▶")
                if st.button(label, key=f"next_{cur}", disabled=not ready, use_container_width=True):
                    ss.step = nxt
                    st.rerun()
        if not ready:
            st.caption("입력을 마치면 다음으로 넘어갈 수 있어요.")

    # ⚠ 내부 API 의존: '.st-key-navbar' 는 st.container(key="navbar") 가 만드는
    #   Streamlit 내부 클래스명이다(공개 API 아님). Streamlit 1.62.0 에서 확인.
    #   '.block-container' 도 내부 클래스다.
    #   배포 후 Streamlit 버전을 올리면 **여기부터** 확인한다 — 고정 바가 풀리거나
    #   마지막 위젯이 가려지면 (1) st-key-<key> 클래스 규칙이 바뀌었는지,
    #   (2) sticky 가 새 레이아웃 컨테이너에서 먹는지, (3) 본문 컨테이너 클래스명이
    #   바뀌었는지 순으로 본다.
    #   padding-bottom: 고정 바 높이만큼 본문 하단에 여백을 줘 마지막 위젯이 바에
    #   가리지 않게 한다.
    st.markdown(
        """
        <style>
        .st-key-navbar {
            position: sticky; bottom: 0; z-index: 100;
            background: var(--background-color, #ffffff);
            border-top: 1px solid rgba(49, 51, 63, 0.2);
            padding-top: 8px; padding-bottom: 4px;
        }
        .block-container { padding-bottom: 5rem; }

        /* 경계 판정(국면 확인) 블록 — 안내문·질문·라디오를 한 덩어리로 묶는다.
           연파랑(st.info)은 카드 회색과 구분이 약해 연초록으로 바꿨다.
           margin-top 은 카드(_card 의 margin-top:12px)와 같은 값이라 간격이 일정하다.
           ⚠ '.st-key-phase_confirm' 도 navbar 와 같은 내부 클래스 의존이다. */
        .st-key-phase_confirm {
            background: #e9f7ef; border: 1px solid #9ae6b4;
            border-radius: 8px; padding: 14px 16px; margin-top: 12px;
        }
        /* ①게이지·②버퍼 열 묶음도 다른 카드와 같은 간격(12px)을 갖게. */
        .st-key-diag_row { margin-top: 12px; }

        /* 접힘 제목 오른쪽 끝에 '펼쳐보기 / 접기'. 왼쪽 화살표만으로는 펼칠 수 있다는
           것이 잘 안 보인다. details[open] 로 갈라 쓰므로 펼침 상태를 따로 추적하지
           않는다 — Streamlit 이 다시 그려도 CSS 가 알아서 맞춘다.
           제목이 길어 글자가 라벨 아래로 지나가지 않도록 오른쪽 여백을 준다.
           ⚠ stExpander 도 navbar 와 같은 내부 구조 의존이다. */
        /* ⚠ `details[open] summary` 로 쓰면 안 된다. 후손을 다 잡아서, 카드 접힘이
           펼쳐진 순간 그 안의 상환 스케줄 접힘까지 '접기'가 된다(닫혀 있어도).
           자식 선택자 `>` 로 자기 summary 만 잡는다. */
        [data-testid="stExpander"] details > summary { position: relative; padding-right: 68px; }
        [data-testid="stExpander"] details > summary::after {
            content: "펼쳐보기";
            position: absolute; right: 10px; top: 50%; transform: translateY(-50%);
            font-size: 0.82rem; color: #2b6cb0; white-space: nowrap;
        }
        [data-testid="stExpander"] details[open] > summary::after { content: "접기"; }

        /* AI 실패 배너 — 배경색만 있어 경계가 흐리고 그냥 지나친다.
           **색은 노랑 그대로 둔다.** 주황으로 올리면 아래 함정 경고(노랑)보다 세져
           "돈 얘기보다 시스템 얘기가 더 급한 것"처럼 위계가 뒤집힌다.
           대신 왼쪽 굵은 세로선 + 전체 테두리로 경계만 세운다.
           ⚠ '.st-key-advice_banner' 와 stAlert 도 navbar 와 같은 내부 구조 의존이다. */
        .st-key-advice_banner [data-testid="stAlert"] {
            border: 1px solid #d69e2e;
            border-left: 6px solid #b7791f;
        }

        /* 슬라이더로 다시 계산되는 영역 — 연한 하늘색 상자.
           위쪽(대비 줄·지원 내역)은 원래 조합 기준이라 안 바뀌는데, 경계가 안 보이면
           어디까지가 조정 결과인지 알 수 없다. 곡률은 다른 카드와 같은 8px.
           안쪽 여백을 줘 내용이 테두리에 붙지 않게 하고, 자식 요소의 왼쪽 여백을
           0으로 맞춰 슬라이더 제목이 슬라이더보다 튀어나오지 않게 한다.
           ⚠ '.st-key-adjust_' 도 navbar 와 같은 내부 클래스 의존이다. */
        [class*="st-key-adjust_"] {
            background: #EAF3FC; border-radius: 8px;
            padding: 14px 16px; margin: 6px 0 4px 0;
        }
        [class*="st-key-adjust_"] > div { padding-left: 0; padding-right: 0; }

        /* 상자 안 요소 간 간격을 일정하게(슬라이더·결과 카드·그래프·경고·스케줄). */
        [class*="st-key-adjust_"] div[data-testid="stVerticalBlock"] { gap: 14px; }

        /* (결과 카드 높이 맞춤은 CSS 로 열을 stretch 시키지 않고, 양쪽 카드에 같은
           min-height 를 줘서 맞춘다 — Streamlit 내부 구조에 덜 기대는 쪽을 골랐다.) */

        /* 상환 스케줄 접힘 블록은 **통째로 흰 배경** — 하늘색 상자 위에 얹힌 형태.
           제목 줄(summary)과 펼친 안쪽까지 전부. 모서리를 깎으려면 overflow 를 숨긴다. */
        [class*="st-key-adjust_"] [data-testid="stExpander"] {
            background: #ffffff; border-radius: 8px; overflow: hidden;
        }
        [class*="st-key-adjust_"] [data-testid="stExpander"] details,
        [class*="st-key-adjust_"] [data-testid="stExpander"] summary,
        [class*="st-key-adjust_"] [data-testid="stExpander"] details > div {
            background: #ffffff;
        }
        /* 한글은 CSS 기본값이 글자 단위로 끊긴다("줄었습니 / 다").
           본문 텍스트를 어절 단위로 끊어 좁은 칸에서도 자연스럽게 접히게 한다. */
        .block-container p, .block-container li,
        .block-container h1, .block-container h2, .block-container h3 { word-break: keep-all; }

        /* ── 좁은 화면에서 가로로 넘치는 두 곳만 막는다 ──────────────────
           범위를 딱 둘로 못박는다. 터치 타깃 크기·카드 재배치·그래프 비율·폰트
           스케일은 하지 않는다 — 며칠짜리라 마감 안에 넣을 범위가 아니다.

           640px 는 Streamlit 이 st.columns 를 세로로 쌓기 시작하는 폭과 같은 대역이라,
           이 선에서 함께 접히면 화면이 한 번에 바뀐다.

           **넓은 화면에는 이 블록 밖에 규칙이 없다.** 미디어 쿼리 안에서만 인라인
           스타일을 덮으므로 데스크톱은 한 픽셀도 안 바뀐다. !important 가 필요한 것은
           덮을 대상이 인라인 스타일이기 때문이다. */
        @media (max-width: 640px) {
            /* ① 진행바 — 세그먼트 9개가 flex:1 + nowrap 이라, flex 항목이
                  min-content(=글자 폭) 아래로 안 줄어들어 가로로 넘쳤다.
                  줄바꿈을 열고 한 줄에 셋씩 놓는다. 줄 수만 늘고 글자는 그대로다. */
            .sn-progress { flex-wrap: wrap !important; }
            .sn-seg { flex: 1 1 30% !important; min-width: 0 !important; }

            /* ② 진단 화면 매출 요약 — 네 항목이 nowrap 이라 넘쳤다.
                  줄바꿈을 열고, 줄이 바뀌면 뜻을 잃는 세로 구분선은 감춘다.
                  항목 자체의 nowrap 은 유지한다 — "최근 3개월 평균 매출"이 중간에
                  끊기면 오히려 읽기 나쁘다. */
            .sn-revsum { flex-wrap: wrap !important; gap: 4px 14px; }
            .sn-revline { display: none !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _seg_style(is_output: bool, state: str) -> str:
    """세그먼트 한 칸의 인라인 스타일.

    지나온 칸=채워진 색, 현재=강조, 남은 칸=회색. 입력·출력(진단·결과) 모두 **파랑 계열**로
    통일하고, 성격 차이는 **색 대비가 아니라 테두리 굵기**가 맡는다 — 출력 칸은 진한 파랑에
    3px 테두리를 회색 상태에서도 유지해 '여기서 뭔가 나온다'가 보이게 한다.
    """
    # 높이를 **고정**한다. 예전엔 높이 지정이 없어 두 가지로 칸 높이가 흔들렸다.
    #   ① 현재 칸이 볼드(font-weight:700)가 되면서 줄 상자 높이가 미세하게 커진다
    #   ② box-sizing 기본값(content-box)이라 진단·결과 칸의 3px 테두리가 높이에 6px
    #      더해진다 — 입력 칸과 애초에 높이가 달랐다
    # height + line-height 로 글자 굵기와 무관하게 같은 높이가 되고, border-box 로
    # 테두리를 높이 안에 넣는다. 글자는 flex 로 세로 가운데 정렬한다.
    base = (
        "flex:1;white-space:nowrap;font-size:11px;box-sizing:border-box;"
        "height:26px;line-height:1;display:flex;align-items:center;justify-content:center;"
        "padding:0 2px;border-radius:6px;"
    )
    if is_output:  # 진단·결과 — 진한 파랑 + 3px 테두리(항상 유지)
        if state == "cur":
            return base + "border:3px solid #0d3a6b;background:#1c5091;color:#fff;font-weight:700;"
        if state == "done":
            return base + "border:3px solid #1c5091;background:#bcd4f2;color:#123a63;font-weight:600;"
        return base + "border:3px solid #7aa7dd;background:#fff;color:#2a5a92;"  # 남음, 테두리 유지
    # 입력 단계 (테두리 없음)
    if state == "cur":
        return base + "background:#2b6cb0;color:#fff;font-weight:700;"
    if state == "done":
        return base + "background:#d6e4ff;color:#1a4b8c;"
    return base + "background:#f0f2f6;color:#9aa0a6;"  # 남음


def render_progress(cur: int, fields: list[str]) -> None:
    """세그먼트 진행바 (한 줄, HTML). 번호는 입력 단계에만, 진단·결과는 구분해 표시한다.

    카운터는 **답할 단계 기준** '3 / 7'. 출력 화면에서는 그 화면 이름('진단'·'결과')을 쓴다.
    조건부 단계(⑥ 추가 확인)가 건너뛰어지면 그 칸을 빼고 카운터도 /6 으로 줄인다.
    한 줄에 안 들어가면: 폰트 11px·좌우 패딩 축소로 대응(white-space:nowrap, flex:1 균등).
    """
    segments = [s for s in SEGMENT_ORDER if not (s == S_CONDITIONAL and not fields)]
    answer_steps = [s for s in segments if s not in OUTPUT_STEPS]
    total = len(answer_steps)

    # 카운터 — 출력 화면은 그 화면 이름을 그대로(진단 / 결과), 입력은 'pos / total'
    #
    # 완료 화면에서는 진행바를 **없애지 않고** 모든 칸을 '지나온' 상태로 둔다.
    # 칸을 하나 더하는 방식은 쓰지 않는다 — 완료는 단계가 아니라 끝이라, 칸을 더하면
    # "아직 할 일이 남았다"로 읽힌다. 통째로 숨기면 화면이 갑자기 비어 어색하다.
    # cur 가 마지막 단계보다 커서 아래 루프가 자연히 전부 'done' 을 만든다.
    if cur == S_DONE:
        counter = "완료"
    elif cur in OUTPUT_STEPS:
        # 화면 이름을 그대로 쓰면 바로 아래 칩과 **같은 말이 두 번** 나온다("진단"·"진단").
        # 카운터의 일은 '어디쯤 왔나'이므로, 지나온 입력 단계 수를 말한다.
        counter = f"입력 {sum(1 for s in answer_steps if s < cur)} / {total} 완료"
    else:
        pos = answer_steps.index(cur) + 1 if cur in answer_steps else total
        counter = f"{pos} / {total}"

    cells = []
    for step in segments:
        state = "done" if step < cur else ("cur" if step == cur else "todo")
        cells.append(
            f'<div class="sn-seg" style="{_seg_style(step in OUTPUT_STEPS, state)}">'
            f"{STEP_TITLES[step]}</div>"
        )
    # 클래스를 붙이는 이유는 좁은 화면용 미디어 쿼리가 인라인 스타일을 덮어야 해서다.
    # 넓은 화면에서는 이 클래스에 걸린 규칙이 없어 지금과 똑같이 그려진다.
    bar = (
        '<div class="sn-progress" style="display:flex;gap:4px;width:100%;'
        'margin:2px 0 8px 0;">'
        + "".join(cells)
        + "</div>"
    )
    counter_html = (
        f'<div style="font-size:12px;font-weight:700;color:#2b6cb0;margin-bottom:3px;">'
        f"{counter}</div>"
    )
    st.markdown(counter_html + bar, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# 위젯 래퍼 — 답을 ss.answers 에 저장한다 (배선 버그 방지)
# ══════════════════════════════════════════════════════════
#
# ⚠ 입력 위젯은 **반드시 이 래퍼**를 통해 만든다. st.number_input / st.selectbox /
#   st.checkbox / st.radio / st.text_input / st.slider / st.date_input /
#   st.file_uploader 를 render_* 안에서 직접 부르면 안 된다.
#
#   이유(2026-08 실측 버그): 단일 단계 UI라 매 rerun 에 현재 단계 위젯만 렌더되는데,
#   Streamlit 은 이번 run 에서 렌더되지 않은 위젯의 key 를 session_state 에서 지운다.
#   그래서 위젯 값을 직접 읽으면(ss.get("fixed_man")) 다음 단계에서 사라져, 진단·결과
#   단계의 UserProfile 이 고정비 0·부채 0 등 기본값으로 조립되는 배선 버그가 났다.
#
#   래퍼는 값을 우리 소유의 dict(ss.answers[key])에 복사한다 — 이 dict 는 위젯이 아니라
#   purge 되지 않는다. collect_* 는 위젯 key 가 아니라 **ss.answers 에서만** 읽는다.
#   직접 호출을 막는 회귀 테스트: tests/test_app.py::test_no_direct_widget_calls
#
# ── WIDGET WRAPPERS START ── (아래 정의부만 st.<widget> 직접 호출 허용)


def _answers() -> dict:
    """답 저장소. 위젯이 아니라서 purge 되지 않는다."""
    return st.session_state.setdefault("answers", {})


def _num(label, key, default, **kw):
    ans = _answers()
    ans[key] = st.number_input(label, value=ans.get(key, default), key=f"{key}_w", **kw)
    return ans[key]


def _check(label, key, default, **kw):
    ans = _answers()
    ans[key] = st.checkbox(label, value=ans.get(key, default), key=f"{key}_w", **kw)
    return ans[key]


def _text(label, key, default, **kw):
    ans = _answers()
    ans[key] = st.text_input(label, value=ans.get(key, default), key=f"{key}_w", **kw)
    return ans[key]


def _slider(label, key, min_value, max_value, default, **kw):
    ans = _answers()
    ans[key] = st.slider(label, min_value, max_value, value=ans.get(key, default), key=f"{key}_w", **kw)
    return ans[key]


def _select(label, key, options, default, **kw):
    ans = _answers()
    current = ans.get(key, default)
    idx = options.index(current) if current in options else 0
    ans[key] = st.selectbox(label, options, index=idx, key=f"{key}_w", **kw)
    return ans[key]


def _radio(label, key, options, default, **kw):
    """라디오. 저장된 답이 선택지에 없으면 **`default` 로** 돌아간다.

    옛 코드는 그때 0번을 골랐다. 자격 확인 질문의 0번은 '예'라서, 선택지가 바뀐
    뒤(예: 택1 질문이 객관식이 되면서 옛 '예'가 사라진 뒤) 답이 조용히 '예'가 되어
    **자격을 주는 쪽으로 기운다.** 확인되지 않은 것은 통과시키지 않는 규칙과 반대다.

    **`default=None` 이면 아무것도 고르지 않은 상태로 시작한다.** 기본값을 주면
    사용자가 안 골랐는데 고른 것처럼 되는 자리가 있다(간편 모드 국면 선택).
    그때는 호출부가 `_nav(ready=...)` 로 다음 단계를 막는다.
    """
    ans = _answers()
    if default is None:
        current = ans.get(key)
        idx = options.index(current) if current in options else None
    else:
        current = ans.get(key, default)
        fallback = options.index(default) if default in options else len(options) - 1
        idx = options.index(current) if current in options else fallback
    ans[key] = st.radio(label, options, index=idx, key=f"{key}_w", **kw)
    return ans[key]


def _date(label, key, default, **kw):
    ans = _answers()
    ans[key] = st.date_input(label, value=ans.get(key, default), key=f"{key}_w", **kw)
    return ans[key]


def _file(label, key, **kw):
    """file_uploader 는 재렌더 시 None 을 돌려준다(네비게이션으로 돌아오면 파일 소실).
    그래서 새로 올렸을 때만 저장소를 갱신한다. 소스를 바꾸면 호출부가 ans[key]를 비운다.
    한계: 업로드 후 위젯의 x 로 취소해도 저장소엔 남는다(복귀와 구분 불가) — 소스를
    데모/없어요로 바꾸면 지워진다.
    """
    ans = _answers()
    f = st.file_uploader(label, key=f"{key}_w", **kw)
    if f is not None:
        ans[key] = f
    return ans.get(key)


# ── WIDGET WRAPPERS END ──


# ── 데모 시나리오별 재무·목표 프리셋 ──
#
# ⚠ 목표(goal_months: A 24 · B 18 · C 12)는 **baseline(무지원 생존)보다 위**여야
#   filter_by_goal 이 실제로 거른다(목표 ≤ baseline 이면 전부 통과해 satisfied 가 자명).
#   재무값을 손대면 baseline 이 움직이므로 목표도 다시 잡아야 한다 —
#   tests/test_app.py::test_demo_goal_above_baseline 이 이 관계를 지킨다.
#   값 근거는 실측(2026-08-21): A 흑자+하드룰②+함정1·2·baseline13, B 거치절벽(goal>거치12),
#   C 회복불가+satisfied=False+새출발 처방. A 의 36+ 미달은 trend_revenue 한계(planning 5.5).
DEMO_PRESETS: dict[str, dict] = {
    # `confirm_yes` — 자격 확인 질문에 '예'로 답한 상태로 시작할 **후보 id** 목록.
    #
    # 데모는 **자격 요건이 충족된 사용자**를 전제한다. 일반 사용자의 기본값은 여전히
    # '잘 모르겠어요'이고(모름을 통과시키면 신청했다가 거절당한다 — 기획서 5.2.5 ⑥),
    # 판정 규칙도 그대로다. 프리셋은 그 질문에 대한 **답만** 미리 채운다.
    # 재무값을 미리 넣는 것과 성격이 같고, 화면(⑥ 단계)에 라디오로 그대로 보이며
    # 사용자가 바꿀 수 있다. 시연 중에 '아니오'로 바꾸면 그 정책이 빠지는 것도 보인다.
    #
    # 해시가 아니라 **후보 id 로 적는다** — 정책 데이터의 질문 문구가 바뀌면 해시는
    # 달라지는데, 프리셋에 해시를 박아 두면 조용히 어긋난다(confirm_key_of 가 옮긴다).
    # 고정비 435 는 **ICR 을 1.34 로 놓으려고** 고른 값이다(2026-08-30).
    #   ICR_RISK(1.0) 이상 ICR_SAFE(1.5) 미만이면 1점이다.
    # 430 도 1점이지만 판정값이 1.49 라 경계에서 0.01 뿐이었다. 한 칸 내려 여유를 뒀다.
    #
    # 격자로 고를 때의 ICR(1.20)과 판정 화면의 ICR(1.32)이 다른데, **분모가 다르다.**
    #   격자   예측선 1개월차(650만)로 영업이익을 계산했다
    #   판정   `recent_3m_avg`(663.5만)를 쓴다 (`_phase_inputs`)
    # 값을 다시 고를 일이 있으면 판정 쪽 기준으로 봐야 한다.
    # 예전 값(300)은 지표 4개가 모두 0점이라 **총점 0 + 하드 규칙 ②**로 일시적자금난이
    # 됐다. 화면에서는 신호등이 전부 초록인데 국면만 자금난이라 "다 정상인데 왜?"로
    # 읽혔다. A 가 첫 시나리오라 거기서 생긴 의구심이 B·C 까지 간다.
    # 이제는 ICR 하나에 주황이 뜨고 **총점 1점으로** 일시적자금난 구간에 들어간다.
    # 지표와 판정이 같은 것을 말한다.
    #
    # 매출감소 지표는 손댈 수 없다 — DROP_WARN 이 0.15 인데 A 의 반기 감소율이
    # 14.91% 다. 점수를 붙이려면 15% 를 넘겨야 하고, 그러면 **미래 자격 시연이
    # 사라진다**(지금 자격선 0.09%p 아래라 예측선이 1개월 뒤 넘어선다).
    # 영업이익은 0점/2점뿐이라 흑자를 지켜야 하고(적자면 ICR 도 2점), 연체는
    # 자격 필터를 건드린다. 그래서 남는 조절 손잡이가 고정비뿐이다.
    #
    # 잃는 것: 하드 규칙 ②(36개월 미생존)가 데모에서 발동하지 않는다. 총점 0 일 때만
    # 걸리는 규칙인데 A 가 유일한 발동처였다. 규칙 자체는 test_phase 가 검증한다.
    "A": dict(fixed_man=435, var_man=180, cash_man=1_900, debt_man=5_000, debt_pay_man=200,
              rate_known=True, rate_pct=8.0, overdue_days=0,
              # 목표 24 -> 18 (2026-08-31). 24 로 두면 역산이 거기서 멈춰 **다섯 그룹이
              # 전부 생존 24 에 눌려 붙는다.** 두 축 중 하나가 상수가 되어 부채만으로
              # 순위가 갈리고, 늦게 받는 융자는 구조적으로 항상 진다 — 같은 목표를
              # 채우려면 그 사이 잔고가 더 마른 만큼 더 빌려야 하기 때문이다.
              # 그래서 미래 자격 조합이 상위 3개에 전부 지배당해 화면에 못 떴다.
              # 18 로 내리면 좌표가 (24, 5,000) · (19, 4,208) · (18, 4,160) 으로
              # 흩어져 "더 오래 버티려면 빚이 는다"가 카드 사이에서 보이고, 미래
              # 카드도 공개된다. baseline 9 의 두 배라 개선 폭도 읽힌다.
              goal_months_on=True, goal_months=18,
              cond_bstart_known=True, cond_bstart=datetime.date(2019, 3, 1),
              # A 는 이 답이 결과를 바꾸지 않는다(제외되는 두 후보가 상위 5위 밖).
              # 그래도 넣어 둔다 — 시나리오를 오갈 때 답이 남아 생기는 혼선을 막는다.
              # `__q` 는 같은 질문이 덮는 둘째 후보다. 질문 key 가 같아 답이 함께
              # 붙지만, 프리셋에 한쪽만 적으면 목록이 실제 질문과 어긋난다.
              confirm_yes=("seoul_urgent_selfemp__1y4y", "seoul_urgent_selfemp__1y4y__q",
                           "fund_credit_weak__2y3y")),
    "B": dict(fixed_man=430, var_man=180, cash_man=1_500, debt_man=2_500, debt_pay_man=70,
              rate_known=True, rate_pct=5.0, overdue_days=0,
              goal_months_on=True, goal_months=18,
              cond_bstart_known=True, cond_bstart=datetime.date(2019, 3, 1),
              # 없으면 1위가 긴급자영업자금 -> 성장기반자금으로 바뀌고 부채가 219만 늘어난다.
              confirm_yes=("seoul_urgent_selfemp__1y4y", "seoul_urgent_selfemp__1y4y__q",
                           "fund_credit_weak__2y3y", "seoul_standup__fund")),
    # C 의 is_closed=True 는 의도적이다. **끄면 두 가지가 같이 사라진다.**
    #   ① 회복불가 국면의 폐업 지원 시연 — 끄면 pool 에 철거비·컨설팅·법률자문이 안 들어와
    #      생존 축이 baseline 6 = 최선 6 으로 죽는다. 그러면 두 축 중 하나가 상수라
    #      파레토 정렬을 쓰는 이유가 화면에서 안 읽힌다(부채는 4,026만 줄어드는데도).
    #   ② 점포철거비의 `단가_평당 x 전용면적` 계산 경로 검증 — cond_area=20평이 실제로
    #      쓰여 400만원이 나온다(상한 600만은 30평부터라 여기선 안 걸린다). 폐업을 끄면
    #      면적이 아무 데도 안 쓰이는 죽은 값이 된다.
    # 잃는 것: 소상공인 경영안정 바우처(비용감면 월 25만)가 '영업중' 전제라 pool 에서 빠진다.
    #      C 에서만 빠지고 A·B 는 영향 없다. 유입이 아니라 월 25만 비용감면이라
    #      생존·종료부채에 미치는 영향이 철거비 400만보다 작다(실측: 생존 6→8 은 철거비 몫).
    "C": dict(fixed_man=450, var_man=180, cash_man=1_900, debt_man=7_000, debt_pay_man=100,
              rate_known=True, rate_pct=9.0, overdue_days=120,
              goal_months_on=True, goal_months=12,
              cond_bstart_known=True, cond_bstart=datetime.date(2019, 3, 1),
              cond_area=20.0, is_closed=True,
              # 없으면 다시서기가 빠져 최선 조합(새출발+철거비+다시서기)이 사라지고
              # 생존이 8 -> 7 로 줄어, 폐업을 켜서 만든 6 -> 8 시연이 무너진다.
              confirm_yes=("seoul_standup__fund",)),
}


# 세 프리셋 키의 **합집합**. 손으로 유지하는 목록을 두지 않는다 — 프리셋에 키를 추가하고
# 목록 갱신을 빠뜨리면 조용히 틀리는(값만 새는) 배선 버그가 재발한다. 항상 dict 에서 뽑는다.
_PRESET_KEYS = frozenset(k for preset in DEMO_PRESETS.values() for k in preset)


def apply_demo_preset(scenario_key: str) -> None:
    """데모 시나리오의 재무·목표 프리셋을 ss.answers 에 넣는다. **시나리오가 바뀔 때만.**

    _applied_scenario 로 추적해 시나리오 변경 시에만 덮어쓴다 — 사용자가 값을 고친 뒤엔
    유지되고, 다른 시나리오로 바꾸면 그 프리셋으로 리셋된다. 재무/목표 위젯은 이 단계(②)가
    아니라 뒤 단계에서 렌더되므로, ans 를 먼저 바꿔두면 그 단계에 갈 때 반영된다.

    **한 프리셋에만 있는 키는 다른 시나리오로 갈 때 지운다**(합집합 − 이번 프리셋).
    dict.update 는 없는 키를 지우지 않아서, C(is_closed=True·cond_area=20)를 보고 A 로
    넘어가면 A 가 폐업 상태로 계산된다 — 화면은 멀쩡하고 값만 틀리는 종류라 시연 중에
    알아채기 어렵다. 지우면 위젯이 다시 그려질 때 제 기본값(False·0)으로 돌아간다.

    사용자 수정 보존은 깨지지 않는다: 이 정리는 _applied_scenario 가 바뀌는 순간에만
    돌고, 같은 시나리오 안에서 고친 값은 애초에 여기까지 오지 않는다.
    """
    ans = _answers()
    if ans.get("_applied_scenario") == scenario_key:
        return
    # 프리셋이 덮기 **전에** 사용자의 원래 답을 보관한다(`restore_from_demo` 가 되돌린다).
    #
    # ② 의 기본값이 '데모 시나리오'라, 사용자가 데모를 고르지 않아도 이 화면에 **도착만
    # 하면** 프리셋이 한 번 돈다. 그때 ① 에서 체크한 폐업 여부(`is_closed`)가 사라졌다 —
    # A·B 프리셋에 그 키가 없어 위 pop 에 걸리기 때문이다. 데모로 시연하면 C 프리셋이
    # 값을 다시 넣어 줘서 안 드러나고, **자기 값을 입력하는 실제 사용자에게만** 났다.
    #
    # 보관 시점은 **프리셋 적용 직전**이다. ① 을 떠날 때 뜨면 ① 키만 잡히고 프리셋이
    # 건드리는 나머지(`cond_area` 등)를 놓친다. 그리고 **처음 한 번만** 뜬다 — 시나리오를
    # 바꿀 때 또 뜨면 앞 프리셋 값이 원본으로 둔갑한다.
    if "_preset_backup" not in ans:
        ans["_preset_backup"] = {k: ans[k] for k in _PRESET_KEYS if k in ans}
    preset = DEMO_PRESETS.get(scenario_key)
    if preset:
        # **앞 프리셋이 넣은 키만** 지운다. 예전에는 (합집합 − 이번 프리셋)을 지웠는데,
        # 그러면 프리셋이 처음 적용될 때 **사용자가 ① 에서 답한 `is_closed` 까지** 지웠다
        # — 그 키가 C 프리셋에만 있어 합집합에 들어가기 때문이다.
        # 앞 프리셋을 기준으로 삼으면 첫 적용에서는 지울 것이 없고, 시나리오를 갈아탈
        # 때만 앞 값이 걷힌다(원래 이 정리를 넣은 이유가 그것이다).
        # 걷어낸 자리는 **사용자가 답했던 값**으로 되돌린다. 그냥 지우면 사용자가 ① 에서
        # False 로 답한 것과 답한 적 없는 것(None)이 구분되지 않는다.
        previous = DEMO_PRESETS.get(ans.get("_applied_scenario")) or {}
        backup = ans.get("_preset_backup") or {}
        for stale in previous.keys() - preset.keys():
            if stale in backup:
                ans[stale] = backup[stale]
            else:
                ans.pop(stale, None)  # 답한 적 없으면 위젯 기본값으로 돌아간다
        # 자격 확인 답은 key 가 질문 문구에서 나오므로 위 목록으로는 못 지운다.
        # 시나리오를 바꿀 때 앞 시나리오의 답이 남지 않게 접두로 걷어낸다.
        for stale in [k for k in ans if str(k).startswith("confirm_")]:
            ans.pop(stale, None)
        ans.update(preset)
        # 후보 id -> 질문 key 로 옮긴다(프리셋에 해시를 박지 않으려고).
        # 해당 후보에 확인 질문이 없으면 조용히 건너뛴다.
        db = load_db()
        for candidate_id in ans.pop("confirm_yes", ()) or ():
            if key := confirm_key_of(db, candidate_id):
                ans[key] = CONFIRM_YES
    ans["_applied_scenario"] = scenario_key


def restore_from_demo() -> None:
    """데모를 떠날 때(파일 업로드 · 없어요) **프리셋이 덮기 전의 답으로 되돌린다.**

    ② 의 기본값이 '데모 시나리오'라 사용자가 데모를 고르지 않아도 프리셋이 한 번
    적용된다. 그대로 두면 ① 에서 체크한 폐업 여부가 사라진 채로 진행된다.

    프리셋이 넣은 값은 **전부** 걷어낸다. 일부만 되돌리면 이번 시나리오의 재무값이
    사용자 입력에 섞여 어느 값이 자기 것인지 알 수 없게 된다.

    `_preset_backup` 과 `_applied_scenario` 를 함께 지운다 — 다시 데모로 돌아오면
    프리셋이 새로 적용되고, 그 사이 고친 값이 새 원본으로 잡혀야 한다.
    """
    ans = _answers()
    backup = ans.pop("_preset_backup", None)
    if backup is None:
        return  # 데모를 거친 적이 없거나 이미 되돌렸다
    for key in _PRESET_KEYS:
        ans.pop(key, None)
    # 자격 확인 답은 key 가 질문 문구 해시라 위 목록으로 못 지운다(접두로 걷어낸다).
    for key in [k for k in ans if str(k).startswith("confirm_")]:
        ans.pop(key, None)
    ans.update(backup)
    ans.pop("_applied_scenario", None)


# ── 답 수집 (위젯을 그리지 않고 ss.answers 에서만 읽는다) ──


def collect_revenue() -> dict:
    """매출 자료 답을 ss.answers 에서 읽어 요약을 만든다. 보이는 단계와 무관하게 호출된다."""
    ans = _answers()
    source = ans.get("rev_source", "데모 시나리오")
    summary = None
    precise = True
    rough = 0
    ready = True
    if source == "데모 시나리오":
        summary = summarize_scenario(ans.get("scenario_key", "B"), 12)
    elif source == "파일 업로드":
        f = ans.get("pos_file")
        if f is None:
            ready = False
        else:
            try:
                summary = summarize_uploaded(f.getvalue(), f.name)
            except Exception:
                ready = False
    else:
        precise = False
        rough = won(ans.get("rough_rev_man", 800))
    current_revenue = summary.recent_3m_avg if (precise and summary is not None) else rough
    return dict(
        summary=summary, precise=precise, rough=rough,
        # 아직 안 고른 상태(None)면 임시로 이 값을 쓴다. 진단 화면이 `ready=False` 로
        # 다음 단계를 막고 있어 이 값이 결과 화면까지 가지는 않는다.
        phase_simple=ans.get("phase_simple") or "일시적자금난",
        current_revenue=current_revenue, ready=ready,
    )


def compute_profile(revenue: dict) -> UserProfile:
    """session_state 의 재무·매출 답으로 UserProfile 을 조립한다.

    매출선이 앙상블에서 나왔는지를 `revenue["forecast_is_ensemble"]` 에, 그 앙상블에
    세 모델이 다 들어갔는지를 `revenue["forecast_all_models"]` 에 적어 둔다.
    미래 자격이 앞의 플래그를 본다 — `revenue` 는 매출에서 나온 값들을 담아 화면까지
    가는 자루이므로 여기 두는 것이 맞고, 나중에 매출선만 보고 되짚지 않아도 된다.
    뒤의 플래그는 아무것도 가르지 않는다(`revenue_line` 참조).
    """
    finance = collect_finance()
    monthly_revenue, from_ensemble, all_models = revenue_line(
        revenue["precise"], revenue["summary"], revenue["rough"]
    )
    revenue["forecast_is_ensemble"] = from_ensemble
    revenue["forecast_all_models"] = all_models
    return build_profile(finance, monthly_revenue, revenue["current_revenue"])


def compute_phase_and_annual(revenue: dict, elig: dict, profile: UserProfile) -> tuple[str, int | None]:
    """국면과 (자격 판정용) 연매출을 구한다. 애매하면 사용자가 고른 값이 우선."""
    if revenue["precise"] and revenue["summary"] is not None:
        result = judge_phase(profile, revenue["summary"], revenue["current_revenue"], elig["is_closed"])
        phase = _answers().get("phase_confirm", result.phase) if result.ambiguous else result.phase
        annual = revenue["summary"].rolling_annual_revenue
    else:
        phase = "회복불가" if elig["is_closed"] else revenue["phase_simple"]
        annual = revenue["current_revenue"] * 12 if revenue["current_revenue"] else None
    return phase, annual


def collect_conditional(fields: list[str]) -> dict:
    """조건부 답을 ss.answers 에서 읽는다. fields 에 없는 항목은 기본값(제외 방향)."""
    a = _answers()
    out = dict(business_started=None, store_area_pyeong=None, policy_loan_count_5y=0)
    if "business_started" in fields and a.get("cond_bstart_known"):
        out["business_started"] = a.get("cond_bstart")
    if "store_area_pyeong" in fields:
        area = a.get("cond_area", 0)
        out["store_area_pyeong"] = area if area and area > 0 else None
    if "policy_loan_count_5y" in fields:
        out["policy_loan_count_5y"] = int(a.get("cond_loan_count", 0))
    return out


def collect_goals() -> tuple[int | None, int | None]:
    """희망 금액 / 목표 개월수 답을 ss.answers 에서 읽는다(둘 다 선택)."""
    a = _answers()
    ti = won(a.get("goal_amt_man", 3_000)) if a.get("goal_amt_on") else None
    tm = int(a.get("goal_months", DEFAULT_TARGET_MONTHS)) if a.get("goal_months_on", True) else None
    return ti, tm


# ── 단계별 위젯 렌더 (지금 보이는 한 단계만 그린다) ──


def render_situation(fields: list[str]) -> None:
    st.markdown("안녕하세요. 상황을 몇 가지로 나눠 여쭤볼게요. **먼저 기본 상황부터요.**")
    _text("사업장 지역", "region", "서울")
    _select("재창업 상태", "restart", ("안함", "준비중", "이미함"), "안함")
    _check("폐업했거나 폐업 예정", "is_closed", False)
    _check("대출도 함께 고려", "include_loans", True)
    _nav(S_SITUATION, fields)


# 간편 모드에서 국면을 고를 때 보여줄 설명. 이름만 나열하면 "개선여지"와 "회복불가"의
# 차이를 알 수 없어 아무거나 고르게 된다.
#
# 첫 항목은 **국면 값**이고 `phase.PHASE_ORDER` 의 키와 같아야 한다.
#
# **표기를 그 값으로 통일한다.** 한때 화면용으로 "개선 여지"처럼 띄어 쓴 이름을 따로
# 들었는데, 정밀 모드의 진단 박스·하드 규칙 문장·LLM 프롬프트가 전부 붙여 쓴 값을
# 쓰고 있어 한 화면에 두 표기가 섞였다. 훑을 곳이 적은 쪽으로 맞춘다.
#
# '정상'도 넣는다. 지표 셋이 전부 0점이면 가능한 국면에 오르는데, 선택지에 없으면
# 고를 수 없는 것이 후보로 뜬다. 정책 데이터의 `적용국면` 은 셋뿐이라 '정상'을 고르면
# 후보가 0개가 되고 화면이 그 사실을 말한다 — 정밀 모드와 같은 동작이다.
#
# 설명은 phase.py 의 국면 정의(모범규준 B·C·D 등급에 대응)를 옮긴 것이고, 예시는 그
# 정의를 가게 말로 바꾼 것이다. 순서는 `PHASE_ORDER` 와 같다(좋음 → 나쁨).
_PHASE_CHOICES: list[tuple[str, str, str]] = [
    (
        "정상",
        "매출·비용·부채 어느 쪽에도 위험 신호가 없습니다.",
        "지금 벌이로 비용과 상환을 감당하고 있고 현금도 남습니다.",
    ),
    (
        "일시적자금난",
        "매출이 줄었지만 반등이나 계절 요인일 수 있고, 아직 버틸 현금이 남아 있습니다.",
        "최근 몇 달 손님이 줄었지만 작년 이맘때도 비슷했습니다.",
    ),
    (
        "개선여지",
        "완만한 하락이 이어집니다. 구조를 바꾸면 회복할 수 있는 상태입니다.",
        "1년 넘게 조금씩 줄고, 이자는 내지만 남는 것이 거의 없습니다.",
    ),
    (
        "회복불가",
        "급격한 하락이 이어지고 버틸 현금이 곧 바닥납니다.",
        "매출이 반으로 줄었고 몇 달 안에 결정을 내려야 합니다.",
    ),
]


def render_revenue(revenue: dict, fields: list[str]) -> None:
    st.markdown("**매출 자료가 있으신가요?** 있으면 추세를 분석해 더 정확히 진단해 드려요.")
    source = _radio(
        "자료 유형", "rev_source",
        ("데모 시나리오", "파일 업로드", "없어요 (간편 진행)"), "데모 시나리오",
    )
    ready = True
    if source == "데모 시나리오":
        from data.synthetic_pos import SCENARIOS

        scenario_key = _select(
            "시나리오", "scenario_key", list(SCENARIOS.keys()), "B",
            format_func=lambda k: f"{k} — {SCENARIOS[k].name}",
        )
        apply_demo_preset(scenario_key)  # 시나리오의 재무·목표 프리셋을 ans 에 적용(변경 시에만)
        # 12개월 고정. 그보다 짧으면 반기 비교(직전 6개월 vs 그 전 6개월)가 계산되지
        # 않아 자격·국면 판정이 달라진다. generate(months=)는 개발용이라 노출하지 않는다.
        st.caption("추세·반기 비교를 위해 12개월치로 생성합니다.")
        _answers().pop("pos_file", None)  # 소스가 바뀌면 업로드 잔재 제거 (#②)
    elif source == "파일 업로드":
        # 데모를 거쳐 왔다면 프리셋이 덮은 답을 되돌린다(② 기본값이 데모라 그냥 지나가도
        # 한 번 적용된다). 되돌릴 것이 없으면 아무 일도 안 한다.
        restore_from_demo()
        # 기간이 길수록 비교할 수 있는 단위가 는다. 각 단위는 "직전 N개월 합계 vs 그 전
        # N개월 합계"라 2N개월이 필요하다(분기 6 · 반기 12 · 연도 24). 정책은 신청자가
        # 유리한 단위를 고르게 하므로(소진공 긴급경영안정자금 월·분기·반기·연도),
        # 가진 자료를 잘라 올리면 될 자격이 안 되는 것으로 판정된다.
        st.caption(
            "**업력 전체 기간의 파일을 올려 주세요.** 2년이 넘으면 최근 2년만 "
            "올리셔도 됩니다."
        )
        uploaded = _file("POS 파일 (CSV/Excel)", "pos_file", type=["csv", "xlsx", "xls"])
        if uploaded is None:
            ready = False
        else:
            try:
                uploaded_summary = summarize_uploaded(uploaded.getvalue(), uploaded.name)
            except Exception as error:
                st.error(f"POS 파일을 읽지 못했습니다: {error}")
                ready = False
            else:
                st.caption(uploaded_summary.describe())
                # 반기 비교에 필요한 길이(6개월 x 2)에 못 미치면 **더 올리라고** 안내한다.
                # "좁게 판정됐습니다"로 끝나는 문구는 알리기만 하고 풀지 않는다.
                #
                # 업력이 12개월 미만이라 애초에 비교할 기간이 없는 경우에는 이 안내로
                # 할 수 있는 일이 없는데, 그때 월·분기로만 판정하는 것은 결함이 아니다 —
                # 비교할 '그 앞 6개월'이 존재하지 않고 소진공도 그렇게 판정한다
                # (2026-09-01).
                half_months = 2 * pos.COMPARE_WINDOWS["반기"]
                if len(uploaded_summary.monthly_revenue) < half_months:
                    st.warning(
                        f"올리신 자료는 {len(uploaded_summary.monthly_revenue)}개월치입니다. "
                        f"{half_months}개월이 안 되면 **최근 6개월과 그 앞 6개월을 견주는 "
                        "비교**를 할 수 없어, 일부 정책이 제한될 수 있습니다. "
                        "**자료가 더 있으시면 전체를 올려 주세요.**"
                    )
    else:
        # 여기서는 **월 매출 대략값만** 받는다. 국면 선택은 진단 화면으로 옮겼다 —
        # 이 자리에서는 아무 근거 없이 고르게 되는데, ③④⑤ 를 받고 나면 지표 셋을
        # 계산할 수 있어 가능한 국면이 좁혀진다(`simple_mode_phases`).
        restore_from_demo()  # 데모 프리셋이 덮은 답을 되돌린다(위 분기와 같은 이유)
        _num("월 매출 대략값 (만원)", "rough_rev_man", 800, min_value=0, step=10)
        _answers().pop("pos_file", None)  # 소스가 바뀌면 업로드 잔재 제거 (#②)
    _nav(S_REVENUE, fields, ready=ready)


def render_cost(fields: list[str]) -> None:
    st.markdown("**한 달 비용과 보유 현금**을 알려주세요.")
    _num("월 고정비 (만원)", "fixed_man", 500, min_value=0, step=10)
    _num(
        "월 변동비 (만원)", "var_man", 200, min_value=0, step=10,
        help="재료비·수수료 등 매출에 비례하는 비용. 비율은 코드가 계산합니다.",
    )
    _num("보유 현금 (만원)", "cash_man", 1_000, min_value=0, step=50)
    _nav(S_COST, fields)


def render_debt(fields: list[str]) -> None:
    """④ 부채 규모 — 총잔액·월상환·금리."""
    st.markdown("**기존 부채 규모**를 알려주세요. 없으면 0으로 두시면 됩니다.")
    _num("부채 총 잔액 (만원)", "debt_man", 3_000, min_value=0, step=50)
    _num("월 상환액 (만원)", "debt_pay_man", 80, min_value=0, step=5)
    if _check("부채 금리를 안다", "rate_known", False):
        _num("연 금리 (%)", "rate_pct", 5.0, min_value=0.0, max_value=30.0, step=0.1)
    _nav(S_DEBT, fields)


def render_debt2(fields: list[str]) -> None:
    """⑤ 연체·신용 — 연체일수·체납·고금리부채·직접대출잔액·신용점수."""
    st.markdown("**연체·신용 상황**이 궁금해요. 해당 없으면 그대로 두시면 됩니다.")
    _num("연체 일수", "overdue_days", 0, min_value=0, step=1)
    _check("국세·지방세 체납 중", "tax_delinquent", False)
    # 체납과 같은 성격의 사실이라 같은 자리에 둔다. ⑥ 추가 확인이 아닌 이유는
    # 두 가지다 — ⑥의 규칙은 '예 = 통과'인데 이 답은 '예'가 자격을 뺏어 방향이
    # 반대고, 후보를 만들기 전에 답이 있어야 자격 필터가 처음부터 맞는다.
    _check(
        "새출발기금·신용회복위원회 채무조정을 이행 중", "in_debt_restructuring", False,
        help="이행 중이면 소진공 일반 정책자금과 상환연장은 신청할 수 없습니다. "
             "대신 재도전특별자금(채무조정 유형)이 그 경우를 위한 자금입니다.",
    )
    if _select(
        "고금리(연 7%+) 부채", "high_rate_choice", ("모름", "없음", "금액 입력"), "모름",
        help="대환대출 대상이자 자격 요건. '모름'과 '없음'은 다르게 취급됩니다.",
    ) == "금액 입력":
        _num("그중 고금리 부분 (만원)", "high_rate_man", 0, min_value=0, step=50)
    _num(
        "소진공 직접대출 잔액 (만원)", "policy_bal_man", 0, min_value=0, step=50,
        help="상환연장제도의 대상 금액입니다.",
    )
    if _check("신용점수를 안다", "credit_known", False):
        _num("개인신용점수", "credit_score_val", 850, min_value=0, max_value=1000, step=1)
    _nav(S_DEBT2, fields)


def collect_finance() -> dict:
    """재무 답을 ss.answers 에서 원 단위 dict 로 조립한다. 폴백은 위젯 기본값과 일치시킨다
    (단계를 안 거쳤을 때 0이 아니라 기본값이 들어가도록)."""
    a = _answers()
    choice = a.get("high_rate_choice", "모름")
    if choice == "없음":
        high_rate = 0
    elif choice == "금액 입력":
        high_rate = won(a.get("high_rate_man", 0))
    else:
        high_rate = None
    return dict(
        fixed_cost=won(a.get("fixed_man", 500)),
        variable_cost=won(a.get("var_man", 200)),
        cash=won(a.get("cash_man", 1_000)),
        debt=won(a.get("debt_man", 3_000)),
        debt_payment=won(a.get("debt_pay_man", 80)),
        debt_rate=(a.get("rate_pct", 5.0) / 100 if a.get("rate_known") else None),
        high_rate_debt=high_rate,
        overdue_days=int(a.get("overdue_days", 0)),
        policy_loan_balance=won(a.get("policy_bal_man", 0)),
    )


def collect_eligibility() -> dict:
    """자격 답을 ss.answers 에서 dict 로 조립한다."""
    a = _answers()
    return dict(
        region=a.get("region", "서울"),
        restart_status=a.get("restart", "안함"),
        is_closed=a.get("is_closed", False),
        tax_delinquent=a.get("tax_delinquent", False),
        in_debt_restructuring=a.get("in_debt_restructuring", False),
        include_loans=a.get("include_loans", True),
        credit_score=(int(a.get("credit_score_val", 850)) if a.get("credit_known") else None),
    )


def _phase_box(phase: str) -> None:
    """'진단 국면: X' 만 감싸는 박스. 폭을 내용에 맞추고 화면 가운데, 텍스트도 가운데."""
    # 폭은 아래 카드 두 개를 합친 것과 같게(width:100%) — 좁으면 아래 카드와 어긋나 보인다.
    # 글자는 그 안에서 가운데 정렬. 곡률은 카드(_CARD_RADIUS)와 맞춘다.
    st.markdown(
        f"<div style='width:100%;box-sizing:border-box;border:2px solid #2b6cb0;"
        f"border-radius:{_CARD_RADIUS};padding:12px 16px;margin:8px 0 0 0;"
        "text-align:center;font-size:1.4rem;font-weight:700;'>"
        f"진단 국면: <span style='color:#2b6cb0'>{phase}</span></div>",
        unsafe_allow_html=True,
    )


def hard_rule_sentence(result) -> str | None:
    """하드 규칙으로 국면이 확정됐을 때 국면 박스 아래 띄울 **한 줄**. 아니면 None.

    판정 원인만 말한다. 지표 점수를 언급하지 않는다 — 묻지도 않은 것을 해명하면
    오히려 "점수가 몇 점인데 왜?"라는 오해를 산다(게이지와 지표 표는 그대로 있다).

    phase.py 의 하드 규칙 **3개와 1:1** 이다. 규칙을 늘리면 여기도 늘려야 한다.
    모르는 이름이 오면 규칙 이름을 그대로 넣은 문장으로 떨어지므로 화면에서 티가 난다.
        ① "폐업·휴업"           -> 회복불가
        ② "{horizon}개월 미생존" -> 일시적자금난 (이름에 horizon 이 박혀 접미로 판별한다)
        ③ "단일 지표 2점"        -> 개선여지
    """
    rule = getattr(result, "hard_rule", None)
    if not rule:
        return None
    phase = result.phase
    if rule == "폐업·휴업":
        # 점수와 판정이 어긋나지 않는다(폐업이면 점수도 대개 높다). 앞머리가 필요 없다.
        return f"폐업·휴업으로 인해 {phase} 국면으로 판정되었습니다."
    if rule.endswith("미생존"):  # ② "36개월 미생존"
        window = rule.replace(" 미생존", "")
        return (
            f"현재 기준으로는 정상이나, 지원 없이는 {window} 안에 자금이 소진돼 "
            f"{phase} 국면으로 판정되었습니다."
        )
    if rule == "단일 지표 2점":
        return (
            f"총점은 낮으나, 위험 신호가 뚜렷한 지표가 있어 "
            f"{phase} 국면으로 판정되었습니다."
        )
    return f"{rule}(으)로 인해 {phase} 국면으로 판정되었습니다."


def _hard_rule_line(result) -> None:
    """하드 규칙 문장을 국면 박스 아래 **옅은 주황 카드**로.

    맨 텍스트로 두면 국면 박스에 붙어 떠 보인다. 아래 카드들(옅은 회색)과 구분되게
    주황 계열을 쓰되, 게이지·신호등의 주황(#DE8A47)보다 훨씬 옅게 잡아 상태색으로
    읽히지 않게 한다. 폭·곡률·간격은 다른 카드와 같다.
    """
    sentence = hard_rule_sentence(result)
    if not sentence:
        return
    st.markdown(
        f"<div style='width:100%;box-sizing:border-box;background:#FCEFE0;"
        f"border-radius:{_CARD_RADIUS};padding:12px 18px;margin-top:12px;"
        "text-align:center;color:#4a5568;font-size:0.95rem;line-height:1.5;'>"
        f"{sentence}</div>",
        unsafe_allow_html=True,
    )


# 신호등 팔레트 — 초록 → 노랑 → 주황 → 빨강의 직관적 흐름은 유지하되, 원색 대신
# 채도를 낮춘 톤을 쓴다(금융 앱 톤). 게이지와 지표 표가 **같은 팔레트**를 공유해
# 한 화면에 두 색 언어가 섞이지 않게 한다.
# 색만으로 구분하지 않는다 — 점수 숫자를 항상 함께 찍어 색각 이상이 있어도 읽힌다.
_GREEN, _AMBER, _ORANGE, _RED = "#2E9E6B", "#E0B341", "#DE8A47", "#CF4E4E"
_SCORE_COLORS = {0: _GREEN, 1: _AMBER, 2: _RED}
_INK = "#1F2933"  # 본문 먹색 (순검정보다 부드럽다)


def _darken(hex_color: str, ratio: float = 0.3) -> str:
    """색을 그 색의 **어두운 톤**으로. 구간 테두리에 쓴다 — 검정 테두리는 얇은 띠에서
    투박해 보이고, 각 영역의 어두운 톤이면 경계는 살면서 색을 해치지 않는다."""
    value = hex_color.lstrip("#")
    rgb = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return "#" + "".join(f"{int(channel * (1 - ratio)):02x}" for channel in rgb)


def _indicator_table_html(rows: list[tuple[str, int, str | None]]) -> str:
    """지표 4종을 HTML 표로. 신호등 원(CSS) + 점수 + 상태 문장. 카드 안에 넣어 쓴다.

    st.dataframe 을 쓰지 않는다 — CSS 원을 넣을 수 없고 스크롤·정렬 UI가 붙어 산만하다.
    이모지 신호등도 쓰지 않는다(OS·브라우저마다 모양과 크기가 달라 행 높이가 흔들린다).

    헤더 행은 두지 않았다. 4행짜리 표라 헤더가 시각 무게만 더하고, 열 내용이
    (지표 이름 / "N점" / 설명 문장)으로 이미 자명하다.

    폭은 table-layout:fixed 로 고정한다. 설명 문장 길이가 제각각이라 셋째 열이
    남는 폭을 다 쓰게 하고, 지표·점수 열은 좁게 묶는다.
    """
    # 표 테두리는 카드 배경(#f0f2f6)보다 훨씬 어두워야 묻히지 않는다. 셀 배경을 흰색으로
    # 깔아 카드 위에 표가 '얹힌' 형태로 보이게 한다.
    border = "1px solid #4a5568"
    # word-break:keep-all — CSS 기본값은 한글을 **글자 단위**로 끊어서 "줄었습니다"가
    # "줄었습니 / 다"처럼 잘린다. keep-all 이면 어절 단위로 끊긴다.
    pad = "padding:11px 14px;word-break:keep-all;"  # 글자가 테두리에 붙지 않게 여백도
    cells = []
    for name, score, msg in rows:
        # score 가 None 이면 **재지 못한 지표**다(간편 모드의 매출감소). 0점으로 채우지
        # 않는다 — 0점은 "문제 없음"이라는 판정이고, 그렇게 채우면 낙관 쪽으로 기운다.
        if score is None:
            cells.append(
                f"<tr><td style='{pad}border:{border};color:#9aa0a6;'>{name}</td>"
                f"<td style='{pad}border:{border};white-space:nowrap;color:#9aa0a6;'>—</td>"
                f"<td style='{pad}border:{border};color:#9aa0a6;'>{msg or '측정 불가'}</td></tr>"
            )
            continue
        detail = msg if msg else ("문제 없음" if score == 0 else "위험 신호")
        color = _SCORE_COLORS.get(score, "#718096")
        dot = (
            f"<span style='display:inline-block;width:15px;height:15px;border-radius:50%;"
            f"background:{color};margin-right:9px;vertical-align:-3px;'></span>"
        )
        weight = "700" if score == 2 else "600" if score == 1 else "400"
        cells.append(
            f"<tr><td style='{pad}border:{border};color:#2d3748;'>{name}</td>"
            f"<td style='{pad}border:{border};white-space:nowrap;color:{color};font-weight:{weight};'>"
            f"{dot}{score}점</td>"
            f"<td style='{pad}border:{border};color:#2d3748;'>{detail}</td></tr>"
        )
    return (
        "<table style='table-layout:fixed;width:100%;border-collapse:collapse;background:#fff;"
        f"font-size:1rem;line-height:1.45;border:{border};'>"
        # 점수 열은 신호등 원 + 'N점'만 들어가면 되니 좁게 고정하고, 남는 폭은 설명(셋째
        # 열)에 준다 — 설명이 제일 길어 줄바꿈이 여기서 생긴다.
        "<colgroup><col style='width:30%'><col style='width:74px'><col></colgroup>"
        "<tbody>" + "".join(cells) + "</tbody></table>"
    )


MAX_PHASE_SCORE = 8  # 지표 4종 x 2점

# 국면 구간. 경계는 phase.py 상수에서 가져온다(숫자를 다시 쓰지 않는다).
# 색은 지표 표 신호등(_SCORE_COLORS)과 **같은 계열**이다 — 한 화면에 두 색 언어가
# 섞이지 않게. 초록(정상) - 노랑(일시적) - 주황(개선여지) - 빨강(회복불가).
_PHASE_BANDS = [
    (0, 0, "정상", _GREEN),
    (1, SCORE_TEMPORARY, "일시적자금난", _AMBER),
    (SCORE_TEMPORARY + 1, SCORE_IMPROVABLE, "개선여지", _ORANGE),
    (SCORE_IMPROVABLE + 1, MAX_PHASE_SCORE, "회복불가", _RED),
]


# 카드 — 진단 화면의 네 덩어리(게이지 · 버퍼 · 매출 요약 · 지표 표)를 같은 모양으로 감싼다.
# 곡률은 지원 내역 블록(st.container(border=True) = 0.5rem)과 맞춘다. 배경은 Streamlit
# 보조 배경(#f0f2f6, 진행바 '남은 단계'에서도 쓴다)이라 화면 전체가 한 계열로 묶인다.
_CARD_RADIUS = "8px"
_CARD_BG = "#f0f2f6"
_CARD = f"background:{_CARD_BG};border-radius:{_CARD_RADIUS};"


def _card(inner: str, extra: str = "") -> str:
    """옅은 회색 둥근 카드로 감싼 HTML 조각."""
    return f"<div style='{_CARD}padding:14px 16px;{extra}'>{inner}</div>"


# 말굽 게이지 — 아래가 트인 270°. 시작 225°(좌하) → 시계방향 → 끝 -45°(우하).
# 안쪽 흰 공간이 넓어야 눈금 숫자가 들어간다.
GAUGE_START_DEG, GAUGE_SWEEP_DEG = 225.0, 270.0
# 세로만 눌러 납작하게(가로는 그대로). 호가 위로 올라가 아래에 '5 / 8점' 자리가 생긴다.
GAUGE_SQUASH = 0.84


def _gauge_point(cx: float, cy: float, r: float, frac: float) -> tuple[float, float]:
    """게이지 호 위의 점. frac 0=시작(좌하), 1=끝(우하). y는 화면 좌표(아래로 증가).

    **눈금 숫자·국면 라벨·바늘이 모두 이 함수를 쓴다.** 세로 압축을 여기 한 곳에 두면
    셋이 같이 움직여 어긋나지 않는다.
    """
    angle = math.radians(GAUGE_START_DEG - frac * GAUGE_SWEEP_DEG)
    return cx + r * math.cos(angle), cy - r * math.sin(angle) * GAUGE_SQUASH


def _score_frac(score: float) -> float:
    """점수 → 게이지 위 위치(0~1). **바늘과 눈금 숫자가 이 함수 하나만 쓴다.**

    두 군데서 각도를 계산하면 바늘이 가리키는 자리와 숫자가 어긋난다.

    점수 N 은 그 점수가 **차지하는 칸의 왼쪽 경계**에 놓는다(칸 가운데가 아니다).
    칸 가운데에 두면 3점이 개선여지 구간 한복판에 찍혀, 실제로는 그 구간이 막 시작된
    값인데도 "한가운데"로 읽혔다. 경계에 놓으면 3이 노랑↔주황 경계에 서서
    "막 넘어온 값"이 그대로 보인다.
    9칸이므로 0 → 0, 8 → 8/9 이고 **오른쪽 끝(9)은 비워 둔다**(9점은 없는 값이다).
    """
    return min(max(score, 0), MAX_PHASE_SCORE) / (MAX_PHASE_SCORE + 1)


def _phase_gauge_svg(total: int) -> str:
    """국면 눈금 — 자동차 계기판형 게이지(SVG) 문자열. 카드로 감싸 쓴다.

    **정밀 모드 전용이다.** 간편 모드에는 쓰지 않는다 — 0~8 눈금이 있어 점수를 말하는
    장치인데 그쪽은 점수를 재지 않았고, 구간만 강조해도 형태 자체가 계산된 값처럼
    읽힌다. 간편 모드는 순서만 보이는 세로 축을 쓴다(`render_phase_axis`).

    이전 가로 막대는 9칸을 균등 분할해, 0점 하나뿐인 '정상'과 3점 폭인 '개선여지'가
    같은 비중으로 보였다. 여기서는 호의 길이를 **점수 폭에 비례**시킨다.

    축은 0~8점이 아니라 **9칸**(점수 0..8 각각 한 칸)으로 잡는다. 그래야 구간 폭이
    담고 있는 점수 개수와 정확히 같아지고(정상 1 · 일시적 2 · 개선여지 3 · 회복불가 3),
    바늘을 칸 가운데(score + 0.5)에 놓을 수 있다.

    **아래가 트인 270° 말굽**(GAUGE_START_DEG/SWEEP_DEG). 안쪽 흰 공간이 넓어야
    눈금 숫자 0~8 이 들어간다. 호 양 끝은 둥글게(끝점에 반지름 t/2 원을 깔고 띠를 그 위에
    덮어, 바깥으로 나온 반쪽만 둥근 마개로 남는다).

    검정 윤곽 — 바깥·안쪽 가장자리를 각각 검정 호로 두르고, **9칸 전부**에 방사형 칸
    구분선을 긋는다(점수 하나가 한 칸). 국면이 바뀌는 경계는 굵게(1.4), 같은 국면 안의
    칸 경계는 가늘게(0.7) 그어 두 위계가 함께 읽히게 한다. 국면 경계값은 phase.py 의
    SCORE_TEMPORARY·SCORE_IMPROVABLE 에서 온다(_PHASE_BANDS).

    안쪽 = 눈금 숫자(호 안쪽 가장자리에 바짝, 수평, 검정, 작게), 바깥 = **국면명만**
    (`<textPath>` 로 곡률을 따라 휜다). 호 바깥에 점수 범위를 또 쓰지 않는다 — 안쪽 눈금이
    이미 점수를 말한다. 가운데도 비운다. 점수는 호 아래 '5 / 8점' 하나뿐이고, 숫자·분모가
    **같은 크기**인 균일한 한 덩어리다.

    눈금 각도는 `_score_frac()` **하나에서만** 나온다. 바늘도 같은 함수를 쓰므로 바늘이
    5를 가리키면 그 자리에 5가 있다.

    라벨이 뒤집히는 구간은 보조 호를 **반대 방향**으로 그린다. textPath 의 글자 윗방향은
    진행 방향의 왼쪽인데, 시계방향으로 그리면 그 방향이 바깥(반지름 방향)이 된다.
    화면 아래쪽(sin θ < 0, 좌하단)에서는 그게 아래를 향해 글자가 뒤집히므로, 그 구간만
    반시계로 그려 되돌린다.

    viewBox 는 그린 것들의 **실제 경계**로 잡는다(_bbox). 여백을 남겨두면 카드 안에서
    아무리 키워도 그만큼 비어 보인다.

    하드 규칙으로 확정된 경우에도 그대로 보여준다(원인 문구가 위에서 맥락을 설명한다).
    """
    cells = MAX_PHASE_SCORE + 1  # 9칸
    cx, cy = 0.0, 0.0  # 원점 기준으로 그리고, 마지막에 viewBox 로 잘라낸다
    thick = 34.0
    r_mid = 126.0
    r_in, r_out = r_mid - thick / 2, r_mid + thick / 2
    r_tick = r_in - 11.0           # 눈금 숫자 — 띠 안쪽에 바짝
    r_name = r_out + 16.0          # 바깥 라벨 — 국면명 한 줄
    edge = "#111111"
    xs: list[float] = []
    ys: list[float] = []

    def mark(x: float, y: float, pad: float = 0.0) -> None:
        xs.extend((x - pad, x + pad))
        ys.extend((y - pad, y + pad))

    def arc_d(r: float, f0: float, f1: float) -> str:
        """f0 → f1 타원호(시계방향). 세로 압축 때문에 ry = r x GAUGE_SQUASH 다."""
        x0, y0 = _gauge_point(cx, cy, r, f0)
        x1, y1 = _gauge_point(cx, cy, r, f1)
        large = 1 if abs(f1 - f0) * GAUGE_SWEEP_DEG > 180 else 0
        return f"M {x0:.1f} {y0:.1f} A {r} {r * GAUGE_SQUASH:.1f} 0 {large} 1 {x1:.1f} {y1:.1f}"

    bands, labels, seps = [], [], []
    for lo, hi, name, color in _PHASE_BANDS:
        f0, f1 = lo / cells, (hi + 1) / cells
        # 띠를 **채운 부채꼴**로 그린다(두꺼운 선이 아니라). 타원에서는 굵은 선의 가장자리가
        # 타원이 아니라서, 선으로 그리면 윤곽선(r_in·r_out 타원호)이 띠 가장자리와 어긋난다.
        xi0, yi0 = _gauge_point(cx, cy, r_in, f0)
        xi1, yi1 = _gauge_point(cx, cy, r_in, f1)
        large = 1 if (f1 - f0) * GAUGE_SWEEP_DEG > 180 else 0
        bands.append(
            f"<path d='{arc_d(r_out, f0, f1)} L {xi1:.1f} {yi1:.1f} "
            f"A {r_in} {r_in * GAUGE_SQUASH:.1f} 0 {large} 0 {xi0:.1f} {yi0:.1f} Z' "
            f"fill='{color}'/>"
        )
        mid = (f0 + f1) / 2
        # 호 바깥은 **국면명만**. 점수는 호 안쪽 눈금(0~8)이 이미 말하고 있어서
        # 바깥에 '3-5점'을 또 쓰면 같은 정보가 두 번 나온다.
        #
        # 라벨은 곡률 없이 **수평**으로 놓는다(textPath 를 쓰지 않는다). 휘어 놓으면
        # 구간마다 기운 각도가 달라 어떤 라벨은 눕고 어떤 라벨은 뒤집혀 읽힌다.
        # 폭이 모자라면 폰트를 줄이지 않고 **줄을 나눈다**(일시적자금난 → 일시적/자금난).
        size = 13.0
        lines = [name] if len(name) <= 4 else [name[: (len(name) + 1) // 2], name[(len(name) + 1) // 2:]]
        half_w = max(len(line) for line in lines) * size / 2
        half_h = len(lines) * size * 1.15 / 2
        # 라벨 상자가 호(r_out)에 닿지 않게 바깥으로 민다. 상자의 반폭·반높이를 각도에
        # 투영한 만큼 더 밀어야, 가로로 긴 라벨(회복불가)도 옆구리가 호에 걸리지 않는다.
        theta = math.radians(GAUGE_START_DEG - mid * GAUGE_SWEEP_DEG)
        radius = r_out + 10 + half_w * abs(math.cos(theta)) + half_h * abs(math.sin(theta))
        lx, ly = _gauge_point(cx, cy, radius, mid)
        spans = "".join(
            f"<tspan x='{lx:.1f}' dy='{0 if i == 0 else size * 1.15:.1f}'>{line}</tspan>"
            for i, line in enumerate(lines)
        )
        labels.append(
            f"<text x='{lx:.1f}' y='{ly - (len(lines) - 1) * size * 1.15 / 2:.1f}' "
            f"text-anchor='middle' dominant-baseline='central' font-family='inherit' "
            f"font-size='{size}' font-weight='700' fill='{edge}'>{spans}</text>"
        )
        mark(lx - half_w, ly - half_h)
        mark(lx + half_w, ly + half_h)

    # 칸 구분선 — **점수 하나가 한 칸**이라 9칸 전부를 나눈다(경계 8곳).
    # 국면이 바뀌는 경계(_PHASE_BANDS 의 lo, 즉 phase.py 상수에서 온 값)는 굵게,
    # 같은 국면 안의 칸 경계는 가늘게 그어 두 위계가 함께 읽히게 한다.
    # 굵기는 띠 두께(thick)에 비례시킨다 — 띠를 바꿔도 선만 가늘어지지 않게.
    # 윤곽선만 더 굵게(구간·칸 경계선은 그대로).
    w_edge, w_phase, w_cell = thick * 0.13, thick * 0.115, thick * 0.06
    phase_edges = {band[0] / cells for band in _PHASE_BANDS if band[0]}
    for i in range(1, cells):
        f = i / cells
        bx0, by0 = _gauge_point(cx, cy, r_in, f)
        bx1, by1 = _gauge_point(cx, cy, r_out, f)
        seps.append(
            f"<line x1='{bx0:.1f}' y1='{by0:.1f}' x2='{bx1:.1f}' y2='{by1:.1f}' "
            f"stroke='{edge}' stroke-width='{w_phase if f in phase_edges else w_cell:.2f}'/>"
        )

    # 호 양 끝의 둥근 마개 — 띠를 그 위에 덮어 바깥 반쪽만 남긴다.
    # 세로 압축 때문에 끝면(r_in~r_out)의 실제 길이가 각도마다 달라, 그 길이의 절반을
    # 반지름으로 삼아야 마개가 띠 두께에 딱 맞는다.
    caps = []
    for f, color in (
        (0.0, _PHASE_BANDS[0][3]),
        (1.0, _PHASE_BANDS[-1][3]),
    ):
        ix, iy = _gauge_point(cx, cy, r_in, f)
        ox, oy = _gauge_point(cx, cy, r_out, f)
        ex, ey = (ix + ox) / 2, (iy + oy) / 2
        cap_r = math.hypot(ox - ix, oy - iy) / 2
        caps.append(
            f"<circle cx='{ex:.1f}' cy='{ey:.1f}' r='{cap_r:.1f}' fill='{color}' "
            f"stroke='{edge}' stroke-width='{w_edge:.2f}'/>"
        )
        mark(ex, ey, cap_r + w_edge)

    # 바깥·안쪽 윤곽선 (마개 원과 접선으로 이어진다)
    outlines = [
        f"<path d='{arc_d(r, 0.0, 1.0)}' fill='none' stroke='{edge}' "
        f"stroke-width='{w_edge:.2f}'/>"
        for r in (r_in, r_out)
    ]
    for f in (i / 40 for i in range(41)):  # 호 전체를 경계 계산에 넣는다
        mark(*_gauge_point(cx, cy, r_out, f), w_edge)

    # 눈금 숫자 0~8 — 수평, 검정, 작게. 각도는 바늘과 같은 _score_frac 에서만 나온다.
    ticks = []
    for score in range(cells):
        tx, ty = _gauge_point(cx, cy, r_tick, _score_frac(score))
        ticks.append(
            f"<text x='{tx:.1f}' y='{ty:.1f}' text-anchor='middle' dominant-baseline='central' "
            f"font-family='inherit' font-size='11.5' font-weight='600' fill='{edge}'>{score}</text>"
        )
        mark(tx, ty, 7.0)

    # 바늘 — **회전축을 호가 트인 아래쪽으로 내린다**(중심이 아니다). 아래에 '5 / 8점'이
    # 들어갈 자리는 남긴다.
    #
    # ⚠ 축을 내리면 "중심에서 눈금 각도만큼 회전"이 통하지 않는다. 축에서 본 방향과
    # 원 중심에서 본 방향이 다르기 때문이다. 그래서 **눈금 좌표를 조준**한다 —
    # 눈금 숫자를 놓을 때 쓴 그 점(_gauge_point + _score_frac)을 구하고, 새 축에서
    # 그 점을 향하는 방향으로 그린다. 눈금과 바늘이 같은 좌표를 쓰므로 어긋날 수 없다.
    # 회전축과 점수를 **함께** 같은 만큼 내린다(둘 사이 간격은 그대로). 호·눈금·라벨은
    # 움직이지 않는다 — 이 값은 바늘과 점수에만 더한다.
    drop = 12.0
    pivot_y = cy + r_mid * GAUGE_SQUASH * 0.42 + drop
    target_x, target_y = _gauge_point(cx, cy, r_tick, _score_frac(total))  # 눈금 숫자의 자리
    span = math.hypot(target_x - cx, target_y - pivot_y) or 1.0
    ux, uy = (target_x - cx) / span, (target_y - pivot_y) / span
    tip_x, tip_y = target_x - ux * 9.0, target_y - uy * 9.0  # 숫자 안쪽 가장자리에서 멈춘다
    px, py = -uy, ux  # 뿌리의 폭 방향
    root = 7.0
    needle = (
        f"<polygon points='{tip_x:.1f},{tip_y:.1f} "
        f"{cx + px * root:.1f},{pivot_y + py * root:.1f} "
        f"{cx - px * root:.1f},{pivot_y - py * root:.1f}' fill='{edge}'/>"
        f"<circle cx='{cx}' cy='{pivot_y:.1f}' r='{root + 1.5}' fill='{edge}'/>"
        f"<circle cx='{cx}' cy='{pivot_y:.1f}' r='3.2' fill='#ffffff'/>"
    )
    # 호 아래 한 줄 — 게이지 중심선(x = cx) 위, **말굽 양 끝과 같은 높이**. 글자 기준선을
    # 호 끝의 세로 위치에 맞춰 나란히 보이게 한다.
    score_size = 17.0
    score_y = _gauge_point(cx, cy, r_mid, 0.0)[1] + drop  # 회전축과 같은 만큼 내린다
    caption = f"{total} / {MAX_PHASE_SCORE}점"
    score_text = (
        f"<text x='{cx}' y='{score_y:.1f}' text-anchor='middle' font-family='inherit' "
        f"font-size='{score_size:.0f}' font-weight='700' fill='{edge}'>{caption}</text>"
    )
    # 글자 상자를 실제 크기로만 잡는다(넉넉한 정사각으로 잡으면 아래 여백이 부풀어
    # viewBox 가 세로로 늘어난다). 기준선 위로 대문자 높이, 아래로 디센더.
    # 폭 계수는 한글이 1글자당 거의 한 칸이라 숫자 섞인 '5 / 8점'보다 크게 잡는다.
    score_half_w = len(caption) * score_size * 0.55 / 2
    mark(cx - score_half_w, score_y - score_size * 0.78)
    mark(cx + score_half_w, score_y + score_size * 0.26)

    # viewBox — 여백을 **위아래끼리 · 좌우끼리** 맞춘다(위=아래, 좌=우. 두 쌍의 값이
    # 서로 같을 필요는 없다). 경계는 국면 라벨과 '5 / 8점'까지 포함한 실제 내용 범위다.
    #
    # 좌우는 게이지 **중심선(cx) 기준 대칭**으로 잡는다 — 왼쪽 '일시적/자금난'은 두 줄,
    # 오른쪽 '회복불가'는 한 줄이라 폭이 달라, 그냥 내용 상자에 맞추면 호가 한쪽으로
    # 쏠려 보인다. 넓은 쪽에 맞춰 양쪽을 같게 한다.
    # 위아래는 내용 위끝·아래끝에 같은 여백을 준다(호는 아래가 트여 있어 중심선 대칭이
    # 성립하지 않는다 — 대칭으로 잡으면 아래에 빈 공간만 생긴다).
    pad = 4.0
    half_x = max(abs(min(xs) - cx), abs(max(xs) - cx)) + pad
    min_x, width = cx - half_x, half_x * 2
    min_y, height = min(ys) - pad, max(ys) - min(ys) + pad * 2
    return (
        f"<svg viewBox='{min_x:.1f} {min_y:.1f} {width:.1f} {height:.1f}' "
        "preserveAspectRatio='xMidYMid meet' style='width:100%;height:100%;display:block;'>"
        + "".join(caps) + "".join(bands) + "".join(outlines) + "".join(seps)
        + "".join(ticks) + "".join(labels) + needle + score_text + "</svg>"
    )


def render_diagnosis(revenue: dict, elig: dict, profile: UserProfile, fields: list[str]) -> None:
    """진단 — 국면 박스 + 핵심 수치(버퍼·월 영업이익·매출 추세) + 지표 4종 + 국면 눈금.

    reasons 는 점수 붙은 지표만 담아 4개 중 일부만 나온다. phase_indicator_rows 로
    0점짜리까지 4개 전부(0점=문제 없음), 2점은 눈에 띄게 보여준다.
    """
    precise, summary = revenue["precise"], revenue["summary"]
    is_closed = elig["is_closed"]

    if precise and summary is not None:
        result = judge_phase(profile, summary, revenue["current_revenue"], is_closed)
        phase = st.session_state.get("phase_confirm", result.phase) if result.ambiguous else result.phase

        _phase_box(phase)
        _hard_rule_line(result)  # 하드 규칙으로 확정됐으면 원인 한 줄 (국면 박스 바로 아래)

        # ①게이지 · ②버퍼 를 좌우 카드로 — 세로로 흩어지면 한 화면에 안 들어온다.
        # 두 카드의 높이를 같게 맞추고(min-height) 내용은 세로 가운데 정렬한다.
        # (영업이익은 아래 지표 표에 '영업이익 · N점'으로 이미 있어 중복이라 뺐다)
        # 두 카드는 **폭·높이가 같다**(좌우 균등 분할 + 같은 height). 내용은 각 카드 안에서
        # 가로·세로 모두 가운데. 게이지 SVG 는 viewBox 가 내용에 딱 맞고
        # preserveAspectRatio=meet 이라, 카드 안쪽(패딩 제외)을 잘리지 않는 선에서 꽉 채운다.
        # 카드 간격을 화면 전체에서 일정하게: 국면 박스 아래 margin 을 0으로 두고,
        # ①②가 든 열 묶음에도 나머지 카드와 같은 margin-top(12px)을 준다
        # (.st-key-diag_row). 그러지 않으면 이 줄만 간격이 달라 보인다.
        # 높이는 게이지 viewBox 비율(약 1.7:1)에 맞춘다 — 카드가 그보다 높으면 게이지는
        # 폭에 맞춰 커진 뒤 위아래에 빈 공간만 남아 작아 보인다. 두 카드 모두 같은 값.
        card_h = "height:210px;display:flex;align-items:center;justify-content:center;"
        with st.container(key="diag_row"):
            left, right = st.columns([1, 1], vertical_alignment="center")
            left.markdown(
                _card(
                    f"<div style='width:100%;height:100%;'>{_phase_gauge_svg(result.total)}</div>",
                    card_h + "padding:8px;",
                ),
                unsafe_allow_html=True,
            )
            buf = months_label(result.buffer_months) if result.buffer_months is not None else "—"
            right.markdown(
                _card(
                    # 라벨과 숫자가 붙어 보이지 않게 사이 여백을 주고, 둘 다 한 단계씩
                    # 키운다(라벨은 본문보다 조금 크게, 숫자는 눈에 확 들어오게).
                    "<div style='text-align:center;'>"
                    "<div style='color:#4a5568;font-size:1.12rem;line-height:1.45;"
                    "margin-bottom:16px;'>지원 없이 버틸 수 있는 개월 수</div>"
                    f"<div style='font-size:3.1rem;font-weight:800;line-height:1.1;'>{buf}</div></div>",
                    card_h,
                ),
                unsafe_allow_html=True,
            )

        # ③ 매출 요약 — 네 항목을 카드 **안쪽 폭** 기준으로 균등 분산(왼쪽 쏠림 방지).
        # flex 가 카드 안에 있어 padding 을 뺀 폭으로 다시 계산된다.
        # 진단의 근거라 caption 이 아니라 본문 색으로 둔다.
        # 형광펜 — **변화율 세 항목만** 칠한다. '최근 3개월 평균 매출'은 절대값,
        # '직전 3개월 대비'는 기준 설명이라 강조 대상이 아니다. 항목명과 수치를 한
        # 덩어리로 칠해야 어색하지 않다(수치만 칠하면 항목명이 떠 보인다).
        # 색은 옅은 온기 있는 노랑. 게이지 노랑(#E0B341)보다 채도를 크게 낮춰,
        # 이미 색이 많은 진단 화면에서 상태색이 아니라 '형광펜'으로 읽히게 한다.
        # 강조는 **글자 굵기로만**. 형광펜(배경색)은 뺐다 — 세로 padding 을 가진 상자와
        # 맨 글자를 같은 높이로 맞추는 데 실패했고(인라인·baseline·flex 세 방식 모두
        # 미세하게 어긋났다), 배경이 빠지면 그 문제가 아예 사라진다. 강조 효과는 굵기로
        # 충분하고, 배경이 없으니 폭도 항목당 12px 씩 줄어 카드 여백으로 돌릴 수 있다.
        em = "font-weight:700;"
        summary_items = [
            f"최근 3개월 평균 매출 {summary.recent_3m_avg / MAN:,.0f}만원",
            f"직전 3개월 대비<span style='{em}'>매출 {summary.revenue_change_3m:+.1%}</span>",
            f"<span style='{em}'>손님 수 {summary.order_count_change_3m:+.1%}</span>",
            f"<span style='{em}'>객단가 {summary.avg_order_value_change_3m:+.1%}</span>",
        ]
        # 배치 (나) — 항목은 **내용 폭**으로 두고, 사이 여백을 전부 같게 한다.
        # (가) 안(flex:1 로 칸을 나누고 칸 안에서 가운데 정렬)은 실측에서 전제가 깨졌다:
        # 칸1·2 의 글자가 카드 폭의 1/4 보다 넓어 그 칸들은 글자 폭에 고정되고 칸3·4 만
        # 같아진다(199/195/133/133). 칸도 구분선 간격도 균등하지 않으면서 항목별 여백만
        # 6/6/18/24px 로 제각각이 된다. (나)는 남는 폭을 6개 틈에 똑같이 나눠 주므로
        # **항목 사이 여백이 전부 같다** — 실제로 균등해 보이는 쪽이다.
        # 구분자는 문자가 아니라 1px 세로선(div)이라 폭 계산에 끼어들지 않는다.
        # 클래스는 좁은 화면용 미디어 쿼리가 인라인 스타일을 덮으려고 붙인다.
        # 넓은 화면에서는 이 클래스에 걸린 규칙이 없어 지금과 똑같이 그려진다.
        cell = "display:flex;align-items:center;gap:5px;"
        line = ("<div class='sn-revline' style='width:1px;background:#cbd5e0;"
                "align-self:stretch;'></div>")
        st.markdown(
            _card(
                "<div class='sn-revsum' style='display:flex;flex-wrap:nowrap;"
                "align-items:stretch;"
                "justify-content:space-between;line-height:1.45;color:#2d3748;"
                "font-size:0.88rem;white-space:nowrap;'>"
                + line.join(f"<div class='sn-revcell' style='{cell}'>{item}</div>"
                            for item in summary_items)
                + "</div>",
                # 형광펜을 빼서 생긴 폭(항목당 12px)을 **카드 좌우 여백으로** 돌린다 —
                # 첫 항목이 카드 왼쪽 벽에 붙어 답답했다.
                "margin-top:12px;padding:12px 22px;",
            ),
            unsafe_allow_html=True,
        )

        # ④ 지표 4종 — 제목까지 한 카드 안에. 0점짜리도 전부 보여준다.
        st.markdown(
            _card(
                f"<div style='font-weight:700;margin-bottom:9px;'>"
                f"지표 점수 (총 {result.total} / {MAX_PHASE_SCORE}점)</div>"
                + _indicator_table_html(
                    phase_indicator_rows(profile, summary, revenue["current_revenue"])
                ),
                "margin-top:12px;",
            ),
            unsafe_allow_html=True,
        )

        for warning in summary.warnings:
            st.caption(f"⚠ {warning}")

        if result.ambiguous:
            # 안내문 + 질문 + 라디오를 **한 블록**으로. 따로 두면 안내문과 질문이
            # 분리돼 보인다. 배경·간격은 .st-key-phase_confirm CSS 가 맡는다.
            with st.container(key="phase_confirm"):
                st.markdown(
                    "<div style='font-weight:700;margin-bottom:2px;'>경계에 가까운 판정입니다.</div>"
                    "<div style='color:#4a5568;'>점수가 국면 경계에 걸쳐 있어요. "
                    "어느 국면이라고 생각하시는지 골라 주세요.</div>",
                    unsafe_allow_html=True,
                )
                # 라벨은 숨긴다 — 바로 위 안내가 이미 묻고 있어 '국면이 맞나요?'는 중복이다.
                _radio(
                    "국면이 맞나요?", "phase_confirm",
                    list(neighbor_phases(result)), result.phase,
                    horizontal=True, label_visibility="collapsed",
                )
        _nav(S_DIAGNOSIS, fields)
        return

    # ── 간편 모드 ── **한 장으로 끝낸다.**
    # POS 가 없어도 ③④⑤ 에서 받은 값으로 지표 4종 중 **셋**이 계산된다. 그 셋으로
    # 가능한 국면을 좁히고, 남은 것 중에서 사용자가 고른다(`simple_mode_phases`).
    #
    # 확정 박스(`_phase_box`)와 무지원 생존 개월 박스는 두지 않는다. 국면은 시뮬레이션의
    # **입력 조건**이지 결과가 아니라, 결과값과 같은 층위로 놓으면 확정된 진단처럼 읽힌다.
    #
    # '추정치' 고지는 여기 두지 않는다 — 결과 화면 상단이 이미 말한다. 고를 자리에서
    # 미리 말하면 아직 나오지도 않은 결과를 깎는다.
    phases, rows = simple_mode_phases(profile, revenue["current_revenue"], is_closed)
    auto = len(phases) == 1
    if auto:
        # 지표 셋만으로 이미 하나다. 물을 것이 없으므로 확정하고 넘어간다.
        phase = phases[0]
        _answers()["phase_simple"] = phase
        st.markdown(f"**입력하신 값으로 국면이 하나로 정해집니다 — {phase}.**")
    else:
        phase = _answers().get("phase_simple")
        st.markdown(
            f"**입력하신 값으로는 아래 {len(phases)}개 중 하나입니다.** "
            "어느 쪽에 더 가깝다고 보시나요?"
        )
    if note := exclusion_note(phases, rows, is_closed):
        st.markdown(note)
    render_indicator_expander(rows)
    render_phase_axis(phases, phase)

    if auto:
        st.caption("이 국면으로 정책 추천이 진행됩니다.")
    else:
        # default=None — **아직 안 고른 상태**로 시작한다. 기본값을 주면 사용자가
        # 고르지 않았는데 고른 것처럼 되고, 그러면 좁혀 놓은 의미가 없다.
        phase = _radio(
            "국면 선택", "phase_simple", list(phases), None,
            horizontal=len(phases) <= 3, label_visibility="collapsed",
        )
        # 아직 안 골랐을 때의 안내는 여기 두지 않는다 — `_nav` 이 ready=False 면
        # "입력을 마치면 다음으로 넘어갈 수 있어요"를 이미 띄운다.
        if phase:
            st.caption("선택하신 국면으로 정책 추천이 진행됩니다.")
    _nav(S_DIAGNOSIS, fields, ready=auto or phase is not None)


def render_indicator_expander(rows: list[tuple[str, int, str | None]]) -> None:
    """계산 근거 — 접어 둔다. 펼치면 잴 수 있었던 지표와 점수 방향이 나온다.

    접힘으로 두는 이유는 이 화면의 주인공이 국면 선택이기 때문이다. 그렇다고 빼면
    "좁혀집니다"가 근거 없는 주장이 된다 — 접어 두면 한 줄만 차지하면서 계산 근거를
    보여준다는 이 서비스의 핵심을 지킬 수 있다.

    매출감소는 점수 대신 '측정 불가'로 남긴다. **총점은 보여주지 않는다** — 넷 중
    셋으로 낸 합계는 8점 만점 기준의 국면 경계와 견줄 수 없다.
    """
    with st.expander("어떻게 판단했는지 보기", key="exp_simple_scores"):
        st.caption(
            "지표마다 **0점이 좋고 2점이 나쁩니다.** 매출 자료가 없어 매출 감소는 "
            "재지 못했고, 그래서 국면이 하나로 정해지지 않고 범위로 남습니다."
        )
        st.markdown(
            _indicator_table_html(
                list(rows) + [("매출 감소", None, "매출 자료가 없어 측정 불가")]
            ),
            unsafe_allow_html=True,
        )


def render_phase_axis(phases: list[str], picked: str | None) -> None:
    """국면 넷을 **좋음 → 나쁨** 세로 축으로 보여준다. 가능한 것에만 테두리.

    **게이지를 쓰지 않는다.** 게이지에는 0~8 눈금이 있어 점수를 말하는 장치인데,
    간편 모드는 점수를 재지 않았다. 구간만 강조해도 그 형태 자체가 계산된 값처럼
    읽힌다(정밀 모드는 실제 점수가 있어 그대로 둔다).

    대신 **순서**만 보인다. 국면 순서는 `phase.PHASE_ORDER` 가 정한 값이라 근거가
    있고, 점수를 보여주는 것과 다르다. 카드가 세로로 쌓이므로 축도 세로로 둔다.

    걸러진 국면을 숨기지 않는다 — 숨기면 "왜 이것만 나오나"에 답이 없고, 다 똑같이
    보이면 좁혀졌다는 사실이 안 보인다.
    """
    cards = []
    for value, body, example in _PHASE_CHOICES:
        live = value in phases
        chosen = value == picked
        edge = ("2.5px solid #2b6cb0" if chosen else
                "1.5px solid #9dbbe0" if live else "1px solid #e3e6ea")
        ink = _INK if live else "#a8adb3"
        sub = "#2d3748" if live else "#bcc0c5"
        cards.append(
            f"<div style='border:{edge};border-radius:{_CARD_RADIUS};"
            f"padding:9px 13px;margin-bottom:7px;background:#fff;'>"
            f"<div style='font-weight:700;color:{ink};'>{value}</div>"
            f"<div style='color:{sub};line-height:1.5;font-size:0.92rem;'>{body}</div>"
            f"<div style='color:{sub};line-height:1.5;font-size:0.92rem;'>"
            f"예) {example}</div></div>"
        )
    # 축 — 위(좋음) 파랑에서 아래(나쁨) 빨강으로. 색만으로 뜻을 전하지 않게
    # '좋음'·'나쁨' 글자를 함께 둔다.
    axis = (
        "<div style='display:flex;flex-direction:column;align-items:center;"
        "flex:0 0 auto;padding:2px 0;'>"
        "<div style='font-size:0.78rem;color:#2b6cb0;font-weight:700;'>좋음</div>"
        f"<div style='color:{_GREEN};line-height:1;font-size:0.8rem;'>&#9650;</div>"
        "<div style='flex:1;width:5px;border-radius:3px;"
        f"background:linear-gradient(#2b6cb0,{_RED});'></div>"
        f"<div style='color:{_RED};line-height:1;font-size:0.8rem;'>&#9660;</div>"
        "<div style='font-size:0.78rem;color:#CF4E4E;font-weight:700;'>나쁨</div>"
        "</div>"
    )
    st.markdown(
        _card(
            "<div style='display:flex;gap:12px;align-items:stretch;'>"
            + axis
            + f"<div style='flex:1;'>{''.join(cards)}</div></div>",
            "margin:8px 0;",
        ),
        unsafe_allow_html=True,
    )


def render_conditional(fields: list[str], questions: list[dict] | None = None) -> None:
    """⑥ 조건부 추가 질문 + **자격 확인**. 둘 다 "후보에 오른 정책이 요구하는 것"이라
    성격이 같아 한 단계에 담는다(진행바에 칸을 더하지 않아도 된다).

    ⚠ 이 화면의 답이 후보 풀을 바꾼다(예: 정책자금 지원 횟수가 소진공 자금을 뺀다).
      그래서 위쪽 답을 고치면 아래 자격 확인 질문이 나타나거나 사라질 수 있다.
      논리적으로는 정확한 동작이지만 화면이 살짝 움직인다.
    """
    st.markdown("**후보에 오른 정책이 요구하는 항목만** 몇 가지 더 여쭤볼게요.")
    if "business_started" in fields:
        if _check("사업 개시 연월을 입력", "cond_bstart_known", False):
            _date(
                "사업 개시일", "cond_bstart", datetime.date(2019, 3, 1),
                min_value=datetime.date(2000, 1, 1), max_value=datetime.date.today(),
            )
        else:
            st.caption("입력하지 않으면 업력·사업영위 기간 판정이 필요한 정책은 제외됩니다.")
    if "store_area_pyeong" in fields:
        if _num("점포 전용면적 (평)", "cond_area", 0.0, min_value=0.0, step=1.0) <= 0:
            st.caption("면적을 모르면 점포철거비는 금액을 계산할 수 없어 제외됩니다.")
    if "policy_loan_count_5y" in fields:
        _num("최근 5년 정책자금 지원 횟수", "cond_loan_count", 0, min_value=0, step=1)

    # ── 자격 확인 ──
    # '아니오'와 '잘 모르겠어요'는 똑같이 그 조합을 제외한다(기획서 5.2.5 ⑥).
    # 확인하지 못한 자격을 있다고 가정하면 신청했다가 거절당한다.
    #
    # 답에 따른 결과는 적지 않는다. "'아니오'를 고르면 제외합니다"는 협박처럼 읽히고,
    # 사실대로 답하는 것 말고 다른 선택지가 있는 것처럼 보인다.
    for item in questions or []:
        st.markdown("---")
        options = item.get("options") or (CONFIRM_YES, CONFIRM_NO, CONFIRM_UNSURE)
        # 선택지가 문장이면 가로 라디오에 안 들어가 줄바꿈이 지저분해진다.
        # 짧은 예/아니오/모름만 가로로 두고, 택1 객관식은 세로로 늘어놓는다.
        short = len(options) <= 3 and max(len(o) for o in options) <= 12
        if short:
            # 항목이 여럿(①②③)이면 "하나라도"를 밝힌다 — 택1 요건이라 전부 해당해야
            # 하는 것으로 읽히면 안 된다. 단문 예/아니오에는 그 말이 군더더기다.
            lines = confirm_question_lines(item["question"])
            st.markdown(
                "**아래 항목 중 하나라도 해당사항이 있는지 체크해주세요.**"
                if len(lines) > 1 else "**해당사항이 있는지 체크해주세요.**"
            )
            for line in lines:
                st.markdown(line)
        else:
            # 객관식은 선택지가 이미 갈래를 담고 있다. 질문 원문까지 풀어 쓰면
            # 같은 문장이 목록과 라디오에 두 번 나온다.
            st.markdown("**해당하는 것에 체크해주세요.**")
        _radio(
            "해당하시나요?", item["key"], options, options[-1],
            horizontal=short, label_visibility="collapsed",
        )
    _nav(S_CONDITIONAL, fields)


def render_goals(fields: list[str]) -> None:
    """⑦ 희망 금액 / 목표 생존 개월수 (둘 다 선택). 답은 collect_goals 이 읽는다."""
    st.markdown("**마지막이에요.** 필요한 금액이나 기간이 정해져 있나요? (선택)")
    if _check("필요한 금액이 정해져 있음", "goal_amt_on", False):
        _num("희망 금액 (만원)", "goal_amt_man", 3_000, min_value=0, step=100)
    if _check("목표 생존 개월수 지정", "goal_months_on", True):
        # "버티고 싶은 기간"으로 물으면 희망을 묻는 말이 되어 "영원히"가 답이 된다.
        # 필요 최소치를 묻는 쪽으로 쓴다.
        st.markdown(
            "사업 상황을 다시 살펴볼 여유를 얻기 위해, "
            "꼭 필요한 최소한의 버팀 기간을 알려주세요."
        )
        _slider(
            "목표 생존 개월수", "goal_months", 1, SIM_MONTHS, DEFAULT_TARGET_MONTHS,
            label_visibility="collapsed",
        )
    _nav(S_GOALS, fields)


# ══════════════════════════════════════════════════════════
# 메인 — 단계를 위에서 아래로 쌓는다
# ══════════════════════════════════════════════════════════


def render_results_step(
    db: PolicyDB, revenue: dict, elig: dict, profile: UserProfile,
    phase: str, annual: int | None, fields: list[str],
) -> None:
    """결과 단계 — session_state 의 모든 답으로 파이프라인을 돌려 카드를 그린다."""
    conditional = collect_conditional(fields)
    ctx = build_context(phase, elig, conditional, revenue["summary"] if revenue["precise"] else None, annual)
    target_inflow, target_months = collect_goals()

    drop_ids = excluded_by_confirm(
        st.session_state.get("_confirm_questions") or [], _answers()
    )
    with st.spinner("추천 조합 계산 중..."):
        (
            groups, reports, evals, extras,
            baseline_survival, goal_satisfied, resolve_info,
        ) = run_pipeline(db, profile, ctx, target_inflow, target_months, drop_ids)
        # 자격은 되지만 최종 후보에 못 든 것들 — LLM 이 "왜 이건 안 골랐나"를 쓸 재료.
        picked = {m.label for g in groups for m in g.members}
        rejected = sorted(
            (ev for ev in evals if ev.label not in picked),
            key=lambda v: (-v.survival_months, v.debt_at_exit),
        )
        prompt_text = to_prompt_text(
            groups, reports,
            baseline_survival=baseline_survival,
            goal_months=target_months,
            goal_satisfied=goal_satisfied,
            extras=extras,
            context=ctx,
            profile=profile,
            rejected=rejected,
        )

    # 순서 재배열은 하지 않는다 — render_results 가 picks 를 돌며 조합을 직접 찾는다.
    # 그룹 단위로 재배열하던 옛 방식은 한 그룹에서 두 개를 고르면 하나를 잃었다.
    advice = get_advice(prompt_text, [m.label for g in groups for m in g.members])

    # 미래 자격 카드 — 화면에 **실제로 그려질 셋**을 비교 대상으로 넘긴다.
    # LLM 픽이 있으면 그 셋, 없으면 폴백(groups[:3])이다. render_results 가 카드를
    # 고르는 규칙과 같아야 "기다릴 값어치"가 사용자가 보는 것과 맞는다.
    by_label = {m.label: g for g in groups for m in g.members}
    if advice is not None and advice.ok and advice.picks:
        shown = [by_label[p.label] for p in advice.picks if p.label in by_label]
    else:
        shown = groups[:3]
    future_groups, future_reports, future_months = run_future_pipeline(
        db, profile, ctx, revenue["summary"], profile.monthly_revenue, shown,
        target_months, drop_ids, _answers(),
        from_ensemble=revenue.get("forecast_is_ensemble", False),
    )

    render_results(
        groups, reports, extras, profile, ctx,
        baseline_survival, goal_satisfied, target_months, resolve_info,
        prompt_text, revenue["precise"], advice, bool(drop_ids),
        future_groups=future_groups, future_reports=future_reports,
        future_months=future_months,
    )
    st.divider()
    _nav(S_RESULTS, fields)  # ◀ 이전으로 돌아가 답을 고칠 수 있다


def render_done_step(fields: list[str]) -> None:
    """완료 화면 — 체크 표시와 인사 한 줄.

    체크는 **SVG** 로 그린다. 이모지(✓·✅)는 OS·브라우저마다 모양과 크기가 달라
    가운데 정렬이 흔들린다.
    세로로도 화면 가운데쯤 오게 위에 여백을 둔다(위쪽에 붙으면 끝난 느낌이 안 난다).
    ◀ 이전으로 결과 화면에 돌아갈 수 있다 — 완료를 눌렀다고 막지 않는다.

    agent.py 가 붙으면 이 화면에 요약·저장·공유 같은 것이 들어갈 수 있다.
    지금은 최소한만 두고, 아래에 덧붙이기만 하면 되게 남겨 둔다.
    """
    check = (
        "<svg viewBox='0 0 100 100' width='92' height='92' aria-hidden='true'>"
        "<circle cx='50' cy='50' r='44' fill='none' stroke='#2E9E6B' stroke-width='6'/>"
        "<path d='M30 52 L44 66 L71 37' fill='none' stroke='#2E9E6B' stroke-width='8' "
        "stroke-linecap='round' stroke-linejoin='round'/></svg>"
    )
    st.markdown(
        "<div style='display:flex;flex-direction:column;align-items:center;"
        "justify-content:center;min-height:46vh;gap:18px;'>"
        f"{check}"
        "<div style='font-size:1.25rem;font-weight:700;color:#1F2933;'>"
        "이용해주셔서 감사합니다.</div></div>",
        unsafe_allow_html=True,
    )
    _nav(S_DONE, fields)


def main() -> None:
    st.set_page_config(page_title="소상공인 나침반", page_icon="🧭", layout="centered")
    ss = st.session_state
    if "step" not in ss:
        ss.step = S_SITUATION
    _answers()  # 답 저장소 초기화 (위젯이 아니라서 purge 되지 않는다)

    # 헤더는 시작 화면에서만 크게. ② 이후로는 한 줄로 줄인다 — 큰 헤더가 화면 위쪽을
    # 200px 가까이 먹어서 ◀ 이전 ▶ 다음 이 한 화면에 안 들어왔다.
    # '처음부터'는 없애지 않는다(갈 곳이 없어진다). 작은 헤더 오른쪽 끝에 둔다.
    header = st.columns([5, 1], vertical_alignment="center")
    with header[0]:
        if ss.step == S_SITUATION:
            st.title("🧭 소상공인 나침반")
            st.caption("융자·보조금·채무조정을 36개월 현금흐름으로 함께 계산해 정책 조합을 추천합니다.")
        else:
            st.markdown(
                "<div style='font-size:1.05rem;font-weight:700;padding:2px 0;'>"
                "🧭 소상공인 나침반</div>",
                unsafe_allow_html=True,
            )
    with header[1]:
        if st.button("↻ 처음부터"):
            ss.clear()
            st.rerun()

    db = load_db()
    cur = ss.step

    # 답은 전부 session_state 에 있으므로, 지금 어느 단계를 보든 필요한 파생값을 계산할 수
    # 있다(국면·연매출·조건부 질문 목록). 이 값들로 진행 표시와 ⑥ 건너뛰기를 결정한다.
    revenue = collect_revenue()
    elig = collect_eligibility()
    profile = compute_profile(revenue)
    phase, annual = compute_phase_and_annual(revenue, elig, profile)
    fields = needed_conditional_fields(phase, annual, elig["include_loans"])
    # 후보에 붙은 자격 확인 질문. ⑥ 단계가 이것도 함께 묻고, 답이 후보를 거른다.
    # 풀을 구하려면 컨텍스트가 필요한데, 그 컨텍스트가 ⑥ 답을 쓰므로 서로 물린다 —
    # 지금까지의 답으로 풀을 구하고, 답이 바뀌면 Streamlit 이 다시 그린다.
    ctx_now = build_context(
        phase, elig, collect_conditional(fields),
        revenue["summary"] if revenue["precise"] else None, annual,
    )
    questions = confirm_questions(eligible_candidates(db, profile, ctx_now), ctx_now)
    # 진행바·단계 이동(_step_after/_step_before)도 이 목록을 봐야 ⑥ 을 건너뛸지 안다.
    ss["_confirm_questions"] = questions

    render_progress(cur, fields)
    st.divider()

    # 지금 보이는 한 단계만 그린다 (스크롤이 생기지 않게)
    if cur == S_SITUATION:
        render_situation(fields)
    elif cur == S_REVENUE:
        render_revenue(revenue, fields)
    elif cur == S_COST:
        render_cost(fields)
    elif cur == S_DEBT:
        render_debt(fields)
    elif cur == S_DEBT2:
        render_debt2(fields)
    elif cur == S_DIAGNOSIS:
        render_diagnosis(revenue, elig, profile, fields)
    elif cur == S_CONDITIONAL:
        if not fields and not questions:  # 물을 것이 하나도 없으면 건너뛴다
            ss.step = S_GOALS
            st.rerun()
        render_conditional(fields, questions)
    elif cur == S_GOALS:
        render_goals(fields)
    elif cur == S_DONE:
        render_done_step(fields)
    else:
        render_results_step(db, revenue, elig, profile, phase, annual, fields)


if __name__ == "__main__":
    main()
