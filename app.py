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
    후보 그룹 3~8개 → [LLM 최종 3개 선정]  ← 아직 미구현(seam)
        ▼
    화면: 국면 진단 · 추천 카드 · 현금흐름 그래프 · 융자 슬라이더


대화형 흐름
-----------
사이드바 폼이 아니라 **위에서 아래로 쌓이는 대화**다. 질문은 `st.chat_message`
말풍선으로 던지고, 사용자는 말풍선 안의 위젯(number_input·selectbox·button)으로
답한다. 답한 위젯은 그 자리에 남아 언제든 다시 만질 수 있고, 고치면 그 아래가
Streamlit 재실행으로 자동 재계산된다.

    진행 상태는 `st.session_state.step`(정수) 하나로 관리한다.
    step 이상 도달한 단계만 렌더링하고, 프론티어 단계에만 "다음 ▶" 버튼을 둔다.
    지나간 단계의 위젯은 계속 살아 있어(=session_state) 앞 답을 고칠 수 있다.

질문은 관련된 것끼리 한 말풍선에 2~4개씩 묶어 묻는다. 하나씩 쪼개면 답할 횟수가
너무 많아진다.

    ① 기본 상황(지역·재창업·폐업·대출포함)
    ② 매출 자료(POS 업로드/데모/없어요)
    ③ 월 비용(고정비·변동비·현금)
    ④ 부채(총잔액·월상환·금리·연체 …)
    ⑤ 진단 결과 표시 + (애매하면) 국면 확인
    ⑥ needed_conditional_fields() 가 돌려준 것만
    ⑦ 조건 설정(희망 금액·목표 개월수)
    → 결과

자유 텍스트 파싱은 아직 없다 — LLM이 필요하므로 agent.py 이후로 미룬다.
지금은 위젯으로 답하되 흐름만 대화형이다.


LLM 최종 3개 선정도 아직 없다
-----------------------------
`select_final_three()` 자리를 만들어 두고, 지금은 코드가 만든 후보 그룹을
레이어 순으로 그대로 보여준다. agent.py 가 생기면 그 함수 안만 갈아끼운다.


명명 규칙: 코드 식별자는 영어, 도메인 값은 한글. 주석은 한글.
"""

from __future__ import annotations

import datetime
import io
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import optimizer, pos
from core.cashflow import (
    SIM_MONTHS,
    Applied,
    UserProfile,
    flat_revenue,
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
from core.optimizer import CandidateGroup, apply_links, prerequisites, select, to_prompt_text
from core.phase import (
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


@st.cache_data(show_spinner=False)
def summarize_scenario(key: str, months: int) -> pos.PosSummary:
    """데모 시나리오(A/B/C)의 합성 POS에서 지표를 뽑는다."""
    from data.synthetic_pos import generate

    return pos.summarize(pos.load(generate(key, months=months)))


# ══════════════════════════════════════════════════════════
# LLM 자리 (미구현 seam)
# ══════════════════════════════════════════════════════════


def select_final_three(
    prompt_text: str, groups: list[CandidateGroup], context: UserContext
) -> list[CandidateGroup]:
    """최종 3개 선정 — **LLM의 몫이지만 아직 미구현이다.**

    agent.py 가 생기면 여기서 호출한다:

        return agent.select_final_three(prompt_text, groups, context)

    지금은 코드가 만든 후보 그룹을 **레이어 순 그대로** 돌려준다.
    순위가 아니라 좁히기까지가 코드의 역할이므로(optimizer 참조),
    최종 선정 없이도 후보 그룹을 나란히 보여 주는 것 자체가 의미가 있다.
    """
    return groups  # TODO(agent): LLM 최종 3개 선정으로 교체


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
    """36개월 예상 매출선. forecast.py 가 없어 임시 추세선을 쓴다.

        정밀   POS 최근 3개월 평균을 현재값으로, 3개월 증감률을 월율로 환산해 외삽
        간편   현 수준 유지(추세 없음)
    """
    if precise and summary is not None:
        current = max(summary.recent_3m_avg, 1)
        # 3개월 증감률을 월율로: (1+change_3m)^(1/3) - 1
        monthly_change = (1 + summary.revenue_change_3m) ** (1 / 3) - 1
        return trend_revenue(current, monthly_change)
    return flat_revenue(max(rough_revenue, 1))


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
        store_area_pyeong=conditional["store_area_pyeong"],
        business_started=conditional["business_started"],
        revenue_drop_by_window=drop_by_window,
        revenue_drop_rate=drop_rate,
        revenue_growth_rate=growth_rate,
        annual_revenue=annual_revenue,
        include_loans=elig["include_loans"],
        policy_loan_count_5y=conditional["policy_loan_count_5y"],
    )


def _phase_inputs(
    profile: UserProfile, summary: pos.PosSummary, current_revenue: int
) -> tuple[float, float, float]:
    """국면 지표의 세 입력값 (월 영업이익 · 월 이자비용 · 매출감소율)."""
    operating_profit = current_revenue * (1 - profile.variable_ratio) - profile.fixed_cost
    rate = profile.existing_debt_rate or 0.0
    interest_cost = profile.existing_debt * rate / 12
    drop_rate = summary.drop_rate(("분기", "반기"))
    return operating_profit, interest_cost, drop_rate


def judge_phase(
    profile: UserProfile, summary: pos.PosSummary, current_revenue: int, is_closed: bool
) -> PhaseResult:
    """지표 4종 + 하드 규칙으로 국면을 판정한다."""
    operating_profit, interest_cost, drop_rate = _phase_inputs(profile, summary, current_revenue)
    buffer_months = simulate(profile, []).survival_months
    return judge(
        operating_profit=operating_profit,
        interest_cost=interest_cost,
        revenue_drop_rate=drop_rate,
        overdue_days=profile.overdue_days,
        is_closed=is_closed,  # 하드 규칙 ①: 폐업·휴업 확정이면 회복불가
        buffer_months=buffer_months,
    )


def phase_indicator_rows(
    profile: UserProfile, summary: pos.PosSummary, current_revenue: int
) -> list[tuple[str, int, str | None]]:
    """4개 지표의 (이름, 점수, 사유문장). 진단 화면에서 0점짜리도 전부 보여주려고 쓴다.

    PhaseResult.reasons 는 문장이 붙은 지표만 담고(그마저 ICR 2점은 문장이 None),
    지표별로 매핑되지 않는다. 그래서 판정과 **같은 입력**으로 score_* 를 다시 불러
    지표별 (점수, 문장)을 얻는다. 계산을 바꾸는 게 아니라 표시용으로 재사용하는 것이다.
    """
    operating_profit, interest_cost, drop_rate = _phase_inputs(profile, summary, current_revenue)
    return [
        ("ICR (이자 대비 영업이익)", *score_icr(operating_profit, interest_cost)),
        ("매출 감소", *score_revenue_drop(drop_rate)),
        ("연체", *score_overdue(profile.overdue_days)),
        ("영업이익", *score_operating_profit(operating_profit)),
    ]


# ══════════════════════════════════════════════════════════
# 파이프라인 실행
# ══════════════════════════════════════════════════════════


def run_pipeline(
    db: PolicyDB,
    profile: UserProfile,
    ctx: UserContext,
    target_inflow: int | None,
    target_months: int | None,
):
    """CLAUDE.md의 파이프라인을 입력 경로에 맞게 호출한다.

    Returns:
        (groups, reports, evals, extras, baseline_survival, goal_satisfied, resolve_info)
    """
    pool = eligible_candidates(db, profile, ctx)
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
    result = evaluated.result
    principal = result.principal_during[at - 1] if 0 < at <= len(result.principal_during) else 0.0
    # 만기일시 융자면 '만기 일시상환', 아니면 거치 종료 후 원금 상환 시작
    is_bullet = any(
        it.candidate.is_loan and it.candidate.get("상환방식") == "만기일시"
        for it in evaluated.combo.items
    )
    label = "만기 일시상환" if is_bullet else "원금 상환 시작"
    return at, f"{label} {principal / MAN:,.0f}만원"


def render_flow_chart(evaluated: Evaluated, key: str, report=None) -> None:
    """36개월 현금흐름 **막대**그래프. 잔고(만원)만 그린다.

    - 꺾은선 대신 월별 막대라 24개월차 절벽 같은 급락이 눈에 잘 들어온다.
    - x축 0~horizon 고정, y축 하한 0(balance_at 은 음수가 없다). shortfall_at 은
      역산용 가상 궤적이라 그리지 않는다(오해 방지).
    - 소진 시점 세로선(N개월차 소진), 함정 CLIFF 시점 세로 점선(만기 일시상환 …).
    - 값 라벨은 시작·최고점·절벽 직후 3곳만(37개 전부 붙이면 안 읽힌다).
    """
    result = evaluated.result
    months = list(range(len(result.balance_at)))  # 0 .. horizon
    balance = [b / MAN for b in result.balance_at]

    marker = _cliff_marker(evaluated, report)
    cliff_at = marker[0] if marker is not None else None

    # 절벽 이후 구간 막대는 옅은 색으로 — 위험 구간을 눈에 띄게(과하지 않게)
    base_color, after_color = "#2b6cb0", "#a9c7ef"
    bar_colors = [
        after_color if (cliff_at is not None and m >= cliff_at) else base_color
        for m in months
    ]

    # 값 라벨: 시작(t=0)·최고점·절벽 직후 3곳만
    peak_idx = max(range(len(balance)), key=lambda i: balance[i])
    label_idx = {0, peak_idx}
    if cliff_at is not None and cliff_at < len(balance):
        label_idx.add(cliff_at)
    text = [f"{balance[i]:,.0f}" if i in label_idx else "" for i in months]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=months, y=balance, marker_color=bar_colors,
            text=text, textposition="outside", cliponaxis=False,
            textfont=dict(size=14),  # 값 라벨을 본문 크기로 키운다
            hovertemplate="%{x}개월차 · %{y:,.0f}만원<extra></extra>",  # 툴팁도 만원 정수
        )
    )
    y_top = max(balance) if any(b > 0 for b in balance) else 1
    fig.update_xaxes(range=[-0.5, SIM_MONTHS + 0.5], dtick=6, title_text="개월차")
    # (만원)은 우상단 '단위: 만원'이 대신한다. SI 접두어(40k) 없이 40,000 로.
    fig.update_yaxes(
        range=[0, y_top * 1.22], title_text="현금잔고",  # 값 라벨이 막대와 겹치지 않게 위 여백 확보
        tickformat=",d", exponentformat="none",
    )

    # 소진 시점 강조 — 끝까지 버티면 표시하지 않는다
    if not result.survives_full_term:
        fig.add_vline(
            x=result.survival_months,
            line=dict(color="crimson", width=2, dash="dash"),
            annotation_text=f"{result.survival_months}개월차 소진",
            annotation_position="top",
        )

    # 절벽(급락) 시점 — 함정 판정에 걸린 경우에만. 막대 위에 겹쳐 그린다.
    if marker is not None:
        at, note = marker
        fig.add_vline(
            x=at, line=dict(color="#dd6b20", width=2, dash="dot"),
            annotation_text=note, annotation_position="bottom",
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


def render_support_detail(evaluated: Evaluated, baseline_first_payment: float) -> None:
    """지원 내역 — '무엇을 얼마나 받는지'. 둥근 테두리 박스로 감싼다.

    simulate가 만든 문장(result.notes)은 원 단위라, 표시 직전에 money_to_man 으로
    만원 단위로 바꾼다(계산은 그대로 원 단위, 화면 표기만 변환).
    """
    result = evaluated.result
    with st.container(border=True):
        st.markdown("**지원 내역**")
        for note in result.notes:
            st.markdown(f"- {money_to_man(note)}")

        inflow = evaluated.combo.total_inflow
        forgiven = result.forgiven
        delta_payment = result.payment_during[0] - baseline_first_payment
        parts = [f"현금 유입 {fmt_man(inflow)}"]
        if forgiven > 0:
            parts.append(f"부채 감소 {fmt_man(forgiven)} (추정치)")
        parts.append(f"첫 달 상환 부담 {delta_payment / MAN:+,.0f}만원")
        st.caption(" · ".join(parts))


def render_traps(report) -> None:
    """함정 경고 + 회색지대 원본 숫자."""
    for trap in report.traps:
        st.warning(f"⚠ {trap.message}")
    if report.burden_ratio_now is not None:
        st.caption(
            f"월 상환액 {report.payment_now / MAN:,.0f}만원 / "
            f"월 영업이익 {report.operating_now / MAN:,.0f}만원 ({report.burden_ratio_now:.0%})"
        )
    else:
        st.caption(
            f"월 상환액 {report.payment_now / MAN:,.0f}만원 "
            f"(월 영업이익 {report.operating_now / MAN:,.0f}만원 — 적자라 부담률 산출 불가)"
        )


def render_loan_slider(member: Evaluated, profile: UserProfile, key: str) -> Evaluated:
    """융자 금액 슬라이더. 움직이면 이 카드만 다시 계산한다(순위는 고정).

    조정 대상은 신규유입 융자 하나뿐이다. 슬라이더가 바뀌면 그 금액으로
    조합을 다시 시뮬레이션해 그래프·2축·함정을 갱신한 Evaluated를 돌려준다.
    """
    loan_item = next(
        (
            it
            for it in member.combo.items
            if it.candidate.is_loan and it.candidate.effect == "신규유입"
        ),
        None,
    )
    if loan_item is None:
        return member  # 조정할 융자가 없으면 슬라이더도 없다

    cand = loan_item.candidate
    ceiling = int(cand.get("금액_한도") or 0)
    floor = int(cand.get("금액_하한") or 0)
    # 신청 단위는 재원마다 다르다(소진공 100만·서울시 만). 후보 데이터에 이미 있다.
    unit = int(cand.get("신청금액_단위") or 1_000_000)
    if ceiling <= floor:
        return member  # 조정 폭이 없다

    # 슬라이더는 만원 단위로 보여주고, step 도 신청 단위(만원)로 잡는다.
    min_man, max_man = floor // MAN, ceiling // MAN
    step_man = max(1, unit // MAN)
    default_man = min(max(loan_item.effective_amount // MAN, min_man), max_man)
    picked_man = st.slider(  # transient-widget: 결과 카드 what-if. 단계 답이 아니라
        # 되돌아가면 초기화되는 게 맞다(ss.answers 를 안 탄다).
        f"융자 금액 조정 — {cand.policy_name}",
        min_value=min_man,
        max_value=max_man,
        value=int(default_man),
        step=step_man,
        format="%d만원",
        key=key,
    )
    # 슬라이더 값을 실제 신청 가능 금액으로 스냅한다(단위 올림·금액_하한 처리).
    # 같은 규칙을 UI에서 새로 구현하지 않고 combos.snap_amount 를 그대로 통과시킨다.
    # 스냅 결과가 슬라이더 표시와 어긋나면 스냅된 값을 화면에 쓴다.
    amount = snap_amount(cand, picked_man * MAN)
    grant_sum = sum(
        it.effective_amount
        for it in member.combo.items
        if it.candidate.effect == "신규유입" and not it.candidate.is_loan
    )
    st.caption(f"총 유입 내역 — 보조금(확정) {fmt_man(grant_sum)} + 융자(조정) {fmt_man(amount)}")

    if amount == loan_item.effective_amount:
        return member

    # 이 조합만 다시 계산한다. 순위(select)는 다시 돌리지 않는다.
    new_items = tuple(
        Applied(it.candidate, amount) if it is loan_item else it for it in member.combo.items
    )
    adjusted = Evaluated(combo=Combo(items=new_items), result=simulate(profile, new_items))
    st.caption(
        f"→ 조정 후: 생존 {months_label(adjusted.result.survival_months)} · "
        f"종료 시점 부채 {fmt_man(adjusted.result.debt_at_exit)}"
    )
    return adjusted


def build_schedule_df(result) -> pd.DataFrame:
    """상환 스케줄 표(만원). SimResult 가 이자/원금을 나눠 담으므로 그대로 읽는다.

    payment_during == interest_during + principal_during 이 보장되므로(cashflow.py),
    debt_at 차분으로 역산하던 근사(융자 실행 달에 어긋남)가 필요 없다.

    상환이 끝난 뒤 **끝에서부터 전부 0인 행은 잘라낸다.** 거치 중 원금만 0인 중간
    행은 다른 열이 0이 아니라 남는다. 기존 부채가 남아 잔액이 0이 아니면 자연히
    36회차까지 나온다.
    """
    money_cols = ("원금상환액(월)", "대출이자(월)", "상환원리금", "대출잔액")
    rows = [
        {
            "회차": m,
            "원금상환액(월)": round(result.principal_during[m - 1] / MAN),
            "대출이자(월)": round(result.interest_during[m - 1] / MAN),
            "상환원리금": round(result.payment_during[m - 1] / MAN),
            "대출잔액": round(result.debt_at[m] / MAN),
        }
        for m in range(1, len(result.payment_during) + 1)
    ]
    while rows and all(rows[-1][c] == 0 for c in money_cols):
        rows.pop()
    df = pd.DataFrame(rows)
    return df.set_index("회차") if not df.empty else df


def render_group_card(
    idx: int,
    group: CandidateGroup,
    reports: dict,
    profile: UserProfile,
    ctx: UserContext,
    baseline_first_payment: float,
    expanded: bool,
) -> None:
    """후보 그룹 하나를 카드로 그린다."""
    member = group.members[0]  # 대표. 같은 좌표라 2축 성능은 동일하다.
    trap_icon = " ⚠" if group.has_trap else ""
    # 제목: 후보 N │ 정책명(파랑) │ 성과. LLM 최종 선정 전이라 번호는 순위가 아니다.
    # st.expander 라벨의 마크다운 색상 지시자(:blue[…])는 Streamlit 1.62 문서상 지원되나
    # 렌더링은 브라우저에서 확인해야 한다. 안 먹으면 색 대신 다른 방식으로 바꾼다.
    perf = f"생존 {months_label(group.survival_months)} · 종료 시점 부채 {fmt_man(group.debt_at_exit)}"
    title = f"**후보 {idx}**  │  :blue[{member.label}]{trap_icon}  │  {perf}"
    with st.expander(title, expanded=expanded):
        st.caption(f"레이어 {group.layer}")

        # 같은 성능의 다른 상환조건 멤버가 있으면 함께 보인다
        if len(group.members) > 1:
            others = " / ".join(m.label for m in group.members[1:])
            st.caption(f"같은 성능의 다른 조합: {others}")

        render_support_detail(member, baseline_first_payment)

        # 융자 슬라이더 — 움직이면 이 카드의 그래프·표·2축·함정만 갱신(순위는 고정)
        shown = render_loan_slider(member, profile, key=f"loan_{idx}")

        # 함정 판정을 먼저 구한다(그래프의 절벽 마커가 이 리포트를 쓴다).
        # 슬라이더로 조정했으면 조정본으로 다시 판정한다.
        if shown is member:
            report = reports[member.label]
        else:
            report = detect_traps(profile, shown, ctx.phase)

        render_flow_chart(shown, key=f"chart_{idx}", report=report)

        # 상환 스케줄 표 — 그래프와 같은 시뮬레이션 결과(shown)를 쓴다.
        # 슬라이더를 움직이면 그래프와 함께 이 표도 바뀐다.
        #
        # 표에 내용이 더 있다는 걸 알린다. st.dataframe 은 캔버스 기반 그리드라
        # 스크롤바를 항상 보이게 하는 CSS가 먹지 않는다(마우스를 올려야 나타난다).
        # 그래서 CSS에 의존하지 않는 방법을 쓴다 — 제목에 총 회차, 표 아래에 스크롤 안내.
        schedule = build_schedule_df(shown.result)
        rows = len(schedule)
        title = (
            f"상환 스케줄 보기 (총 {rows}회차 · 단위: 만원)" if rows
            else "상환 스케줄 보기 (단위: 만원)"
        )
        with st.expander(title):
            st.caption("대출잔액은 기존 부채와 신규 융자를 합친 값입니다.")
            if schedule.empty:
                st.caption("상환 일정이 없습니다.")
            else:
                st.dataframe(schedule, use_container_width=True, height=280)
                st.caption("↕ 스크롤하여 전체 보기")

        render_traps(report)

        # 창 밖 만기 일시상환
        if shown.result.balloon_after_horizon > 0:
            st.warning(
                f"⚠ {shown.result.balloon_due_at}개월차에 원금 "
                f"{fmt_man(shown.result.balloon_after_horizon)}이 일시상환으로 남습니다 "
                f"(36개월 시뮬레이션 밖)"
            )

        # 자격 확인 필요 · 계산 전제 등 주의 문구 (optimizer가 프롬프트에 쓰는 것과 동일)
        for note in optimizer._caveats(member.combo.items, ctx):
            st.caption(f"※ {note}")

        # 신청 전 해야 할 일 · 신청처 — 카드에 오른 조합에 대해서만
        for line in prerequisites(member.combo.items):
            st.info(f"신청 전 확인: {line}")
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
) -> None:
    """추천 조합 화면 — 카드·부가안내·심사용 프롬프트."""
    if not precise:
        st.info("간편 모드입니다. 매출 추세가 없어 모든 수치는 **입력값 기준 추정치**입니다.")

    # 두 축이 무엇인지 상단에 한 번 설명한다 (카드마다 축별 강점 라벨이 붙으므로).
    st.info(
        "두 축은 **생존 개월수**(지원을 받으면 몇 개월 버티는가)와 "
        "**종료 시점 부채**(그때 빚이 얼마 남는가)입니다."
    )
    st.caption(f"지원을 받지 않으면 {months_label(baseline_survival)} 버틸 수 있습니다.")

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

    if not groups:
        st.warning("조건에 맞는 조합을 찾지 못했습니다. 자격 조건이나 입력값을 조정해 보세요.")
        return

    # LLM 최종 선정 전이라 후보 번호는 순위가 아니다 (agent.py 가 붙으면 지울 줄).
    st.caption("LLM 최종 선정 전이라 번호는 순위가 아닙니다.")
    st.caption(
        "ℹ 슬라이더로 금액을 바꾸면 그 카드의 그래프·표·2축이 다시 계산됩니다. 카드 순서는 고정입니다."
    )

    baseline_first_payment = simulate(profile, []).payment_during[0]
    for idx, group in enumerate(groups, start=1):
        render_group_card(
            idx, group, reports, profile, ctx,
            baseline_first_payment,
            expanded=(idx == 1),  # 1순위 기본 펼침
        )

    # 부가 추천 (컨설팅·법률자문 등)
    if extras:
        st.markdown("**함께 안내할 것** — 순위엔 안 들지만 자격이 되는 지원")
        for cand in extras:
            note = cand.get("_순위제외근거") or cand.get("조건") or ""
            st.markdown(f"- {cand}" + (f" — {note[:80]}" if note else ""))

    # LLM 프롬프트 (심사용) — agent.py 가 이 텍스트를 받아 최종 3개를 고른다
    with st.expander("심사용: LLM 프롬프트 보기"):
        st.info("최종 3개 선정은 LLM의 몫이며 아직 미구현입니다. 아래는 코드가 만든 프롬프트입니다.")
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


def _step_after(cur: int, fields: list[str]) -> int:
    """다음 단계. 물을 것이 없으면 ⑥ 추가 확인을 건너뛴다."""
    nxt = cur + 1
    if nxt == S_CONDITIONAL and not fields:
        nxt = S_GOALS
    return min(nxt, S_RESULTS)


def _step_before(cur: int, fields: list[str]) -> int:
    """이전 단계. 건너뛴 ⑥은 되돌아갈 때도 건너뛴다."""
    prv = cur - 1
    if prv == S_CONDITIONAL and not fields:
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
            if cur < S_RESULTS:
                label = "결과 보기 ▶" if _step_after(cur, fields) == S_RESULTS else "다음 ▶"
                if st.button(label, key=f"next_{cur}", disabled=not ready, use_container_width=True):
                    ss.step = _step_after(cur, fields)
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
    base = (
        "flex:1;white-space:nowrap;text-align:center;font-size:11px;"
        "padding:4px 2px;border-radius:6px;"
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
    if cur in OUTPUT_STEPS:
        counter = STEP_TITLES[cur]
    else:
        pos = answer_steps.index(cur) + 1 if cur in answer_steps else total
        counter = f"{pos} / {total}"

    cells = []
    for step in segments:
        state = "done" if step < cur else ("cur" if step == cur else "todo")
        cells.append(
            f'<div style="{_seg_style(step in OUTPUT_STEPS, state)}">{STEP_TITLES[step]}</div>'
        )
    bar = (
        '<div style="display:flex;gap:4px;width:100%;margin:2px 0 8px 0;">'
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
    ans = _answers()
    current = ans.get(key, default)
    idx = options.index(current) if current in options else 0
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
        phase_simple=ans.get("phase_simple", "일시적자금난"),
        current_revenue=current_revenue, ready=ready,
    )


def compute_profile(revenue: dict) -> UserProfile:
    """session_state 의 재무·매출 답으로 UserProfile 을 조립한다."""
    finance = collect_finance()
    monthly_revenue = build_monthly_revenue(revenue["precise"], revenue["summary"], revenue["rough"])
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


def render_revenue(revenue: dict, fields: list[str]) -> None:
    st.markdown("**매출 자료가 있으신가요?** 있으면 추세를 분석해 더 정확히 진단해 드려요.")
    source = _radio(
        "자료 유형", "rev_source",
        ("데모 시나리오", "파일 업로드", "없어요 (간편 진행)"), "데모 시나리오",
    )
    ready = True
    if source == "데모 시나리오":
        from data.synthetic_pos import SCENARIOS

        _select(
            "시나리오", "scenario_key", list(SCENARIOS.keys()), "B",
            format_func=lambda k: f"{k} — {SCENARIOS[k].name}",
        )
        # 12개월 고정. 그보다 짧으면 반기 비교(직전 6개월 vs 그 전 6개월)가 계산되지
        # 않아 자격·국면 판정이 달라진다. generate(months=)는 개발용이라 노출하지 않는다.
        st.caption("추세·반기 비교를 위해 12개월치로 생성합니다.")
        _answers().pop("pos_file", None)  # 소스가 바뀌면 업로드 잔재 제거 (#②)
    elif source == "파일 업로드":
        uploaded = _file("POS 파일 (CSV/Excel)", "pos_file", type=["csv", "xlsx", "xls"])
        if uploaded is None:
            ready = False
        else:
            try:
                st.caption(summarize_uploaded(uploaded.getvalue(), uploaded.name).describe())
            except Exception as error:
                st.error(f"POS 파일을 읽지 못했습니다: {error}")
                ready = False
    else:
        _num("월 매출 대략값 (만원)", "rough_rev_man", 800, min_value=0, step=10)
        _select("국면 (직접 선택)", "phase_simple", ("일시적자금난", "개선여지", "회복불가"), "일시적자금난")
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
        include_loans=a.get("include_loans", True),
        credit_score=(int(a.get("credit_score_val", 850)) if a.get("credit_known") else None),
    )


def _phase_box(phase: str) -> None:
    """'진단 국면: X' 만 감싸는 박스. 폭을 내용에 맞추고 화면 가운데, 텍스트도 가운데."""
    st.markdown(
        "<div style='display:flex;justify-content:center;margin:8px 0 14px 0;'>"
        "<div style='border:2px solid #2b6cb0;border-radius:12px;padding:12px 32px;"
        "text-align:center;font-size:1.4rem;font-weight:700;'>"
        f"진단 국면: <span style='color:#2b6cb0'>{phase}</span></div></div>",
        unsafe_allow_html=True,
    )


def _phase_gauge(total: int) -> None:
    """0-8점 국면 눈금 + 현재 위치 세모. 경계는 phase.py 상수에서 가져온다(재입력 금지).

    범위 표기는 하이픈(1-2점)을 쓴다 — '~'는 마크다운 취소선(~~)으로 오인돼 글자가 사라진다.
    """
    bands = [
        (0, 0, "정상", "#9aa0a6"),
        (1, SCORE_TEMPORARY, "일시적자금난", "#2b6cb0"),
        (SCORE_TEMPORARY + 1, SCORE_IMPROVABLE, "개선여지", "#dd6b20"),
        (SCORE_IMPROVABLE + 1, 8, "회복불가", "#c0392b"),
    ]
    cells = []
    for score in range(9):
        color = next(c for lo, hi, _, c in bands if lo <= score <= hi)
        # 현재 점수 칸: 큰 세모 + 바로 아래 '현재 N점'을 한 덩어리로.
        if score == total:
            mark = (
                "<div style='color:#c0392b;font-weight:800;font-size:20px;line-height:1;"
                "margin-top:1px;'>▲</div>"
                f"<div style='color:#c0392b;font-size:11px;font-weight:700;white-space:nowrap;'>"
                f"현재 {total}점</div>"
            )
        else:
            mark = ""
        cells.append(
            f"<div style='flex:1;text-align:center;font-size:11px;'>"
            f"<div style='height:10px;background:{color};border-radius:2px;'></div>{mark}</div>"
        )
    st.markdown(
        "<div style='display:flex;gap:2px;margin-top:6px;'>" + "".join(cells) + "</div>",
        unsafe_allow_html=True,
    )
    legend = " ─ ".join(
        (f"{lo}점 {name}" if lo == hi else f"{lo}-{hi}점 {name}") for lo, hi, name, _ in bands
    )
    st.caption(legend)


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

        # 큰 숫자는 '버틸 수 있는 개월 수' 하나만, 가운데 정렬.
        # (영업이익은 아래 지표 목록에 '영업이익 · N점'으로 이미 있어 중복이라 뺐다)
        _, mid, _ = st.columns([1, 2, 1])
        buf = months_label(result.buffer_months) if result.buffer_months is not None else "—"
        mid.metric("지원 없이 버틸 수 있는 개월 수", buf)

        # 매출 요약 — 한 줄, │ 구분, 본문 검정(caption 아님). 진단의 근거라 묻히면 안 된다.
        st.markdown(
            f"최근 3개월 평균 매출 {summary.recent_3m_avg / MAN:,.0f}만원 │ "
            f"직전 3개월 대비 {summary.revenue_change_3m:+.1%} │ "
            f"손님 수 {summary.order_count_change_3m:+.1%} │ "
            f"객단가 {summary.avg_order_value_change_3m:+.1%}"
        )

        if result.hard_rule:
            st.markdown(f"**판정 근거: {result.hard_rule} (확정)**")

        # 지표 4종 — 0점짜리도 전부. 2점은 눈에 띄게.
        st.markdown(f"**지표 점수 (총 {result.total} / 8점)**")
        for name, score, msg in phase_indicator_rows(profile, summary, revenue["current_revenue"]):
            detail = msg if msg else ("문제 없음" if score == 0 else "위험 신호")
            if score == 2:
                st.markdown(f"- :red[**{name} · {score}점**] — {detail}")
            elif score == 1:
                st.markdown(f"- **{name} · {score}점** — {detail}")
            else:
                st.markdown(f"- :gray[{name} · {score}점 — {detail}]")
        if result.hard_rule:
            st.caption("하드 규칙으로 확정돼 점수는 판정에 쓰이지 않았습니다.")

        _phase_gauge(result.total)

        for warning in summary.warnings:
            st.caption(f"⚠ {warning}")

        if result.ambiguous:
            st.info("경계에 가까운 판정입니다. 맞는지 골라 주세요.")
            _radio(
                "국면이 맞나요?", "phase_confirm",
                list(neighbor_phases(result)), result.phase, horizontal=True,
            )
        _nav(S_DIAGNOSIS, fields)
        return

    # 간편 모드 — 국면을 직접 선택했으므로 지표 판정이 없다
    phase = "회복불가" if is_closed else revenue["phase_simple"]
    _phase_box(phase)
    st.markdown("매출 추세가 없어 입력하신 값과 선택하신 국면으로 진행합니다. 결과 수치는 **추정치**입니다.")
    _nav(S_DIAGNOSIS, fields)


def render_conditional(fields: list[str]) -> None:
    """⑥ 조건부 추가 질문. fields 에 든 것만 묻는다(답은 collect_conditional 이 읽는다)."""
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
    _nav(S_CONDITIONAL, fields)


def render_goals(fields: list[str]) -> None:
    """⑦ 희망 금액 / 목표 생존 개월수 (둘 다 선택). 답은 collect_goals 이 읽는다."""
    st.markdown("**마지막이에요.** 필요한 금액이나 버티고 싶은 기간이 정해져 있나요? (선택)")
    if _check("필요한 금액이 정해져 있음", "goal_amt_on", False):
        _num("희망 금액 (만원)", "goal_amt_man", 3_000, min_value=0, step=100)
    if _check("목표 생존 개월수 지정", "goal_months_on", True):
        _slider("몇 개월 이상 버티고 싶으세요?", "goal_months", 1, SIM_MONTHS, DEFAULT_TARGET_MONTHS)
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

    with st.spinner("추천 조합 계산 중..."):
        (
            groups, reports, evals, extras,
            baseline_survival, goal_satisfied, resolve_info,
        ) = run_pipeline(db, profile, ctx, target_inflow, target_months)
        prompt_text = to_prompt_text(
            groups, reports,
            baseline_survival=baseline_survival,
            goal_months=target_months,
            goal_satisfied=goal_satisfied,
            extras=extras,
            context=ctx,
        )
        # LLM 자리 (지금은 groups 그대로 반환)
        groups = select_final_three(prompt_text, groups, ctx)

    render_results(
        groups, reports, extras, profile, ctx,
        baseline_survival, goal_satisfied, target_months, resolve_info,
        prompt_text, revenue["precise"],
    )
    st.divider()
    _nav(S_RESULTS, fields)  # ◀ 이전으로 돌아가 답을 고칠 수 있다


def main() -> None:
    st.set_page_config(page_title="소상공인 나침반", page_icon="🧭", layout="centered")
    ss = st.session_state
    if "step" not in ss:
        ss.step = S_SITUATION
    _answers()  # 답 저장소 초기화 (위젯이 아니라서 purge 되지 않는다)

    header = st.columns([5, 1])
    with header[0]:
        st.title("🧭 소상공인 나침반")
        st.caption("융자·보조금·채무조정을 36개월 현금흐름으로 함께 계산해 정책 조합을 추천합니다.")
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
        if not fields:  # 안전장치: 물을 것이 없으면 건너뛴다
            ss.step = S_GOALS
            st.rerun()
        render_conditional(fields)
    elif cur == S_GOALS:
        render_goals(fields)
    else:
        render_results_step(db, revenue, elig, profile, phase, annual, fields)


if __name__ == "__main__":
    main()
