"""app.py UI 헬퍼 검증. pytest 없이 `python -m tests.test_app` 로 실행.

계산이 아니라 **화면 표기 헬퍼**를 검증한다. money_to_man 은 cashflow 가 만든
원 단위 문장을 만원 단위로 바꾸는데, '->' 가 붙는 문장에서 공백을 먹거나 개월수·
비율을 잘못 건드리기 쉬워 회귀 테스트를 남긴다.
"""
from __future__ import annotations

import ast
import contextlib
import datetime
import logging
import os
import pathlib
import re

import streamlit

import app
from core import agent, forecast
from core.cashflow import SIM_MONTHS, Applied, UserProfile, simulate, trend_revenue
from core.combos import Combo, evaluate
from core.metrics import detect_all


def test_money_to_man():
    """원 단위 금액을 만원으로. '->' · 원 유무 · 문장 중간 혼합을 모두 본다."""
    cases = [
        # 기본: 원이 붙은 금액
        ("보조금 경영안정 바우처 6,000,000원", "보조금 경영안정 바우처 600만원"),
        ("융자 500,000,000원 @ 3.00%", "융자 50,000만원 @ 3.00%"),
        # ── '->' 케이스 (같은 실수가 재발하면 여기서 걸린다) ──
        # ① 앞엔 원 없음 + 뒤엔 원 있음: 공백을 먹어 '9,500만원->' 이 되면 안 된다
        ("95,000,000 -> 24,000,000원", "9,500만원 -> 2,400만원"),
        # ② 월 상환 변화(둘 다 원 단위, 앞은 원 생략)
        ("월 상환 333,333 -> 258,889원", "월 상환 33만원 -> 25만원"),
        # ③ 문장 중간에 섞이고 개월수·비율은 건드리지 않는다
        (
            "대상 20,000,000원 · 월 상환 333,333 -> 258,889원 (60개월 -> 144개월, 금리 +0.2%)",
            "대상 2,000만원 · 월 상환 33만원 -> 25만원 (60개월 -> 144개월, 금리 +0.2%)",
        ),
        # ④ 원이 아예 없는 순수 숫자(잔액 등)도 만원으로, 뒤 공백 유지
        ("잔액 30,000,000 남음", "잔액 3,000만원 남음"),
        # 라벨 속 '2년거치 3년' 같은 짧은 숫자는 건드리지 않는다
        ("신용취약소상공인자금(2년거치 3년 균등분할)", "신용취약소상공인자금(2년거치 3년 균등분할)"),
    ]
    for src, expected in cases:
        got = app.money_to_man(src)
        assert got == expected, f"money_to_man({src!r})\n  = {got!r}\n  ≠ {expected!r}"
    print(f"   money_to_man {len(cases)}건 통과 ('->' · 원 유무 · 문장 중간 포함)")


def test_months_label():
    """생존 개월 표기 — 상한(36)에 걸리면 '36개월 이상', 아니면 'N개월'."""
    assert app.months_label(SIM_MONTHS) == f"{SIM_MONTHS}개월 이상"
    assert app.months_label(SIM_MONTHS + 5) == f"{SIM_MONTHS}개월 이상"
    assert app.months_label(18) == "18개월"
    assert app.months_label(0) == "0개월"
    print("   months_label 경계 통과")


def test_no_direct_widget_calls():
    """render_* 는 위젯 래퍼(_num/_select/…)만 써야 한다. st.number_input 등을 직접 부르면
    단계 이동 시 값이 purge 돼 '조용히 0' 배선 버그가 재발한다.

    예외는 'transient-widget:' 마커가 붙은 줄만. **마커 개수를 1로 못박는다** —
    render_loan_slider 의 what-if 슬라이더 하나뿐. 예외를 늘리려면 이 테스트를 고쳐야
    하므로 '정말 transient 인가'를 한 번 더 의식적으로 보게 된다.
    """
    import pathlib

    src = pathlib.Path(app.__file__).read_text(encoding="utf-8").splitlines()
    lo = next(i for i, l in enumerate(src) if "WIDGET WRAPPERS START" in l)
    hi = next(i for i, l in enumerate(src) if "WIDGET WRAPPERS END" in l)
    stateful = (
        "st.number_input(", "st.selectbox(", "st.radio(", "st.checkbox(",
        "st.text_input(", "st.slider(", "st.date_input(", "st.file_uploader(",
    )
    violations, markers = [], 0
    for i, line in enumerate(src):
        if lo <= i <= hi:  # 래퍼 정의부만 직접 호출 허용
            continue
        if not any(w in line for w in stateful):
            continue
        if "transient-widget:" in line:
            markers += 1
        else:
            violations.append((i + 1, line.strip()))
    assert not violations, f"래퍼 밖 직접 위젯 호출(래퍼로 바꿀 것): {violations}"
    assert markers == 1, (
        f"transient-widget 예외가 {markers}개다(1개여야 함). 예외를 늘렸다면 정말 단계 답이 "
        "아닌 transient 컨트롤인지 확인하고 이 개수를 의식적으로 갱신할 것."
    )
    print(f"   직접 위젯 호출 0건 · transient 예외 {markers}건(고정)")


# ── AppTest — 실제 Streamlit 세션으로 배선을 검증한다(헤드리스 프로필 조립을 안 탄다) ──


def _stub_llm() -> None:
    """테스트는 **실제 LLM 을 부르지 않는다.** 폴백 응답으로 고정한다.

    `agent._api_key()` 가 .env 를 읽게 되면서 테스트가 실제 API 를 호출하기 시작했고,
    세 가지가 한꺼번에 나빠졌다.
      · 돌릴 때마다 돈이 나간다(호출당 약 $0.03, 한 번 돌리면 수십 회).
      · 같은 코드인데 통과했다 실패했다 한다. LLM 이 매번 다른 조합을 골라 카드 구성이
        달라지고 `_loan_sliders(at)[0]` 이 다른 조합을 가리킨다(A→B→A 왕복이 이걸로 깨졌다).
      · 느리다. 한 번에 몇 분씩 걸렸다.

    LLM 응답이 필요한 테스트는 각자 고정 응답으로 갈아끼운다(test_every_pick_gets_a_card).
    실제 호출은 `tests/test_llm_live.py` 에서 환경변수로 명시할 때만 한다.
    """
    agent.select_final_three = lambda p, l, c=None: agent.Advice(
        ok=False, failure=agent.Failure.NO_KEY, detail="테스트 — LLM 을 부르지 않는다"
    )


def _fresh_app():
    from streamlit.testing.v1 import AppTest

    _stub_llm()
    at = AppTest.from_file(app.__file__)  # 절대경로 (상대경로는 tests/ 기준으로 풀린다)
    at.run(timeout=120)
    return at


def test_no_live_api_call_by_default():
    """기본 경로에서 실제 API 호출이 안 나가는지 — 카나리아.

    anthropic 클라이언트 생성 자체를 막아 두고 결과 화면까지 간다. `_stub_llm()` 이
    풀리거나 누가 우회 경로를 만들면 여기서 즉시 걸린다. 조용히 다시 새는 것을 막는
    것이 목적이라, 통과해도 아무 일이 없는 게 정상이다.
    """
    import anthropic

    real = anthropic.Anthropic

    def blocked(*a, **k):
        raise AssertionError(
            "테스트가 실제 Anthropic API 를 부르려 했다. _stub_llm() 이 풀렸는지, "
            "또는 select_final_three 를 우회하는 경로가 생겼는지 확인하라."
        )

    anthropic.Anthropic = blocked
    try:
        at = _fresh_app()
        _to_results(at)
        assert not at.exception, f"예외 발생: {[str(e.value)[:120] for e in at.exception]}"
    finally:
        anthropic.Anthropic = real
    print("   기본 경로에서 실제 API 호출 없음 ✓")


def _ans(at) -> dict:
    try:
        return dict(at.session_state["answers"])
    except (KeyError, AttributeError):
        return {}


def _advance(at) -> None:
    """현재 단계의 '다음' 버튼을 눌러 한 단계 진행."""
    cur = at.session_state["step"]
    at.button(key=f"next_{cur}").click()
    at.run(timeout=120)


def _back(at) -> None:
    cur = at.session_state["step"]
    at.button(key=f"prev_{cur}").click()
    at.run(timeout=120)


def test_answers_survive_navigation():
    """기본 흐름(데모 시나리오 B) 통과 시 ③(비용)·④(부채) 답이 진단·이후 단계에서
    살아 있는지. 이번 배선 버그(렌더 안 된 위젯 key 가 purge 돼 프로필이 0으로 조립됨)의
    핵심 회귀. 기본 시나리오가 B 라 apply_demo_preset('B') 가 B 프리셋값을 넣는다 —
    그래서 기대값이 위젯 기본값(500·200·…)이 아니라 B 프리셋(430·180·1500·2500)이다."""
    at = _fresh_app()
    seen = {}
    while at.session_state["step"] < app.S_DIAGNOSIS:
        _advance(at)
        seen[at.session_state["step"]] = _ans(at)

    pB = app.DEMO_PRESETS["B"]
    a = _ans(at)  # 진단 단계에서
    assert a.get("fixed_man") == pB["fixed_man"], f"fixed_man 유실: {a.get('fixed_man')}"
    assert a.get("var_man") == pB["var_man"], f"var_man 유실: {a.get('var_man')}"
    assert a.get("cash_man") == pB["cash_man"], f"cash_man 유실: {a.get('cash_man')}"
    assert a.get("debt_man") == pB["debt_man"], f"debt_man 유실: {a.get('debt_man')}"
    # ⑤(연체·신용) 단계에서도 ③의 fixed_man 이 살아 있었는지 명시 확인 — 버그의 핵심
    assert seen.get(app.S_DEBT2, {}).get("fixed_man") == pB["fixed_man"], (
        "⑤ 연체·신용 단계에서 ③ 비용 값이 사라졌다 — 배선 버그 재발"
    )
    # 목표도 B 프리셋(18)이 들어갔는지 — 시나리오 선택이 목표 기본값을 갈아끼우는지 확인(#②)
    assert a.get("goal_months") == pB["goal_months"], (
        f"데모 B 목표 기본값 미적용: {a.get('goal_months')} (기대 {pB['goal_months']})"
    )
    print(f"   ③④ 답이 진단까지 생존 · ⑤에서도 fixed_man={pB['fixed_man']} · 목표={pB['goal_months']} 유지")


def test_edit_and_return():
    """앞 단계로 돌아가 고친 값이 반영되는지(편집→진행→복귀→재편집)."""
    at = _fresh_app()
    while at.session_state["step"] < app.S_COST:
        _advance(at)
    at.number_input(key="fixed_man_w").set_value(700)
    at.run(timeout=120)
    assert _ans(at)["fixed_man"] == 700

    while at.session_state["step"] < app.S_DIAGNOSIS:
        _advance(at)
    assert _ans(at)["fixed_man"] == 700, "진행 후 편집값 유실"

    while at.session_state["step"] > app.S_COST:
        _back(at)
    assert at.number_input(key="fixed_man_w").value == 700, "복귀 시 편집값이 안 보임"

    at.number_input(key="fixed_man_w").set_value(900)
    at.run(timeout=120)
    while at.session_state["step"] < app.S_DIAGNOSIS:
        _advance(at)
    assert _ans(at)["fixed_man"] == 900, "재편집값 미반영"
    print("   편집→진행→복귀→재편집 반영 OK (700→900)")


def test_slider_does_not_leak():
    """결과 카드 what-if 슬라이더로 굴린 값이 저장소(추천 금액)에 새지 않고,
    앞 단계로 갔다 오면 원래 계산된 금액으로 리셋되는지."""
    at = _fresh_app()
    while at.session_state["step"] < app.S_RESULTS:
        _advance(at)

    sliders = [s for s in at.slider if s.key and s.key.startswith("loan_")]
    if not sliders:
        print("   (융자 슬라이더 있는 카드 없음 — 스킵)")
        return
    v0 = sliders[0].value  # 계산된 기본값
    sliders[0].set_value(v0 + 100)
    at.run(timeout=120)

    # 저장소에 loan_ 키가 새지 않았는지 — collect_* 가 여기서만 읽으므로 추천엔 영향 없음
    assert not any(str(k).startswith("loan_") for k in _ans(at)), "슬라이더 값이 ss.answers 로 샜다"

    _back(at)
    while at.session_state["step"] < app.S_RESULTS:
        _advance(at)
    v1 = [s for s in at.slider if s.key and s.key.startswith("loan_")][0].value
    assert v1 == v0, f"복귀 후 슬라이더가 리셋 안 됨(굴린 값이 남음): {v0} → {v1}"
    print(f"   슬라이더 값 누출 없음 · 복귀 시 계산값으로 리셋({v0})")


def _demo_baseline(scenario_key: str) -> int:
    """데모 프리셋으로 무지원(조합 없음) 생존 개월수를 계산. app 의 데모 경로를 그대로 탄다."""
    from core.cashflow import simulate

    preset = app.DEMO_PRESETS[scenario_key]
    summary = app.summarize_scenario(scenario_key, 12)
    won = lambda man: int(man) * 10_000  # noqa: E731
    finance = dict(
        fixed_cost=won(preset["fixed_man"]), variable_cost=won(preset["var_man"]),
        cash=won(preset["cash_man"]), debt=won(preset["debt_man"]),
        debt_payment=won(preset["debt_pay_man"]), debt_rate=preset["rate_pct"] / 100,
        high_rate_debt=None, overdue_days=preset["overdue_days"], policy_loan_balance=0,
    )  # high_rate_debt 는 baseline(조합 없음)에 영향 없음 — 대환은 조합에서만 쓴다
    profile = app.build_profile(finance, app.build_monthly_revenue(True, summary, 0), summary.recent_3m_avg)
    return simulate(profile, []).survival_months


def test_demo_goal_above_baseline():
    """데모 목표(A 24·B 18·C 12)가 baseline(무지원 생존)보다 위인지 — 세 시나리오 전부.

    목표 ≤ baseline 이면 filter_by_goal 이 전부 통과시켜 satisfied 가 자명해지고,
    역산이 ~0 융자를 줘 B 의 거치 절벽도 안 뜬다. 재무값을 손대 baseline 이 움직이면
    이 관계가 깨질 수 있어 여기서 못박는다(app.DEMO_PRESETS 주석 참조)."""
    for key in ("A", "B", "C"):
        base = _demo_baseline(key)
        goal = app.DEMO_PRESETS[key]["goal_months"]
        assert base < goal, (
            f"[{key}] baseline {base}개월 ≥ 목표 {goal}개월 — 목표가 baseline 위가 아니라 "
            f"filter_by_goal 이 무의미하다. 재무값을 손댔으면 DEMO_PRESETS 의 목표를 다시 잡을 것."
        )
        print(f"   [{key}] baseline {base} < 목표 {goal} ✓")
    print("   데모 목표 > baseline 불변식 통과 (A/B/C)")


def test_debt_relief_reaches_confirmation():
    """C 의 baseline 이 채무조정 **적용시점(확정까지 걸리는 개월)** 이상인지.

    새출발기금은 `적용시점_개월`(부실차주 6개월, 캠코 통화 "신청~확정 5~6개월"의 상한)
    이후에야 감면이 반영된다. 그런데 파산하면 그 시점에서 계산이 멈추므로(가정 F1,
    cashflow.py:100 · planning.md:447) **baseline 이 적용시점보다 짧으면 감면이
    통째로 사라진다** — 종료부채가 baseline 과 같아져 C 의 핵심 처방(부채 4,026만 감소)이
    화면에서 증발한다. 재무값을 만져 baseline 이 5로 떨어지면 조용히 그렇게 된다.

    "확정 전까지 계속 상환한다"는 보수적 가정은 심사 중 상환 의무를 확인하지 못해서다
    (planning.md:1470-1475, 미해결 C-37 — 캠코도 담당자가 모른다고 답했다).
    모델링을 바꾸는 게 아니라, 그 가정 위에서 데모가 성립하는지를 지킨다."""
    import app as _app
    from core import loader

    db = loader.load()
    relief_at = max(
        int(c.get("적용시점_개월", 0) or 0)
        for c in db.candidates
        if "새출발" in str(c) and c.effect in ("부채감면", "이자조정")
    )
    base = _demo_baseline("C")
    assert base >= relief_at, (
        f"C baseline {base}개월 < 새출발기금 적용시점 {relief_at}개월 — 확정 전에 파산해 "
        f"감면이 통째로 사라진다(가정 F1). C 재무값을 손댔으면 baseline 을 다시 잡을 것."
    )
    print(f"   [C] baseline {base} >= 채무조정 적용시점 {relief_at} ✓")


def test_drop_window_label_matches():
    """매출 감소 근거 문장의 기간이 **실제 채택된 비교 단위**와 같은지.

    감소율은 분기·반기 중 큰 쪽을 쓰는데(제도가 "여러 단위 중 하나만 충족하면 된다"),
    문장은 오래도록 "최근 3개월"로 고정돼 있었다. 반기가 채택되면 6개월 값을 3개월이라
    말하게 되고, 화면 위쪽 매출 요약 바(항상 3개월)와 숫자가 어긋나 보인다.
    판정에는 영향이 없어(둘 다 15~40% 구간이면 같은 1점) 눈에 안 띄는 종류라 못박는다.
    """
    from core import pos as _pos

    won = lambda man: int(man) * 10_000  # noqa: E731
    for key in ("A", "B", "C"):
        preset = app.DEMO_PRESETS[key]
        summary = app.summarize_scenario(key, 12)
        finance = dict(
            fixed_cost=won(preset["fixed_man"]), variable_cost=won(preset["var_man"]),
            cash=won(preset["cash_man"]), debt=won(preset["debt_man"]),
            debt_payment=won(preset["debt_pay_man"]), debt_rate=preset["rate_pct"] / 100,
            high_rate_debt=None, overdue_days=preset["overdue_days"], policy_loan_balance=0,
        )
        profile = app.build_profile(
            finance, app.build_monthly_revenue(True, summary, 0), summary.recent_3m_avg
        )
        rate, window = summary.drop_rate_detail(("분기", "반기"))
        months = _pos.COMPARE_WINDOWS[window]
        rows = app.phase_indicator_rows(profile, summary, summary.recent_3m_avg)
        sentence = next(msg for name, _, msg in rows if name == "매출 감소")
        if sentence is None:  # 감소율이 경계 미만이면 0점이라 문장이 없다
            assert rate < 0.15, f"[{key}] 문장이 없는데 감소율이 {rate:.1%}다"
            print(f"   [{key}] 감소율 {rate:.1%} < 15% → 문장 없음(0점) ✓")
            continue
        assert f"최근 {months}개월" in sentence and f"그 전 {months}개월" in sentence, (
            f"[{key}] 채택 단위는 {window}({months}개월)인데 문장은 '{sentence}' 다 — "
            "기간 표현이 실제 계산과 어긋난다."
        )
        print(f"   [{key}] 채택 {window}({months}개월) {rate:.0%} → '{sentence}' ✓")

    # 단위별로 문장이 실제로 갈리는지(같은 값이라도 개월 수가 다르면 다른 문장)
    from core.phase import score_revenue_drop

    assert "최근 6개월" in score_revenue_drop(0.21, 6)[1]
    assert "최근 3개월" in score_revenue_drop(0.21, 3)[1]
    print("   score_revenue_drop 이 개월 수를 문장에 반영 ✓")


def test_reason_sentences_end_with_period():
    """근거 문장은 전부 완결 문장이라 마침표로 끝난다(명사구 폴백은 제외).

    표로 바뀌면서 문장 끝이 제각각이면 눈에 띈다. 화면에서 붙이지 않고 phase.py 에서
    통일한다 — 같은 문장이 LLM 프롬프트로도 나가기 때문이다."""
    from core.phase import (
        score_icr, score_operating_profit, score_overdue, score_revenue_drop,
    )

    produced = [
        score_icr(-50_000, 300_000),        # 적자 → 이자 감당 불가
        score_icr(400_000, 300_000),        # 흑자지만 여유 없음
        score_icr(100_000, 300_000),        # 흑자, 감당 못함
        score_revenue_drop(0.21, 6),
        score_revenue_drop(0.45, 3),
        score_overdue(15),
        score_overdue(120),
        score_operating_profit(-170_000),
    ]
    for _, msg in produced:
        assert msg is not None and msg.endswith("."), f"마침표 없음: {msg!r}"
    # ICR 적자 케이스가 비어 있지 않은지 — 표에서 한 칸만 비면 값이 빠진 것처럼 보인다
    assert score_icr(-50_000, 300_000)[1], "ICR 적자 케이스의 근거 문장이 비어 있다"
    print(f"   근거 문장 {len(produced)}종 전부 마침표로 끝남 · ICR 적자 문장 채워짐 ✓")


def _demo_evals(key: str):
    """데모 시나리오 하나의 (profile, 전체 조합 평가, 그룹, 리포트)."""
    import datetime as _dt  # noqa: F401  (프리셋의 date 값 사용)

    preset = app.DEMO_PRESETS[key]
    summary = app.summarize_scenario(key, 12)
    won = lambda man: int(man) * 10_000  # noqa: E731
    finance = dict(
        fixed_cost=won(preset["fixed_man"]), variable_cost=won(preset["var_man"]),
        cash=won(preset["cash_man"]), debt=won(preset["debt_man"]),
        debt_payment=won(preset["debt_pay_man"]), debt_rate=preset["rate_pct"] / 100,
        high_rate_debt=None, overdue_days=preset["overdue_days"], policy_loan_balance=0,
    )
    profile = app.build_profile(
        finance, app.build_monthly_revenue(True, summary, 0), summary.recent_3m_avg
    )
    phase = {"A": "일시적자금난", "B": "개선여지", "C": "회복불가"}[key]
    ctx = app.build_context(
        phase,
        dict(region="서울", restart_status="안함", is_closed=preset.get("is_closed", False),
             tax_delinquent=False, include_loans=True, credit_score=None),
        dict(business_started=preset["cond_bstart"], store_area_pyeong=preset.get("cond_area"),
             policy_loan_count_5y=0),
        summary, summary.rolling_annual_revenue,
    )
    groups, reports, evals, *_ = app.run_pipeline(
        app.load_db(), profile, ctx, None, preset["goal_months"]
    )
    return profile, evals, groups, reports


def _assert_split_identity(res, where: str) -> int:
    """합계 == 기존 + 신규 를 생존 구간 전체에서 확인한다. 검사한 회차 수를 돌려준다.

    노출한 배열은 **약정 스케줄**이고 합계 배열은 **실제로 나간 돈**이라, 파산 후에는
    합계가 0으로 동결돼(가정 F1) 항등식이 깨진다. 그래서 **생존 구간만** 본다 —
    화면 표도 그 범위까지만 그린다.
    """
    for m in range(1, res.survival_months + 1):
        i = m - 1
        for name, total, ex, nw in (
            ("상환", res.payment_during, res.existing_payment_during, res.new_payment_during),
            ("이자", res.interest_during, res.existing_interest_during, res.new_interest_during),
            ("원금", res.principal_during, res.existing_principal_during, res.new_principal_during),
        ):
            assert abs(total[i] - (ex[i] + nw[i])) < 1.0, (
                f"[{where}] {m}회차 {name}: 합계 {total[i]:,.0f} != "
                f"기존 {ex[i]:,.0f} + 신규 {nw[i]:,.0f}"
            )
        assert abs(res.debt_at[m] - (res.existing_debt_at[m] + res.new_debt_at[m])) < 1.0, (
            f"[{where}] {m}시점 부채: 합계 {res.debt_at[m]:,.0f} != "
            f"기존 {res.existing_debt_at[m]:,.0f} + 신규 {res.new_debt_at[m]:,.0f}"
        )
    return res.survival_months


def _profile_for_split(**over):
    """항등식 검사용 프로필. 경로마다 필요한 값만 바꿔 쓴다."""
    base = dict(
        cash_on_hand=1_900 * 10_000,
        monthly_revenue=trend_revenue(663 * 10_000, -0.01),
        fixed_cost=435 * 10_000, variable_ratio=0.27,
        existing_debt=5_000 * 10_000, existing_debt_payment=200 * 10_000,
        existing_debt_rate=0.08, high_rate_debt=None, overdue_days=0,
        policy_loan_balance=0,
    )
    base.update(over)
    return UserProfile(**base)


def test_simresult_split_identity():
    """SimResult 의 기존/신규 분해가 합계와 어긋나지 않는지.

    **기존분을 변형하는 경로를 각각 따로 단언한다.** 대환전환은 기존 스케줄을 비율로
    축소하고, 상환연장은 축소한 뒤 늘어난 기간으로 새 스케줄을 붙인다. 서로 다른
    변형이라 한쪽이 통과해도 다른 쪽은 모른다.

    예전에는 데모 A·B·C 를 훑어 "그 경로를 타는 조합이 있으면 검사한다"는 구조였다.
    두 문제가 있었다.
      ① 프리셋 값이나 목표 개월수를 바꾸면 커버리지가 조용히 사라진다. 실제로 A 의
         고정비를 올리자 대환 조합이 목표 필터에서 전부 탈락해 0개가 됐다.
      ② 두 효과를 합집합으로 봐서 대환 하나만 있어도 통과했다. **상환연장 경로는
         한 번도 검증된 적이 없다** — 데모 셋 다 policy_loan_balance 가 0이라
         후보에 오르지 못하는데 아무도 눈치채지 못했다.
    항등식은 계산 코드의 성질이지 데모의 성질이 아니므로 여기서 조합을 직접 만든다.
    """
    db = app.load_db()
    conversion = next(c for c in db.candidates if c.effect == "대환전환")
    extension = next(c for c in db.candidates if c.effect == "상환연장")

    checked = 0
    cases = [
        # 대환: 고금리 부채가 있어야 자격이 선다
        ("대환전환", _profile_for_split(high_rate_debt=5_000 * 10_000), [Applied(conversion)]),
        # 상환연장: 소진공 직접대출 잔액이 있어야 대상이 잡힌다
        ("상환연장", _profile_for_split(policy_loan_balance=2_000 * 10_000), [Applied(extension)]),
        # 둘은 슬롯이 달라(융자 / 채무조정) 한 조합에 함께 들어갈 수 있다
        ("대환전환+상환연장",
         _profile_for_split(high_rate_debt=5_000 * 10_000,
                            policy_loan_balance=2_000 * 10_000),
         [Applied(conversion), Applied(extension)]),
    ]
    for label, profile, items in cases:
        res = simulate(profile, items)
        months = _assert_split_identity(res, label)
        assert months > 0, f"[{label}] 생존이 0개월이라 검사할 구간이 없다"
        checked += months

    # 데모에도 있으면 함께 본다. 없다고 실패시키지 않는다 — 위에서 이미 각 경로를 봤다.
    demo_checked = demo_adjusted = 0
    for key in ("A", "B", "C"):
        _, evals, _, _ = _demo_evals(key)
        for ev in evals:
            if {"상환연장", "대환전환"} & {it.candidate.effect for it in ev.combo.items}:
                demo_adjusted += 1
            demo_checked += _assert_split_identity(ev.result, f"{key} {ev.label}")

    print(f"   합계 == 기존+신규 · 경로별 자립 검사 {len(cases)}종 {checked}회차 "
          f"(대환·상환연장 각각) + 데모 {demo_checked:,}회차"
          f"{f' · 그중 변형 조합 {demo_adjusted}개' if demo_adjusted else ' · 데모에는 변형 조합 없음'} ✓")


def test_month_display_sums():
    """화면에 나가는 모든 회차에서 **표시값의 덧셈이 맞는지**.

    round(a+b) != round(a)+round(b) 라서 각자 반올림하면 화면의 덧셈이 틀어진다
    (이 테스트를 넣기 전 실측에서 1,132회차 중 216건이 어긋났고, 절벽 시점도 있었다.
    "154만원 (원금 139 + 이자 15)" 같은 문장의 합이 조합에 따라 안 맞았다).
    month_display 가 잔차를 이자에 몰아 항상 맞게 한다."""
    from core.cashflow import month_display

    checked = 0
    for key in ("A", "B", "C"):
        _, evals, _, _ = _demo_evals(key)
        for ev in evals:
            res = ev.result
            for m in range(1, res.survival_months + 1):
                d = month_display(res, m)
                checked += 1
                assert d.total == d.principal + d.interest, f"[{key}] {m}회차 총액 {d}"
                assert d.principal == d.existing_principal + d.new_principal, f"[{key}] {m}회차 원금 {d}"
                assert d.interest == d.existing_interest + d.new_interest, f"[{key}] {m}회차 이자 {d}"
                assert d.debt == d.existing_debt + d.new_debt, f"[{key}] {m}회차 잔액 {d}"
    print(f"   표시값 덧셈 일치 · {checked:,}회차 ✓")


def _assert_cliff_message_matches_table(result, report, where: str) -> bool:
    """절벽 경고 문장의 숫자가 상환 스케줄 표의 같은 회차와 맞는지. 검사했으면 True."""
    import re as _re

    if not report.cliff_at:
        return False
    cliff = next((t for t in report.traps if t.at == report.cliff_at), None)
    if cliff is None:
        return False
    rows = app.schedule_rows(result)
    assert len(rows) >= report.cliff_at, f"[{where}] 절벽 회차가 표 밖이면 나란히 볼 수 없다"
    row = rows[report.cliff_at - 1]
    total, principal, interest = (
        int(_re.search(r"월 상환액 ([\d,]+)만원", cliff.message).group(1).replace(",", "")),
        int(_re.search(r"원금 ([\d,]+)", cliff.message).group(1).replace(",", "")),
        int(_re.search(r"이자 ([\d,]+)", cliff.message).group(1).replace(",", "")),
    )
    assert total == principal + interest, f"[{where}] 문장 덧셈 불일치: {cliff.message}"
    assert principal == row["원금(기존)"] + row["원금(신규)"], (
        f"[{where}] 문장 원금 {principal} != 표 {row['원금(기존)']}+{row['원금(신규)']}"
    )
    assert interest == row["이자(기존)"] + row["이자(신규)"], (
        f"[{where}] 문장 이자 {interest} != 표 {row['이자(기존)']}+{row['이자(신규)']}"
    )
    return True


def _cliff_case():
    """절벽 함정이 반드시 걸리는 조합을 직접 만든다.

    거치가 있는 융자를 붙이면 거치가 끝나는 달에 신규 상환의 원금이 0 -> 양수로
    바뀐다(`find_cliff_at` 이 찾는 시점). 다만 **거치가 끝나는 것만으로는 함정이
    걸리지 않는다** — 판정 기준이 `절벽 시점 <= 생존 < 36` 이라, 36개월을 다 버티면
    절벽이 있어도 함정이 아니다(그 부담을 실제로 맞지 않는다).

    그래서 현금을 절벽 시점보다 조금 더 버틸 만큼만 준다. 실측으로 고른 값이다:
    2년거치(25회차 절벽) · 현금 3,000만 -> 생존 26개월. 25 <= 26 < 36 이라 걸린다.
    """
    db = app.load_db()
    grace = next(
        c for c in db.candidates
        if c.is_loan and int(c.get("거치_개월", 0) or 0) == 24
        and "직접대출" in c.label
    )
    profile = _profile_for_split(cash_on_hand=3_000 * 10_000)
    combo = Combo(items=(Applied(grace, amount=3_000 * 10_000),))
    evaluated = evaluate(profile, [combo])[0]
    reports = detect_all(profile, [evaluated], "일시적자금난")
    return evaluated, reports[evaluated.label], f"자립({grace.label[:24]})"


def test_trap_message_matches_table():
    """함정 경고의 숫자와 상환 스케줄 표의 숫자가 **같은 회차에서 일치**하는지.

    문장과 표가 각자 반올림하면 같은 13회차인데 이자가 15 / 14 로 갈린다.
    둘 다 month_display 를 거치게 해서 막는다.

    **절벽이 있는 조합을 직접 만들어 검사한다.** 예전에는 데모 A·B·C 에 절벽이
    있으면 검사하고 없으면 `assert seen` 으로 실패시켰는데, 프리셋을 만지면
    커버리지가 조용히 사라지는 구조였다(같은 이유로
    `test_simresult_split_identity` 를 자립시켰다). 데모 순회는 남기되 없다고
    실패시키지 않는다.
    """
    evaluated, report, where = _cliff_case()
    assert _assert_cliff_message_matches_table(evaluated.result, report, where), (
        "자립 케이스에 절벽 함정이 안 걸렸다 — `절벽 시점 <= 생존 < 36` 을 만족하는지 "
        "확인할 것(현금을 늘리면 36개월을 다 버텨 함정이 사라진다)"
    )

    seen = 0
    for key in ("A", "B", "C"):
        _, _, groups, reports = _demo_evals(key)
        for group in groups:
            member = group.members[0]
            if _assert_cliff_message_matches_table(
                member.result, reports[member.label], f"{key} {member.label[:24]}"
            ):
                seen += 1
    print(f"   함정 경고 ↔ 표 숫자 일치 · 자립 1개 + 데모 {seen}개 ✓")


def test_schedule_stops_at_exhaustion():
    """표가 자금 소진 시점에서 끊기는지. 그래프는 거기서 끝나는데 표만 36회차까지
    이어지면 어긋나 보이고, 잔고가 마른 뒤의 상환은 일어나지 않는 일이다.
    36개월을 버티는 조합은 끝까지 보여준다."""
    stopped = full = 0
    for key in ("A", "B", "C"):
        _, evals, _, _ = _demo_evals(key)
        for ev in evals:
            rows = app.schedule_rows(ev.result)
            if ev.result.survives_full_term:
                full += 1
                assert len(rows) <= len(ev.result.payment_during)
            else:
                stopped += 1
                assert len(rows) <= ev.result.survival_months, (
                    f"[{key}] 생존 {ev.result.survival_months}개월인데 표가 {len(rows)}행이다"
                )
    print(f"   소진 조합 {stopped}개는 생존 시점까지만 · 완주 조합 {full}개는 끝까지 ✓")


def test_confirm_keys_are_stable():
    """자격 확인 질문의 위젯 key 가 **프로세스를 나눠도 같은 값**인지.

    파이썬 hash() 는 PYTHONHASHSEED 로 랜덤화돼 앱을 다시 띄우면 값이 바뀐다. 그것을
    key 로 쓰면 저장된 답이 날아가고 데모 프리셋도 조용히 안 걸린다. 시연 중 재시작하면
    터지는 종류라 못박는다(내용 기반 sha1 로 고쳤다)."""
    import subprocess
    import sys

    code = (
        "import app;"
        "print(app.confirm_key('테스트 질문입니다'));"
        "print(app.stable_key('card', '조합 이름'))"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
            cwd=str(pathlib.Path(app.__file__).parent),
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONUTF8": "1"},
        ).stdout.strip()
        for seed in ("0", "12345")
    ]
    assert runs[0] == runs[1], f"프로세스마다 key 가 다르다:\n{runs[0]}\n{runs[1]}"
    assert runs[0].splitlines()[0].startswith("confirm_")
    print(f"   PYTHONHASHSEED 를 바꿔도 key 동일 ✓ ({runs[0].splitlines()[0]})")


def test_confirm_excludes_unsure_like_no():
    """'아니오'와 '잘 모르겠어요'가 **똑같이** 제외되는지 (기획서 932~935).

    확인하지 못한 자격을 있다고 가정하면 신청했다가 거절당한다. 답하지 않은 것도
    확인되지 않은 것이라 같이 제외한다.
    예외는 `_확인필요_자격_지금가능: true` — 지금 못 했어도 신청 전에 하면 되므로
    남기고 prerequisites 가 안내한다."""
    now_possible = {"key": "k1", "ids": ["c_now"], "now_possible": True, "question": "q"}
    blocking = {"key": "k2", "ids": ["c_block"], "now_possible": False, "question": "q"}
    questions = [now_possible, blocking]

    assert app.excluded_by_confirm(questions, {"k2": app.CONFIRM_YES}) == set()
    for answer in (app.CONFIRM_NO, app.CONFIRM_UNSURE):
        assert app.excluded_by_confirm(questions, {"k2": answer}) == {"c_block"}, answer
    assert app.excluded_by_confirm(questions, {}) == {"c_block"}, "무응답도 제외해야 한다"
    # '지금가능'은 어떤 답이어도 남는다
    assert "c_now" not in app.excluded_by_confirm(questions, {"k1": app.CONFIRM_NO})
    print("   아니오 = 모름 = 무응답 → 제외 · '지금가능'은 예외 ✓")


def test_demo_confirm_presets_match_real_questions():
    """데모 프리셋의 `confirm_yes` 후보 id 가 **실제 질문 key 로 옮겨지는지**.

    프리셋에 해시를 박아 두면 정책 데이터의 질문 문구가 바뀌었을 때 조용히 어긋나
    시연 화면이 달라진다. id 로 적고 confirm_key_of 가 옮기게 했으니, 그 id 가
    실제로 확인 질문을 가진 후보인지 여기서 확인한다."""
    db = app.load_db()
    checked = 0
    for key, preset in app.DEMO_PRESETS.items():
        for candidate_id in preset.get("confirm_yes", ()):
            resolved = app.confirm_key_of(db, candidate_id)
            assert resolved, (
                f"[{key}] 프리셋의 {candidate_id} 에 자격 확인 질문이 없다 — "
                "정책 데이터가 바뀌었으면 프리셋도 고쳐야 한다."
            )
            assert resolved.startswith("confirm_")
            checked += 1
    assert checked, "confirm_yes 를 가진 프리셋이 하나도 없다"
    print(f"   프리셋 confirm_yes {checked}건이 실제 질문 key 로 해석됨 ✓")


def _drive(scenarios: list[str]) -> dict:
    """시나리오를 순서대로 골라가며 결과 단계까지 몰고, 최종 answers 를 돌려준다."""
    at = _fresh_app()
    while at.session_state["step"] < app.S_REVENUE:
        _advance(at)
    for key in scenarios:
        at.selectbox(key="scenario_key_w").set_value(key)
        at.run(timeout=120)
    while at.session_state["step"] < app.S_RESULTS:
        _advance(at)
    return _ans(at)


def test_preset_switch_no_leak():
    """시나리오를 갈아탈 때 앞 시나리오의 프리셋 값이 남지 않는지.

    dict.update 는 없는 키를 지우지 않는다. C 만 가진 is_closed=True·cond_area=20 이
    남으면 **A 가 폐업 상태로 계산되는데 화면은 멀쩡해 보인다** — 시연 중에 알아채기
    어려운 종류라 여기서 못박는다. apply_demo_preset 이 (합집합 - 이번 프리셋) 키를
    지워서 막는다.

    왕복(A→B→C→A)이 핵심이다. 전환 경로가 달라도 결과가 같아야 한다."""
    c_only = {k for k in app.DEMO_PRESETS["C"]} - {k for k in app.DEMO_PRESETS["A"]}
    assert c_only, "C 전용 키가 없다면 이 테스트의 전제가 깨진 것이다(프리셋 구조 변경?)"

    fresh_a = _drive(["A"])
    for path in (["C", "A"], ["A", "B", "C", "A"]):
        got = _drive(path)
        for key in c_only:
            assert not got.get(key), (
                f"{'→'.join(path)} 후 C 전용 키 {key}={got.get(key)!r} 가 남았다 — "
                "A 가 C 의 상태로 계산된다."
            )
        diff = {
            k: (fresh_a.get(k), got.get(k))
            for k in set(fresh_a) | set(got)
            if k != "_applied_scenario" and fresh_a.get(k) != got.get(k)
        }
        assert not diff, f"{'→'.join(path)} 의 A 가 처음 A 와 다르다: {diff}"
        print(f"   {'→'.join(path):<12} 후 A 가 처음 A 와 동일 ✓")

    to_c = _drive(["A", "C"])
    assert to_c.get("is_closed") is True and to_c.get("cond_area") == app.DEMO_PRESETS["C"]["cond_area"], (
        f"A→C 전환 시 C 프리셋이 안 들어갔다: is_closed={to_c.get('is_closed')} cond_area={to_c.get('cond_area')}"
    )
    print("   A→C 전환 시 C 전용 값(폐업·면적) 정상 주입 ✓")


def test_conditional_gating():
    """묻지 않은 조건부 항목은 프로필에 싣지 않는지.

    ⑦ 추가 확인 단계는 국면·자격에 따라 묻는 항목이 달라진다. store_area_pyeong 은
    C(회복불가·폐업)에서만 묻는다. C 에서 답한 뒤 A 로 가면 A 는 그 값을 묻지도 않으므로
    프로필에 실려선 안 된다. collect_conditional 이 fields 로 게이팅해서 막는다
    (collect_revenue 가 소스로 게이팅하는 것과 같은 방식).

    답 자체를 지우지는 않는다 — A→C→A 로 오갈 때 A 에서 답한 것을 또 물으면 성가시다."""
    import app as _app

    summary = _app.summarize_scenario("C", 12)
    annual = summary.rolling_annual_revenue
    c_fields = _app.needed_conditional_fields("회복불가", annual, True)
    a_fields = _app.needed_conditional_fields("일시적자금난", annual, True)
    only_c = set(c_fields) - set(a_fields)
    assert "store_area_pyeong" in only_c, (
        f"전제가 깨졌다 — store_area_pyeong 이 C 전용이 아니다. C={c_fields} A={a_fields}"
    )

    # A 의 fields 로 게이팅하면, ans 에 면적이 남아 있어도 프로필엔 안 실려야 한다
    ans = {"cond_area": 20.0, "cond_bstart_known": True, "cond_bstart": datetime.date(2019, 3, 1)}
    loaded = {"store_area_pyeong": (ans.get("cond_area") if "store_area_pyeong" in a_fields else None)}
    assert loaded["store_area_pyeong"] is None, "A 에서 묻지 않은 면적이 프로필에 실렸다"
    print(f"   C 전용 항목 {sorted(only_c)} 은 A 에서 프로필에 안 실림 ✓")


def _loan_sliders(at):
    return [s for s in at.slider if s.key and s.key.startswith("loan_")]


_LOAN_CELL = re.compile(r"융자\(조정\)</div><div style='[^']*'>([\d,]+)만원")


def _card_loan_amount(at) -> int | None:
    """'융자(조정)' 칸에 적힌 만원 값. 슬라이더 표시값과 같아야 한다.

    둘이 갈리는 게 되돌리기 버그의 증상이었다 — 파이썬은 추천 금액으로 계산하는데
    브라우저 눈금은 굴린 자리에 남아 화면에 두 값이 다르게 떴다.
    """
    for block in at.markdown:
        hit = _LOAN_CELL.search(block.value)
        if hit:
            return int(hit.group(1).replace(",", ""))
    return None


@contextlib.contextmanager
def _streamlit_warnings():
    """Streamlit 이 로거로 내는 경고를 모은다(화면 요소가 아니라 stderr 로 나간다)."""
    seen: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    handler = _Collect()
    logger = logging.getLogger("streamlit")
    logger.addHandler(handler)
    try:
        yield seen
    finally:
        logger.removeHandler(handler)


def _to_results(at) -> None:
    while at.session_state["step"] < app.S_RESULTS:
        _advance(at)


def test_slider_resets_when_inputs_change():
    """앞 단계로 돌아가 조건을 바꾸고 오면 슬라이더가 **새 계산값**으로 돌아오는지.

    슬라이더는 카드 안의 what-if 라 되돌아가면 초기화되는 게 맞다(transient-widget).
    그 성질을 지키는 것은 위젯 key 가 아니라 **Streamlit 의 purge** 다 — 결과 화면을
    떠나면 loan_* 이 렌더되지 않아 키가 지워지고, 돌아오면 새 기본값으로 다시 만들어진다.

    ⚠ **Streamlit 내부 동작에 기대는 부분이다**('.st-key-' 클래스 의존과 같은 성격).
      버전이 올라가 purge 가 사라지면 굴린 값이 새 계산을 덮어써 **화면 금액이 어긋나는데
      눈에 잘 안 띈다.** 그래서 여기서 못박는다.
      키에 금액을 섞어 내용이 바뀌면 새 위젯이 되게 하는 방법도 있지만, 그러면 슬라이더를
      움직일 때마다 키가 바뀌어 조작이 끊긴다.
    """
    at = _fresh_app()
    _to_results(at)
    sliders = _loan_sliders(at)
    if not sliders:
        print("   (융자 슬라이더가 있는 카드 없음 — 스킵)")
        return
    moved = sliders[0].value + 500
    sliders[0].set_value(moved)
    at.run(timeout=120)
    assert _loan_sliders(at)[0].value == moved

    _back(at)  # 결과 -> ⑦ 조건 설정
    left = [k for k in at.session_state.filtered_state if str(k).startswith("loan_")]
    assert not left, f"결과 화면을 떠났는데 loan_ 키가 남았다: {left}"
    goal = [s for s in at.slider if s.key == "goal_months_w"]
    assert goal, "⑦ 에 목표 개월수 슬라이더가 없다"
    goal[0].set_value(30)
    at.run(timeout=120)

    _to_results(at)
    after = _loan_sliders(at)[0].value
    assert after != moved, f"조건을 바꿨는데 굴린 값({moved})이 그대로 남았다"
    print(f"   조건 변경 후 복귀 — 굴린 값 {moved} 안 남고 새 계산값 {after} ✓")


def test_slider_does_not_leak_across_scenarios():
    """A 에서 굴린 슬라이더 값이 B 를 거쳐 A 로 돌아왔을 때 남지 않는지.

    카드 key 를 조합 이름으로 잡으므로(순서 의존을 없애려고), 시나리오가 달라도 같은
    조합이면 key 가 같다. 그때 값이 새면 다른 시나리오의 금액이 화면에 뜬다.
    여기서도 purge 가 막아 주지만, 위와 같은 이유로 확인해 둔다.
    """
    at = _fresh_app()
    while at.session_state["step"] < app.S_REVENUE:
        _advance(at)
    at.selectbox(key="scenario_key_w").set_value("A")
    at.run(timeout=120)
    _to_results(at)
    sliders = _loan_sliders(at)
    if not sliders:
        print("   (A 에 융자 슬라이더가 있는 카드 없음 — 스킵)")
        return
    original = sliders[0].value
    sliders[0].set_value(original + 700)
    at.run(timeout=120)

    for scenario in ("B", "A"):
        while at.session_state["step"] > app.S_REVENUE:
            _back(at)
        at.selectbox(key="scenario_key_w").set_value(scenario)
        at.run(timeout=120)
        _to_results(at)

    back_again = _loan_sliders(at)[0].value
    assert back_again == original, (
        f"A→B→A 왕복 후 굴린 값이 남았다: {back_again} (원래 {original})"
    )
    print(f"   A→B→A 왕복 — 굴린 값 {original + 700} 안 남고 {back_again} 로 복귀 ✓")


def test_reset_button_restores_only_its_card():
    """↺ 추천 금액으로 — 그 카드만 되돌리고 나머지 카드는 굴린 값을 지키는지.

    ⚠ 이 테스트가 지키는 것은 **on_click 콜백**이다. `if st.button(...): pop;
    st.rerun()` 로 바꾸면 rerun 이 스크립트를 그 자리에서 끊어 아래 카드들의
    슬라이더가 등록되지 않고, Streamlit 이 그 상태를 지운다. 그러면 카드 하나를
    되돌렸을 뿐인데 아래 카드들이 전부 기본값으로 돌아간다. 화면을 봐야 알 수 있는
    종류라 여기서 못박는다(실제로 그렇게 구현했다가 이 테스트에 걸렸다).

    되돌린 카드의 아래 내용(카드 4칸·그래프·경고·표)이 함께 갱신되는지도 본다.
    """
    at = _fresh_app()
    _to_results(at)
    sliders = _loan_sliders(at)
    if len(sliders) < 2:
        print(f"   (융자 슬라이더가 {len(sliders)}개뿐 — 스킵)")
        return

    def screen(at):
        return [m.value for m in at.markdown] + [w.value for w in at.warning]

    keys = [s.key for s in sliders]
    base = {s.key: s.value for s in sliders}
    before = screen(at)

    # ── ① 카드 하나만 굴렸다 되돌리면 그 아래 내용까지 원래대로 ──
    # 나머지 카드를 건드리지 않은 상태라 화면 전체가 처음과 같아야 한다. 카드 4칸·
    # 그래프 라벨·함정 경고·상환 표가 전부 markdown/warning 에 들어 있다.
    sliders[0].set_value(base[keys[0]] + 900)
    at.run(timeout=120)
    assert screen(at) != before, "굴렸는데 화면이 안 바뀌었다"
    at.button(key=f"reset_{keys[0]}").click()
    at.run(timeout=120)
    restored = _loan_sliders(at)[0]
    assert restored.value == base[keys[0]]
    assert screen(at) == before, "되돌린 카드의 아래 내용이 원래 값으로 안 돌아왔다"

    # 눈금이 실제로 움직이는지. set_value 가 False 면 파이썬은 추천 금액으로 계산하는데
    # 브라우저는 굴린 자리를 그대로 들고 있어 화면에 두 값이 다르게 뜬다.
    # AppTest 의 .value 는 proto 를 되읽은 값이라 이 차이를 못 본다 — 그래서 플래그를
    # 직접 본다. 세션에서 key 를 지우는 방식으로 되돌리면 여기서 걸린다.
    assert restored.proto.set_value, (
        "슬라이더 proto 의 set_value 가 False 다 — 브라우저 눈금이 안 움직인다. "
        "되돌리기가 세션 대입이 아니라 pop 으로 바뀌었는지 확인하라"
    )
    assert _card_loan_amount(at) == restored.value, (
        f"슬라이더({restored.value})와 융자(조정) 카드({_card_loan_amount(at)})가 다르다"
    )

    # ── ② 전부 굴린 뒤 하나만 되돌리면 나머지는 굴린 값 유지 ──
    for i, slider in enumerate(_loan_sliders(at)):
        slider.set_value(slider.value + 300 + i * 100)
    at.run(timeout=120)
    expected = {k: base[k] + 300 + i * 100 for i, k in enumerate(keys)}
    moved = {s.key: s.value for s in _loan_sliders(at)}
    assert moved == expected, f"굴리기 실패\n  {moved}\n  {expected}"

    at.button(key=f"reset_{keys[0]}").click()
    at.run(timeout=120)
    after = {s.key: s.value for s in _loan_sliders(at)}

    assert after[keys[0]] == base[keys[0]], (
        f"되돌린 카드가 추천 금액이 아니다: {after[keys[0]]} (추천 {base[keys[0]]})"
    )
    kept = {k: after[k] for k in keys[1:]}
    assert kept == {k: expected[k] for k in keys[1:]}, (
        "되돌리지 않은 카드까지 초기화됐다 — st.rerun() 으로 바뀌었는지 확인하라\n"
        f"  지금: {kept}\n  기대: {({k: expected[k] for k in keys[1:]})}"
    )
    print(
        f"   되돌리기 — 카드1 {expected[keys[0]]}→{after[keys[0]]} · "
        f"나머지 {len(kept)}개 굴린 값 유지 · 아래 내용도 원래대로 ✓"
    )


def test_every_pick_gets_a_card():
    """LLM 이 **어느 조합을 골랐든** 카드와 순위가 붙는지.

    ⚠ 옛 구조는 그룹을 돌며 `group.members[0].label` 이 picks 에 있는지 물었다.
    그래서 LLM 이 대표 아닌 멤버를 고르면 그 pick 이 갈 자리가 없어 통째로 사라졌다
    (why·trap_note 까지). 화면에는 순위가 하나 빠진 것처럼만 보여 눈치채기 어렵다.

    멤버가 여럿인 그룹은 드물다 — A·B·C 15개 그룹 중 2개뿐이고(A 의 L2 가 2개,
    B 의 L3 가 3개) 나머지는 전부 1개다. 그래서 이 버그가 오래 안 보였다.
    **그 두 경우를 콕 집어** 대표가 아닌 멤버를 고르게 하고 카드가 나오는지 본다.
    """
    from core import optimizer

    captured: dict = {}
    real_prompt = optimizer.to_prompt_text
    real_select = agent.select_final_three

    def spy_prompt(groups, *a, _c=captured, **k):
        _c["groups"] = groups
        return real_prompt(groups, *a, **k)

    # app.py 가 `from core.optimizer import to_prompt_text` 로 이름을 묶으므로
    # **앱을 만들기 전에** 갈아끼워야 한다.
    optimizer.to_prompt_text = spy_prompt
    try:
        for scenario in ("A", "B"):
            captured.clear()
            agent.select_final_three = lambda p, l, c=None: agent.Advice(
                ok=False, failure=agent.Failure.NO_KEY, detail="탐색용"
            )
            at = _fresh_app()
            while at.session_state["step"] < app.S_REVENUE:
                _advance(at)
            at.selectbox(key="scenario_key_w").set_value(scenario)
            at.run(timeout=120)
            _to_results(at)

            groups = captured.get("groups") or []
            multi = next((g for g in groups if len(g.members) > 1), None)
            assert multi is not None, (
                f"[{scenario}] 멤버가 여럿인 그룹이 없다 — 이 테스트가 짚으려는 경우가 "
                "사라졌다. 조합 생성이 바뀌었는지 확인하라"
            )
            # 대표가 **아닌** 멤버를 1위로. 이게 옛 구조에서 사라지던 pick 이다.
            others = [g for g in groups if g is not multi][:2]
            chosen = [multi.members[-1].label] + [g.members[0].label for g in others]

            def fake(prompt_text, labels, client=None, _picked=chosen):
                assert all(x in labels for x in _picked), "고른 label 이 목록에 없다"
                return agent.Advice(
                    ok=True, phase_reading="검증용", excluded=[],
                    picks=[
                        agent.Pick(label=x, axis_focus="부채", why=f"검증용 {i}",
                                   trap_note=None, steps=[], before_apply=None)
                        for i, x in enumerate(_picked, 1)
                    ],
                )

            agent.select_final_three = fake
            # 캐시를 비운다 — 프롬프트가 그대로면 앞서 캐시된 실패 응답이 재사용된다.
            at.session_state["_advice"] = None
            while at.session_state["step"] > app.S_REVENUE:
                _back(at)
            _to_results(at)

            titles = [e.label for e in at.expander if "│" in e.label]
            ranked = [t for t in titles if "위**" in t]
            assert len(ranked) == 3, (
                f"[{scenario}] 고른 3개 중 {len(ranked)}개만 순위가 붙었다\n  "
                + "\n  ".join(titles)
            )
            for want, title in zip(chosen, ranked):
                assert want in title, (
                    f"[{scenario}] 고른 조합이 카드에 안 나왔다\n"
                    f"  고른 것: {want}\n  카드: {title}"
                )
            print(f"   [{scenario}] 멤버 {len(multi.members)}개 그룹의 대표 아닌 멤버를 "
                  f"골라도 카드 3개에 순위가 붙는다 ✓")
    finally:
        optimizer.to_prompt_text = real_prompt
        agent.select_final_three = real_select


def test_slider_init_paths_and_no_warning():
    """슬라이더 초기화 두 경로가 어긋나지 않고, Streamlit 경고가 안 나는지.

    되돌리기가 세션에 값을 대입하므로 `value=` 를 함께 넘기면 Streamlit 이
    "default value 로 만들었는데 Session State API 로도 값을 넣었다"고 경고하며
    스택까지 찍는다. 화면에는 안 뜨지만 시연 중 터미널에 나오면 결함처럼 보인다.
    그래서 세션에 값이 있으면 `value=` 를 넘기지 않는데, 그 분기가 두 경로를 만든다.

        세션에 값 없음   value=추천금액 으로 생성 (첫 진입 · 퍼지 후 복귀)
        세션에 값 있음   value= 없이 생성        (굴린 뒤 · 되돌린 뒤)

    value 만 갈리고 min·max·step 은 같아야 한다.
    """
    with _streamlit_warnings() as warned:
        at = _fresh_app()
        while at.session_state["step"] < app.S_GOALS:
            _advance(at)
        _to_results(at)
        sliders = _loan_sliders(at)
        if not sliders:
            print("   (융자 슬라이더가 있는 카드 없음 — 스킵)")
            return

        key = sliders[0].key
        base = sliders[0].value
        first = sliders[0].proto            # 경로 A — value= 로 생성
        assert _card_loan_amount(at) == base, "첫 진입부터 슬라이더와 카드가 다르다"

        def slider_of(at):
            return [s for s in _loan_sliders(at) if s.key == key][0]

        slider_of(at).set_value(base + 300)
        at.run(timeout=120)
        at.button(key=f"reset_{key}").click()
        at.run(timeout=120)
        second = slider_of(at).proto        # 경로 B — 세션 값으로 생성
        assert slider_of(at).value == base

        assert (first.min, first.max, first.step) == (second.min, second.max, second.step), (
            f"두 경로의 min·max·step 이 다르다\n"
            f"  value= 경로: {first.min}~{first.max} step {first.step}\n"
            f"  세션  경로: {second.min}~{second.max} step {second.step}"
        )

        # 퍼지 후 복귀 — 세션이 비므로 다시 value= 경로. 새 계산값으로 시작해야 한다.
        _back(at)
        assert not [k for k in at.session_state.filtered_state if str(k).startswith("loan_")]
        _advance(at)
        assert slider_of(at).value == base, "복귀 후 추천 금액으로 시작하지 않는다"

        # 목표를 바꾸면 추천 금액 자체가 달라진다. 옛 값이 남으면 안 된다.
        slider_of(at).set_value(base + 400)
        at.run(timeout=120)
        while at.session_state["step"] > app.S_GOALS:
            _back(at)
        at.slider(key="goal_months_w").set_value(30)
        at.run(timeout=120)
        _to_results(at)
        after_goal = _loan_sliders(at)[0]
        assert after_goal.value != base + 400, "목표를 바꿨는데 굴린 값이 남았다"
        assert _card_loan_amount(at) == after_goal.value

    session_warnings = [w for w in warned if "Session State API" in w]
    assert not session_warnings, (
        "Streamlit 이 세션 대입 경고를 냈다 — 세션에 값이 있을 때 value= 를 넘기는지 "
        f"확인하라\n  {session_warnings[0]}"
    )
    print(
        f"   슬라이더 두 경로 일치({first.min}~{first.max} step {first.step}) · "
        f"목표 변경 후 {after_goal.value} · Streamlit 경고 0건 ✓"
    )


def test_reset_button_only_where_slider_exists():
    """슬라이더가 없는 카드에는 되돌리기 버튼도 없어야 한다.

    C(회복불가)처럼 융자가 배제된 국면에서는 조정할 융자가 없는 카드가 나온다.
    """
    at = _fresh_app()
    while at.session_state["step"] < app.S_REVENUE:
        _advance(at)
    for scenario in ("A", "B", "C"):
        while at.session_state["step"] > app.S_REVENUE:
            _back(at)
        at.selectbox(key="scenario_key_w").set_value(scenario)
        at.run(timeout=120)
        _to_results(at)
        slider_keys = {s.key for s in _loan_sliders(at)}
        reset_keys = {
            b.key[len("reset_"):] for b in at.button
            if b.key and b.key.startswith("reset_loan_")
        }
        assert reset_keys == slider_keys, (
            f"[{scenario}] 버튼과 슬라이더가 짝이 안 맞는다\n"
            f"  슬라이더만: {slider_keys - reset_keys}\n"
            f"  버튼만: {reset_keys - slider_keys}"
        )
        print(f"   [{scenario}] 슬라이더 {len(slider_keys)}개 = 되돌리기 버튼 {len(reset_keys)}개 ✓")


@contextlib.contextmanager
def _capture_messages():
    """st.error/warning/caption/info 로 나간 문구를 모은다.

    app.py 는 `import streamlit as st` 라 모듈 속성을 갈아끼우면 그대로 잡힌다.
    """
    seen: list[tuple[str, str]] = []
    names = ("error", "warning", "caption", "info")
    orig = {n: getattr(streamlit, n) for n in names}
    for name in names:
        setattr(
            streamlit, name,
            lambda msg, *a, _n=name, **k: seen.append((_n, str(msg))),
        )
    try:
        yield seen
    finally:
        for name, fn in orig.items():
            setattr(streamlit, name, fn)


def test_zero_candidates_message_by_cause():
    """후보 0개일 때 원인별로 다른 문구가 뜨는지.

    자격 확인으로 빠진 경우와 애초에 자격이 없는 경우는 사용자가 할 수 있는 일이 다르다.
    이 분기는 앞선 커밋에서 넣었지만 옛 경고가 먼저 return 해 실행되지 않았다.
    화면 문구를 직접 보는 테스트가 없어 못 잡았다.
    """
    def run(dropped: bool) -> str:
        with _capture_messages() as seen:
            # groups 가 비면 profile·ctx 를 쓰기 전에 return 한다.
            app.render_results(
                [], {}, [], None, None,
                baseline_survival=6, goal_satisfied=True, target_months=None,
                resolve_info=None, prompt_text="", precise=True,
                advice=None, dropped_by_confirm=dropped,
            )
        return " / ".join(text for _, text in seen)

    by_confirm = run(True)
    assert "확인되지 않은 자격은 없는 것으로 보고 제외했습니다" in by_confirm
    assert "⑥ 추가 확인" in by_confirm
    assert "조건에 맞는 조합을 찾지 못했습니다" not in by_confirm, (
        "옛 경고가 살아 있다 — 원인별 분기가 다시 죽었다"
    )

    no_eligible = run(False)
    assert "현재 조건으로 자격이 되는 정책이 없습니다" in no_eligible
    assert "소상공인 지원기관" in no_eligible
    assert "⑥ 추가 확인" not in no_eligible

    # 자격을 얻으려고 상황을 다르게 적으라는 안내는 쓰지 않는다.
    for text in (by_confirm, no_eligible):
        for banned in ("폐업", "재창업"):
            assert banned not in text, f"'{banned}' 를 바꿔 보라는 안내는 쓰지 않는다"
    print("   후보 0개 — 자격확인 제외 / 애초에 자격 없음 문구가 갈린다 ✓")


def test_no_unreachable_code():
    """이른 return 뒤에 놓여 실행되지 않는 코드를 찾는다.

    같은 조건을 두 번 검사하면(`if not groups:` 가 둘) 뒤 블록이 죽는데, 화면은
    옛 문구를 그대로 띄워서 눈에 띄지 않는다. 조건 안의 이름이 그 사이에 재대입되면
    정상이므로 건너뛴다.
    """
    term = (ast.Return, ast.Raise, ast.Break, ast.Continue)

    def names_in(node):
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    def assigned_in(stmts):
        return {
            n.id for s in stmts for n in ast.walk(s)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }

    root = pathlib.Path(app.__file__).parent
    files = sorted(root.glob("*.py")) + sorted((root / "core").glob("*.py"))
    problems = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            for attr in ("body", "orelse", "finalbody"):
                stmts = getattr(node, attr, None)
                if not (isinstance(stmts, list) and stmts
                        and isinstance(stmts[0], ast.stmt)):
                    continue
                for i, stmt in enumerate(stmts[:-1]):
                    if isinstance(stmt, term):
                        problems.append(
                            f"{path.name}:{stmts[i + 1].lineno} "
                            f"— {stmt.lineno}행 {type(stmt).__name__} 뒤"
                        )
                guards: dict[str, int] = {}
                for i, stmt in enumerate(stmts):
                    if not isinstance(stmt, ast.If):
                        continue
                    cond = ast.unparse(stmt.test)
                    if cond in guards:
                        prev = guards[cond]
                        if not (names_in(stmt.test) & assigned_in(stmts[prev + 1:i])):
                            problems.append(
                                f"{path.name}:{stmt.lineno} — `if {cond}` 를 "
                                f"{stmts[prev].lineno}행이 이미 걸러 return 한다"
                            )
                    if stmt.body and isinstance(stmt.body[-1], term) and not stmt.orelse:
                        guards[cond] = i

    assert not problems, "도달하지 않는 코드\n  " + "\n  ".join(problems)
    print(f"   도달 불가 코드 없음 · 파일 {len(files)}개 ✓")


def main() -> None:
    print("=" * 50)
    print("app.py UI 헬퍼·배선 검증")
    print("=" * 50)
    # 이 파일은 UI 배선을 본다. 매출 예측이 섞이면 두 가지가 어긋난다.
    #   ① 이 파일은 PYTHONUTF8=1 로 돌리는데 그러면 Prophet 이 죽어(CLAUDE.md 참조)
    #      두 모델짜리 앙상블을 검사하게 된다 — 배포는 세 모델이라 다른 것을 본다.
    #   ② 배선 검사가 예측값에 흔들린다. 예측은 test_forecast 가 따로 본다.
    # 앱 경로로 앙상블이 실제로 불리는지는 test_forecast 가 확인한다.
    forecast.ENABLED = False
    test_money_to_man()
    test_no_live_api_call_by_default()
    test_months_label()
    test_no_direct_widget_calls()
    test_demo_goal_above_baseline()
    test_debt_relief_reaches_confirmation()
    test_drop_window_label_matches()
    test_reason_sentences_end_with_period()
    test_confirm_keys_are_stable()
    test_confirm_excludes_unsure_like_no()
    test_demo_confirm_presets_match_real_questions()
    test_zero_candidates_message_by_cause()
    test_no_unreachable_code()
    test_simresult_split_identity()
    test_month_display_sums()
    test_trap_message_matches_table()
    test_schedule_stops_at_exhaustion()
    test_conditional_gating()
    test_preset_switch_no_leak()
    test_answers_survive_navigation()
    test_edit_and_return()
    test_slider_does_not_leak()
    test_slider_resets_when_inputs_change()
    test_slider_does_not_leak_across_scenarios()
    test_every_pick_gets_a_card()
    test_reset_button_restores_only_its_card()
    test_slider_init_paths_and_no_warning()
    test_reset_button_only_where_slider_exists()
    print("\n전부 통과")


if __name__ == "__main__":
    main()
