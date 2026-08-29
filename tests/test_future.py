"""미래 자격 후보 검증.

`PYTHONUTF8=1 python -m tests.test_future` 로 실행.

**예측 모델에 기대지 않는다.** 검사 대상은 "예측선이 주어졌을 때 자격을 어떻게 다시
판정하는가"이지 예측값 자체가 아니다. 그래서 계열을 손으로 만들어 넘기고, 데모를
쓰는 검사에서는 앙상블을 꺼 추세선으로 고정한다(`test_forecast` 가 예측을 본다).
"""

from __future__ import annotations

import app
from core import forecast, future
from core.cashflow import UserProfile, trend_revenue
from core.combos import UserContext, eligible_candidates

MAN = 10_000
DB = app.load_db()


def _profile(**over) -> UserProfile:
    base = dict(
        cash_on_hand=1_900 * MAN, monthly_revenue=trend_revenue(663 * MAN, -0.01),
        fixed_cost=435 * MAN, variable_ratio=0.27, existing_debt=5_000 * MAN,
        existing_debt_payment=200 * MAN, existing_debt_rate=0.08,
        high_rate_debt=None, overdue_days=0, policy_loan_balance=0,
    )
    base.update(over)
    return UserProfile(**base)


def test_window_drops_sign_and_gaps():
    """감소를 양수로 담고, 2N개월이 없는 단위는 **넣지 않는다**(0으로 채우지 않는다)."""
    falling = [100.0] * 6 + [80.0] * 6          # 12개월: 뒤 6개월이 20% 낮다
    out = future._window_drops(falling)
    assert out["분기"] == 0.0, out             # 최근 3개월 vs 그 전 3개월 = 둘 다 80
    assert abs(out["반기"] - 0.2) < 1e-9, out  # 480 vs 600 -> 20% 감소
    assert "연도" not in out, "24개월이 없으면 연도 단위를 넣으면 안 된다"

    rising = [80.0] * 6 + [100.0] * 6
    assert future._window_drops(rising)["반기"] < 0, "증가는 음수로 담긴다"
    print("   감소=양수 · 증가=음수 · 자료 없는 단위는 생략 ✓")


def test_shift_context_moves_only_revenue():
    """`shift_context` 는 매출에서 나오는 값과 국면만 바꾼다."""
    ctx = UserContext(phase="일시적자금난", region="서울", credit_score=850,
                      annual_revenue=8_000 * MAN)
    past = [100.0] * 12
    forecast_line = [50] * 6            # 앞으로 반토막
    moved = future.shift_context(ctx, past, forecast_line, 3, "개선여지")

    assert moved.phase == "개선여지"
    assert moved.revenue_drop_rate > 0, "매출이 반토막인데 감소율이 양수가 아니다"
    assert moved.revenue_growth_rate == 0.0, "감소 중인데 증가율이 붙었다"
    for name in ("region", "credit_score", "annual_revenue"):
        assert getattr(moved, name) == getattr(ctx, name), f"{name} 이 바뀌었다"
    assert ctx.revenue_drop_rate == 0.0, "원본 컨텍스트가 바뀌었다"
    print(f"   매출·국면만 이동 · 감소율 {moved.revenue_drop_rate:.1%} · 원본 보존 ✓")


def test_phase_at_uses_debt_at_that_month():
    """이자비용이 **그 시점 잔액**으로 계산되는지. t=0 에서는 현재 부채와 같다.

    고정하면 이자를 과대평가해 ICR 이 낮아지고 국면이 비관 쪽으로 기운다.
    """
    profile = _profile()
    past = [663.0 * MAN] * 12
    line = [663 * MAN] * 36

    at0 = future.phase_at(profile, past, line, 0, False)
    at6 = future.phase_at(profile, past, line, 6, False)
    # 매출을 평평하게 줬으므로 영업이익은 같다. ICR 차이는 이자(=부채)에서만 온다.
    assert at6.scores["ICR"] <= at0.scores["ICR"], (
        "부채가 줄었는데 ICR 점수가 나빠졌다"
    )
    print(f"   t=0 ICR점수 {at0.scores['ICR']} · t=6 {at6.scores['ICR']} "
          f"(부채 상환 반영) ✓")


def test_delayed_does_not_touch_original():
    """후보 사본이 원본을 오염시키지 않는다. `Candidate` 는 frozen dataclass 다."""
    loan = next(c for c in DB.candidates if c.is_loan)
    before = loan.get("지급시점_개월")
    moved = future.delayed(loan, 2)
    assert loan.get("지급시점_개월") == before, "원본이 바뀌었다"
    assert moved.get("지급시점_개월") == (before or 0) + 2
    assert future.delayed(loan, 0) is loan, "0개월이면 사본을 만들 이유가 없다"

    # 원래 값이 있으면 더한다 — 두 지연은 겹치지 않는다
    relief = next((c for c in DB.candidates if c.get("적용시점_개월")), None)
    if relief is not None:
        base = int(relief.get("적용시점_개월"))
        assert future.delayed(relief, 3).get("적용시점_개월") == base + 3
        print(f"   원본 보존 · 지연 누적({base} + 3) ✓")
    else:
        print("   원본 보존 · 지연 누적 ✓")


def test_no_future_when_inputs_missing():
    """예측선이나 과거 실적이 없으면 아무것도 찾지 않는다(간편 모드)."""
    profile = _profile()
    ctx = UserContext(phase="일시적자금난", region="서울")
    assert future.future_candidates(DB, profile, ctx, [], [100] * 6) == {}
    assert future.future_candidates(DB, profile, ctx, [100.0] * 12, []) == {}
    print("   과거 없음 / 예측 없음 → 빈 결과 ✓")


def _demo(key: str):
    """데모 하나의 (preset, summary, 매출선, profile, ctx, drop_ids)."""
    preset = app.DEMO_PRESETS[key]
    summary = app.summarize_scenario(key, 12)
    rev = app.build_monthly_revenue(True, summary, 0)
    closed = bool(preset.get("is_closed", False))
    profile = app.build_profile(
        dict(fixed_cost=preset["fixed_man"] * MAN, variable_cost=preset["var_man"] * MAN,
             cash=preset["cash_man"] * MAN, debt=preset["debt_man"] * MAN,
             debt_payment=preset["debt_pay_man"] * MAN,
             debt_rate=preset["rate_pct"] / 100, high_rate_debt=None,
             overdue_days=int(preset["overdue_days"]), policy_loan_balance=0),
        rev, rev[0])
    phase = app.judge_phase(profile, summary, max(summary.recent_3m_avg, 1), closed)
    ctx = app.build_context(
        phase.phase,
        dict(region="서울", restart_status="안함", is_closed=closed,
             tax_delinquent=False, include_loans=True, credit_score=850),
        dict(store_area_pyeong=preset.get("cond_area"),
             business_started=preset["cond_bstart"], policy_loan_count_5y=0),
        summary, int(max(summary.recent_3m_avg, 1) * 12))
    pool = eligible_candidates(DB, profile, ctx)
    yes = {app.confirm_key_of(DB, c) for c in preset.get("confirm_yes", ())}
    drop = app.excluded_by_confirm(
        app.confirm_questions(pool), {k: app.CONFIRM_YES for k in yes if k})
    return preset, summary, rev, profile, ctx, drop


def test_already_eligible_gets_no_future_card():
    """이미 자격이 있는 후보는 '미래'가 아니다. 데모 B·C 는 미래 후보가 없다."""
    for key in ("B", "C"):
        preset, summary, rev, profile, ctx, drop = _demo(key)
        past = [float(v) for v in summary.monthly_revenue.values]
        found = future.future_candidates(DB, profile, ctx, past, rev, ctx.is_closed)
        assert not found, f"[{key}] 미래 후보가 생겼다: {found}"
    print("   데모 B·C — 미래 후보 0개(이미 자격이 있다) ✓")


def test_future_pipeline_leaves_main_result_alone():
    """미래 파이프라인을 돌려도 본 결과가 한 글자도 안 바뀐다.

    미래 후보를 현재 후보와 같은 풀에 넣으면 순위가 흔들린다. 그건 "지금 신청할 수
    있는 것들"의 비교를 "몇 달 뒤에나 되는 것"이 흔드는 셈이라 맞지 않는다.
    """
    for key in ("A", "B", "C"):
        preset, summary, rev, profile, ctx, drop = _demo(key)
        before = app.run_pipeline(DB, profile, ctx, None, preset["goal_months"], drop)
        app.run_future_pipeline(
            DB, profile, ctx, summary, rev, before[0][:3], preset["goal_months"], drop)
        after = app.run_pipeline(DB, profile, ctx, None, preset["goal_months"], drop)

        assert len(before[0]) == len(after[0]), f"[{key}] 그룹 수가 달라졌다"
        for b, a in zip(before[0], after[0]):
            assert b.members[0].label == a.members[0].label, f"[{key}] 순위가 달라졌다"
            assert b.survival_months == a.survival_months, f"[{key}] 생존이 달라졌다"
            assert b.debt_at_exit == a.debt_at_exit, f"[{key}] 부채가 달라졌다"
        assert len(before[1]) == len(after[1]), f"[{key}] 함정 리포트 수가 달라졌다"
    print("   본 결과 불변 — 그룹·순위·두 축·함정 (A/B/C) ✓")


def test_simple_mode_has_no_future_card():
    """간편 모드(POS 없음)에서는 미래 자격을 계산하지 않는다."""
    profile = _profile()
    ctx = UserContext(phase="일시적자금난", region="서울")
    groups, reports, months = app.run_future_pipeline(
        DB, profile, ctx, None, app.build_monthly_revenue(False, None, 800 * MAN), [])
    assert groups == [] and reports == {} and months == {}
    print("   간편 모드 → 미래 카드 없음 ✓")


def main() -> None:
    print("=" * 62)
    print("미래 자격 후보 검증")
    print("=" * 62)
    forecast.ENABLED = False  # 예측값이 아니라 자격 판정을 본다. 추세선으로 고정.
    for fn in (
        test_window_drops_sign_and_gaps,
        test_shift_context_moves_only_revenue,
        test_phase_at_uses_debt_at_that_month,
        test_delayed_does_not_touch_original,
        test_no_future_when_inputs_missing,
        test_already_eligible_gets_no_future_card,
        test_future_pipeline_leaves_main_result_alone,
        test_simple_mode_has_no_future_card,
    ):
        fn()
    print("\n" + "=" * 62)
    print("전부 통과")
    print("=" * 62)


if __name__ == "__main__":
    main()
