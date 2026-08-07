"""현금흐름 엔진 검증.

pytest 없이도 `python -m tests.test_cashflow`로 돌려볼 수 있다.
기획서의 데모 시나리오 3개(A 무부채 / B 거치절벽 / C 연체·회복불가)를 염두에 둔 케이스.
"""

from __future__ import annotations

from core import loader
from core.cashflow import (
    NET_DEBT_AT,
    SIM_MONTHS,
    Applied,
    UserProfile,
    simulate,
    trend_revenue,
)

db = loader.load()
MAN = 10_000  # 만원


def snack_shop_profile() -> UserProfile:
    """기획서 예시 1 - 충무로 분식집. 매출 하락, 기존 대출 있음.

    ※ v29까지의 예시 수치는 그대로 쓰면 앞뒤가 안 맞았다.
      매출 1,440만 / 고정비 720만 / 변동비율 32%로 계산하면
      영업 현금흐름이 +259만원(흑자)인데 "월 순이익 -85만원"이라 적혀 있었다.
      순이익 쪽을 기준으로 고정비를 역산했다.
      (1,440 - 1,440*0.32 - fixed_cost = -85 -> 약 1,064만원)
    """
    return UserProfile(
        cash_on_hand=800 * MAN,
        monthly_revenue=trend_revenue(1_440 * MAN, -0.015),  # 월 1.5% 하락
        fixed_cost=1_064 * MAN,
        variable_ratio=0.32,
        existing_debt=4_000 * MAN,
        existing_debt_payment=65 * MAN,
        existing_debt_rate=0.07,  # 대환 대상(7% 이상)
    )


def report(title: str, r) -> None:
    print(f"\n-- {title}")
    print(f"   생존 {r.survival_label} · {NET_DEBT_AT}개월 순부채 {r.net_debt_at_30/MAN:,.0f}만원")
    for line in r.notes:
        print(f"   · {line}")


def test_baseline():
    """무지원 = 현재 버퍼. 조합을 안 넣으면 나온다."""
    p = snack_shop_profile()
    r = simulate(p)
    report("무지원 (현재 버퍼)", r)
    assert r.survival_months < SIM_MONTHS, "적자 상태인데 36개월을 버티면 안 됨"
    assert len(r.balance_at) == SIM_MONTHS + 1
    return r


def test_grace_cliff():
    """일반경영안정 7,000만원 - 25개월차에 상환이 시작되는지."""
    p = snack_shop_profile()
    c = db.find("fund_general__2y3y")
    r = simulate(p, [Applied(c, 7_000 * MAN)])
    report("일반경영안정 7,000만원 (2년거치 3년)", r)

    during_grace = r.payment_during[23]  # 24개월차
    after_grace = r.payment_during[24]  # 25개월차
    print(f"   24개월차 {during_grace/MAN:,.0f}만원 -> 25개월차 {after_grace/MAN:,.0f}만원")
    assert after_grace > during_grace * 1.5, "거치 종료 시점에 상환액이 급증해야 함(절벽)"
    return r


def test_bullet_repayment():
    """서울시 성장기반자금 2년만기 일시상환 - 24개월차 낭떠러지."""
    p = snack_shop_profile()
    c = db.find("seoul_growth_base__2y_bullet")
    r = simulate(p, [Applied(c, 5_000 * MAN)])
    report("서울시 성장기반 5,000만원 (2년만기 일시상환)", r)

    before = r.payment_during[22]
    at_maturity = r.payment_during[23]  # 24개월차에 원금 전액
    print(f"   23개월차 {before/MAN:,.0f}만원 -> 24개월차 {at_maturity/MAN:,.0f}만원")
    assert at_maturity > 4_000 * MAN, "만기에 원금 전액이 나가야 함"
    drop = r.debt_at[23] - r.debt_at[24]
    print(f"   부채잔액 {r.debt_at[23]/MAN:,.0f} -> {r.debt_at[24]/MAN:,.0f}만원")
    assert drop > 4_900 * MAN, "만기 상환으로 신규 융자 잔액이 사라져야 함"
    return r


def test_refinance():
    """대환은 유입이 없고 월 상환액만 낮춘다."""
    p = snack_shop_profile()
    base = simulate(p)
    c = db.find("fund_refinance__2y8y")
    r = simulate(p, [Applied(c, 4_000 * MAN)])
    report("대환대출 4,000만원 (7% -> 4.5%, 2년거치 8년)", r)

    assert sum(r.inflow_during) == 0, "대환은 현금 유입이 없어야 함"
    print(f"   월 상환 {base.payment_during[0]/MAN:,.1f}만원 -> {r.payment_during[0]/MAN:,.1f}만원")
    assert r.payment_during[0] < base.payment_during[0], "월 상환 부담이 줄어야 함"
    assert r.survival_months > base.survival_months, "생존 개월수가 늘어야 함"
    return r


def test_debt_forgiveness():
    """새출발기금 부실차주 - 원금 감면 + 월 상환액 급감."""
    p = UserProfile(
        cash_on_hand=300 * MAN,
        monthly_revenue=trend_revenue(1_100 * MAN, -0.03),
        fixed_cost=950 * MAN,
        variable_ratio=0.32,
        existing_debt=9_500 * MAN,
        existing_debt_payment=165 * MAN,
        overdue_months=4,
    )
    base = simulate(p)
    report("무지원 (연체 4개월)", base)

    c = db.find("new_start__default")
    r = simulate(p, [Applied(c)])
    report("새출발기금 부실차주 (원금 80% 감면 추정)", r)

    assert r.net_debt_at_30 < base.net_debt_at_30, "순부채가 줄어야 함"
    assert r.forgiven > 0
    assert sum(r.inflow_during) == 0, "채무조정은 현금 유입이 없어야 함"
    return r


def test_repayment_extension():
    """소진공 상환연장제도 - 원금 감면 없이 월 상환액만 낮춤."""
    p = snack_shop_profile()
    p.policy_loan_balance = 3_000 * MAN
    base = simulate(p)
    c = db.find("repay_extension__max5y")
    r = simulate(p, [Applied(c)])
    report("상환연장제도 (대상 3,000만원, 최대 5년 연장)", r)

    print(f"   월 상환 {base.payment_during[0]/MAN:,.1f}만원 -> {r.payment_during[0]/MAN:,.1f}만원")
    assert r.payment_during[0] < base.payment_during[0], "월 상환액이 줄어야 함"
    assert r.forgiven == 0, "상환유예는 원금 감면이 없어야 함"
    return r


def test_cost_relief():
    """경영안정 바우처 25만원 - 해당 월 고정비 감소."""
    p = snack_shop_profile()
    base = simulate(p)
    c = db.find("voucher_mgmt__default")
    r = simulate(p, [Applied(c)])
    report("경영안정 바우처 25만원", r)
    assert r.balance_at[1] > base.balance_at[1], "첫 달 잔고가 바우처만큼 늘어야 함"
    return r


def test_refinance_then_forgiveness():
    """대환 + 채무조정을 함께 적용해도 기존 부채가 이중 계산되지 않는지.

    조합 문법상 융자1 + 채무조정1이 가능하므로 실제로 생길 수 있는 조합이다.
    대환이 기존 부채 일부를 가져간 뒤, 채무조정은 '남은 부분'에만 적용돼야 한다.
    """
    p = UserProfile(
        cash_on_hand=300 * MAN,
        monthly_revenue=trend_revenue(1_100 * MAN, -0.03),
        fixed_cost=950 * MAN,
        variable_ratio=0.32,
        existing_debt=9_500 * MAN,
        existing_debt_payment=165 * MAN,
        existing_debt_rate=0.07,
        overdue_months=4,
    )
    r = simulate(
        p,
        [
            Applied(db.find("fund_refinance__2y8y"), 3_000 * MAN),  # 3,000만 전환
            Applied(db.find("new_start__default")),                  # 나머지 감면
        ],
    )
    report("대환 3,000만 + 새출발기금 (동시 적용)", r)

    # 대환이 3,000만을 가져갔으므로 채무조정 대상은 6,500만이어야 한다.
    # 감면율 80% -> 감면액 5,200만 (9,500만의 80%인 7,600만이 아님)
    expected = 6_500 * MAN * 0.8
    print(f"   감면액 {r.forgiven/MAN:,.0f}만원 (기대 {expected/MAN:,.0f}만원)")
    assert abs(r.forgiven - expected) < 1 * MAN, (
        f"채무조정이 기존 부채 전액(9,500만)에 적용되면 이중 계산이다. "
        f"실제 {r.forgiven:,.0f} / 기대 {expected:,.0f}"
    )
    assert sum(r.inflow_during) == 0, "대환·채무조정 모두 현금 유입이 없어야 함"
    return r


def test_combination():
    """대환 + 긴급경영안정 - 기획서 예시 1의 1순위 조합."""
    p = snack_shop_profile()
    r = simulate(
        p,
        [
            Applied(db.find("fund_refinance__2y8y"), 4_000 * MAN),
            Applied(db.find("fund_emergency__2y3y"), 3_000 * MAN),
        ],
    )
    report("대환 4,000만 + 긴급경영안정 3,000만", r)
    assert sum(r.inflow_during) == 3_000 * MAN, "융자 유입만 잡혀야 함(대환은 0)"
    return r


def main() -> None:
    print("=" * 62)
    print("현금흐름 엔진 검증 - 현금효과 6종 + 조합")
    print("=" * 62)
    for fn in (
        test_baseline,
        test_grace_cliff,
        test_bullet_repayment,
        test_refinance,
        test_debt_forgiveness,
        test_refinance_then_forgiveness,
        test_repayment_extension,
        test_cost_relief,
        test_combination,
    ):
        fn()
    print("\n" + "=" * 62)
    print("전부 통과")
    print("=" * 62)


if __name__ == "__main__":
    main()
