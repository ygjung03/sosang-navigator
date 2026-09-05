"""현금흐름 엔진 검증.

pytest 없이도 `python -m tests.test_cashflow`로 돌려볼 수 있다.
기획서의 데모 시나리오 3개(A 무부채 / B 거치절벽 / C 연체·회복불가)를 염두에 둔 케이스.
"""

from __future__ import annotations

from core import loader
from core.cashflow import (
    SIM_MONTHS,
    SIM_MONTHS,
    Applied,
    UserProfile,
    simulate,
    trend_revenue,
)

db = loader.load()
MAN = 10_000  # 만원


def snack_shop_profile() -> UserProfile:
    """충무로 분식집 - 매출 하락, 기존 대출 있음 (개선여지 국면).

    ※ 영세 소상공인 기준으로 잡았다.
      경영안정 바우처 자격이 '연매출 1억 400만원 미만'이라, 그보다 큰 매출로
      프로필을 잡으면 보조금이 자격 필터에서 전부 빠져 조합이 융자뿐이 된다.
      월 800만원(연 9,600만원)이면 바우처 자격 안에 들고, 정부가 정의한
      '영세 소상공인' 구간이라 이 서비스의 대상과도 맞는다.

        월 매출     800만
        변동비      256만 (32%)
        고정비      600만
        월 영업이익 -56만
        월 상환      50만
        -> 월 순현금 -106만, 보유현금 500만이면 대략 4~5개월
    """
    return UserProfile(
        cash_on_hand=500 * MAN,
        monthly_revenue=trend_revenue(800 * MAN, -0.015),  # 월 1.5% 하락
        fixed_cost=600 * MAN,
        variable_ratio=0.32,
        existing_debt=3_000 * MAN,
        existing_debt_payment=50 * MAN,
        existing_debt_rate=0.07,  # 대환 대상(7% 이상)
        high_rate_debt=3_000 * MAN,
    )


def long_lived_profile() -> UserProfile:
    """상환 스케줄 검증용 — 25개월 이상 살아남는 프로필.

    거치절벽(25개월차)과 만기일시(24개월차)를 검증하려면 그 시점까지
    살아 있어야 한다. snack_shop_profile은 3,000만원을 빌려도 18개월에
    소진되어, 파산 이후 계산을 동결한 뒤로는 24~25개월 상환액이 0이 된다.
    (동결 전에는 죽은 뒤에도 계산이 이어져 검증이 통과했으나,
     실제로 일어날 수 없는 시나리오를 검증하고 있었다)

        월 매출     1,200만
        변동비        384만 (32%)
        고정비        700만
        월 영업이익   +116만  -> 흑자
    """
    return UserProfile(
        cash_on_hand=2_000 * MAN,
        monthly_revenue=trend_revenue(1_200 * MAN, -0.003),
        fixed_cost=700 * MAN,
        variable_ratio=0.32,
        existing_debt=1_000 * MAN,
        existing_debt_payment=20 * MAN,
        existing_debt_rate=0.05,
    )


def report(title: str, r) -> None:
    print(f"\n-- {title}")
    print(f"   생존 {r.survival_label} · {SIM_MONTHS}개월 부채 {r.debt_at_exit/MAN:,.0f}만원")
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
    """거치 종료 시점에 상환액이 급증하는지 (25개월차).

    ※ 대리대출(대리분기)은 25개월차가 아니라 27개월차부터 원금이 나간다.
      3개월마다 상환하므로 거치 종료 후 첫 상환 달이 27개월차다.
      여기서는 매월 상환하는 **직접대출**로 검증한다.
    """
    p = long_lived_profile()
    c = db.find("fund_credit_weak__2y3y")
    r = simulate(p, [Applied(c, 3_000 * MAN)])
    report("일반경영안정 3,000만원 (2년거치 3년)", r)

    during_grace = r.payment_during[23]  # 24개월차
    after_grace = r.payment_during[24]  # 25개월차
    print(f"   24개월차 {during_grace/MAN:,.0f}만원 -> 25개월차 {after_grace/MAN:,.0f}만원")
    assert after_grace > during_grace * 1.5, "거치 종료 시점에 상환액이 급증해야 함(절벽)"
    return r


def test_bullet_repayment():
    """서울시 성장기반자금 2년만기 일시상환 - 실행 후 24개월차 낭떠러지."""
    p = long_lived_profile()
    c = db.find("seoul_growth_base__2y_bullet")
    r = simulate(p, [Applied(c, 3_000 * MAN)])
    report("서울시 성장기반 3,000만원 (2년만기 일시상환)", r)

    # 실행이 지급시점만큼 밀리면 만기도 같이 밀린다. 달을 박지 않고 지급시점에서 센다.
    payout = int(c.get("지급시점_개월", 0) or 0)
    before = r.payment_during[payout + 22]
    at_maturity = r.payment_during[payout + 23]  # 실행 후 24개월차에 원금 전액
    print(f"   {payout + 23}개월차 {before/MAN:,.0f}만원 -> "
          f"{payout + 24}개월차 {at_maturity/MAN:,.0f}만원")
    assert at_maturity > 2_500 * MAN, "만기에 원금 전액이 나가야 함"
    drop = r.debt_at[payout + 23] - r.debt_at[payout + 24]
    print(f"   부채잔액 {r.debt_at[payout + 23]/MAN:,.0f} -> "
          f"{r.debt_at[payout + 24]/MAN:,.0f}만원")
    assert drop > 2_900 * MAN, "만기 상환으로 신규 융자 잔액이 사라져야 함"
    return r


def test_refinance():
    """대환은 유입이 없고 월 상환액만 낮춘다."""
    p = snack_shop_profile()
    base = simulate(p)
    c = db.find("fund_refinance__2y8y")
    r = simulate(p, [Applied(c, 3_000 * MAN)])
    report("대환대출 3,000만원 (7% -> 4.5%, 2년거치 8년)", r)

    assert sum(r.inflow_during) == 0, "대환은 현금 유입이 없어야 함"
    print(f"   월 상환 {base.payment_during[0]/MAN:,.1f}만원 -> {r.payment_during[0]/MAN:,.1f}만원")
    assert r.payment_during[0] < base.payment_during[0], "월 상환 부담이 줄어야 함"
    assert r.survival_months > base.survival_months, "생존 개월수가 늘어야 함"
    return r


def test_debt_forgiveness():
    """새출발기금 부실차주 - 원금 감면 + 월 상환액 급감.

    ※ 채무조정은 확정까지 6개월이 걸린다(적용시점_개월). 그 전에 파산하면
      효과가 나타나기 전에 시뮬레이션이 끝나므로, 6개월을 버틸 수 있는
      프로필이어야 검증이 성립한다.
      감면 대상도 '지금' 잔액이 아니라 '확정 시점' 잔액이다.
    """
    p = UserProfile(
        cash_on_hand=1_500 * MAN,
        monthly_revenue=trend_revenue(900 * MAN, -0.02),
        fixed_cost=650 * MAN,
        variable_ratio=0.32,
        existing_debt=6_000 * MAN,
        existing_debt_payment=150 * MAN,
        existing_debt_rate=0.07,
        overdue_days=120,
    )
    base = simulate(p)
    report("무지원 (연체 120일)", base)

    c = db.find("new_start__default")
    r = simulate(p, [Applied(c)])
    report("새출발기금 부실차주 (감면율 60% · 6개월 뒤 확정)", r)

    assert r.debt_at_exit < base.debt_at_exit, "부채가 줄어야 함"
    assert r.forgiven > 0
    assert sum(r.inflow_during) == 0, "채무조정은 현금 유입이 없어야 함"
    return r


def test_repayment_extension():
    """소진공 상환연장제도 - 원금 감면 없이 월 상환액만 낮춤."""
    p = snack_shop_profile()
    p.policy_loan_balance = 2_000 * MAN
    base = simulate(p)
    c = db.find("repay_extension__max7y")
    r = simulate(p, [Applied(c)])
    report("상환연장제도 (대상 2,000만원, 최대 5년 연장)", r)

    # 상환연장은 약정까지 1개월 이상 걸린다(적용시점_개월). 첫 달은 원래 금액을 낸다.
    # 소진공 안내: "약정 전까지 기존 대출에서 발생한 이자는 약정직전 수납 필요"
    at = int(db.find("repay_extension__max7y").get("적용시점_개월", 0))
    print(f"   {at}개월차까지는 원래 금액: "
          f"{base.payment_during[at-1]/MAN:,.1f}만 = {r.payment_during[at-1]/MAN:,.1f}만")
    print(f"   {at+1}개월차부터 감소: "
          f"{base.payment_during[at]/MAN:,.1f}만 -> {r.payment_during[at]/MAN:,.1f}만")
    assert abs(r.payment_during[at - 1] - base.payment_during[at - 1]) < 1, (
        "적용 시점 전까지는 원래 금액을 내야 함"
    )
    assert r.payment_during[at] < base.payment_during[at], "적용 후 월 상환액이 줄어야 함"
    assert r.forgiven == 0, "상환연장은 원금 감면이 없어야 함"
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

    ※ 감면 대상은 '지금' 잔액이 아니라 '확정 시점(6개월 뒤)' 잔액이다.
      그 사이 갚은 만큼은 이미 줄어 있으므로 기대값을 고정 숫자로 쓸 수 없다.
      대신 **감면액이 남은 부채의 감면율만큼인지**를 검증한다.
    """
    p = UserProfile(
        cash_on_hand=1_500 * MAN,
        monthly_revenue=trend_revenue(900 * MAN, -0.02),
        fixed_cost=650 * MAN,
        variable_ratio=0.32,
        existing_debt=6_000 * MAN,
        existing_debt_payment=150 * MAN,
        existing_debt_rate=0.07,
        high_rate_debt=6_000 * MAN,
        overdue_days=120,
    )
    r = simulate(
        p,
        [
            Applied(db.find("fund_refinance__2y8y"), 2_000 * MAN),  # 2,000만 전환
            Applied(db.find("new_start__default")),                  # 나머지 감면
        ],
    )
    report("대환 2,000만 + 새출발기금 (동시 적용)", r)

    # 대환이 2,000만을 가져갔으므로 채무조정 대상은 '남은 4,000만'이고,
    # 6개월간 상환한 만큼 더 줄어 있다. 상한은 4,000만 x 감면율이다.
    rate = float(db.find("new_start__default").get("원금감면율_추정"))
    ceiling = 4_000 * MAN * rate           # 6개월간 한 푼도 안 갚았을 때의 상한
    all_debt = 6_000 * MAN * rate          # 전액에 적용했을 때 (이중 계산)
    print(f"   감면액 {r.forgiven/MAN:,.0f}만원 "
          f"(상한 {ceiling/MAN:,.0f}만 · 이중계산이면 {all_debt/MAN:,.0f}만)")
    assert r.forgiven <= ceiling + 1, (
        f"채무조정이 기존 부채 전액에 적용되면 이중 계산이다. "
        f"실제 {r.forgiven:,.0f} / 상한 {ceiling:,.0f}"
    )
    assert r.forgiven > ceiling * 0.5, (
        f"감면액이 지나치게 작다. 대환이 덜어낸 몫이 과다 계산됐을 수 있다. "
        f"실제 {r.forgiven:,.0f} / 상한 {ceiling:,.0f}"
    )
    assert sum(r.inflow_during) == 0, "대환·채무조정 모두 현금 유입이 없어야 함"
    return r


def test_combination():
    """대환 + 긴급경영안정 - 기획서 예시 1의 1순위 조합."""
    p = snack_shop_profile()
    r = simulate(
        p,
        [
            Applied(db.find("fund_refinance__2y8y"), 3_000 * MAN),
            Applied(db.find("fund_emergency__2y3y"), 2_000 * MAN),
        ],
    )
    report("대환 3,000만 + 긴급경영안정 2,000만", r)
    assert sum(r.inflow_during) == 2_000 * MAN, "융자 유입만 잡혀야 함(대환은 0)"
    return r


def test_payment_identity():
    """payment_during == interest_during + principal_during 를 **전수** 검증한다.

    cases_phase.py 33개 케이스 각각에서 자격을 통과한 조합을 모두 만들어
    시뮬레이션하고, 모든 달에서 항등식이 성립하는지 본다. 손으로 케이스를
    만들지 않아도 상환방식 네 가지(균등분할·분기균등·대리분기·만기일시)와
    거치 중/거치 종료 후/만기 도달, 파산(셋 다 0), 이자>상환액(원금상환 0)이
    자연스럽게 다 걸린다.
    """
    from core.combos import (
        build_combos,
        default_loan_amounts,
        eligible_candidates,
        evaluate,
        fixed_amounts,
        grant_amounts,
    )
    from tests.cases_phase import CASES, context_of, profile_of

    checked = 0
    for name, c in CASES.items():
        profile = profile_of(c)
        ctx = context_of(c)
        # 무지원 베이스라인 + 자격을 통과한 모든 조합
        results = [simulate(profile)]
        pool = eligible_candidates(db, profile, ctx)
        amounts = default_loan_amounts(db, pool, profile, target_months=6)
        amounts.update(fixed_amounts(profile, pool))
        amounts.update(grant_amounts(db, pool, ctx))
        results += [ev.result for ev in evaluate(profile, build_combos(db, pool, amounts))]

        for r in results:
            for m in range(SIM_MONTHS):
                diff = abs(r.payment_during[m] - (r.interest_during[m] + r.principal_during[m]))
                assert diff < 1e-3, (
                    f"{name} · {m + 1}개월차: payment {r.payment_during[m]:,.4f} != "
                    f"interest {r.interest_during[m]:,.4f} + principal {r.principal_during[m]:,.4f} "
                    f"(차이 {diff})"
                )
            checked += 1

    print("\n-- 이자/원금 항등식 전수 검증")
    print(f"   {len(CASES)}개 케이스 · 조합 {checked}개 · 모든 달 "
          "payment == interest + principal 성립")
    assert checked > 50, f"조합이 너무 적다({checked}개) — 커버리지 부족"


def test_survival_means_last_month_with_cash():
    """`survival_months` 가 **잔고가 남아 있던 마지막 달**인지 전수 검증한다.

    "0 이하가 되는 첫 시점"으로 읽으면 한 칸 밀린다. 상환 스케줄을 자르는 자리와
    함정 판정이 이 뜻에 기대고 있어, 한 칸 어긋나면 조용히 틀린 표가 나간다.

        balance_at[survival]     > 0     아직 남아 있다
        balance_at[survival + 1] == 0    여기서 말랐다
        payment_during[survival:]        전부 0 (못 갚았다)
    """
    from core.combos import (
        build_combos,
        default_loan_amounts,
        eligible_candidates,
        evaluate,
        fixed_amounts,
        grant_amounts,
    )
    from tests.cases_phase import CASES, context_of, profile_of

    checked = died = 0
    for name, c in CASES.items():
        profile = profile_of(c)
        ctx = context_of(c)
        results = [simulate(profile)]
        pool = eligible_candidates(db, profile, ctx)
        amounts = default_loan_amounts(db, pool, profile, target_months=6)
        amounts.update(fixed_amounts(profile, pool))
        amounts.update(grant_amounts(db, pool, ctx))
        results += [ev.result for ev in evaluate(profile, build_combos(db, pool, amounts))]

        for r in results:
            months = r.survival_months
            checked += 1
            if months >= SIM_MONTHS:
                assert r.balance_at[SIM_MONTHS] > 0, f"{name}: 36개월 생존인데 잔고 0"
                continue
            died += 1
            assert r.balance_at[months] > 0, (
                f"{name}: survival={months} 인데 "
                f"balance_at[{months}]={r.balance_at[months]:,.0f} <= 0"
            )
            assert r.balance_at[months + 1] <= 0.5, (
                f"{name}: survival={months} 인데 "
                f"balance_at[{months + 1}]={r.balance_at[months + 1]:,.0f} > 0"
            )
            assert all(v < 0.5 for v in r.payment_during[months:]), (
                f"{name}: survival={months} 뒤에도 상환액이 있다"
            )

    print()
    print("-- 생존 개월수의 뜻 전수 검증")
    print(f"   조합 {checked}개(그중 파산 {died}개) · "
          "survival = 잔고가 남아 있던 마지막 달")
    assert died > 20, f"파산 사례가 너무 적다({died}개) — 검사가 헐겁다"


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
        test_payment_identity,
        test_survival_means_last_month_with_cash,
    ):
        fn()
    print("\n" + "=" * 62)
    print("전부 통과")
    print("=" * 62)


if __name__ == "__main__":
    main()
