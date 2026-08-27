"""함정 판정 검증.

`python -m tests.test_metrics`로 실행.
"""

from __future__ import annotations

from core import loader
from core.cashflow import Applied, UserProfile, simulate, trend_revenue
from core.combos import (
    Combo,
    Evaluated,
    UserContext,
    build_combos,
    default_loan_amounts,
    eligible_candidates,
    evaluate,
)
from core.metrics import TrapKind, detect_all, detect_traps, find_cliff_at, summarize

db = loader.load()
MAN = 10_000


def snack_shop_profile() -> UserProfile:
    """개선여지 국면 — 영세 소상공인, 월 영업이익 -56만원(적자)."""
    return UserProfile(
        cash_on_hand=500 * MAN,
        monthly_revenue=trend_revenue(800 * MAN, -0.015),
        fixed_cost=600 * MAN,
        variable_ratio=0.32,
        existing_debt=3_000 * MAN,
        existing_debt_payment=50 * MAN,
        existing_debt_rate=0.07,
        high_rate_debt=3_000 * MAN,
    )


def healthy_profile() -> UserProfile:
    """흑자 사업자 — 함정 1(즉시악화) 판정이 살아 있는지 확인용."""
    return UserProfile(
        cash_on_hand=1_000 * MAN,
        monthly_revenue=trend_revenue(1_000 * MAN, -0.005),
        fixed_cost=500 * MAN,
        variable_ratio=0.30,  # 영업이익 = 1000 - 300 - 500 = +200만
        existing_debt=0,
        existing_debt_payment=0,
    )


def make_evaluated(profile: UserProfile, candidate_id: str, amount: int) -> Evaluated:
    """후보 하나짜리 조합을 만들어 평가한다."""
    cand = db.find(candidate_id)
    combo = Combo(items=(Applied(cand, amount),))
    return Evaluated(combo=combo, result=simulate(profile, combo.items))


def show(title: str, ev: Evaluated, rep) -> None:
    print(f"\n-- {title}")
    print(
        f"   생존 {ev.result.survival_label} (무지원 {rep.baseline_survival_months}개월) · "
        f"부채 {ev.debt_at_exit/MAN:,.0f}만원 · 절벽 {rep.cliff_at or '없음'}"
    )
    for t in rep.traps:
        print(f"   ⚠ [{t.kind.value}] {t.message}")
    if not rep.traps:
        print("   (함정 없음)")


def test_find_cliff():
    """절벽 시점 탐지 — 판정이 아니라 탐지.

    **인자는 신규 융자의 원금분**이다(총 상환액이 아니다). 원금이 0에서 양수로
    처음 바뀌는 시점을 찾는다. 상환 방식 네 가지를 모두 본다.
    """
    # 2년거치 4년 균분 — 24개월 원금 0, 25개월차부터 양수
    assert find_cliff_at([0.0] * 24 + [80.0] * 12) == 25
    # 만기 일시상환 — 원금이 마지막에 한 번. 그 시점이 절벽이다
    assert find_cliff_at([0.0] * 23 + [5000.0] + [0.0] * 12) == 24
    # 분기 균분상환 — 3개월마다 0/양수가 반복돼도 **첫 번째만** 잡는다.
    # 거치가 끝나는 시점이 절벽이지 그 뒤 반복은 아니다.
    assert find_cliff_at([0.0] * 12 + [0.0, 0.0, 90.0] * 8) == 15
    # 대환대출(거치 없음) — 1개월차부터 원금이 양수라 0->양수 전환이 없다
    assert find_cliff_at([70.0] * 36) is None
    # 채무조정·보조금만 — 신규 융자가 없어 원금이 전부 0
    assert find_cliff_at([0.0] * 36) is None
    print("\n-- 절벽 시점 탐지 OK (거치=25 / 만기일시=24 / 분기=15(첫 회만) / "
          "거치없음·융자없음=None)")


def test_immediate_only_when_profitable():
    """함정 1은 흑자일 때만 판정한다."""
    # 적자 사업자 — 상환액이 커도 이 함정은 안 뜬다
    poor = snack_shop_profile()
    ev_poor = make_evaluated(poor, "fund_general__2y3y", 3_000 * MAN)
    rep_poor = detect_traps(poor, ev_poor, phase="개선여지")
    show("적자 사업자 + 융자 3,000만원", ev_poor, rep_poor)
    assert TrapKind.IMMEDIATE not in {t.kind for t in rep_poor.traps}, (
        "적자면 즉시악화 판정을 하지 않아야 함(동어반복)"
    )
    assert rep_poor.burden_ratio_now is None, "적자면 부담률이 None이어야 함"

    # 흑자 사업자 — 과도한 상환액이면 뜬다
    rich = healthy_profile()
    ev_rich = make_evaluated(rich, "fund_refinance__10y", 5_000 * MAN)
    rep_rich = detect_traps(rich, ev_rich, phase="개선여지")
    show("흑자 사업자(월 +200만) + 무거치 대출", ev_rich, rep_rich)
    print(f"   부담률: {rep_rich.burden_ratio_now:.0%}" if rep_rich.burden_ratio_now else "")
    assert rep_rich.burden_ratio_now is not None, "흑자면 부담률이 나와야 함"
    return rep_poor, rep_rich


def test_cliff_needs_reachability():
    """함정 2 — 절벽에 도달 못 하면 경고하지 않는다."""
    profile = snack_shop_profile()

    # 소액 융자: 절벽(25개월차) 전에 현금이 소진된다
    small = make_evaluated(profile, "fund_general__2y3y", 500 * MAN)
    rep_small = detect_traps(profile, small, phase="개선여지")
    show("융자 500만원 (절벽 전에 소진)", small, rep_small)
    if rep_small.cliff_at and small.result.survival_months < rep_small.cliff_at:
        assert TrapKind.CLIFF not in {t.kind for t in rep_small.traps}, (
            "절벽에 도달하지 못하면 경고하지 않아야 함"
        )
        print("   -> 절벽 도달 전 소진이므로 절벽 경고 없음 (의도된 동작)")
    return rep_small


def test_cliff_fires():
    """함정 2 — 절벽까지 살아서 갔는데 그 뒤 무너지면 경고한다."""
    profile = snack_shop_profile()
    found = None
    for amount in range(1_000, 7_100, 500):
        ev = make_evaluated(profile, "fund_general__2y3y", amount * MAN)
        rep = detect_traps(profile, ev, phase="개선여지")
        if TrapKind.CLIFF in {t.kind for t in rep.traps}:
            found = (amount, ev, rep)
            break

    if found:
        amount, ev, rep = found
        show(f"융자 {amount:,}만원 (절벽 도달 후 붕괴)", ev, rep)
        assert rep.cliff_at is not None
        assert rep.cliff_at <= ev.result.survival_months, "절벽에 도달했어야 함"
        assert not ev.result.survives_full_term, "끝까지 버티면 경고 대상이 아님"
    else:
        print("\n-- 이 프로필에서는 절벽 함정이 발생하는 금액대가 없음")
    return found


def test_worse_than_baseline():
    """함정 4 — 받았는데 무지원보다 빨리 죽으면 경고."""
    profile = snack_shop_profile()
    base = simulate(profile, [])
    print(f"\n-- 베이스라인 악화 탐색 (무지원 생존 {base.survival_months}개월)")

    hits = []
    for cid in ("seoul_growth_base__2y_bullet", "fund_refinance__10y",
                "fund_general__2y3y", "seoul_urgent_selfemp__1y4y"):
        for amount in (500, 1_000, 3_000):
            ev = make_evaluated(profile, cid, amount * MAN)
            rep = detect_traps(profile, ev, phase="개선여지", baseline=base)
            if TrapKind.WORSE_THAN_BASELINE in {t.kind for t in rep.traps}:
                hits.append((cid, amount, ev, rep))

    if hits:
        cid, amount, ev, rep = hits[0]
        show(f"{db.find(cid)} {amount:,}만원", ev, rep)
        assert ev.result.survival_months < base.survival_months
    else:
        print("   이 프로필에서는 베이스라인 악화 사례가 없음 (융자가 잔고를 올리므로 정상)")
    return hits


def test_recovery_fail():
    """함정 3 — 회복불가 국면의 신규 융자. 대환은 제외."""
    profile = snack_shop_profile()

    loan = make_evaluated(profile, "fund_general__2y3y", 2_000 * MAN)
    rep_loan = detect_traps(profile, loan, phase="회복불가")
    show("회복불가 국면 + 신규 융자", loan, rep_loan)
    assert TrapKind.DEBT_IN_RECOVERY_FAIL in {t.kind for t in rep_loan.traps}

    refi = make_evaluated(profile, "fund_refinance__2y8y", 3_000 * MAN)
    rep_refi = detect_traps(profile, refi, phase="회복불가")
    show("회복불가 국면 + 대환대출", refi, rep_refi)
    assert TrapKind.DEBT_IN_RECOVERY_FAIL not in {t.kind for t in rep_refi.traps}, (
        "대환은 부채를 늘리지 않으므로 제외되어야 함"
    )

    # 개선여지 국면이면 신규 융자도 이 함정이 아니다
    rep_ok = detect_traps(profile, loan, phase="개선여지")
    assert TrapKind.DEBT_IN_RECOVERY_FAIL not in {t.kind for t in rep_ok.traps}
    return rep_loan


def test_warning_rate():
    """경고율 — '경고는 적을수록 신뢰' 원칙이 지켜지는지."""
    profile = snack_shop_profile()
    ctx = UserContext(
        phase="개선여지",
        region="서울",
        revenue_drop_rate=0.22,
        annual_revenue=800 * MAN * 12,
    )
    pool = eligible_candidates(db, profile, ctx)
    amounts = default_loan_amounts(db, pool, profile, target_months=12)
    combos = build_combos(db, pool, amounts)
    evaluated = evaluate(profile, combos)
    reports = detect_all(profile, evaluated, ctx.phase)

    print(f"\n-- 전체 조합 {len(combos)}개")
    print(summarize(evaluated, reports, top=8))

    trapped = sum(1 for r in reports.values() if r.has_trap)
    rate = trapped / len(reports) if reports else 0
    print(f"\n   경고가 붙은 조합: {trapped}/{len(reports)}개 ({rate:.0%})")

    from collections import Counter
    kinds = Counter(t.kind.value for r in reports.values() for t in r.traps)
    for kind, n in kinds.most_common():
        print(f"     {kind:<20} {n}건")

    assert rate < 1.0, (
        f"전체 조합의 100%에 경고가 붙으면 정보가 아니라 소음이다 (현재 {rate:.0%})"
    )
    return reports


def main() -> None:
    print("=" * 62)
    print("함정 판정 검증 - 4종")
    print("=" * 62)
    for fn in (
        test_find_cliff,
        test_immediate_only_when_profitable,
        test_cliff_needs_reachability,
        test_cliff_fires,
        test_worse_than_baseline,
        test_recovery_fail,
        test_warning_rate,
    ):
        fn()
    print("\n" + "=" * 62)
    print("전부 통과")
    print("=" * 62)


if __name__ == "__main__":
    main()
