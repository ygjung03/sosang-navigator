"""조합 생성 검증.

`python -m tests.test_combos`로 실행.
"""

from __future__ import annotations

from core import loader
from core.cashflow import UserProfile, trend_revenue
from core.combos import (
    Combo,
    UserContext,
    build_combos,
    default_loan_amounts,
    eligible_candidates,
    evaluate,
    filter_by_goal,
    required_amount,
    required_loan_amount,
    summarize,
)

db = loader.load()
MAN = 10_000


def snack_shop_profile() -> UserProfile:
    """충무로 분식집 — 매출 하락, 기존 대출 있음 (개선여지 국면).

    영세 소상공인 기준으로 잡았다. 경영안정 바우처 자격이
    '연매출 1억 400만원 미만'이라 그보다 큰 매출로 프로필을 잡으면
    보조금이 자격 필터에서 전부 빠져 조합이 융자뿐이 된다.
    월 800만원(연 9,600만원)이면 바우처 자격 안에 든다.
    """
    return UserProfile(
        cash_on_hand=500 * MAN,
        monthly_revenue=trend_revenue(800 * MAN, -0.015),
        fixed_cost=600 * MAN,
        variable_ratio=0.32,
        existing_debt=3_000 * MAN,
        existing_debt_payment=50 * MAN,
        existing_debt_rate=0.07,
        high_rate_debt=3_000 * MAN,  # 전액이 7% 이상 고금리
    )


def snack_shop_context() -> UserContext:
    return UserContext(
        phase="개선여지",
        region="서울",
        credit_score=None,  # 모름 -> 점수 제한 정책도 통과
        restart_status="안함",
        revenue_drop_rate=0.22,  # 전년 대비 22% 감소
        annual_revenue=800 * MAN * 12,  # 9,600만원 -> 바우처 자격 통과
    )


def test_eligibility():
    """자격 필터가 국면·부채 요건을 제대로 거르는지."""
    profile = snack_shop_profile()
    ctx = snack_shop_context()
    pool = eligible_candidates(db, profile, ctx)

    print(f"\n-- 자격 통과 후보 {len(pool)}개 (개선여지 국면)")
    for c in pool:
        print(f"   [{c.method}] {c}")

    assert pool, "자격을 통과한 후보가 하나도 없으면 안 됨"
    assert all("개선여지" in c.phases for c in pool), "국면 필터가 안 걸림"
    # 영세 사업자(연 9,600만원)라 경영안정 바우처 자격에 들어야 한다
    assert any("바우처" in str(c) for c in pool), (
        "연매출 1억 400만원 미만이면 경영안정 바우처가 통과해야 함"
    )
    # 상환유예는 정책자금 직접대출 잔액이 0이므로 빠져야 한다
    assert not any(c.effect == "상환유예" for c in pool), (
        "정책자금 대출이 없는데 상환연장제도가 통과되면 안 됨"
    )
    return pool


def test_no_loans_option():
    """'대출 없이' 옵션이 융자를 하드 제약으로 배제하는지 (기획서 5.2.1)."""
    profile = snack_shop_profile()
    ctx = snack_shop_context()
    ctx.include_loans = False
    pool = eligible_candidates(db, profile, ctx)

    print(f"\n-- 대출 제외 시 후보 {len(pool)}개")
    for c in pool:
        print(f"   [{c.method}] {c}")
    assert not any(c.is_loan for c in pool), "융자가 배제되어야 함"
    return pool


def test_grammar():
    """조합 문법 제약 — 융자1·채무조정1·보조금2를 넘지 않는지."""
    profile = snack_shop_profile()
    ctx = snack_shop_context()
    pool = eligible_candidates(db, profile, ctx)
    combos = build_combos(db, pool)

    print(f"\n-- 생성된 조합 {len(combos)}개")
    for combo in combos:
        methods = [c.method for c in combo.candidates]
        assert methods.count("융자") <= 1, f"융자 2개 이상: {combo.label}"
        assert methods.count("채무조정") <= 1, f"채무조정 2개 이상: {combo.label}"
        assert methods.count("보조금") <= 2, f"보조금 3개 이상: {combo.label}"

    # 재원 간 중복이 false이므로 소진공 + 서울시 융자 조합은 없어야 한다
    for combo in combos:
        loans = [c for c in combo.candidates if c.is_loan]
        assert len({c.source for c in loans}) <= 1, (
            f"재원이 다른 융자가 섞임: {combo.label}"
        )
    return combos


def test_required_amount():
    """필요액 역산 — 곱셈이 아니라 곡선 최저점 기준."""
    profile = snack_shop_profile()

    need_6 = required_amount(profile, 6)
    need_12 = required_amount(profile, 12)
    print(f"\n-- 필요액 역산 (무지원 곡선 최저점)")
    print(f"   6개월 목표  : {need_6 / MAN:,.0f}만원")
    print(f"   12개월 목표 : {need_12 / MAN:,.0f}만원")

    assert need_12 > need_6, "목표가 길수록 더 많은 돈이 필요해야 함"

    # 융자는 거치 이자 때문에 최저점보다 더 빌려야 한다
    cand = db.find("fund_emergency__2y3y")
    loan_6 = required_loan_amount(profile, cand, 6)
    print(f"   6개월 목표를 융자로: {loan_6 / MAN:,.0f}만원 (이자 반영)")
    assert loan_6 >= need_6, "거치 이자만큼 더 빌려야 함"
    return need_6, loan_6


def test_goal_filter():
    """목표 충족 조합을 '모두' 남기는지 (기획서 5.2.1)."""
    profile = snack_shop_profile()
    ctx = snack_shop_context()
    pool = eligible_candidates(db, profile, ctx)
    combos = build_combos(db, pool)
    evaluated = evaluate(profile, combos)

    passed, satisfied = filter_by_goal(evaluated, target_months=12)
    print(f"\n-- 목표 12개월 충족 조합 {len(passed)}개 (충족 여부: {satisfied})")
    if satisfied:
        assert all(e.survival_months >= 12 for e in passed)
        assert len(passed) > 1, "충족 조합이 하나뿐이면 '모두 남긴다'를 검증 못 함"

    # 달성 불가능한 목표를 주면 전체를 돌려준다
    impossible, ok = filter_by_goal(evaluated, target_months=999)
    assert not ok and len(impossible) == len(evaluated), (
        "충족 조합이 없으면 전체를 돌려줘야 함"
    )
    return passed


def test_closure_filter():
    """폐업 요건 — 영업 중인 사용자에게 점포철거비가 추천되면 안 된다."""
    profile = snack_shop_profile()
    ctx = snack_shop_context()
    ctx.phase = "회복불가"  # 철거비가 걸리는 국면
    ctx.is_closed = False  # 그러나 아직 영업 중

    pool = eligible_candidates(db, profile, ctx)
    names = [str(c) for c in pool]
    print(f"\n-- 회복불가 + 영업중 후보 {len(pool)}개")
    for n in names:
        print(f"   {n}")
    assert not any("철거비" in n for n in names), (
        "영업 중인데 점포철거비가 통과되면 안 됨"
    )

    # 폐업 예정이면 통과해야 한다
    ctx.is_closed = True
    pool_closed = eligible_candidates(db, profile, ctx)
    assert any("철거비" in str(c) for c in pool_closed), (
        "폐업 예정이면 철거비가 나와야 함"
    )
    print(f"   -> 폐업 예정으로 바꾸면 {len(pool_closed)}개 (철거비 포함)")
    return pool


def test_default_loan_amounts():
    """융자 디폴트가 정책 한도가 아니라 필요액이어야 한다 (기획서 5.4)."""
    profile = snack_shop_profile()
    ctx = snack_shop_context()
    pool = eligible_candidates(db, profile, ctx)

    amounts = default_loan_amounts(db, pool, profile, target_months=12)
    print("\n-- 융자 디폴트 금액 (12개월 목표 기준)")
    for cid, amt in amounts.items():
        cand = db.find(cid)
        ceiling = int(cand.get("금액_한도", 0))
        print(f"   {str(cand):<44} {amt/MAN:>8,.0f}만원 / 한도 {ceiling/MAN:>8,.0f}만원")
        assert amt <= ceiling, "디폴트가 한도를 넘으면 안 됨"

    # 5억 한도짜리가 5억으로 잡히면 안 된다
    growth = amounts.get("seoul_growth_base__2y3y")
    if growth:
        assert growth < 50_000_000, (
            f"한도 5억을 그대로 쓰면 비현실적: {growth:,}원"
        )
    return amounts


def test_pipeline():
    """자격 필터 -> 디폴트 금액 -> 조합 -> 시뮬레이션 -> 2축 순위."""
    profile = snack_shop_profile()
    ctx = snack_shop_context()
    pool = eligible_candidates(db, profile, ctx)
    amounts = default_loan_amounts(db, pool, profile, target_months=12)
    combos = build_combos(db, pool, amounts)
    evaluated = evaluate(profile, combos)

    print(f"\n-- 파이프라인: 후보 {len(pool)} -> 조합 {len(combos)} -> 평가 완료")
    print(summarize(evaluated, top=8))

    assert len(evaluated) == len(combos)
    # 무지원 베이스라인보다 나은 조합이 하나는 있어야 한다
    from core.cashflow import simulate

    base = simulate(profile, [])
    best = max(e.survival_months for e in evaluated)
    print(f"\n   무지원 {base.survival_months}개월 -> 최선 조합 {best}개월")
    assert best > base.survival_months, "지원을 받아도 나아지지 않으면 이상함"
    return evaluated


def main() -> None:
    print("=" * 62)
    print("조합 생성 검증 - 자격 필터 · 문법 제약 · 필요액 역산")
    print("=" * 62)
    for fn in (
        test_eligibility,
        test_no_loans_option,
        test_grammar,
        test_required_amount,
        test_goal_filter,
        test_closure_filter,
        test_default_loan_amounts,
        test_pipeline,
    ):
        fn()
    print("\n" + "=" * 62)
    print("전부 통과")
    print("=" * 62)


if __name__ == "__main__":
    main()
