"""후보 선정 검증.

`python -m tests.test_optimizer`로 실행.
"""

from __future__ import annotations

from core import loader
from core.cashflow import UserProfile, simulate, trend_revenue
from core.combos import (
    UserContext,
    build_combos,
    default_loan_amounts,
    eligible_candidates,
    evaluate,
)
from core.metrics import detect_all
from core.optimizer import (
    MIN_CANDIDATES,
    group_by_outcome,
    non_dominated_sort,
    select,
    summarize,
    to_prompt_text,
)

db = loader.load()
MAN = 10_000

# 여섯 가지 상황. 한 케이스만 보고 판단하면 잘못된 결론에 도달한다.
CASES = {
    "A 무부채·자금난": dict(rev=800, fix=600, cash=500, debt=0, pay=0, hi=0,
                        phase="일시적자금난", trend=-0.01),
    "B 부채·개선여지": dict(rev=800, fix=600, cash=500, debt=3_000, pay=50, hi=3_000,
                        phase="개선여지", trend=-0.015),
    "C 연체·회복불가": dict(rev=700, fix=700, cash=200, debt=6_000, pay=110, hi=6_000,
                        phase="회복불가", trend=-0.03),
    "D 흑자·완만하락": dict(rev=1_000, fix=500, cash=1_500, debt=2_000, pay=40, hi=0,
                        phase="일시적자금난", trend=-0.005),
    "E 급락·고부채": dict(rev=600, fix=650, cash=300, debt=8_000, pay=150, hi=8_000,
                       phase="회복불가", trend=-0.04),
    "F 소규모·저부채": dict(rev=500, fix=380, cash=300, debt=1_000, pay=20, hi=1_000,
                        phase="개선여지", trend=-0.02),
}


def build(case: dict):
    """케이스 하나를 파이프라인 끝까지 돌린다."""
    profile = UserProfile(
        cash_on_hand=case["cash"] * MAN,
        monthly_revenue=trend_revenue(case["rev"] * MAN, case["trend"]),
        fixed_cost=case["fix"] * MAN,
        variable_ratio=0.32,
        existing_debt=case["debt"] * MAN,
        existing_debt_payment=case["pay"] * MAN,
        existing_debt_rate=0.07 if case["debt"] else None,
        high_rate_debt=case["hi"] * MAN,
    )
    ctx = UserContext(
        phase=case["phase"],
        region="서울",
        revenue_drop_rate=0.22,
        annual_revenue=case["rev"] * MAN * 12,
    )
    pool = eligible_candidates(db, profile, ctx)
    amounts = default_loan_amounts(db, pool, profile, target_months=12)
    evaluated = evaluate(profile, build_combos(db, pool, amounts))
    reports = detect_all(profile, evaluated, ctx.phase)
    return profile, ctx, evaluated, reports


def test_grouping():
    """같은 좌표는 한 그룹으로 묶이는지."""
    _, _, evaluated, _ = build(CASES["B 부채·개선여지"])
    groups = group_by_outcome(evaluated)
    print(f"\n-- 좌표 그룹핑: 조합 {len(evaluated)}개 → 그룹 {len(groups)}개")

    assert sum(g.size for g in groups) == len(evaluated), "조합이 누락되면 안 됨"
    # 같은 그룹 안의 멤버는 좌표가 같아야 한다
    for g in groups:
        for m in g.members:
            assert m.survival_months == g.survival_months
    # 그룹 간 좌표는 서로 달라야 한다
    coords = [(g.survival_months, round(g.debt_at_exit / MAN)) for g in groups]
    assert len(coords) == len(set(coords)), "그룹 좌표가 중복되면 안 됨"
    return groups


def test_layering():
    """비지배 정렬이 층을 제대로 벗기는지."""
    _, _, evaluated, _ = build(CASES["B 부채·개선여지"])
    groups = group_by_outcome(evaluated)
    selected = non_dominated_sort(groups, need=MIN_CANDIDATES)

    print(f"\n-- 비지배 정렬: 그룹 {len(groups)}개 → 선정 {len(selected)}개")
    by_layer: dict[int, int] = {}
    for g in selected:
        by_layer[g.layer] = by_layer.get(g.layer, 0) + 1
    print(f"   레이어별 그룹 수: {dict(sorted(by_layer.items()))}")

    assert len(selected) >= MIN_CANDIDATES or len(selected) == len(groups)
    # 1층은 아무에게도 지배당하지 않아야 한다
    layer1 = [g for g in selected if g.layer == 1]
    for a in layer1:
        for b in groups:
            dominated = (
                b.survival_months >= a.survival_months
                and b.debt_at_exit <= a.debt_at_exit
                and (b.survival_months > a.survival_months
                     or b.debt_at_exit < a.debt_at_exit)
            )
            assert not dominated, f"1층인데 지배당함: {a.label()}"
    return selected


def test_pruning():
    """그룹 안에서 함정 있는 멤버가 빠지는지."""
    _, _, evaluated, reports = build(CASES["C 연체·회복불가"])
    groups = select(evaluated, reports)

    print(f"\n-- 그룹 내 정리 (회복불가)")
    pruned = [g for g in groups if g.dropped]
    for g in pruned[:3]:
        print(f"   생존 {g.survival_months}개월 · 부채 {g.debt_at_exit/MAN:,.0f}만")
        print(f"     남김 {g.size}개 / 제외 {len(g.dropped)}개")

    for g in groups:
        clean = [m for m in g.members if not reports[m.label].has_trap]
        dirty = [m for m in g.members if reports[m.label].has_trap]
        # 함정 없는 멤버가 남아 있으면 함정 있는 멤버가 함께 있으면 안 된다
        if clean:
            assert not dirty, f"함정 없는 대안이 있는데 함정 있는 것이 남음: {g.label()}"
        # 구성 개수가 섞이면 안 된다
        sizes = {len(m.combo.items) for m in g.members}
        assert len(sizes) == 1, f"구성 개수가 섞임: {sizes}"
    return groups


def test_all_traps_kept():
    """멤버 전부에 함정이 있으면 아무도 빼지 않는지."""
    _, _, evaluated, reports = build(CASES["C 연체·회복불가"])
    groups = select(evaluated, reports)

    all_trapped = [
        g for g in groups
        if g.members and all(reports[m.label].has_trap for m in g.members)
    ]
    print(f"\n-- 전원 함정인 그룹 {len(all_trapped)}개 (전부 유지되어야 함)")
    for g in all_trapped[:3]:
        print(f"   생존 {g.survival_months}개월 · 멤버 {g.size}개 · 제외 {len(g.dropped)}개")
    # 전원 함정이면 함정을 이유로 제외된 것이 없어야 한다
    for g in all_trapped:
        assert g.size >= 1, "그룹이 통째로 사라지면 안 됨"
    return all_trapped


def test_edge_cases():
    """경계 상황 — 그룹이 need보다 적거나 없을 때.

    실제 데이터로만 검증하면 이런 상황을 놓친다. 정책이 늘거나 줄면
    (자격 필터가 대부분을 거르면) 충분히 발생한다.
    """
    from core.optimizer import CandidateGroup, non_dominated_sort

    def G(survival, debt):
        return CandidateGroup(survival_months=survival, debt_at_exit=debt)

    print("\n-- 경계 케이스")

    # 그룹이 need보다 적으면 있는 대로 다 보낸다
    out = non_dominated_sort([G(10, 100), G(8, 50)], need=5)
    print(f"   그룹 2개 / need=5  → {len(out)}개  레이어 {[g.layer for g in out]}")
    assert len(out) == 2, "있는 대로 다 보내야 함"

    out = non_dominated_sort([G(10, 100)], need=5)
    assert len(out) == 1

    assert non_dominated_sort([], need=5) == []
    assert select([], {}) == []

    # 완전 트레이드오프 — 한 층을 통째로 넣으므로 need를 넘겨도 자르지 않는다
    tradeoff = [G(20, 900), G(15, 600), G(10, 300), G(5, 100)]
    out = non_dominated_sort(tradeoff, need=2)
    print(f"   완전 트레이드오프 4개 / need=2 → {len(out)}개  레이어 {[g.layer for g in out]}")
    assert len(out) == 4, "층을 자르면 그 자리에 임의 기준이 필요해진다"
    assert all(g.layer == 1 for g in out), "서로 지배하지 않으므로 전부 1층"

    # 일직선 지배 — 층이 하나씩 벗겨진다
    chain = [G(20, 100), G(15, 200), G(10, 300), G(5, 400), G(3, 500), G(1, 600)]
    out = non_dominated_sort(chain, need=5)
    print(f"   체인 6개 / need=5   → {len(out)}개  레이어 {[g.layer for g in out]}")
    assert len(out) == 5
    assert [g.layer for g in out] == [1, 2, 3, 4, 5], "한 층에 하나씩 들어가야 함"
    return out


def test_six_cases():
    """여섯 케이스에서 후보 규모가 안정적인지."""
    print(f"\n{'케이스':<20} {'조합':>5} {'그룹':>5} {'선정':>5} {'멤버':>5} {'층':>3}")
    results = {}
    for name, case in CASES.items():
        _, _, evaluated, reports = build(case)
        if not evaluated:
            print(f"{name:<20} 조합 없음")
            continue
        groups = select(evaluated, reports)
        members = sum(g.size for g in groups)
        layers = max(g.layer for g in groups)
        print(f"{name:<20} {len(evaluated):>5} "
              f"{len(group_by_outcome(evaluated)):>5} {len(groups):>5} {members:>5} {layers:>3}")
        results[name] = (len(groups), members)
        assert len(groups) <= 12, f"{name}: 후보 그룹이 너무 많음 ({len(groups)})"
        assert members <= 15, f"{name}: LLM에 넘길 조합이 너무 많음 ({members})"
    return results


def test_prompt_text():
    """LLM 프롬프트용 직렬화."""
    profile, ctx, evaluated, reports = build(CASES["B 부채·개선여지"])
    groups = select(evaluated, reports)
    baseline = simulate(profile, [])

    print(f"\n-- 후보 요약")
    print(summarize(groups))

    text = to_prompt_text(groups, reports, baseline.survival_months)
    print(f"\n-- LLM 프롬프트 ({len(text)}자)")
    print(text[:1400])

    assert "레이어" in text and "생존" in text and "부채" in text
    assert len(text) < 8000, "프롬프트가 너무 길면 LLM 판단이 산만해진다"
    return text


def main() -> None:
    print("=" * 62)
    print("후보 선정 검증 - 좌표 그룹핑 · 비지배 정렬 · 그룹 내 정리")
    print("=" * 62)
    for fn in (
        test_grouping,
        test_layering,
        test_pruning,
        test_all_traps_kept,
        test_edge_cases,
        test_six_cases,
        test_prompt_text,
    ):
        fn()
    print("\n" + "=" * 62)
    print("전부 통과")
    print("=" * 62)


if __name__ == "__main__":
    main()
