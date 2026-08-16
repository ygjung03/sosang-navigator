"""국면 판정 검증.

`python -m tests.test_phase`로 실행.

케이스는 `tests/cases_phase.py`에 있다. 수익성 계층 넷과 각 지표의 경계값
바로 위아래를 짝으로 포함한다.

⚠ **라벨 일치율은 검증 기준이 아니다.** 케이스의 `phase` 라벨은 손으로 붙인
  것이라 규칙에 맞춰 고치면 100%가 되는데 그것은 순환논리다.
  여기서 보는 것은 다음 셋이다.
    1. 경계값에서 판정이 실제로 뒤집히는가
    2. 하드 규칙이 의도한 상황에서만 발동하는가
    3. 분포가 한쪽으로 쏠리지 않는가
"""

from __future__ import annotations

import collections

from core.cashflow import SIM_MONTHS, simulate
from core.phase import (
    AMBIGUOUS_SCORES,
    describe,
    judge,
    neighbor_phases,
    score_icr,
    score_operating_profit,
    score_overdue,
    score_revenue_drop,
)
from tests.cases_phase import (
    CASES,
    MAN,
    icr,
    interest_cost,
    operating_profit,
    profile_of,
)


def judge_case(name: str):
    """케이스 하나를 판정한다."""
    c = CASES[name]
    buffer = simulate(profile_of(c), horizon=SIM_MONTHS).survival_months
    return judge(
        operating_profit=operating_profit(c),
        interest_cost=interest_cost(c),
        revenue_drop_rate=c.get("drop", 0),
        overdue_days=c["od"],
        is_closed=c["closed"],
        buffer_months=buffer,
    )


# ══════════════════════════════════════════════════════════
# 1. 지표별 경계값
# ══════════════════════════════════════════════════════════
def test_icr_boundary() -> None:
    """이자보상배율 1.0 / 1.5 에서 점수가 바뀐다."""
    cost = 100 * MAN
    # 1.5 경계 — 이상이면 0점, 미만이면 1점
    assert score_icr(150 * MAN, cost)[0] == 0
    assert score_icr(149 * MAN, cost)[0] == 1
    # 1.0 경계 — 이상이면 1점, 미만이면 2점
    assert score_icr(100 * MAN, cost)[0] == 1
    assert score_icr(99 * MAN, cost)[0] == 2
    # 부채가 없으면 이자 부담 자체가 없다
    assert score_icr(-50 * MAN, 0)[0] == 0
    # 적자면 비율이 음수라 문구를 만들지 않는다(영업이익 지표가 담당)
    point, message = score_icr(-50 * MAN, cost)
    assert point == 2 and message is None
    print("  ✓ ICR 1.0 / 1.5 경계, 무부채·적자 처리")


def test_revenue_drop_boundary() -> None:
    """매출 감소율 15% / 40% 에서 점수가 바뀐다."""
    assert score_revenue_drop(0.149)[0] == 0
    assert score_revenue_drop(0.15)[0] == 1
    assert score_revenue_drop(0.399)[0] == 1
    assert score_revenue_drop(0.40)[0] == 2
    # 매출이 늘었으면 당연히 0점
    assert score_revenue_drop(-0.1)[0] == 0
    print("  ✓ 매출 감소 15% / 40% 경계")


def test_overdue_boundary() -> None:
    """연체 10일 / 90일 에서 점수가 바뀐다."""
    assert score_overdue(9)[0] == 0
    assert score_overdue(10)[0] == 1
    assert score_overdue(89)[0] == 1
    assert score_overdue(90)[0] == 2
    print("  ✓ 연체 10일 / 90일 경계")


def test_operating_profit_sign() -> None:
    """영업이익은 부호만 본다. 0원도 적자로 취급한다."""
    assert score_operating_profit(1)[0] == 0
    assert score_operating_profit(0)[0] == 2
    assert score_operating_profit(-1)[0] == 2
    print("  ✓ 영업이익 부호")


# ══════════════════════════════════════════════════════════
# 2. 하드 규칙
# ══════════════════════════════════════════════════════════
def test_closed_forces_unrecoverable() -> None:
    """폐업·휴업이면 점수와 무관하게 회복불가다.

    'J1 폐업·흑자'는 네 지표가 모두 0점이고 36개월을 버티는데도
    폐업 확정이므로 회복불가여야 한다.
    """
    result = judge_case("J1 폐업·흑자")
    assert result.total == 0, result.total
    assert result.phase == "회복불가"
    assert result.hard_rule == "폐업·휴업"
    assert not result.ambiguous, "하드 규칙은 확정 판정이므로 애매하지 않다"
    print(f"  ✓ 폐업 확정 → 회복불가 (점수 {result.total}점인데도)")


def test_zero_score_but_dies_within_horizon() -> None:
    """0점이어도 창 안에 자금이 소진되면 정상이 아니다.

    'B1 거치종료·원금부담'은 ICR이 6.93으로 이자는 넉넉히 내지만
    원금 상환 부담으로 14개월 뒤 파산한다. 네 지표는 모두 0점이다.
    """
    result = judge_case("B1 거치종료·원금부담")
    assert result.total == 0
    assert result.phase == "일시적자금난"
    assert result.hard_rule == "36개월 미생존"
    assert result.buffer_months is not None and result.buffer_months < SIM_MONTHS
    # 반대 케이스 — 0점이고 36개월을 버티면 정상이다
    normal = judge_case("H1 0점·36개월 생존")
    assert normal.total == 0 and normal.phase == "정상"
    assert normal.hard_rule is None
    print(
        f"  ✓ 0점 + {result.buffer_months}개월 생존 → 일시적자금난 / "
        f"0점 + 36개월 생존 → 정상"
    )


def test_single_severe_indicator() -> None:
    """한 지표라도 2점이면 가벼운 국면이 될 수 없다.

    2000=2 (이자도 못 냄)와 0110=2 (가벼운 신호 둘)는 총점이 같지만
    전자는 개선여지, 후자는 일시적자금난이어야 한다.
    """
    severe = judge_case("E4 ICR 0.98")
    mild = judge_case("G2 연체 11일")
    assert severe.total == mild.total == 2, (severe.total, mild.total)
    assert max(severe.scores.values()) == 2
    assert max(mild.scores.values()) == 1
    assert severe.phase == "개선여지" and severe.hard_rule == "단일 지표 2점"
    assert mild.phase == "일시적자금난" and mild.hard_rule is None
    print("  ✓ 같은 2점이라도 단일 지표 2점이면 개선여지로 올라간다")


# ══════════════════════════════════════════════════════════
# 3. 애매 판정
# ══════════════════════════════════════════════════════════
def test_ambiguous_only_at_boundaries() -> None:
    """애매는 경계 점수에서만, 그리고 하드 규칙이 없을 때만 생긴다."""
    for name in CASES:
        result = judge_case(name)
        if result.ambiguous:
            assert result.total in AMBIGUOUS_SCORES, (name, result.total)
            assert result.hard_rule is None, (name, result.hard_rule)
    print("  ✓ 애매는 경계 점수 + 하드 규칙 미적용일 때만")


def test_neighbor_phases_exclude_normal() -> None:
    """3지선다에 '정상'이 들어가면 안 된다.

    애매는 2점 이상에서만 생기므로 정상일 수 없다.
    """
    for name in CASES:
        result = judge_case(name)
        options = neighbor_phases(result)
        if result.ambiguous:
            assert "정상" not in options, (name, options)
            assert result.phase in options
            assert 2 <= len(options) <= 3, options
        else:
            assert options == (result.phase,)
    print("  ✓ 3지선다에 '정상' 없음, 판정 국면을 포함")


# ══════════════════════════════════════════════════════════
# 4. 전체 분포
# ══════════════════════════════════════════════════════════
def test_distribution_not_skewed() -> None:
    """분포가 한 국면에 쏠리지 않는다.

    쏠리면 국면 판정의 실익이 없다. 어느 국면도 전체의 절반을 넘지 않아야 한다.
    """
    dist = collections.Counter(judge_case(name).phase for name in CASES)
    total = len(CASES)
    print(f"\n  분포 ({total}개): {dict(dist)}")
    for phase, count in dist.items():
        assert count <= total / 2, f"{phase}가 {count}/{total}로 쏠렸다"
    assert len(dist) == 4, f"네 국면이 모두 나와야 한다: {dict(dist)}"

    ambiguous = sum(1 for name in CASES if judge_case(name).ambiguous)
    ratio = ambiguous / total
    print(f"  애매: {ambiguous}/{total} ({ratio:.0%})")
    # 너무 자주 물으면 귀찮고 아예 안 물으면 오판이 그대로 간다
    assert 0.05 <= ratio <= 0.4, f"애매 비율이 {ratio:.0%}"
    print("  ✓ 분포 정상, 네 국면 모두 등장")


def test_describe_runs() -> None:
    """설명 문장이 모든 케이스에서 만들어진다."""
    for name in CASES:
        text = describe(judge_case(name))
        assert "국면:" in text
        assert "버틸 수 있습니다" in text
    print("  ✓ describe() 전 케이스 통과")


def main() -> None:
    print("=" * 62)
    print(f"국면 판정 검증 — 케이스 {len(CASES)}개")
    print("=" * 62)
    for fn in (
        test_icr_boundary,
        test_revenue_drop_boundary,
        test_overdue_boundary,
        test_operating_profit_sign,
        test_closed_forces_unrecoverable,
        test_zero_score_but_dies_within_horizon,
        test_single_severe_indicator,
        test_ambiguous_only_at_boundaries,
        test_neighbor_phases_exclude_normal,
        test_distribution_not_skewed,
        test_describe_runs,
    ):
        fn()
    print("\n" + "=" * 62)
    print("전부 통과")
    print("=" * 62)


if __name__ == "__main__":
    main()
