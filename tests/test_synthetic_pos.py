"""합성 POS 생성기 검증.

`python -m tests.test_synthetic_pos`로 실행.

예측 모델이 찾아내야 할 패턴이 실제로 심어져 있는지, 그리고
그 패턴이 '눈으로 바로 보일 만큼 인위적이지는 않은지'를 함께 본다.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.synthetic_pos import (
    HOLIDAY_EFFECT,
    LUNAR_HOLIDAYS,
    SCENARIOS,
    WEEKDAY_FACTOR,
    daily_revenue,
    generate,
    monthly_revenue,
    trend_curve,
)

MAN = 10_000

# generate() 한 번이 0.6초라 테스트마다 새로 만들면 20초 가까이 걸린다.
# 시나리오와 시드가 같으면 결과가 같으므로(재현성 보장) 한 번만 만들어 재사용한다.
_cache: dict[tuple, pd.DataFrame] = {}


def pos_of(scenario: str = "B", months: int = 12, seed: int = 42) -> pd.DataFrame:
    """같은 조건이면 이미 만든 데이터를 돌려준다."""
    key = (scenario, months, seed)
    if key not in _cache:
        _cache[key] = generate(scenario, months=months, seed=seed)
    return _cache[key]


def test_schema():
    """POS 파서가 기대하는 컬럼이 다 있는지."""
    pos = pos_of("B")
    expected = {
        "order_id", "order_datetime", "menu_id", "menu_name",
        "quantity", "unit_price", "channel", "discount_amount",
    }
    print(f"\n-- 스키마: 라인 {len(pos):,}행 / 주문 {pos['order_id'].nunique():,}건")
    print(f"   컬럼 {sorted(pos.columns)}")
    assert expected.issubset(set(pos.columns)), f"누락: {expected - set(pos.columns)}"
    assert (pos["quantity"] > 0).all(), "수량은 양수여야 함"
    assert (pos["discount_amount"] >= 0).all(), "할인액은 음수가 될 수 없음"

    # 할인이 라인 매출을 넘지 않아야 한다
    line_revenue = pos["quantity"] * pos["unit_price"]
    assert (pos["discount_amount"] <= line_revenue).all(), "할인이 매출을 초과"

    # 쿠폰은 배달 주문에만
    hall = pos[pos["channel"] == "홀"]
    assert (hall["discount_amount"] == 0).all(), "홀 주문에 쿠폰이 붙으면 안 됨"
    return pos


def test_month_alignment():
    """월 경계에 맞춰 생성되는지.

    부분 월이 섞이면 월별 집계에서 첫·마지막 달이 실제보다 낮게 나와
    추세를 잘못 읽게 된다.
    """
    pos = pos_of("B")
    monthly = monthly_revenue(pos)
    print(f"\n-- 월 정렬: {len(monthly)}개월 ({monthly.index[0]} ~ {monthly.index[-1]})")

    assert len(monthly) == 12, f"12개월이어야 하는데 {len(monthly)}개월"
    # 첫 달과 마지막 달이 중간 달들에 비해 극단적으로 낮으면 부분 월이다
    middle_mean = monthly.iloc[1:-1].mean()
    assert monthly.iloc[0] > middle_mean * 0.6, "첫 달이 부분 월로 보임"
    assert monthly.iloc[-1] > middle_mean * 0.6, "마지막 달이 부분 월로 보임"
    return monthly


def test_revenue_scale():
    """연매출이 경영안정 바우처 자격(1억 400만원 미만) 안에 드는지.

    그보다 크면 자격 필터에서 보조금이 전부 빠져 조합이 융자뿐이 된다.
    """
    print(f"\n-- 매출 규모")
    for key in SCENARIOS:
        pos = pos_of(key)
        annual = monthly_revenue(pos).sum()
        monthly_avg = annual / 12
        print(f"   {key}: 연 {annual/MAN:>6,.0f}만원 · 월평균 {monthly_avg/MAN:>5,.0f}만원")
        assert annual < 104_000_000, f"{key}: 바우처 자격 초과 ({annual:,})"
        assert 60_000_000 < annual, f"{key}: 너무 작음 ({annual:,})"


def test_trend_shape():
    """추세 곡선 자체가 의도대로인지 — 노이즈·계절성을 뺀 순수 추세.

    실제 매출은 계절성과 노이즈가 섞여 시드에 따라 순서가 뒤집힐 수 있다.
    시나리오 설계가 맞는지는 추세 곡선으로 확인하는 것이 정확하다.
    """
    print(f"\n-- 추세 곡선 (노이즈·계절성 제외, 12개월 누적)")
    cumulative = {}
    for key, sc in SCENARIOS.items():
        curve = trend_curve(sc, 365)
        cumulative[key] = curve[-1] - 1
        first3 = sum(curve[0:91]) / 91
        last3 = sum(curve[274:365]) / 91
        print(f"   {key} {sc.name:<10} 누적 {cumulative[key]:+6.1%} · "
              f"첫3개월 대비 마지막3개월 {last3/first3-1:+6.1%}")

    assert cumulative["C"] < cumulative["B"] < cumulative["A"] < 0, (
        f"하락 폭은 C < B < A 순이어야 함: {cumulative}"
    )
    assert cumulative["C"] < -0.30, f"C는 급격한 하락이어야 함 ({cumulative['C']:.1%})"
    assert cumulative["A"] > -0.20, f"A는 완만해야 함 ({cumulative['A']:.1%})"
    return cumulative


def test_trend_survives_noise():
    """노이즈를 얹어도 추세 방향이 살아남는지.

    시드 하나로 판단하면 우연에 좌우된다. 여러 시드의 평균으로 본다.
    """
    seeds = [7, 42, 101, 2026, 31337]
    print(f"\n-- 실제 매출의 하락폭 (시드 {len(seeds)}개 평균)")
    means = {}
    for key in SCENARIOS:
        drops = []
        for seed in seeds:
            monthly = monthly_revenue(pos_of(key, seed=seed))
            drops.append(monthly.iloc[-3:].mean() / monthly.iloc[:3].mean() - 1)
        means[key] = sum(drops) / len(drops)
        spread = f"{min(drops):+.1%} ~ {max(drops):+.1%}"
        print(f"   {key}: 평균 {means[key]:+6.1%}  (범위 {spread})")

    assert means["C"] < means["A"], (
        f"노이즈를 얹어도 C가 A보다 많이 떨어져야 함: {means}"
    )
    assert all(v < 0 for v in means.values()), "세 시나리오 모두 하락해야 함"
    return means


def test_trend_curve_is_smooth():
    """기울기 이행이 계단이 아니라 부드러운지.

    계단식으로 뚝 떨어지면 변곡점이 눈에 바로 보여, 예측 모델이
    그것을 찾아냈다는 사실에 의미가 없어진다.
    """
    print(f"\n-- 추세 커브의 일별 변화율 (계단이면 튀는 값이 생긴다)")
    for key, sc in SCENARIOS.items():
        curve = trend_curve(sc, 365)
        ratios = [curve[i + 1] / curve[i] for i in range(len(curve) - 1)]
        jumps = [abs(ratios[i + 1] - ratios[i]) for i in range(len(ratios) - 1)]
        print(f"   {key}: 최대 급변 {max(jumps):.5f} (0.001 미만이면 부드러움)")
        assert max(jumps) < 0.001, f"{key}: 기울기가 계단처럼 바뀜"


def test_weekday_effect():
    """요일 효과가 데이터에 남아 있는지 — 예측 모델이 찾아야 할 패턴."""
    pos = pos_of("B")
    daily = daily_revenue(pos)
    frame = pd.DataFrame({"revenue": daily.values}, index=pd.to_datetime(daily.index))
    by_weekday = frame.groupby(frame.index.weekday)["revenue"].mean()
    normalized = by_weekday / by_weekday.mean()

    names = ["월", "화", "수", "목", "금", "토", "일"]
    print(f"\n-- 요일 효과 (실측 vs 설정)")
    for i in range(7):
        setting = WEEKDAY_FACTOR[i] / np.mean(list(WEEKDAY_FACTOR.values()))
        print(f"   {names[i]}  실측 {normalized[i]:.2f}  설정 {setting:.2f}")

    assert normalized[5] > normalized[0], "토요일이 월요일보다 높아야 함"
    assert normalized[4] > normalized[1], "금요일이 화요일보다 높아야 함"
    return normalized


def test_holiday_effect():
    """명절 효과가 있는지 — 전날 대목, 당일 휴무에 가까움."""
    pos = pos_of("B")
    daily = daily_revenue(pos)

    print(f"\n-- 명절 효과")
    checked = 0
    for holidays in LUNAR_HOLIDAYS.values():
        for holiday in holidays:
            if holiday not in daily.index:
                continue
            eve = holiday - timedelta(days=1)
            if eve not in daily.index:
                continue
            baseline = daily.median()
            print(f"   {holiday}  전날 {daily[eve]/MAN:>5,.0f}만 · "
                  f"당일 {daily[holiday]/MAN:>5,.0f}만 · 중앙값 {baseline/MAN:>5,.0f}만")
            assert daily[holiday] < baseline, "명절 당일은 매출이 낮아야 함"
            checked += 1
    assert checked > 0, "생성 구간에 명절이 하나도 없으면 검증이 안 됨"


def test_not_too_clean():
    """변곡점이 눈으로 바로 보일 만큼 매끈하지는 않은지.

    월별 매출이 단조 감소하면 노이즈가 부족한 것이다. 실제 매출은
    하락 추세에서도 오르는 달이 섞인다.
    """
    print(f"\n-- 노이즈 (하락 추세에도 오르는 달이 섞여야 함)")
    for key in SCENARIOS:
        monthly = monthly_revenue(pos_of(key))
        ups = sum(1 for i in range(1, len(monthly)) if monthly.iloc[i] > monthly.iloc[i - 1])
        daily = daily_revenue(pos_of(key))
        cv = daily.std() / daily.mean()
        print(f"   {key}: 전월 대비 상승한 달 {ups}/11개 · 일 매출 변동계수 {cv:.0%}")
        assert ups >= 2, f"{key}: 단조 감소에 가까워 변곡점이 너무 잘 보인다"
        assert cv > 0.20, f"{key}: 일별 변동이 너무 작다 ({cv:.0%})"


def test_reproducible():
    """같은 시드면 같은 데이터 — 데모 재현성."""
    a = generate("B", months=12, seed=42, end_date=date(2026, 7, 31))
    b = generate("B", months=12, seed=42, end_date=date(2026, 7, 31))
    c = generate("B", months=12, seed=7, end_date=date(2026, 7, 31))
    print(f"\n-- 재현성: seed 42 두 번 {len(a):,}행 / {len(b):,}행, seed 7 {len(c):,}행")
    assert len(a) == len(b) and a.equals(b), "같은 시드면 같은 결과여야 함"
    assert len(a) != len(c) or not a.equals(c), "다른 시드면 달라야 함"


def main() -> None:
    print("=" * 66)
    print("합성 POS 검증 - 스키마 · 추세 · 요일 · 명절 · 노이즈")
    print("=" * 66)
    for fn in (
        test_schema,
        test_month_alignment,
        test_revenue_scale,
        test_trend_shape,
        test_trend_survives_noise,
        test_trend_curve_is_smooth,
        test_weekday_effect,
        test_holiday_effect,
        test_not_too_clean,
        test_reproducible,
    ):
        fn()
    print("\n" + "=" * 66)
    print("전부 통과")
    print("=" * 66)


if __name__ == "__main__":
    main()
