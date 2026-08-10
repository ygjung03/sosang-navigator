"""POS 파싱 검증.

`python -m tests.test_pos`로 실행.
"""

from __future__ import annotations

import io
from datetime import date, timedelta

import pandas as pd

from core.pos import (
    ANNUAL_WINDOW_MONTHS,
    PosSummary,
    load,
    map_columns,
    prepare,
    read_file,
    summarize,
    variable_ratio_from_amount,
)
from data.synthetic_pos import SCENARIOS, generate

MAN = 10_000

_cache: dict[tuple, pd.DataFrame] = {}


def pos_of(scenario: str = "B", months: int = 12) -> pd.DataFrame:
    key = (scenario, months)
    if key not in _cache:
        _cache[key] = generate(scenario, months=months)
    return _cache[key]


def test_column_mapping():
    """실제 POS 파일의 한글 컬럼명이 표준명으로 매핑되는지."""
    raw = pd.DataFrame({
        "판매일자": ["2026-01-02 12:30", "2026-01-02 13:00"],
        "전표번호": ["A001", "A002"],
        "판매수량": [2, 1],
        "판매단가": [4500, 12000],
        "할인금액": [0, 2000],
    })
    mapped = map_columns(raw)
    print(f"\n-- 컬럼 매핑: {list(raw.columns)}")
    print(f"           → {list(mapped.columns)}")
    assert "order_datetime" in mapped.columns
    assert "order_id" in mapped.columns
    assert "quantity" in mapped.columns

    frame = prepare(raw)
    assert frame["revenue"].tolist() == [9000, 10000], "수량×단가−할인이 맞아야 함"
    return frame


def test_revenue_column_given():
    """매출액이 이미 있으면 그대로 쓰는지 (수량·단가가 없어도 동작)."""
    raw = pd.DataFrame({
        "날짜": ["2026-01-02", "2026-01-03"],
        "결제금액": [15000, 8000],
    })
    frame = prepare(raw)
    print(f"\n-- 매출액 컬럼 직접 제공: {frame['revenue'].tolist()}")
    assert frame["revenue"].tolist() == [15000, 8000]
    # 주문번호가 없으면 일시로 대체
    assert "order_id" in frame.columns
    return frame


def test_encoding():
    """CP949로 저장된 CSV를 읽는지. 엑셀에서 저장하면 이 인코딩이 된다."""
    csv = "판매일자,판매수량,판매단가\n2026-01-02,2,4500\n2026-01-03,1,12000\n"
    buffer = io.BytesIO(csv.encode("cp949"))
    frame = read_file(buffer)
    print(f"\n-- CP949 CSV 읽기: {len(frame)}행, 컬럼 {list(frame.columns)}")
    assert len(frame) == 2
    return frame


def test_missing_datetime_raises():
    """일시 컬럼이 없으면 알아들을 수 있는 오류를 내는지."""
    raw = pd.DataFrame({"메뉴": ["떡볶이"], "금액": [4500]})
    try:
        prepare(raw)
    except ValueError as error:
        print(f"\n-- 일시 컬럼 없음: {str(error)[:60]}...")
        assert "일시" in str(error)
        return
    raise AssertionError("오류가 나야 함")


def test_daily_aggregation():
    """일별 집계가 맞는지."""
    pos = load(pos_of("B"))
    summary = summarize(pos)

    total_from_daily = summary.daily_revenue.sum()
    total_from_rows = (pos["quantity"] * pos["unit_price"] - pos["discount_amount"]).sum()
    print(f"\n-- 일별 집계: {len(summary.daily_revenue)}일 / "
          f"합계 {total_from_daily/MAN:,.0f}만원")
    assert abs(total_from_daily - total_from_rows) < 1, "집계 합계가 원본과 달라짐"

    # 객단가 = 매출 / 건수
    sample = summary.daily_revenue.index[10]
    expected = summary.daily_revenue[sample] / summary.daily_order_count[sample]
    assert abs(summary.daily_avg_order_value[sample] - expected) < 1
    return summary


def test_month_alignment():
    """부분 월이 추세 계산에서 제외되는지.

    시작·종료가 월 중간이면 그 달의 매출이 실제보다 낮게 잡혀 하락으로 오독된다.
    """
    full = generate("B", months=12, end_date=date(2026, 7, 31))
    summary_full = summarize(load(full))

    # 일부러 월 중간에서 끊는다
    partial = full[full["order_datetime"] < "2026-07-15"]
    summary_partial = summarize(load(partial))

    print(f"\n-- 부분 월 처리")
    print(f"   온전: 마지막 3개월 대비 {summary_full.revenue_change_3m:+.1%}")
    print(f"   7월 중간에서 절단: {summary_partial.revenue_change_3m:+.1%}")
    print(f"   경고 {len(summary_partial.warnings)}건")
    # 부분 월을 그대로 넣으면 -50% 가까이 나온다. 제외했으므로 그 정도는 아니어야 한다
    assert summary_partial.revenue_change_3m > -0.45, (
        "부분 월이 추세에 섞여 하락이 과장됨"
    )
    return summary_partial


def test_rolling_annual():
    """롤링 연매출과 환산 처리."""
    summary_12 = summarize(load(pos_of("B", months=12)))
    print(f"\n-- 롤링 연매출")
    print(f"   12개월: {summary_12.rolling_annual_revenue/MAN:,.0f}만원 "
          f"(환산 {summary_12.is_annualized})")
    assert not summary_12.is_annualized, "12개월이면 환산이 아니어야 함"

    summary_6 = summarize(load(generate("B", months=6)))
    print(f"   6개월 : {summary_6.rolling_annual_revenue/MAN:,.0f}만원 "
          f"(환산 {summary_6.is_annualized})")
    assert summary_6.is_annualized, "6개월이면 환산이어야 함"
    assert summary_6.rolling_annual_revenue > summary_6.monthly_revenue.sum(), (
        "환산값은 실제 합계보다 커야 함"
    )
    assert any("환산" in w for w in summary_6.warnings)

    # 바우처 자격(1억 400만원 미만) 안에 들어야 데모가 성립한다
    for key in SCENARIOS:
        annual = summarize(load(pos_of(key))).rolling_annual_revenue
        assert annual < 104_000_000, f"{key}: 바우처 자격 초과 ({annual:,})"
    return summary_12, summary_6


def test_trend_indicators():
    """추세 지표가 시나리오별로 구분되는지."""
    print(f"\n-- 추세 지표")
    print(f"{'':4} {'매출변화':>8} {'건수':>8} {'객단가':>8} {'연속하락':>8}")
    changes = {}
    for key in SCENARIOS:
        s = summarize(load(pos_of(key)))
        changes[key] = s.revenue_change_3m
        print(f"{key:4} {s.revenue_change_3m:>+8.1%} {s.order_count_change_3m:>+8.1%} "
              f"{s.avg_order_value_change_3m:>+8.1%} {s.declining_months:>7}개월")

    assert changes["C"] < changes["A"], "C가 A보다 많이 떨어져야 함"
    assert all(v < 0 for v in changes.values()), "세 시나리오 모두 하락"
    return changes


def test_no_driver_judgment():
    """건수·객단가는 판정하지 않고 수치만 제공하는지.

    두 값이 비슷하게 움직이는 경우가 많아 판정 기준을 세우면 근거 없는
    경계가 생긴다. 화면에는 숫자를 그대로 보여준다.
    """
    summary = summarize(load(pos_of("B")))
    print(f"\n-- 진단 문구")
    print(f"   {summary.describe()}")
    assert not hasattr(summary, "decline_driver"), "판정 속성이 남아 있으면 안 됨"
    ratio = summary.count_contribution
    print(f"   건수 기여 비중(참고): {ratio:.0%}" if ratio else "")
    assert "손님 수" in summary.describe() and "객단가" in summary.describe()
    return summary


def test_variable_ratio():
    """사용자가 말한 금액을 변동비율로 환산하는지 (기획서 5.1)."""
    summary = summarize(load(pos_of("B")))
    ratio = variable_ratio_from_amount(summary, 200 * MAN)
    print(f"\n-- 변동비 환산: 월 200만원 / 최근 3개월 평균 "
          f"{summary.recent_3m_avg/MAN:,.0f}만원 → {ratio:.1%}")
    assert 0 < ratio < 1
    # 매출보다 큰 금액을 넣어도 0.95를 넘지 않는다 (입력 오류 방어)
    assert variable_ratio_from_amount(summary, 10_000 * MAN) <= 0.95
    return ratio


def main() -> None:
    print("=" * 66)
    print("POS 파싱 검증 - 컬럼 매핑 · 인코딩 · 집계 · 추세 지표")
    print("=" * 66)
    for fn in (
        test_column_mapping,
        test_revenue_column_given,
        test_encoding,
        test_missing_datetime_raises,
        test_daily_aggregation,
        test_month_alignment,
        test_rolling_annual,
        test_trend_indicators,
        test_no_driver_judgment,
        test_variable_ratio,
    ):
        fn()
    print("\n" + "=" * 66)
    print("전부 통과")
    print("=" * 66)


if __name__ == "__main__":
    main()
