"""
POS 파싱 — 주문 원장을 일별·월별 지표로 집계한다.

파이프라인에서 이 모듈의 자리:

    POS 파일 (CSV/Excel) 또는 합성 데이터
        │
        │  load()      컬럼명 매핑 · 인코딩 처리 · 검증
        ▼
    표준 DataFrame
        │
        │  summarize()  일별 집계 · 추세 지표 · 롤링 연매출
        ▼
    PosSummary
        ├─→ forecast.py   일별 매출·건수·객단가 (예측 입력)
        ├─→ phase.py      추세 지표 (국면 판단 입력)
        ├─→ combos.py     롤링 연매출 (자격 판정)
        └─→ app.py        1차 진단 화면


매출을 건수와 객단가로 나누는 이유
----------------------------------
    매출 = 거래 건수 x 객단가

같은 매출 하락이어도 원인이 다르면 상황이 다르다. 손님 자체가 줄었다면
상권·경쟁·입지 문제에 가깝고, 손님은 그대로인데 덜 쓴다면 메뉴 구성이나
가격 문제일 수 있다.

**다만 어느 쪽이 원인인지 판정하지 않는다.** 두 값이 비슷하게 움직이는 경우가
많아 판정 기준을 세우면 근거 없는 경계가 생기고, 잘못된 단정은 오해를 부른다.
화면에는 두 변화율을 숫자 그대로 보여주고 해석은 사용자와 LLM에게 맡긴다.
회색지대에 원본 숫자만 노출하는 방침(기획서 5.3)과 같은 논리다.

예측(forecast.py)에는 일별 매출만 쓴다. 건수와 객단가를 따로 예측하면
모델이 두 배가 되므로 MVP에서는 진단 표시에만 활용한다.


롤링 연매출
-----------
경영안정 바우처처럼 '연매출 O원 미만'을 요구하는 정책의 자격 판정에 쓴다.
달력 연도가 아니라 **오늘로부터 거슬러 12개월**의 합계다.

POS가 12개월에 못 미치면 (보유 개월 합계 / 보유 개월 수) x 12 로 환산하고
is_annualized 를 True로 둔다. 화면에서 "환산값" 라벨을 붙이기 위한 표식이다.
경영안정 바우처 공고문도 '25년 개업자에 대해 같은 방식(월 평균 x 12)을
쓰므로 근거가 있으나, 계절성이 있으면 어긋나므로 표시가 필요하다.


명명 규칙: 코드 식별자는 영어, 도메인 값은 한글. 주석은 한글.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# 표준 컬럼명. 실제 POS 파일은 제조사마다 이름이 달라 매핑이 필요하다.
COL_DATETIME = "order_datetime"
COL_ORDER_ID = "order_id"
COL_QUANTITY = "quantity"
COL_UNIT_PRICE = "unit_price"
COL_DISCOUNT = "discount_amount"
COL_REVENUE = "revenue"  # 계산해서 채운다

# 실제 POS 파일에서 흔히 쓰이는 컬럼명 -> 표준명.
# 소문자로 바꾸고 공백을 지운 뒤 비교한다.
COLUMN_ALIASES: dict[str, str] = {
    # 일시
    "order_datetime": COL_DATETIME,
    "일시": COL_DATETIME,
    "날짜": COL_DATETIME,
    "판매일자": COL_DATETIME,
    "결제일시": COL_DATETIME,
    "주문일시": COL_DATETIME,
    "거래일시": COL_DATETIME,
    "영업일자": COL_DATETIME,
    "datetime": COL_DATETIME,
    "date": COL_DATETIME,
    # 주문 번호
    "order_id": COL_ORDER_ID,
    "주문번호": COL_ORDER_ID,
    "전표번호": COL_ORDER_ID,
    "영수증번호": COL_ORDER_ID,
    "거래번호": COL_ORDER_ID,
    # 수량
    "quantity": COL_QUANTITY,
    "수량": COL_QUANTITY,
    "판매수량": COL_QUANTITY,
    "qty": COL_QUANTITY,
    # 단가
    "unit_price": COL_UNIT_PRICE,
    "단가": COL_UNIT_PRICE,
    "판매단가": COL_UNIT_PRICE,
    "price": COL_UNIT_PRICE,
    # 할인
    "discount_amount": COL_DISCOUNT,
    "할인금액": COL_DISCOUNT,
    "할인": COL_DISCOUNT,
    "discount": COL_DISCOUNT,
    # 금액이 이미 계산돼 있는 경우
    "revenue": COL_REVENUE,
    "매출액": COL_REVENUE,
    "결제금액": COL_REVENUE,
    "판매금액": COL_REVENUE,
    "금액": COL_REVENUE,
    "합계금액": COL_REVENUE,
    "amount": COL_REVENUE,
}

# 한국어 CSV에서 흔한 인코딩. 위에서부터 시도한다.
ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr")

# 추세 지표를 낼 때 쓰는 구간 길이(개월).
# 3개월은 분기와 맞고 계절 변동을 어느 정도 상쇄한다.
TREND_WINDOW_MONTHS = 3

# 롤링 연매출을 계산하는 기간(개월). 정책 자격이 '연매출' 기준이라 12로 고정.
ANNUAL_WINDOW_MONTHS = 12

# 이 개월 수에 못 미치면 추세 판단을 신뢰하기 어렵다는 경고를 붙인다.
MIN_MONTHS_FOR_TREND = 6


# ══════════════════════════════════════════════════════════
# 결과
# ══════════════════════════════════════════════════════════


@dataclass
class PosSummary:
    """POS에서 뽑아낸 지표 묶음."""

    # ── 예측 입력 (forecast.py) ──
    daily_revenue: pd.Series
    """일별 매출(원). 인덱스는 date. 영업하지 않은 날은 빠져 있다."""

    daily_order_count: pd.Series
    """일별 주문 건수. order_id 기준 고유 개수."""

    daily_avg_order_value: pd.Series
    """일별 객단가(원) = 매출 / 건수."""

    # ── 진단 화면 (1차 화면) ──
    monthly_revenue: pd.Series
    """월별 매출(원). 인덱스는 Period('M')."""

    recent_3m_avg: int
    """최근 3개월 평균 월 매출(원). 변동비율 환산의 분모로도 쓴다."""

    revenue_change_3m: float
    """최근 3개월 vs 직전 3개월 매출 증감률. -0.15 = 15% 감소."""

    revenue_change_yoy: float | None
    """전년 동기 대비 증감률. 12개월 미만이면 None."""

    declining_months: int
    """현재 몇 개월째 연속 하락 중인지. 이번 달부터 거슬러 센다."""

    order_count_change_3m: float
    """건수 증감률 (최근 3개월 vs 직전 3개월)."""

    avg_order_value_change_3m: float
    """객단가 증감률 (최근 3개월 vs 직전 3개월)."""

    # ── 자격 판정 (combos.py) ──
    rolling_annual_revenue: int
    """최근 12개월 매출 합계(원). 12개월 미만이면 환산값."""

    is_annualized: bool
    """12개월에 못 미쳐 환산했는지. True면 화면에 '환산값' 라벨을 붙인다."""

    # ── 메타 ──
    months_covered: float
    """POS가 담고 있는 기간(개월). 소수점 포함."""

    first_date: pd.Timestamp
    last_date: pd.Timestamp

    warnings: list[str] = field(default_factory=list)
    """정확도에 영향을 주는 사항. 화면에 그대로 노출한다."""

    # ── 파생 ──
    @property
    def count_contribution(self) -> float | None:
        """매출 변화 중 '건수 변화'가 차지하는 비중(0~1). 판정이 아니라 참고 수치.

            매출 변화 ≈ 건수 변화 + 객단가 변화   (변화율이 작을 때의 근사)

        어느 쪽이 주된 원인인지 **판정하지 않는다.** 임계값을 세우면 근거 없는
        경계가 생기고, 실제로 두 값이 비슷하게 움직이는 경우가 많아 판정이
        오히려 오해를 부른다. 화면에는 건수·객단가 변화율을 **숫자 그대로**
        보여주고 해석은 사용자와 LLM에게 맡긴다.
        (회색지대에 원본 숫자만 노출하는 것과 같은 방침)

        Returns:
            건수 기여 비중. 두 변화가 모두 0이면 None.
        """
        total = abs(self.order_count_change_3m) + abs(self.avg_order_value_change_3m)
        if total == 0:
            return None
        return abs(self.order_count_change_3m) / total

    def describe(self) -> str:
        """1차 진단 화면에 쓸 한 문단. app.py가 그대로 쓸 수 있게."""
        MAN = 10_000
        lines = [
            f"최근 3개월 평균 매출 {self.recent_3m_avg / MAN:,.0f}만원, "
            f"직전 3개월 대비 {self.revenue_change_3m:+.1%}."
        ]
        if self.declining_months >= 2:
            lines.append(f"{self.declining_months}개월째 하락 중입니다.")
        if self.revenue_change_yoy is not None:
            lines.append(f"전년 동기 대비 {self.revenue_change_yoy:+.1%}.")

        # 건수와 객단가는 판정하지 않고 수치를 그대로 보여준다.
        # 매출이 왜 변했는지는 이 두 숫자에 이미 드러나 있다.
        lines.append(
            f"같은 기간 손님 수 {self.order_count_change_3m:+.1%}, "
            f"객단가 {self.avg_order_value_change_3m:+.1%}."
        )
        return " ".join(lines)


# ══════════════════════════════════════════════════════════
# 로딩
# ══════════════════════════════════════════════════════════


def _normalize(name: str) -> str:
    """컬럼명을 비교 가능한 형태로. 공백·언더스코어를 지우고 소문자로."""
    return str(name).strip().lower().replace(" ", "").replace("_", "")


_ALIAS_LOOKUP = {_normalize(k): v for k, v in COLUMN_ALIASES.items()}


def map_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """실제 POS 파일의 컬럼명을 표준명으로 바꾼다.

    제조사마다 이름이 달라(판매일자 / 결제일시 / 영업일자 …) 매핑이 필요하다.
    모르는 컬럼은 그대로 둔다 — 버리면 나중에 쓸 정보를 잃는다.
    """
    renamed = {}
    for column in frame.columns:
        standard = _ALIAS_LOOKUP.get(_normalize(column))
        if standard:
            renamed[column] = standard
    return frame.rename(columns=renamed)


def read_file(path: str | Path | io.BytesIO) -> pd.DataFrame:
    """CSV 또는 Excel을 읽는다.

    한국어 CSV는 인코딩이 제각각이라(엑셀에서 저장하면 CP949) 순서대로 시도한다.
    """
    if isinstance(path, (str, Path)):
        suffix = Path(path).suffix.lower()
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)

    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            if hasattr(path, "seek"):
                path.seek(0)
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
            continue
        except Exception:
            raise
    raise ValueError(
        f"CSV 인코딩을 판별하지 못했습니다. 시도한 인코딩: {ENCODINGS}"
    ) from last_error


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """표준 컬럼으로 정리하고 매출액을 계산한다.

    Returns:
        order_datetime · revenue · order_id 를 갖춘 DataFrame

    Raises:
        ValueError: 일시 컬럼이 없거나, 매출을 계산할 수 없을 때
    """
    frame = map_columns(frame.copy())

    if COL_DATETIME not in frame.columns:
        raise ValueError(
            f"일시 컬럼을 찾지 못했습니다. "
            f"현재 컬럼: {list(frame.columns)} / "
            f"인식 가능한 이름: 날짜, 판매일자, 결제일시, order_datetime 등"
        )
    frame[COL_DATETIME] = pd.to_datetime(frame[COL_DATETIME], errors="coerce")
    dropped = int(frame[COL_DATETIME].isna().sum())
    frame = frame.dropna(subset=[COL_DATETIME])
    if frame.empty:
        raise ValueError("유효한 일시 값이 하나도 없습니다.")

    # 매출액: 이미 있으면 그대로, 없으면 수량 x 단가 - 할인
    if COL_REVENUE not in frame.columns:
        if COL_QUANTITY in frame.columns and COL_UNIT_PRICE in frame.columns:
            discount = frame.get(COL_DISCOUNT, 0)
            frame[COL_REVENUE] = (
                frame[COL_QUANTITY] * frame[COL_UNIT_PRICE] - discount
            )
        else:
            raise ValueError(
                "매출액을 계산할 수 없습니다. "
                "매출액 컬럼이 있거나, 수량과 단가 컬럼이 모두 있어야 합니다."
            )
    frame[COL_REVENUE] = pd.to_numeric(frame[COL_REVENUE], errors="coerce").fillna(0)

    # 주문 번호가 없으면 일시를 대용으로 쓴다. 건수 집계에만 쓰이므로
    # 같은 시각의 주문은 하나로 묶이지만, 없는 것보다는 낫다.
    if COL_ORDER_ID not in frame.columns:
        frame[COL_ORDER_ID] = frame[COL_DATETIME].astype(str)

    if dropped:
        frame.attrs["dropped_rows"] = dropped
    return frame


def load(source) -> pd.DataFrame:
    """파일 경로·업로드 객체·DataFrame 무엇이든 받아 표준 형태로."""
    if isinstance(source, pd.DataFrame):
        return prepare(source)
    return prepare(read_file(source))


# ══════════════════════════════════════════════════════════
# 집계
# ══════════════════════════════════════════════════════════


def _months_between(first: pd.Timestamp, last: pd.Timestamp) -> float:
    """두 날짜 사이의 개월 수.

    (일수 / 30.44)로 계산하면 2025-08-01 ~ 2026-07-31이 11.99개월이 되어
    "12개월 미만" 경고가 잘못 뜬다. 달력 기준으로 세고 부분 월만 소수로 더한다.
    """
    whole = (last.year - first.year) * 12 + (last.month - first.month)
    # 첫 달에서 쓰지 못한 날수만큼 빼고, 마지막 달에서 채운 날수만큼 더한다
    start_fraction = (first.day - 1) / first.days_in_month
    end_fraction = last.day / last.days_in_month
    return whole - start_fraction + end_fraction


def _pct_change(recent: float, before: float) -> float:
    """증감률. 분모가 0이면 0을 돌려준다(판단 불가를 0으로 표현)."""
    if before == 0:
        return 0.0
    return recent / before - 1


def _count_declining_months(monthly: pd.Series) -> int:
    """마지막 달부터 거슬러 몇 개월 연속 하락했는지 센다."""
    count = 0
    for i in range(len(monthly) - 1, 0, -1):
        if monthly.iloc[i] < monthly.iloc[i - 1]:
            count += 1
        else:
            break
    return count


def summarize(pos: pd.DataFrame) -> PosSummary:
    """표준 POS DataFrame에서 진단·예측·자격 판정에 필요한 지표를 뽑는다."""
    frame = pos if COL_REVENUE in pos.columns else prepare(pos)
    warnings: list[str] = []

    dates = frame[COL_DATETIME]
    first_date, last_date = dates.min(), dates.max()
    months_covered = _months_between(first_date, last_date)

    # ── 일별 집계 ──
    day_index = dates.dt.date
    daily_revenue = frame.groupby(day_index)[COL_REVENUE].sum()
    daily_order_count = frame.groupby(day_index)[COL_ORDER_ID].nunique()
    daily_avg_order_value = (daily_revenue / daily_order_count).fillna(0)

    # ── 월별 집계 ──
    month_index = dates.dt.to_period("M")
    monthly_revenue = frame.groupby(month_index)[COL_REVENUE].sum()
    monthly_order_count = frame.groupby(month_index)[COL_ORDER_ID].nunique()
    monthly_aov = (monthly_revenue / monthly_order_count).fillna(0)

    # 첫 달과 마지막 달은 부분 월일 수 있어 추세를 왜곡한다.
    # 온전한 달만 골라 추세를 계산한다.
    full_months = _full_months(monthly_revenue, first_date, last_date)
    if len(full_months) < len(monthly_revenue):
        warnings.append(
            "시작·종료 달이 온전하지 않아 추세 계산에서 제외했습니다."
        )
    trend_base = monthly_revenue.loc[full_months]
    trend_count = monthly_order_count.loc[full_months]
    trend_aov = monthly_aov.loc[full_months]

    # ── 추세 지표 ──
    window = min(TREND_WINDOW_MONTHS, max(1, len(trend_base) // 2))
    if len(trend_base) >= 2:
        recent = trend_base.iloc[-window:]
        before = trend_base.iloc[-2 * window : -window]
        revenue_change_3m = _pct_change(recent.mean(), before.mean())
        order_count_change = _pct_change(
            trend_count.iloc[-window:].mean(),
            trend_count.iloc[-2 * window : -window].mean(),
        )
        aov_change = _pct_change(
            trend_aov.iloc[-window:].mean(),
            trend_aov.iloc[-2 * window : -window].mean(),
        )
        recent_3m_avg = int(recent.mean())
    else:
        revenue_change_3m = order_count_change = aov_change = 0.0
        recent_3m_avg = int(trend_base.mean()) if len(trend_base) else 0
        warnings.append("데이터가 2개월 미만이라 추세를 계산할 수 없습니다.")

    declining_months = _count_declining_months(trend_base)

    # ── 전년 동기 대비 ──
    revenue_change_yoy: float | None = None
    if len(trend_base) >= ANNUAL_WINDOW_MONTHS + TREND_WINDOW_MONTHS:
        recent = trend_base.iloc[-TREND_WINDOW_MONTHS:]
        year_ago = trend_base.iloc[
            -ANNUAL_WINDOW_MONTHS - TREND_WINDOW_MONTHS : -ANNUAL_WINDOW_MONTHS
        ]
        revenue_change_yoy = _pct_change(recent.mean(), year_ago.mean())

    # ── 롤링 연매출 ──
    rolling_annual_revenue, is_annualized = _rolling_annual(
        monthly_revenue, months_covered
    )
    if is_annualized:
        warnings.append(
            f"매출 자료가 {months_covered:.1f}개월치라 12개월로 환산했습니다. "
            "계절 변동이 있으면 실제와 다를 수 있습니다."
        )

    # ── 정확도 경고 ──
    if months_covered < MIN_MONTHS_FOR_TREND:
        warnings.append(
            f"매출 자료가 {months_covered:.1f}개월치입니다. "
            f"추세 판단의 정확도가 낮습니다."
        )
    elif months_covered < ANNUAL_WINDOW_MONTHS:
        warnings.append(
            f"매출 자료가 {months_covered:.1f}개월치라 계절 변동이 반영되지 않습니다."
        )
    if frame.attrs.get("dropped_rows"):
        warnings.append(
            f"일시를 읽을 수 없는 {frame.attrs['dropped_rows']}행을 제외했습니다."
        )

    return PosSummary(
        daily_revenue=daily_revenue,
        daily_order_count=daily_order_count,
        daily_avg_order_value=daily_avg_order_value,
        monthly_revenue=monthly_revenue,
        recent_3m_avg=recent_3m_avg,
        revenue_change_3m=revenue_change_3m,
        revenue_change_yoy=revenue_change_yoy,
        declining_months=declining_months,
        order_count_change_3m=order_count_change,
        avg_order_value_change_3m=aov_change,
        rolling_annual_revenue=rolling_annual_revenue,
        is_annualized=is_annualized,
        months_covered=months_covered,
        first_date=first_date,
        last_date=last_date,
        warnings=warnings,
    )


def _full_months(
    monthly: pd.Series, first_date: pd.Timestamp, last_date: pd.Timestamp
) -> list:
    """온전한 달만 골라낸다.

    시작일이 1일이 아니거나 종료일이 말일이 아니면 그 달은 부분 월이다.
    부분 월을 추세에 넣으면 매출이 실제보다 낮게 잡혀 하락으로 오독된다.
    다만 전부 제외하면 데이터가 없어지므로, 온전한 달이 2개 미만이면
    원본을 그대로 쓴다.
    """
    periods = list(monthly.index)
    if len(periods) <= 2:
        return periods

    keep = list(periods)
    if first_date.day != 1:
        keep = [p for p in keep if p != periods[0]]
    last_day_of_month = last_date.days_in_month
    if last_date.day != last_day_of_month:
        keep = [p for p in keep if p != periods[-1]]

    return keep if len(keep) >= 2 else periods


def _rolling_annual(
    monthly: pd.Series, months_covered: float
) -> tuple[int, bool]:
    """최근 12개월 매출 합계. 12개월에 못 미치면 환산한다.

    Returns:
        (연매출, 환산 여부)
    """
    if len(monthly) >= ANNUAL_WINDOW_MONTHS:
        return int(monthly.iloc[-ANNUAL_WINDOW_MONTHS:].sum()), False

    total = monthly.sum()
    if months_covered <= 0:
        return 0, True
    # 월 평균 x 12. 경영안정 바우처 공고문도 '25년 개업자에 대해 같은 방식을 쓴다.
    return int(total / months_covered * ANNUAL_WINDOW_MONTHS), True


# ══════════════════════════════════════════════════════════
# 시뮬레이터 연결
# ══════════════════════════════════════════════════════════


def variable_ratio_from_amount(summary: PosSummary, monthly_cost: int) -> float:
    """사용자가 금액으로 말한 변동비를 매출 대비 비율로 바꾼다.

    변동비는 **매출에 비례해 늘어나는 비용 전체**다. 한 가지로 좁히면 업종이
    달라졌을 때 과소 집계된다.

        음식점    재료비, 배달 수수료, 포장재
        소매업    매입원가
        서비스업  시술 재료, 시간제 인건비 (변동비 자체가 작을 수 있다)

    기획서 5.1 — 사용자에게 비율을 묻지 않고 금액을 묻는다. 나눗셈은 코드가 한다.
    POS가 있으면 최근 3개월 평균 매출을 분모로 쓸 수 있어 사용자는 매출을
    말할 필요도 없다.

    Args:
        monthly_cost: 한 달 변동비 총액(원). 사용자가 "보통 한 달에 얼마"로 답한 값.
    """
    if summary.recent_3m_avg <= 0:
        return 0.0
    ratio = monthly_cost / summary.recent_3m_avg
    return float(min(max(ratio, 0.0), 0.95))  # 0~95%로 제한 (100% 이상은 입력 오류)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.synthetic_pos import SCENARIOS, generate

    MAN = 10_000
    for key, scenario in SCENARIOS.items():
        raw = generate(key, months=12)
        summary = summarize(load(raw))

        print("=" * 68)
        print(f"시나리오 {key} — {scenario.name}")
        print("=" * 68)
        print(f"  기간      {summary.first_date.date()} ~ {summary.last_date.date()} "
              f"({summary.months_covered:.1f}개월)")
        print(f"  연매출    {summary.rolling_annual_revenue / MAN:,.0f}만원"
              f"{' (환산)' if summary.is_annualized else ''}")
        print(f"  최근 3개월 평균 {summary.recent_3m_avg / MAN:,.0f}만원")
        print()
        print(f"  {summary.describe()}")
        for warning in summary.warnings:
            print(f"  ⚠ {warning}")
        print()
