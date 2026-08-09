"""
합성 POS 생성기 — 데모 시나리오 3종의 매출 데이터를 만든다.

POS 실데이터를 확보하기 전까지 파이프라인을 end-to-end로 돌리고,
매출 예측 모델(core/forecast.py)이 찾아내야 할 패턴을 심어두기 위한 장치다.

    core/pos.py 가 읽어 일별 매출·건수·객단가로 집계하고,
    core/forecast.py 가 추세·변곡점·요일·명절 효과를 추정한다.


데이터에 심는 것
----------------
    추세      시나리오별로 다른 하락 패턴 (아래 SCENARIOS)
    변곡점    매출이 꺾이기 시작하는 시점. 계단이 아니라 몇 달에 걸쳐 기울기가 이행
    요일      주말 수요 증가. 다만 계수를 고정하지 않고 주마다 흔든다
    명절      설·추석 전후. 업종에 따라 대목이거나 비수기인데 여기서는 대목으로
    노이즈    포아송 + 주간 변동 + 이따금 발생하는 이상치(비·행사·휴무)


"찾기 어렵게" 만드는 이유
-------------------------
※ 이 섹션은 파라미터를 조정할 때 계속 참조하므로 ADR로 옮기지 않고 여기 둔다.

변곡점이 눈으로 바로 보이면 예측 모델이 그것을 찾아냈다는 사실에 의미가 없다.
계단식으로 뚝 떨어뜨리지 않고, 노이즈에 반쯤 묻히도록 만든다.

  · 기울기를 2~3개월에 걸쳐 서서히 바꾼다 (실제 상권 악화가 이런 모양이다)
  · 일별 변동계수를 충분히 크게 둔다
  · 요일 계수도 주마다 흔들어 고정 패턴이 아니게 한다
  · 변곡 시점을 라운드 넘버(1월 1일, 정확히 6개월 전)에 두지 않는다

검증: 만든 데이터를 월별로 집계해 눈으로 훑었을 때 변곡점이 바로 짚이면
노이즈가 부족한 것이다.


원본 구조는 다른 프로젝트의 합성 POS 생성기에서 가져왔다. 주문(order_id)
단위로 여러 라인을 묶는 구조, 라인별 쿠폰 배분 로직이 그것이다. 여기에
추세·변곡점·명절·주간 변동을 얹었다.


명명 규칙: 코드 식별자는 영어, 도메인 값은 한글. 주석은 한글.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════
# 메뉴 구성
# ══════════════════════════════════════════════════════════

# (menu_id, 이름, 단가, 일평균 기본 판매량)
# 기본 판매량은 '요일계수 1.0, 추세 100%'일 때의 하루 판매량이다.
# 월 매출 약 800만원(연 9,600만원)에 맞췄다 — 경영안정 바우처 자격이
# 연매출 1억 400만원 미만이라 그보다 크면 보조금이 자격 필터에서 빠진다.
MENUS = [
    ("M01", "떡볶이", 4_000, 17),
    ("M02", "로제떡볶이", 5_500, 8),
    ("M03", "순대", 4_000, 11),
    ("M04", "김밥", 3_500, 13),
    ("M05", "마라탕", 8_000, 5),
]
MENU_BY_ID = {
    menu_id: {"name": name, "price": price, "base_quantity": qty}
    for menu_id, name, price, qty in MENUS
}

# 요일 계수 (월=0 … 일=6). 주말에 수요가 는다.
# 이 값을 그대로 쓰지 않고 주마다 흔든다(WEEKDAY_JITTER) — 고정 패턴이면
# 예측 모델이 너무 쉽게 맞히고, 실제 매출도 그렇게 규칙적이지 않다.
WEEKDAY_FACTOR = {0: 0.90, 1: 0.88, 2: 1.00, 3: 1.02, 4: 1.22, 5: 1.40, 6: 1.28}
WEEKDAY_JITTER = 0.08  # 요일 계수에 곱해지는 주별 흔들림의 표준편차

# 주문 구성
DELIVERY_RATIO = 0.4  # 배달 주문 비중
ITEMS_PER_ORDER_PROB = {1: 0.6, 2: 0.3, 3: 0.1}  # 주문당 서로 다른 메뉴 수
QUANTITY_PROB = {1: 0.8, 2: 0.2}  # 메뉴당 수량

# 정액 쿠폰 — 배달 주문이고 최소주문금액 이상일 때만
COUPON_CHOICES = [0, 2_000, 3_000]
COUPON_PROB = [0.5, 0.25, 0.25]
_COUPON_CDF = ((0, 2_000, 3_000), (0.5, 0.75, 1.0))
COUPON_MIN_ORDER = 10_000

# 일별 변동
DAILY_NOISE_SD = 0.13  # 하루 단위 승수 노이즈(로그정규)의 표준편차
WEEKLY_WAVE_SD = 0.035  # 주 단위로 이어지는 완만한 파동
OUTLIER_PROB = 0.035  # 이상치 발생 확률 (비·행사·단체주문·조기마감)
CLOSED_PROB = 0.010  # 휴무 확률 (매출 0)

# 명절 — 설·추석 전후. 분식집은 대목으로 잡았다.
# (월, 일)은 음력이라 해마다 다르므로 연도별로 명시한다.
LUNAR_HOLIDAYS = {
    2025: [date(2025, 1, 29), date(2025, 10, 6)],  # 설, 추석
    2026: [date(2026, 2, 17), date(2026, 9, 25)],
    2027: [date(2027, 2, 7), date(2027, 9, 15)],
}
# 명절 당일 기준 오프셋별 승수. 전날까지 대목, 당일·다음날은 휴무에 가깝다.
HOLIDAY_EFFECT = {-3: 1.25, -2: 1.40, -1: 1.55, 0: 0.35, 1: 0.45, 2: 0.90}

# 양력 공휴일 — 어린이날·광복절 등. 명절보다 효과가 작다.
SOLAR_HOLIDAYS = [(1, 1), (3, 1), (5, 5), (6, 6), (8, 15), (10, 3), (10, 9), (12, 25)]
SOLAR_HOLIDAY_FACTOR = 1.18


# ══════════════════════════════════════════════════════════
# 시나리오
# ══════════════════════════════════════════════════════════


@dataclass
class Scenario:
    """매출 추세 시나리오 하나.

    추세는 '월 변화율'의 구간 목록으로 표현한다. 구간이 바뀌는 지점이
    변곡점이며, 기울기를 계단처럼 바꾸지 않고 transition_days에 걸쳐
    서서히 이행시킨다.
    """

    key: str
    name: str
    description: str

    segments: list[tuple[int, float]]
    """(시작 개월차, 월 변화율). -0.015 = 매달 1.5% 감소.

    예: [(0, 0.0), (9, -0.02)] 은 9개월차부터 월 2%씩 감소한다는 뜻.
    """

    transition_days: int = 60
    """기울기가 바뀌는 데 걸리는 기간(일). 길수록 변곡점이 흐릿해진다."""

    seasonal_amplitude: float = 0.0
    """연중 계절 변동의 진폭. 0.12면 최고·최저가 평균 대비 ±12%."""

    seasonal_peak_month: int = 8
    """계절 변동이 정점에 이르는 월."""

    extra_noise: float = 0.0
    """이 시나리오에만 추가되는 일별 노이즈. 불안정한 가게를 표현한다."""


SCENARIOS: dict[str, Scenario] = {
    "A": Scenario(
        key="A",
        name="일시적 자금난",
        description=(
            "1년 중 대부분은 평탄했고 최근 3개월만 완만하게 하락한다. "
            "계절 변동 폭이 커서 '하락처럼 보이지만 계절성일 수도 있는' 구간이 섞인다. "
            "예측 모델이 추세와 계절성을 구분해야 답이 나온다."
        ),
        segments=[(0, 0.004), (8, -0.042)],
        transition_days=70,
        seasonal_amplitude=0.11,
        seasonal_peak_month=8,
    ),
    "B": Scenario(
        key="B",
        name="개선 여지",
        description=(
            "6개월차부터 서서히 꺾여 월 1.5%씩 감소한다. "
            "기울기가 두 달에 걸쳐 바뀌므로 변곡점이 한 지점으로 딱 떨어지지 않는다. "
            "거치기간 절벽을 시연할 주력 시나리오."
        ),
        segments=[(0, -0.002), (6, -0.038)],
        transition_days=60,
        seasonal_amplitude=0.07,
        seasonal_peak_month=8,
    ),
    "C": Scenario(
        key="C",
        name="회복 불가",
        description=(
            "4개월차부터 하락이 시작되어 점점 가팔라진다. "
            "구간을 셋으로 나눠 가속을 표현하고, 일별 변동도 크게 두어 "
            "경영이 불안정한 상태를 나타낸다."
        ),
        segments=[(0, -0.006), (4, -0.045), (8, -0.075)],
        transition_days=45,
        seasonal_amplitude=0.05,
        seasonal_peak_month=8,
        extra_noise=0.05,
    ),
}


# ══════════════════════════════════════════════════════════
# 추세 계산
# ══════════════════════════════════════════════════════════


def _smoothstep(x: float) -> float:
    """0→1로 부드럽게 이어지는 함수. 기울기 이행에 쓴다.

    선형으로 이행하면 이행 시작·종료 지점에 각이 생겨 오히려 눈에 띈다.
    양 끝에서 미분이 0이 되는 곡선을 쓰면 이음매가 보이지 않는다.
    """
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _monthly_rate_at(scenario: Scenario, day_index: int) -> float:
    """day_index일차의 '월 변화율'. 구간 사이는 부드럽게 이행한다."""
    month = day_index / 30.44  # 평균 월 길이
    rate = scenario.segments[0][1]

    for start_month, target_rate in scenario.segments[1:]:
        transition_months = scenario.transition_days / 30.44
        progress = _smoothstep((month - start_month) / transition_months + 0.5)
        rate = rate + (target_rate - rate) * progress
    return rate


def trend_curve(scenario: Scenario, days: int) -> list[float]:
    """0일차를 1.0으로 두었을 때 각 날짜의 추세 승수를 한 번에 계산한다.

    월 변화율을 일 단위로 쪼개 누적한다. 변화율 자체가 시간에 따라
    변하므로 단순 등비가 아니라 누적곱이다.

    날짜마다 처음부터 다시 누적하면 O(n^2)이 되어 1년치에 2천만 번 연산이
    필요하다. 한 번 훑으며 누적해 리스트로 돌려준다.
    """
    curve = [1.0]
    multiplier = 1.0
    for day_index in range(days):
        monthly = _monthly_rate_at(scenario, day_index)
        multiplier *= (1.0 + monthly) ** (1 / 30.44)
        curve.append(multiplier)
    return curve


def _seasonal_multiplier(scenario: Scenario, day: date) -> float:
    """연중 계절 변동. 1년 주기 사인파."""
    if scenario.seasonal_amplitude <= 0:
        return 1.0
    day_of_year = day.timetuple().tm_yday
    peak_day = (scenario.seasonal_peak_month - 1) * 30.44 + 15
    phase = 2 * np.pi * (day_of_year - peak_day) / 365.25
    return 1.0 + scenario.seasonal_amplitude * float(np.cos(phase))


def _holiday_multiplier(day: date) -> float:
    """명절·공휴일 효과."""
    for holidays in LUNAR_HOLIDAYS.values():
        for holiday in holidays:
            offset = (day - holiday).days
            if offset in HOLIDAY_EFFECT:
                return HOLIDAY_EFFECT[offset]
    if (day.month, day.day) in SOLAR_HOLIDAYS:
        return SOLAR_HOLIDAY_FACTOR
    return 1.0


# ══════════════════════════════════════════════════════════
# 주문 생성 헬퍼
# ══════════════════════════════════════════════════════════


def _to_cdf(prob: dict[int, float]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """{값: 확률}을 (값 튜플, 누적확률 튜플)로 미리 바꿔둔다."""
    values = tuple(prob.keys())
    cumulative, total = [], 0.0
    for weight in prob.values():
        total += weight
        cumulative.append(total)
    return values, tuple(cumulative)


# 주문마다 호출되므로(1년치에 3만 번) 미리 누적확률로 변환해 둔다.
# rng.choice는 매 호출마다 배열을 만들고 정규화 검사를 해 느리다.
_ITEMS_CDF = _to_cdf(ITEMS_PER_ORDER_PROB)
_QUANTITY_CDF = _to_cdf(QUANTITY_PROB)


def _draw(rng: np.random.Generator, cdf: tuple[tuple[int, ...], tuple[float, ...]]) -> int:
    """미리 만든 누적확률에서 하나를 뽑는다. 난수 하나와 비교 몇 번이면 끝난다."""
    values, cumulative = cdf
    r = rng.random()
    for value, threshold in zip(values, cumulative):
        if r < threshold:
            return value
    return values[-1]


def _allocate(coupon: int, line_revenues: list[int]) -> list[int]:
    """주문 단위 쿠폰을 라인별 매출 비율로 나눈다.

    1원 단위 오차가 생기면 결제액 합계를 맞추기 위해 첫 라인에 몰아준다.
    """
    total = sum(line_revenues)
    if total == 0 or coupon == 0:
        return [0] * len(line_revenues)
    shares = [coupon * revenue // total for revenue in line_revenues]
    shares[0] += coupon - sum(shares)
    return shares


# ══════════════════════════════════════════════════════════
# 본체
# ══════════════════════════════════════════════════════════


def generate(
    scenario: str | Scenario = "B",
    months: int = 12,
    end_date: date | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """합성 POS 주문 데이터를 만든다.

    Args:
        scenario: 시나리오 키("A"/"B"/"C") 또는 Scenario 객체
        months:   생성 기간(개월). 기획서가 정밀 모드의 전제로 삼은 12개월이 기본.
        end_date: 마지막 날짜. 기본값은 오늘.
        seed:     난수 시드. 같은 시드면 같은 데이터가 나온다(데모 재현성).

    Returns:
        컬럼: order_id, order_datetime, menu_id, menu_name,
              quantity, unit_price, channel, discount_amount

        한 주문(order_id)이 여러 행으로 나뉜다. 같은 order_id로 묶으면
        객단가·장바구니 분석이 가능하다.
        매출액은 quantity * unit_price - discount_amount 로 계산한다.
    """
    sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    rng = np.random.default_rng(seed)

    # 월 경계에 맞춘다. 부분 월이 섞이면 월별 집계에서 첫·마지막 달이
    # 실제보다 낮게 나와 추세를 잘못 읽게 된다.
    if end_date is None:
        today = date.today()
        end_date = date(today.year, today.month, 1) - timedelta(days=1)  # 전월 말일
    start_month = end_date.month - months + 1
    start_year = end_date.year
    while start_month <= 0:
        start_month += 12
        start_year -= 1
    start_date = date(start_year, start_month, 1)
    days = (end_date - start_date).days + 1

    # 주 단위로 이어지는 완만한 파동. 매주 새로 뽑지 않고 이전 값에서 조금씩
    # 움직이게 해 '어떤 주는 계속 잘되고 어떤 주는 계속 안 되는' 현실을 만든다.
    weekly_wave = 1.0
    # 요일 계수도 주마다 흔든다. 고정 패턴이면 예측 모델이 너무 쉽게 맞힌다.
    weekday_shift = {d: 1.0 for d in range(7)}

    trend = trend_curve(sc, days)  # 미리 계산 (O(n))

    rows: list[dict] = []
    order_number = 0

    for day_index in range(days):
        day = start_date + timedelta(days=day_index)

        # 주가 바뀌면 파동과 요일 흔들림을 새로 뽑는다
        if day.weekday() == 0:
            weekly_wave *= float(rng.normal(1.0, WEEKLY_WAVE_SD))
            weekly_wave = min(max(weekly_wave, 0.88), 1.12)  # 발산 방지
            weekday_shift = {
                d: float(rng.normal(1.0, WEEKDAY_JITTER)) for d in range(7)
            }

        # 휴무일
        if rng.random() < CLOSED_PROB:
            continue

        # 그날의 수요 승수 = 추세 x 계절 x 요일 x 주간파동 x 명절 x 노이즈
        multiplier = (
            trend[day_index]
            * _seasonal_multiplier(sc, day)
            * WEEKDAY_FACTOR[day.weekday()]
            * weekday_shift[day.weekday()]
            * weekly_wave
            * _holiday_multiplier(day)
        )
        noise_sd = DAILY_NOISE_SD + sc.extra_noise
        multiplier *= float(rng.lognormal(-(noise_sd**2) / 2, noise_sd))

        # 이따금 발생하는 이상치 — 비 오는 날, 단체 주문, 조기 마감
        if rng.random() < OUTLIER_PROB:
            multiplier *= float(rng.choice([0.45, 0.6, 1.7, 2.1]))

        # ① 메뉴별 그날 수요
        demand_pool: dict[str, int] = {}
        for menu_id, info in MENU_BY_ID.items():
            expected = max(info["base_quantity"] * multiplier, 0.05)
            demand_pool[menu_id] = int(rng.poisson(expected))

        # ② 수요를 주문 단위로 묶는다
        while sum(demand_pool.values()) > 0:
            order_number += 1
            order_id = f"O{order_number:06d}"

            minute = int(rng.integers(11 * 60, 21 * 60))  # 11:00~21:00
            order_datetime = datetime(day.year, day.month, day.day) + timedelta(
                minutes=minute
            )
            is_delivery = rng.random() < DELIVERY_RATIO
            channel = "배달" if is_delivery else "홀"

            available = [m for m, q in demand_pool.items() if q > 0]
            menu_count = min(_draw(rng, _ITEMS_CDF), len(available))
            # rng.choice(replace=False)는 작은 리스트에도 배열을 만들어 느리다.
            # 앞에서부터 필요한 개수만 섞는 부분 셔플로 대체한다.
            for i in range(menu_count):
                j = int(rng.integers(i, len(available)))
                available[i], available[j] = available[j], available[i]
            chosen = available[:menu_count]

            order_lines: list[tuple[str, int]] = []
            for menu_id in chosen:
                quantity = min(demand_pool[menu_id], _draw(rng, _QUANTITY_CDF))
                demand_pool[menu_id] -= quantity
                order_lines.append((menu_id, quantity))

            line_revenues = [
                MENU_BY_ID[m]["price"] * q for m, q in order_lines
            ]
            order_revenue = sum(line_revenues)
            coupon = 0
            if is_delivery and order_revenue >= COUPON_MIN_ORDER:
                coupon = min(_draw(rng, _COUPON_CDF), order_revenue)
            discounts = _allocate(coupon, line_revenues)

            for (menu_id, quantity), discount in zip(order_lines, discounts):
                info = MENU_BY_ID[menu_id]
                rows.append(
                    {
                        "order_id": order_id,
                        "order_datetime": order_datetime,
                        "menu_id": menu_id,
                        "menu_name": info["name"],
                        "quantity": quantity,
                        "unit_price": info["price"],
                        "channel": channel,
                        "discount_amount": int(discount),
                    }
                )

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
# 확인용 집계
# ══════════════════════════════════════════════════════════


def monthly_revenue(pos: pd.DataFrame) -> pd.Series:
    """월별 매출. 생성 결과를 눈으로 확인할 때 쓴다."""
    revenue = pos["quantity"] * pos["unit_price"] - pos["discount_amount"]
    return revenue.groupby(pos["order_datetime"].dt.to_period("M")).sum()


def daily_revenue(pos: pd.DataFrame) -> pd.Series:
    """일별 매출. core/pos.py가 할 일의 최소 버전."""
    revenue = pos["quantity"] * pos["unit_price"] - pos["discount_amount"]
    return revenue.groupby(pos["order_datetime"].dt.date).sum()


if __name__ == "__main__":
    MAN = 10_000
    BAR_UNIT = 150_000  # 막대 한 칸이 나타내는 금액(원). 터미널 폭에 맞춘 값

    for key, sc in SCENARIOS.items():
        pos = generate(key, months=12, seed=42)
        monthly = monthly_revenue(pos)
        daily = daily_revenue(pos)

        print("=" * 66)
        print(f"시나리오 {key} — {sc.name}")
        print("=" * 66)
        print(f"  주문 {pos['order_id'].nunique():,}건 / 라인 {len(pos):,}행")
        print(f"  연매출 {monthly.sum() / MAN:,.0f}만원 "
              f"(바우처 기준 10,400만 {'통과' if monthly.sum() < 104_000_000 else '미달'})")
        print(f"  일 매출 변동계수 {daily.std() / daily.mean():.1%}")
        print()
        print("  월별 매출(만원)")
        first = monthly.iloc[0]
        for period, value in monthly.items():
            bar = "█" * int(value / BAR_UNIT)
            print(f"    {period}  {value / MAN:>6,.0f}  "
                  f"({value / first - 1:+6.1%})  {bar}")
        print()
