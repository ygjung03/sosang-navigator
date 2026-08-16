"""국면 판정 검증용 케이스 데이터 — 수익성 계층을 고르게 채운다.

기존 12개는 전부 영업이익이 음수라 ICR·DSCR이 음수로만 나왔다.
비율이 의미를 갖는 구간(흑자)을 채워 4개 계층이 모두 나타나게 한다.

    ① DSCR >= 1          영업이익으로 원리금을 감당한다
    ② ICR >= 1 > DSCR    이자는 내지만 원금 상환이 버겁다
    ③ 0 < 영업이익, ICR < 1  이자도 못 낸다
    ④ 영업이익 <= 0       비율이 성립하지 않는다
"""
from __future__ import annotations

from core.cashflow import SIM_MONTHS, UserProfile, simulate, trend_revenue
from core.combos import UserContext

MAN = 10_000
VARIABLE_RATIO = 0.32


def profile_of(c: dict) -> UserProfile:
    return UserProfile(
        cash_on_hand=c["cash"] * MAN,
        monthly_revenue=trend_revenue(c["rev"] * MAN, c["trend"]),
        fixed_cost=c["fix"] * MAN,
        variable_ratio=VARIABLE_RATIO,
        existing_debt=c["debt"] * MAN,
        existing_debt_payment=c["pay"] * MAN,
        existing_debt_rate=c.get("rate", 0.07) if c["debt"] else None,
        high_rate_debt=c["debt"] * MAN if c.get("rate", 0.07) >= 0.07 else 0,
        overdue_days=c["od"],
        policy_loan_balance=c.get("policy", 0) * MAN,
    )


def context_of(c: dict) -> UserContext:
    return UserContext(
        phase=c["phase"],
        region="서울",
        is_closed=c["closed"],
        store_area_pyeong=c.get("area"),
        business_years=c.get("years"),
        revenue_drop_rate=c.get("drop", 0.15),
        annual_revenue=c["rev"] * MAN * 12,
    )


# ── 케이스 ────────────────────────────────────────────────
# rev/fix/cash/debt/pay 는 만원 단위. trend 는 월 매출 변화율.
CASES: dict[str, dict] = {
    # ══ ① DSCR >= 1 — 원리금을 감당한다 ══
    "A1 흑자·여유": dict(
        rev=1_200, fix=600, cash=2_000, debt=2_000, pay=40, rate=0.05,
        od=0, closed=False, trend=-0.003, drop=0.05, phase="일시적자금난"),
    "A2 흑자·부채많음": dict(
        rev=2_000, fix=1_000, cash=1_500, debt=8_000, pay=150, rate=0.06,
        od=0, closed=False, trend=-0.005, drop=0.08, phase="일시적자금난"),

    # ══ ② ICR >= 1 > DSCR — 이자는 내지만 원금이 버겁다 ══
    #    거치가 끝나 원금 상환이 시작된 전형. "일시적 자금난"의 핵심 구간
    "B1 거치종료·원금부담": dict(
        rev=1_000, fix=550, cash=800, debt=5_000, pay=145, rate=0.045,
        od=0, closed=False, trend=-0.008, drop=0.12, phase="일시적자금난",
        policy=3_000),
    "B2 원금부담·계절하락": dict(
        rev=900, fix=500, cash=600, debt=4_000, pay=128, rate=0.05,
        od=0, closed=False, trend=-0.015, drop=0.18, phase="일시적자금난",
        policy=2_500),
    "B4 원금부담·버퍼여유": dict(
        rev=1_100, fix=640, cash=1_800, debt=6_000, pay=170, rate=0.05,
        od=0, closed=False, trend=-0.006, drop=0.1, phase="일시적자금난",
        policy=4_000),
    "B3 원금부담·연체시작": dict(
        rev=850, fix=480, cash=400, debt=4_500, pay=110, rate=0.06,
        od=12, closed=False, trend=-0.02, drop=0.22, phase="일시적자금난",
        policy=2_000),

    # ══ ③ 흑자지만 ICR < 1 — 이자도 못 낸다 ══
    #    고금리 부채가 많아 이자만으로 영업이익을 넘는다
    "C1 고금리·소액흑자": dict(
        rev=700, fix=450, cash=500, debt=6_000, pay=90, rate=0.14,
        od=0, closed=False, trend=-0.02, drop=0.25, phase="개선여지"),
    "C2 고금리·연체": dict(
        rev=750, fix=490, cash=350, debt=7_000, pay=110, rate=0.15,
        od=35, closed=False, trend=-0.025, drop=0.3, phase="개선여지"),
    "C3 이자부담·경계": dict(
        rev=950, fix=580, cash=700, debt=6_500, pay=130, rate=0.115,
        od=0, closed=False, trend=-0.012, drop=0.15, phase="일시적자금난"),

    # ══ ④ 영업이익 <= 0 — 비율이 성립하지 않는다 ══
    "D1 적자·버퍼있음": dict(
        rev=800, fix=600, cash=1_500, debt=3_000, pay=60, rate=0.07,
        od=0, closed=False, trend=-0.015, drop=0.2, phase="개선여지",
        policy=2_000),
    # 라벨 정정(2026-08-15): 처음 "개선여지"로 붙였으나 기획서 국면 정의
    # ("급격한 지속 하락, 버퍼 소진 임박")과 정확히 일치해 회복불가로 바꿨다.
    # 매출 35% 급락 · 월 174만원 적자 · 버퍼 1개월 · 연체 45일.
    "D2 적자·연체": dict(
        rev=700, fix=650, cash=400, debt=5_000, pay=100, rate=0.09,
        od=45, closed=False, trend=-0.03, drop=0.35, phase="회복불가"),
    "D3 적자·장기연체": dict(
        rev=700, fix=700, cash=200, debt=6_000, pay=110, rate=0.12,
        od=120, closed=False, trend=-0.03, drop=0.45, phase="회복불가"),
    "D4 급락·고부채": dict(
        rev=600, fix=650, cash=300, debt=8_000, pay=150, rate=0.12,
        od=100, closed=False, trend=-0.04, drop=0.5, phase="회복불가"),
    "D5 폐업·단기연체": dict(
        rev=400, fix=350, cash=300, debt=4_000, pay=80, rate=0.12,
        od=5, closed=True, area=20, trend=-0.05, drop=0.55, phase="회복불가"),
    "D6 폐업·장기연체": dict(
        rev=350, fix=300, cash=400, debt=5_500, pay=95, rate=0.12,
        od=150, closed=True, area=35, trend=-0.045, drop=0.6, phase="회복불가"),
    # ══ 경계 검증용 (2026-08-15 추가) ══
    #    각 지표의 경계값 바로 위/아래를 짝으로 만들어 판정이 갈리는지 본다.

    # ICR 1.5 경계
    "E1 ICR 1.52": dict(
        rev=1_000, fix=600, cash=1_200, debt=6_000, pay=110, rate=0.1,
        od=0, closed=False, trend=-0.008, drop=0.1, phase="일시적자금난"),
    "E2 ICR 1.48": dict(
        rev=990, fix=600, cash=1_200, debt=6_000, pay=110, rate=0.1,
        od=0, closed=False, trend=-0.008, drop=0.1, phase="일시적자금난"),
    # ICR 1.0 경계
    "E3 ICR 1.02": dict(
        rev=900, fix=560, cash=900, debt=6_500, pay=120, rate=0.11,
        od=0, closed=False, trend=-0.01, drop=0.12, phase="개선여지"),
    "E4 ICR 0.98": dict(
        rev=890, fix=560, cash=900, debt=6_500, pay=120, rate=0.11,
        od=0, closed=False, trend=-0.01, drop=0.12, phase="개선여지"),

    # 매출 감소 15% 경계
    "F1 감소 14%": dict(
        rev=900, fix=520, cash=800, debt=3_000, pay=60, rate=0.06,
        od=0, closed=False, trend=-0.012, drop=0.14, phase="일시적자금난"),
    "F2 감소 16%": dict(
        rev=900, fix=520, cash=800, debt=3_000, pay=60, rate=0.06,
        od=0, closed=False, trend=-0.012, drop=0.16, phase="일시적자금난"),
    # 매출 감소 40% 경계
    "F3 감소 39%": dict(
        rev=800, fix=500, cash=600, debt=4_000, pay=80, rate=0.08,
        od=0, closed=False, trend=-0.03, drop=0.39, phase="일시적자금난"),
    "F4 감소 41%": dict(
        rev=800, fix=500, cash=600, debt=4_000, pay=80, rate=0.08,
        od=0, closed=False, trend=-0.03, drop=0.41, phase="개선여지"),

    # 연체 10일 경계
    "G1 연체 9일": dict(
        rev=850, fix=520, cash=700, debt=4_500, pay=90, rate=0.09,
        od=9, closed=False, trend=-0.015, drop=0.18, phase="일시적자금난"),
    "G2 연체 11일": dict(
        rev=850, fix=520, cash=700, debt=4_500, pay=90, rate=0.09,
        od=11, closed=False, trend=-0.015, drop=0.18, phase="일시적자금난"),
    # 연체 90일 경계
    "G3 연체 89일": dict(
        rev=750, fix=520, cash=500, debt=5_500, pay=100, rate=0.1,
        od=89, closed=False, trend=-0.025, drop=0.28, phase="회복불가"),
    "G4 연체 91일": dict(
        rev=750, fix=520, cash=500, debt=5_500, pay=100, rate=0.1,
        od=91, closed=False, trend=-0.025, drop=0.28, phase="회복불가"),

    # 36개월 생존 경계 (하드 규칙 ②)
    "H1 0점·36개월 생존": dict(
        rev=1_500, fix=800, cash=3_000, debt=3_000, pay=50, rate=0.04,
        od=0, closed=False, trend=-0.002, drop=0.04, phase="일시적자금난"),
    "H2 0점·35개월 생존": dict(
        rev=1_100, fix=680, cash=900, debt=4_000, pay=95, rate=0.045,
        od=0, closed=False, trend=-0.004, drop=0.06, phase="일시적자금난"),

    # 무부채 계열 — ICR이 정의되지 않는다
    "I1 무부채·흑자": dict(
        rev=900, fix=500, cash=1_000, debt=0, pay=0,
        od=0, closed=False, trend=-0.005, drop=0.08, phase="일시적자금난"),
    "I2 무부채·급락": dict(
        rev=600, fix=520, cash=400, debt=0, pay=0,
        od=0, closed=False, trend=-0.04, drop=0.45, phase="개선여지"),

    # 휴업 (하드 규칙 ①) — 폐업과 같이 다루는지
    "J1 폐업·흑자": dict(
        rev=700, fix=380, cash=1_500, debt=2_000, pay=40, rate=0.05,
        od=0, closed=True, area=15, trend=-0.01, drop=0.1, phase="회복불가"),

    "D7 무부채·적자": dict(
        rev=800, fix=600, cash=500, debt=0, pay=0,
        od=0, closed=False, trend=-0.01, drop=0.12, phase="개선여지"),
}


# ── 지표 계산 ─────────────────────────────────────────────
def operating_profit(c: dict) -> float:
    """월 영업이익 = 매출 − 변동비 − 고정비."""
    revenue = c["rev"] * MAN
    return revenue - revenue * VARIABLE_RATIO - c["fix"] * MAN


def interest_cost(c: dict) -> float:
    """월 이자비용 = 부채 × 연금리 ÷ 12."""
    return c["debt"] * MAN * c.get("rate", 0.07) / 12


def icr(c: dict) -> float | None:
    """이자보상배율 = 영업이익 / 이자비용. 부채가 없으면 정의되지 않는다."""
    cost = interest_cost(c)
    return operating_profit(c) / cost if cost > 0 else None


def dscr(c: dict) -> float | None:
    """부채상환능력비율 = 영업이익 / 월 상환액(원금+이자)."""
    pay = c["pay"] * MAN
    return operating_profit(c) / pay if pay > 0 else None


def tier(c: dict) -> str:
    """수익성 계층."""
    op = operating_profit(c)
    if op <= 0:
        return "④ 영업이익 ≤ 0"
    d, i = dscr(c), icr(c)
    if d is None:
        return "① DSCR≥1 (무부채)"
    if d >= 1:
        return "① DSCR ≥ 1"
    if i is not None and i >= 1:
        return "② ICR≥1 > DSCR"
    return "③ ICR < 1"


if __name__ == "__main__":
    print("=" * 96)
    print(f"  국면 설계용 케이스 {len(CASES)}개 — 수익성 계층 분포")
    print("=" * 96)
    print(f"{'케이스':<18}{'영업이익':>9}{'이자':>7}{'월상환':>7}"
          f"{'ICR':>7}{'DSCR':>7}{'버퍼':>5}  {'계층':<16}국면")
    print("-" * 96)
    for name, c in CASES.items():
        base = simulate(profile_of(c), horizon=SIM_MONTHS)
        f = lambda x: f"{x:.2f}" if x is not None else "   —"
        print(f"{name:<18}{operating_profit(c)/MAN:>8,.0f}만"
              f"{interest_cost(c)/MAN:>6,.0f}만{c['pay']:>6,.0f}만"
              f"{f(icr(c)):>7}{f(dscr(c)):>7}{base.survival_months:>4}개"
              f"  {tier(c):<16}{c['phase']}")

    print()
    import collections
    dist = collections.Counter(tier(c) for c in CASES.values())
    print("계층 분포:", dict(dist))
