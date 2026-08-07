"""
월별 현금흐름 시뮬레이션 엔진.

성격이 다른 지원(융자·보조금·채무조정)을 비교하기 위한 '공통 화폐'.
모든 조합을 같은 기준점(t=0 = 현재)에서 36개월 시뮬레이션하고, 두 축을 산출한다.

    축 1. 생존 개월수      - 잔고가 0 이하가 되는 첫 시점 (클수록 좋음)
    축 2. 30개월 후 순부채 - 그 시점의 총 부채 잔액 (작을수록 좋음)

기본 계산식 (기획서 5.1):

    balance(0) = 보유 현금
    balance(N) = balance(N-1)
                 + [ revenue(N) - fixed_cost - revenue(N) * variable_ratio ]  <- 영업 현금흐름
                 - 기존부채 상환(N)
                 + 정책 유입(N)
                 - 정책 상환(N)

나눗셈("보유현금 / 월 순손실")을 쓰지 않는 이유는 매출 하락 추세에서
월 적자가 매달 커지기 때문이다. 흑자·적자를 같은 식으로 처리한다.

명명 규칙: 코드 식별자는 영어, JSON 키는 한글 그대로 (loader.py 참조).


═══════════════════════════════════════════════════════════
가정 (ASSUMPTIONS) - 임의 상수를 만들지 않기 위해 명시적으로 관리
═══════════════════════════════════════════════════════════

A1. 상환 방식은 '원금균등분할'로 계산한다.
    공고문은 "균등분할상환"이라고만 하고 원리금균등인지 원금균등인지
    명시하지 않는다. 원금균등이 계산이 단순하고 초기 상환 부담을
    보수적으로(크게) 잡으므로 채택했다. -> 문의목록 A-7 확인 대상.

B1. 기존 부채는 사용자가 '총 잔액'과 '월 상환액'만 입력한다.
    금리를 모르면 월 상환액 전액이 원금 상환이라고 가정한다.
    이자를 임의로 추정해 넣으면 근거 없는 상수가 되기 때문이다.
    -> 부채가 실제보다 빨리 줄어 순부채 축이 낙관적으로 나오지만,
       모든 조합에 동일하게 적용되므로 조합 간 상대 비교는 유지된다.
    사용자가 금리를 입력하면(existing_debt_rate) 정확히 계산한다.

C1. 비용감면(바우처)은 총액 1회성이므로 지급 시점 한 달의 고정비를
    그만큼 줄이는 것으로 처리한다. 실제로는 사용기한 내 분산 소진되나,
    금액이 작아 2축에 미치는 영향이 미미하다.

D1. 상환유예는 '대상 부채 잔액 / (잔여개월 + 연장개월)'로 월 상환액을
    다시 계산한다. 잔여개월은 (대상 잔액 / 현재 월 상환액)으로 역산한다.

E1. 변동비율은 매출과 무관하게 고정이다.
    "매출이 절반이 되면 재료비도 절반"을 가정하는 것인데, 실제로는
    최소 발주량·폐기율 때문에 그만큼 줄지 않는다.
    -> 하락 국면에서 영업 현금흐름이 실제보다 낙관적으로 나온다.
       모든 조합에 동일 적용되므로 상대 비교는 유지된다.
    확장 시 매출 구간별 변동비율로 개선 가능(기획서 8장).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .loader import Candidate

SIM_MONTHS = 36  # 그래프·시뮬레이션 기간 (기획서 확정값)
NET_DEBT_MONTH = 30  # 순부채 측정 시점 (기획서 확정값)


# ══════════════════════════════════════════════════════════
# 입력
# ══════════════════════════════════════════════════════════


@dataclass
class UserProfile:
    """시뮬레이션 입력. 정밀/간편 모드 공통."""

    cash_on_hand: int  # 보유 현금
    monthly_revenue: Sequence[int]  # 길이 SIM_MONTHS. 정밀=POS 추세 예측, 간편=현 수준 유지
    fixed_cost: int  # 월 고정비
    variable_ratio: float  # 0~1. 매출 대비 변동비 비율 (입력은 금액, 환산은 UI 계층)

    # 기존 부채 (MVP 필수, 없으면 0)
    existing_debt: int = 0
    existing_debt_payment: int = 0  # 월 상환액
    existing_debt_rate: float | None = None  # 모르면 None -> 가정 B1
    overdue_months: int = 0  # 연체 개월

    # 정책자금 이력 (상환유예 대상 판정용)
    policy_loan_balance: int = 0  # 소진공 직접대출 잔액

    def __post_init__(self) -> None:
        if len(self.monthly_revenue) < SIM_MONTHS:
            raise ValueError(
                f"monthly_revenue는 {SIM_MONTHS}개월치가 필요합니다 "
                f"(현재 {len(self.monthly_revenue)})"
            )
        if not 0 <= self.variable_ratio < 1:
            raise ValueError(f"variable_ratio는 0 이상 1 미만이어야 합니다: {self.variable_ratio}")

    def operating_cashflow(self, month: int) -> float:
        """m월(1-based)의 영업 현금흐름. 흑자면 양수, 적자면 음수."""
        revenue = self.monthly_revenue[month - 1]
        return revenue - self.fixed_cost - revenue * self.variable_ratio


@dataclass
class Applied:
    """조합에 포함된 후보 + 실제 적용 금액."""

    candidate: Candidate
    amount: int | None = None  # 융자 금액. None이면 정책 한도

    @property
    def effective_amount(self) -> int:
        if self.amount is not None:
            return self.amount
        return int(
            self.candidate.get("금액_한도")
            or self.candidate.get("지급액")
            or self.candidate.get("순유입액")
            or 0
        )


# ══════════════════════════════════════════════════════════
# 상환 스케줄
# ══════════════════════════════════════════════════════════


def loan_schedule(
    principal: float,
    annual_rate: float,
    grace_months: int,
    repay_months: int,
    method: str = "균등분할",
    start_month: int = 0,
    months: int = SIM_MONTHS,
) -> tuple[list[float], list[float]]:
    """월별 상환액과 잔액 배열을 만든다.

    Args:
        method: "균등분할"(원금균등, 가정 A1) 또는 "만기일시"
        start_month: 대출이 실행되는 월(0 = t=0에 실행). 미래 후보는 N을 넣는다.

    Returns:
        (payments[0..months-1], balances[0..months])
        payments[i]는 (i+1)월에 나가는 돈, balances[i]는 i월 말 잔액.
    """
    payments = [0.0] * months
    balances = [0.0] * (months + 1)

    monthly_rate = annual_rate / 12
    grace_end = start_month + grace_months  # 이 월까지는 이자만

    for m in range(1, months + 1):
        if m <= start_month:
            balances[m] = 0.0  # 아직 실행 전
            continue
        if m == start_month + 1:
            balances[m - 1] = principal  # 실행 직후 잔액

        prev = balances[m - 1]
        if prev <= 0:
            balances[m] = 0.0
            continue

        interest = prev * monthly_rate

        if method == "만기일시":
            # '2년만기 일시상환' = 24개월차에 원금 전액 (낭떠러지)
            # grace_months를 만기로 해석한다. 그 전까지는 이자만.
            if m < grace_end:
                payments[m - 1] = interest
                balances[m] = prev
            else:
                payments[m - 1] = interest + prev
                balances[m] = 0.0
        elif m <= grace_end:
            # 거치기간: 이자만
            payments[m - 1] = interest
            balances[m] = prev
        else:
            # 원금균등분할 (가정 A1)
            principal_part = principal / repay_months if repay_months else prev
            principal_part = min(principal_part, prev)
            payments[m - 1] = interest + principal_part
            balances[m] = prev - principal_part

    return payments, balances


def existing_debt_schedule(
    p: UserProfile, months: int = SIM_MONTHS
) -> tuple[list[float], list[float]]:
    """사용자의 기존 부채 상환 스케줄 (가정 B1)."""
    payments = [0.0] * months
    balances = [0.0] * (months + 1)
    balances[0] = p.existing_debt

    if p.existing_debt <= 0 or p.existing_debt_payment <= 0:
        return payments, balances

    monthly_rate = (p.existing_debt_rate or 0.0) / 12
    for m in range(1, months + 1):
        prev = balances[m - 1]
        if prev <= 0:
            balances[m] = 0.0
            continue
        interest = prev * monthly_rate
        principal_part = min(p.existing_debt_payment - interest, prev)
        if principal_part <= 0:
            # 월 상환액이 이자에도 못 미치면 원금이 안 줄고 잔액이 는다
            payments[m - 1] = p.existing_debt_payment
            balances[m] = prev + (interest - p.existing_debt_payment)
            continue
        payments[m - 1] = interest + principal_part
        balances[m] = prev - principal_part

    return payments, balances


# ══════════════════════════════════════════════════════════
# 시뮬레이션
# ══════════════════════════════════════════════════════════


@dataclass
class SimResult:
    balance: list[float]  # [0..months] - balance[0] = 보유현금
    payment: list[float]  # [0..months-1] 총 상환액 (기존 + 신규)
    debt: list[float]  # [0..months] 총 부채 잔액
    inflow: list[float]  # [0..months-1] 정책 유입
    survival_months: int  # 잔고가 0 이하가 되는 첫 시점. 끝까지 버티면 months
    net_debt_at_30: float
    survives_full_term: bool
    forgiven: float = 0.0  # 채무조정 감면액
    notes: list[str] = field(default_factory=list)  # 적용 내역 (사람이 읽는 설명)

    @property
    def survival_label(self) -> str:
        return f"{SIM_MONTHS}개월+" if self.survives_full_term else f"{self.survival_months}개월"


def simulate(
    p: UserProfile,
    applied: Sequence[Applied] = (),
    months: int = SIM_MONTHS,
) -> SimResult:
    """조합을 적용한 36개월 현금흐름을 계산한다.

    applied가 비어 있으면 '현재 버퍼'(무지원 베이스라인)가 나온다.
    """
    inflow = [0.0] * months
    new_payment = [0.0] * months
    new_balance = [0.0] * (months + 1)
    cost_relief = [0.0] * months  # 비용감면: 해당 월 고정비 감소분
    forgiven = 0.0
    notes: list[str] = []

    old_payment, old_balance = existing_debt_schedule(p, months)
    old_payment = list(old_payment)
    old_balance = list(old_balance)

    for a in applied:
        c = a.candidate
        effect = c.effect
        amount = a.effective_amount

        # ── 1) 신규유입: 융자 또는 보조금 ──
        if effect == "신규유입":
            at = int(c.get("지급시점_개월", 0) or 0)
            if at < months:
                inflow[at] += amount
            if c.is_loan:
                sch, bal = loan_schedule(
                    principal=amount,
                    annual_rate=float(c.get("금리_연", 0.0)),
                    grace_months=int(c.get("거치_개월", 0)),
                    repay_months=int(c.get("상환_개월", 0)),
                    method=str(c.get("상환방식", "균등분할")),
                    start_month=at,
                    months=months,
                )
                new_payment = [x + y for x, y in zip(new_payment, sch)]
                new_balance = [x + y for x, y in zip(new_balance, bal)]
                notes.append(
                    f"융자 {c} {amount:,}원 @ {float(c.get('금리_연', 0)):.2%}"
                )
            else:
                notes.append(f"보조금 {c} {amount:,}원")

        # ── 2) 대환전환: 기존 고금리 부채를 저금리로 갈아탐 ──
        elif effect == "대환전환":
            converted = min(amount, p.existing_debt)
            if converted <= 0:
                notes.append(f"대환 {c}: 전환할 기존 부채 없음 -> 효과 0")
                continue
            share = converted / p.existing_debt
            # 기존 부채에서 전환분만큼 상환·잔액을 덜어낸다
            old_payment = [x * (1 - share) for x in old_payment]
            old_balance = [x * (1 - share) for x in old_balance]
            # 새 조건으로 스케줄 생성 (현금 유입은 없음 - 부채가 옮겨갈 뿐)
            sch, bal = loan_schedule(
                principal=converted,
                annual_rate=float(c.get("전환후_금리_연", 0.0)),
                grace_months=int(c.get("거치_개월", 0)),
                repay_months=int(c.get("상환_개월", 0)),
                method=str(c.get("상환방식", "균등분할")),
                start_month=0,
                months=months,
            )
            new_payment = [x + y for x, y in zip(new_payment, sch)]
            new_balance = [x + y for x, y in zip(new_balance, bal)]
            notes.append(
                f"대환 {c} {converted:,}원 -> {float(c.get('전환후_금리_연', 0)):.2%}"
            )

        # ── 3) 부채감면: 원금이 깎이고 새 조건으로 재조정 ──
        elif effect == "부채감면":
            rate = float(c.get("원금감면율_추정", 0.0))
            before = p.existing_debt
            if before <= 0:
                notes.append(f"채무조정 {c}: 조정할 부채 없음 -> 효과 0")
                continue
            cut = before * rate
            forgiven += cut
            after = before - cut
            # 기존 부채 스케줄을 통째로 대체
            old_payment = [0.0] * months
            old_balance = [0.0] * (months + 1)
            sch, bal = loan_schedule(
                principal=after,
                annual_rate=0.0,  # 채무조정은 이자 감면이 원칙 -> 0으로 처리
                grace_months=int(c.get("조정후_거치_개월", 0)),
                repay_months=int(c.get("조정후_상환_개월", 0)),
                method="균등분할",
                start_month=0,
                months=months,
            )
            new_payment = [x + y for x, y in zip(new_payment, sch)]
            new_balance = [x + y for x, y in zip(new_balance, bal)]
            notes.append(
                f"채무조정 {c} {before:,.0f} -> {after:,.0f}원 "
                f"(감면 {cut:,.0f}원, 추정치)"
            )

        # ── 4) 상환유예: 원금 감면 없이 기간만 연장 (가정 D1) ──
        elif effect == "상환유예":
            target = p.policy_loan_balance
            if target <= 0 or p.existing_debt_payment <= 0:
                notes.append(f"상환유예 {c}: 대상 정책자금 대출 없음 -> 효과 0")
                continue
            target = min(target, p.existing_debt)
            share = target / p.existing_debt
            target_payment = p.existing_debt_payment * share
            remaining = math.ceil(target / target_payment) if target_payment else 0
            extension = int(c.get("연장_개월_최대", 0))
            new_monthly = target / (remaining + extension) if (remaining + extension) else 0.0

            old_payment = [x * (1 - share) for x in old_payment]
            old_balance = [x * (1 - share) for x in old_balance]

            left = target
            for m in range(1, months + 1):
                if left <= 0:
                    break
                pay = min(new_monthly, left)
                new_payment[m - 1] += pay
                left -= pay
                new_balance[m] += left
            notes.append(
                f"상환유예 {c} 대상 {target:,.0f}원 · "
                f"월 상환 {target_payment:,.0f} -> {new_monthly:,.0f}원 "
                f"({remaining}개월 -> {remaining + extension}개월)"
            )

        # ── 5) 비용감면: 해당 월 고정비 감소 (가정 C1) ──
        elif effect == "비용감면":
            at = int(c.get("지급시점_개월", 0) or 0)
            if at < months:
                cost_relief[at] += amount
            notes.append(f"비용감면 {c} {amount:,}원")

        # ── 6) 없음: 순위 대상이 아니지만 방어적으로 처리 ──
        elif effect == "없음":
            notes.append(f"현금효과 없음: {c}")

        else:
            raise ValueError(f"알 수 없는 현금효과: {effect!r} ({c.candidate_id})")

    # ── 월별 누적 차감 ──
    balance = [float(p.cash_on_hand)] + [0.0] * months
    payment = [0.0] * months
    debt = [float(p.existing_debt)] + [0.0] * months
    survival = months
    survives = True

    for m in range(1, months + 1):
        operating = p.operating_cashflow(m) + cost_relief[m - 1]
        pay = old_payment[m - 1] + new_payment[m - 1]
        payment[m - 1] = pay
        balance[m] = balance[m - 1] + operating - pay + inflow[m - 1]
        debt[m] = old_balance[m] + new_balance[m]
        if survives and balance[m] <= 0:
            survival = m - 1
            survives = False

    return SimResult(
        balance=balance,
        payment=payment,
        debt=debt,
        inflow=inflow,
        survival_months=survival,
        net_debt_at_30=debt[NET_DEBT_MONTH],
        survives_full_term=survives,
        forgiven=forgiven,
        notes=notes,
    )


# ══════════════════════════════════════════════════════════
# 매출 예측선 (헬퍼)
# ══════════════════════════════════════════════════════════


def trend_revenue(current: int, monthly_change: float, months: int = SIM_MONTHS) -> list[int]:
    """단순 등비 추세. 정밀 모드의 POS 예측선을 대신하는 임시 헬퍼.

    monthly_change -0.02 = 매달 2% 감소.
    실제 서비스에서는 POS 12개월 데이터로 추세를 뽑아 대체한다.
    """
    return [int(current * (1 + monthly_change) ** m) for m in range(1, months + 1)]


def flat_revenue(current: int, months: int = SIM_MONTHS) -> list[int]:
    """간편 모드: 추세가 없으므로 현 수준 유지 가정."""
    return [current] * months
