"""
월별 현금흐름 시뮬레이션 엔진.

성격이 다른 지원(융자·보조금·채무조정)을 비교하기 위한 '공통 화폐'.
모든 조합을 같은 기준점(t=0 = 현재)에서 36개월 시뮬레이션하고, 두 축을 산출한다.

    축 1. 생존 개월수      - 잔고가 0 이하가 되는 첫 시점 (클수록 좋음)
    축 2. 30개월 후 순부채 - 그 시점의 총 부채 잔액 (작을수록 좋음)


핵심 계산식 (기획서 5.1)
------------------------
    balance[n] = balance[n-1]
                 + 영업 현금흐름          (매출 - 고정비 - 변동비)
                 - 상환액                 (기존 부채 + 새로 받은 정책자금)
                 + 유입액                 (융자 실행금 · 보조금)

이 한 줄을 36번 반복하는 것이 전부다.
나머지 코드는 전부 "상환액·유입액 배열을 어떻게 채울 것인가"를 다룬다.

나눗셈("보유현금 / 월 순손실")을 쓰지 않는 이유는 매출 하락 추세에서
월 적자가 매달 커지기 때문이다. 흑자·적자를 같은 식으로 처리한다.


인덱스 규칙 (헷갈리기 쉬움)
---------------------------
    payment[i], inflow[i]   길이 = horizon      (i+1)개월차에 나가는/들어오는 돈
    balance[i], debt[i]     길이 = horizon + 1  i개월차 '말'의 잔고/부채

잔고는 시작 시점이 필요해서 한 칸 더 길다. balance[0] = 지금 이 순간의 보유 현금.
따라서 payment[23]은 24개월차, payment[24]는 25개월차다.


명명 규칙
---------
코드 식별자는 영어, JSON 키는 한글 그대로 (loader.py 참조).
주석·docstring은 한글.


═══════════════════════════════════════════════════════════
가정 (ASSUMPTIONS) - 임의 상수를 만들지 않기 위해 명시적으로 관리
═══════════════════════════════════════════════════════════

A1. 상환 방식은 '원금균등분할'로 계산한다.
    공고문은 "균등분할상환"이라고만 하고 원리금균등인지 원금균등인지
    명시하지 않는다. 원금균등이 계산이 단순하고 초기 상환 부담을
    보수적으로(크게) 잡으므로 채택했다. -> 문의목록 A-7 확인 대상.

    ※ 거치기간에도 이자는 납부한다. '거치'는 원금 상환을 미루는 것이지
      돈을 안 내는 것이 아니다. 그래서 거치가 길수록 총 이자가 늘어
      "생존 개월수는 늘지만 순부채도 는다"는 2축 트레이드오프가 생긴다.

B1. 기존 부채는 사용자가 '총 잔액'과 '월 상환액'만 입력한다.
    금리를 모르면 월 상환액 전액이 원금 상환이라고 가정한다.
    이자를 임의로 추정해 넣으면 근거 없는 상수가 되기 때문이다.
    -> 부채가 실제보다 빨리 줄어 순부채 축이 낙관적으로 나오지만,
       모든 조합에 동일하게 적용되므로 조합 간 상대 비교는 유지된다.
    사용자가 금리를 입력하면(existing_debt_rate) 정확히 계산한다.

C1. 비용감면(바우처)은 총액 1회성이므로 지급 시점 한 달의 고정비를
    그만큼 줄이는 것으로 처리한다. 실제로는 사용기한 내 분산 소진되나,
    금액이 작아 2축에 미치는 영향이 미미하다.

D1. 상환유예는 잔여 상환기간을 (대상 잔액 / 현재 월 상환액)으로 역산한 뒤,
    거기에 연장 개월을 더한 기간으로 다시 분할한다.
    금리는 (기존 금리 + 공고상 가산금리)를 적용한다.

E1. 변동비율은 매출과 무관하게 고정이다.
    "매출이 절반이 되면 재료비도 절반"을 가정하는 것인데, 실제로는
    최소 발주량·폐기율 때문에 그만큼 줄지 않는다.
    -> 하락 국면에서 영업 현금흐름이 실제보다 낙관적으로 나온다.
       모든 조합에 동일 적용되므로 상대 비교는 유지된다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .loader import Candidate

SIM_MONTHS = 36  # 시뮬레이션·그래프 기간 (기획서 확정값)
NET_DEBT_MONTH = 30  # 순부채를 측정하는 시점 (기획서 확정값)


# ══════════════════════════════════════════════════════════
# 입력
# ══════════════════════════════════════════════════════════


@dataclass
class UserProfile:
    """시뮬레이션 입력값 묶음. 정밀/간편 모드 공통.

    C의 struct처럼 '필드를 담는 상자'다. @dataclass가 __init__을 자동 생성한다.
    """

    # ── 현재 재무 상태 ──
    cash_on_hand: int
    """보유 현금(원). 지금 당장 쓸 수 있는 돈. balance[0]의 출발점."""

    monthly_revenue: Sequence[int]
    """앞으로 horizon개월간의 **예상** 월 매출(원).

    과거 POS 기록이 아니라 '미래 예측선'이다.
    monthly_revenue[0] = 1개월차(다음 달), [35] = 36개월차.
    정밀 모드는 POS 추세 분석 결과, 간편 모드는 현 수준 유지 가정.
    """

    fixed_cost: int
    """월 고정비(원). 임대료·인건비 등 매출과 무관하게 나가는 비용."""

    variable_ratio: float
    """변동비율(0~1). 매출 대비 재료비 등의 비율.

    사용자에게는 금액으로 묻고("보통 한 달에 얼마"), 비율 환산은 UI 계층이 한다.
    """

    # ── 기존 부채 (MVP 필수, 없으면 0) ──
    existing_debt: int = 0
    """기존 부채 총 잔액(원). 지금 갚아야 할 빚의 총합."""

    existing_debt_payment: int = 0
    """기존 부채의 월 상환액(원)."""

    existing_debt_rate: float | None = None
    """기존 부채의 연 금리(소수). 모르면 None -> 가정 B1 적용."""

    overdue_months: int = 0
    """연체 개월 수. 채무조정 자격 판정에 쓰인다(엔진 계산에는 미사용)."""

    # ── 정책자금 이력 ──
    policy_loan_balance: int = 0
    """소진공 정책자금 '직접대출' 잔액(원). 상환연장제도의 대상 금액."""

    def __post_init__(self) -> None:
        """@dataclass가 __init__을 만든 직후 자동 호출되는 검증 훅."""
        if len(self.monthly_revenue) < SIM_MONTHS:
            raise ValueError(
                f"monthly_revenue는 {SIM_MONTHS}개월치가 필요합니다 "
                f"(현재 {len(self.monthly_revenue)}개)"
            )
        if not 0 <= self.variable_ratio < 1:
            raise ValueError(
                f"variable_ratio는 0 이상 1 미만이어야 합니다: {self.variable_ratio}"
            )

    def operating_cashflow(self, month_index: int) -> float:
        """N개월차의 영업 현금흐름. 흑자면 양수, 적자면 음수.

        Args:
            month_index: 몇 개월차인지 (1-based). **달력 월이 아니다.**
                         t=0(현재)으로부터의 경과 개월이며, 1 = 다음 달.

        흑자·적자에 분기가 없다. 결과가 양수면 잔고가 늘고 음수면 준다.
        """
        revenue = self.monthly_revenue[month_index - 1]  # 배열은 0-based라 -1
        variable_cost = revenue * self.variable_ratio  # 매출에 비례하는 비용
        return revenue - self.fixed_cost - variable_cost


@dataclass
class Applied:
    """'어떤 후보를 얼마에 적용한다'는 한 쌍.

    Candidate에는 정책 '한도'만 있는데, 기획서 5.4에 따라 사용자가
    한도보다 적게 빌릴 수 있다(슬라이더). 같은 후보를 다른 금액으로
    여러 번 시뮬레이션해야 하므로 금액을 분리해 담는다.

        Candidate = 메뉴판의 "아메리카노, 최대 3잔"
        Applied   = 실제 주문 "아메리카노 2잔"
    """

    candidate: Candidate
    """적용할 정책 후보 (읽기 전용 원본)."""

    amount: int | None = None
    """실제 적용 금액(원). None이면 정책이 정한 기본값(한도·지급액)을 쓴다."""

    @property
    def effective_amount(self) -> int:
        """실제로 계산에 쓸 금액.

        기본값을 읽을 필드가 지원방법마다 다르다.
            융자         -> 금액_한도
            보조금       -> 지급액
            사업화자금   -> 순유입액 (자부담을 뺀 순액)
            채무조정     -> 금액 필드 없음. 감면율로 계산하므로 0
        """
        if self.amount is not None:
            return self.amount
        # or 체인을 쓰지 않는다. 0도 유효한 값인데 or는 0을 falsy로 보고 건너뛴다.
        for key in ("금액_한도", "지급액", "순유입액"):
            value = self.candidate.get(key)
            if value is not None:
                return int(value)
        return 0


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
    horizon: int = SIM_MONTHS,
) -> tuple[list[float], list[float]]:
    """대출 하나의 월별 상환액과 잔액을 계산한다.

    Args:
        principal:    대출 원금(원). 실제로 빌리는 금액.
        annual_rate:  연 금리(소수). 0.0445 = 4.45%.
        grace_months: 거치 개월. 원금을 안 갚고 이자만 내는 기간.
                      만기일시 방식에서는 '만기'로 해석한다.
        repay_months: 상환 개월. 거치가 끝난 뒤 원금을 나눠 갚는 기간.
        method:       "균등분할"(원금균등, 가정 A1) 또는 "만기일시".
        start_month:  대출이 실행되는 개월차. 0 = 지금.
                      미래 자격 후보는 N개월 뒤이므로 N을 넣는다.
        horizon:      시뮬레이션 총 개월수.

    Returns:
        (payments, balances)
            payments[i] : (i+1)개월차에 나가는 돈 (이자 + 원금). 길이 horizon
            balances[i] : i개월차 말에 남은 빚.               길이 horizon + 1
    """
    payments = [0.0] * horizon  # 월별 상환액
    balances = [0.0] * (horizon + 1)  # 월말 잔액

    monthly_rate = annual_rate / 12  # 월 금리
    grace_end_month = start_month + grace_months  # 거치가 끝나는 개월차

    for month in range(1, horizon + 1):  # month = 몇 개월차인지 (1-based)
        # 아직 대출을 받기 전이면 잔액 0
        if month <= start_month:
            balances[month] = 0.0
            continue

        # 대출이 실행되는 순간: 직전 달 말 잔액에 원금을 꽂는다.
        # balances[i]는 'i월 말'이므로, start_month 말에 빚이 생긴 것으로 본다.
        if month == start_month + 1:
            balances[month - 1] = principal

        opening_balance = balances[month - 1]  # 이번 달 시작 시점의 빚
        if opening_balance <= 0:  # 이미 다 갚았으면 계산할 것이 없다
            balances[month] = 0.0
            continue

        interest = opening_balance * monthly_rate  # 이번 달 이자 (남은 빚에 붙는다)

        if method == "만기일시":
            # '2년만기 일시상환' = 24개월차에 원금 전액 (거치 절벽이 아니라 낭떠러지)
            # 만기 전까지는 이자만 내고, 만기 달에 원금을 통째로 갚는다.
            if month < grace_end_month:
                payments[month - 1] = interest
                balances[month] = opening_balance
            else:
                payments[month - 1] = interest + opening_balance
                balances[month] = 0.0

        elif month <= grace_end_month:
            # 거치기간: 원금은 그대로 두고 이자만 낸다
            payments[month - 1] = interest
            balances[month] = opening_balance

        else:
            # 상환기간: 원금을 균등하게 나눠 갚는다 (가정 A1)
            # repay_months가 0이면 나눗셈이 터지므로 잔액 전액으로 방어
            principal_part = principal / repay_months if repay_months else opening_balance
            # 마지막 달에 반올림 오차로 남은 빚보다 많이 갚는 것을 방지
            principal_part = min(principal_part, opening_balance)
            payments[month - 1] = interest + principal_part
            balances[month] = opening_balance - principal_part

    return payments, balances


def existing_debt_schedule(
    profile: UserProfile, horizon: int = SIM_MONTHS
) -> tuple[list[float], list[float]]:
    """사용자가 이미 지고 있는 빚의 상환 스케줄 (가정 B1).

    Returns:
        (payments, balances) — loan_schedule과 같은 형식
    """
    payments = [0.0] * horizon
    balances = [0.0] * (horizon + 1)
    balances[0] = profile.existing_debt  # 지금 시점의 빚

    # 빚이 없거나 상환액을 모르면 계산할 것이 없다
    if profile.existing_debt <= 0 or profile.existing_debt_payment <= 0:
        return payments, balances

    # 금리를 모르면 0 -> 월 상환액 전액이 원금 상환이 된다 (가정 B1)
    monthly_rate = (profile.existing_debt_rate or 0.0) / 12

    for month in range(1, horizon + 1):
        opening_balance = balances[month - 1]
        if opening_balance <= 0:
            balances[month] = 0.0
            continue

        interest = opening_balance * monthly_rate
        principal_part = min(profile.existing_debt_payment - interest, opening_balance)

        if principal_part <= 0:
            # 이자가 월 상환액보다 큰 경우: 갚는데도 빚이 늘어난다.
            # 고금리 다중채무 상황에서 실제로 발생한다.
            payments[month - 1] = profile.existing_debt_payment
            balances[month] = opening_balance + (interest - profile.existing_debt_payment)
            continue

        payments[month - 1] = interest + principal_part
        balances[month] = opening_balance - principal_part

    return payments, balances


# ══════════════════════════════════════════════════════════
# 시뮬레이션 결과
# ══════════════════════════════════════════════════════════


@dataclass
class SimResult:
    """36개월 시뮬레이션 결과."""

    balance: list[float]
    """월말 현금 잔고. 길이 horizon+1. balance[0] = 보유 현금."""

    payment: list[float]
    """월별 총 상환액 (기존 부채 + 신규 정책자금). 길이 horizon."""

    debt: list[float]
    """월말 총 부채 잔액. 길이 horizon+1."""

    inflow: list[float]
    """월별 현금 유입 (융자 실행금·보조금). 길이 horizon."""

    survival_months: int
    """축 1 — 잔고가 0 이하가 되는 첫 시점. 끝까지 버티면 horizon."""

    net_debt_at_30: float
    """축 2 — 30개월차 말의 총 부채 잔액."""

    survives_full_term: bool
    """36개월 내내 잔고가 0 아래로 안 내려갔는지."""

    forgiven: float = 0.0
    """채무조정으로 감면된 원금 총액(원). 상환유예는 감면이 없으므로 0."""

    notes: list[str] = field(default_factory=list)
    """적용 내역을 사람이 읽을 수 있게 정리한 문장들."""

    @property
    def survival_label(self) -> str:
        """화면 표기용. 끝까지 버티면 '36개월+'."""
        if self.survives_full_term:
            return f"{SIM_MONTHS}개월+"
        return f"{self.survival_months}개월"


# ══════════════════════════════════════════════════════════
# 시뮬레이션 본체
# ══════════════════════════════════════════════════════════


def simulate(
    profile: UserProfile,
    applied: Sequence[Applied] = (),
    horizon: int = SIM_MONTHS,
) -> SimResult:
    """조합을 적용한 현금흐름을 계산한다.

    Args:
        profile: 사용자 재무 상황
        applied: 적용할 후보들. **비워두면 무지원 베이스라인(= 현재 버퍼)**이 나온다.
        horizon: 시뮬레이션 총 개월수

    2단계로 나뉜다.
        1단계 — 후보를 하나씩 훑으며 유입·상환·고정비 배열을 채운다 (현금효과 6종 분기)
        2단계 — 채워진 배열로 월별 누적 차감을 돌려 잔고를 구한다
    """
    # ── 1단계에서 채울 배열들 ──
    inflow = [0.0] * horizon  # 정책에서 들어오는 현금
    new_payment = [0.0] * horizon  # 새로 생기는 상환액
    new_balance = [0.0] * (horizon + 1)  # 새로 생기는 부채 잔액
    cost_relief = [0.0] * horizon  # 비용감면으로 줄어드는 고정비

    forgiven = 0.0  # 감면된 원금 누적
    notes: list[str] = []  # 사람이 읽는 적용 내역

    # 기존 부채 스케줄. 대환·채무조정·상환유예가 이 배열을 변형한다.
    old_payment, old_balance = existing_debt_schedule(profile, horizon)
    old_payment = list(old_payment)
    old_balance = list(old_balance)

    # 아직 손대지 않은 기존 부채 원금.
    # profile.existing_debt를 매번 읽으면 여러 효과가 겹칠 때 이중 계산된다.
    # (예: 대환이 4,000만 중 3,000만을 가져간 뒤 채무조정이 오면
    #      남은 1,000만에만 적용되어야 한다)
    remaining_existing = float(profile.existing_debt)

    # ══════════ 1단계: 현금효과별로 배열 채우기 ══════════
    for item in applied:
        cand = item.candidate  # 정책 후보
        effect = cand.effect  # 현금효과 6종 중 하나
        amount = item.effective_amount  # 실제 적용 금액

        # ── 1) 신규유입 — 융자 실행금 또는 보조금이 들어온다 ──
        if effect == "신규유입":
            payout_month = int(cand.get("지급시점_개월", 0) or 0)  # 언제 들어오는지
            if payout_month < horizon:
                inflow[payout_month] += amount

            if cand.is_loan:
                # 융자는 갚아야 하므로 상환 스케줄을 만들어 더한다
                sched_payment, sched_balance = loan_schedule(
                    principal=amount,
                    annual_rate=float(cand.get("금리_연", 0.0)),
                    grace_months=int(cand.get("거치_개월", 0)),
                    repay_months=int(cand.get("상환_개월", 0)),
                    method=str(cand.get("상환방식", "균등분할")),
                    start_month=payout_month,
                    horizon=horizon,
                )
                new_payment = [a + b for a, b in zip(new_payment, sched_payment)]
                new_balance = [a + b for a, b in zip(new_balance, sched_balance)]
                notes.append(
                    f"융자 {cand} {amount:,}원 @ {float(cand.get('금리_연', 0)):.2%}"
                )
            else:
                # 보조금은 유출이 없다
                notes.append(f"보조금 {cand} {amount:,}원")

        # ── 2) 대환전환 — 현금 유입 없이 기존 빚을 저금리로 옮긴다 ──
        elif effect == "대환전환":
            converted = min(amount, remaining_existing)  # 실제로 전환되는 금액
            if converted <= 0:
                notes.append(f"대환 {cand}: 전환할 기존 부채 없음 -> 효과 0")
                continue

            share = converted / remaining_existing  # 기존 부채 중 전환 비율
            # 전환된 만큼 기존 스케줄에서 덜어낸다
            old_payment = [x * (1 - share) for x in old_payment]
            old_balance = [x * (1 - share) for x in old_balance]
            remaining_existing -= converted

            # 새 조건으로 스케줄을 만들어 붙인다 (빚이 옮겨간 것이지 돈이 들어온 게 아니다)
            sched_payment, sched_balance = loan_schedule(
                principal=converted,
                annual_rate=float(cand.get("전환후_금리_연", 0.0)),
                grace_months=int(cand.get("거치_개월", 0)),
                repay_months=int(cand.get("상환_개월", 0)),
                method=str(cand.get("상환방식", "균등분할")),
                start_month=0,
                horizon=horizon,
            )
            new_payment = [a + b for a, b in zip(new_payment, sched_payment)]
            new_balance = [a + b for a, b in zip(new_balance, sched_balance)]
            notes.append(
                f"대환 {cand} {converted:,.0f}원 "
                f"-> {float(cand.get('전환후_금리_연', 0)):.2%}"
            )

        # ── 3) 부채감면 — 원금이 깎이고 새 조건으로 재조정된다 ──
        elif effect == "부채감면":
            forgive_rate = float(cand.get("원금감면율_추정", 0.0))
            before = remaining_existing  # 조정 전 잔액
            if before <= 0:
                notes.append(f"채무조정 {cand}: 조정할 부채 없음 -> 효과 0")
                continue

            cut = before * forgive_rate  # 감면액
            after = before - cut  # 조정 후 잔액
            forgiven += cut

            # 남아 있던 기존 부채 스케줄을 통째로 대체한다
            old_payment = [0.0] * horizon
            old_balance = [0.0] * (horizon + 1)
            remaining_existing = 0.0

            sched_payment, sched_balance = loan_schedule(
                principal=after,
                annual_rate=0.0,  # 채무조정은 이자 감면이 원칙
                grace_months=int(cand.get("조정후_거치_개월", 0)),
                repay_months=int(cand.get("조정후_상환_개월", 0)),
                method="균등분할",
                start_month=0,
                horizon=horizon,
            )
            new_payment = [a + b for a, b in zip(new_payment, sched_payment)]
            new_balance = [a + b for a, b in zip(new_balance, sched_balance)]
            notes.append(
                f"채무조정 {cand} {before:,.0f} -> {after:,.0f}원 "
                f"(감면 {cut:,.0f}원, 추정치)"
            )

        # ── 4) 상환유예 — 원금 감면 없이 기간만 늘려 월 상환액을 낮춘다 (가정 D1) ──
        elif effect == "상환유예":
            # 대상은 소진공 정책자금 직접대출 잔액. 기존 부채를 넘을 수 없다.
            target = min(profile.policy_loan_balance, remaining_existing)
            if target <= 0 or profile.existing_debt_payment <= 0:
                notes.append(f"상환유예 {cand}: 대상 정책자금 대출 없음 -> 효과 0")
                continue

            share = target / remaining_existing  # 기존 부채 중 유예 대상 비율
            target_payment = profile.existing_debt_payment * share  # 대상분의 월 상환액
            remaining_term = math.ceil(target / target_payment)  # 남은 상환 개월(역산)
            extension = int(cand.get("연장_개월_최대", 0))  # 연장 개월
            new_term = remaining_term + extension  # 연장 후 총 상환 개월

            # 연장하면 금리가 가산된다 (공고: 약정금리 + 0.2%p, 단기연체 시 +0.4%p)
            surcharge = float(cand.get("금리_가산_연", 0.0))
            new_rate = (profile.existing_debt_rate or 0.0) + surcharge

            # 대상분을 기존 스케줄에서 덜어내고
            old_payment = [x * (1 - share) for x in old_payment]
            old_balance = [x * (1 - share) for x in old_balance]
            remaining_existing -= target

            # 늘어난 기간으로 새 스케줄을 만든다 (거치 없이 바로 분할 상환)
            sched_payment, sched_balance = loan_schedule(
                principal=target,
                annual_rate=new_rate,
                grace_months=0,
                repay_months=new_term,
                method="균등분할",
                start_month=0,
                horizon=horizon,
            )
            new_payment = [a + b for a, b in zip(new_payment, sched_payment)]
            new_balance = [a + b for a, b in zip(new_balance, sched_balance)]
            notes.append(
                f"상환유예 {cand} 대상 {target:,.0f}원 · "
                f"월 상환 {target_payment:,.0f} -> {sched_payment[0]:,.0f}원 "
                f"({remaining_term}개월 -> {new_term}개월, 금리 +{surcharge:.1%})"
            )

        # ── 5) 비용감면 — 유입이 아니라 그 달의 고정비가 줄어든다 (가정 C1) ──
        elif effect == "비용감면":
            payout_month = int(cand.get("지급시점_개월", 0) or 0)
            if payout_month < horizon:
                cost_relief[payout_month] += amount
            notes.append(f"비용감면 {cand} {amount:,}원")

        # ── 6) 없음 — 순위 대상이 아니지만 방어적으로 처리 ──
        elif effect == "없음":
            notes.append(f"현금효과 없음: {cand}")

        else:
            raise ValueError(f"알 수 없는 현금효과: {effect!r} ({cand.candidate_id})")

    # ══════════ 2단계: 월별 누적 차감 ══════════
    balance = [float(profile.cash_on_hand)] + [0.0] * horizon
    payment = [0.0] * horizon
    debt = [float(profile.existing_debt)] + [0.0] * horizon

    survival_months = horizon  # 끝까지 버티면 horizon
    survives = True  # 아직 잔고가 0 아래로 안 갔는지

    for month in range(1, horizon + 1):
        # 영업 현금흐름 + 비용감면 효과
        operating = profile.operating_cashflow(month) + cost_relief[month - 1]
        # 이번 달 총 상환액
        total_payment = old_payment[month - 1] + new_payment[month - 1]

        payment[month - 1] = total_payment
        balance[month] = (
            balance[month - 1] + operating - total_payment + inflow[month - 1]
        )
        debt[month] = old_balance[month] + new_balance[month]

        # 잔고가 0 이하가 되는 '첫' 시점만 기록한다.
        # month월에 0이 됐다면 (month-1)개월까지 버틴 것.
        if survives and balance[month] <= 0:
            survival_months = month - 1
            survives = False

    return SimResult(
        balance=balance,
        payment=payment,
        debt=debt,
        inflow=inflow,
        survival_months=survival_months,
        net_debt_at_30=debt[NET_DEBT_MONTH],
        survives_full_term=survives,
        forgiven=forgiven,
        notes=notes,
    )


# ══════════════════════════════════════════════════════════
# 매출 예측선 (임시 헬퍼)
# ══════════════════════════════════════════════════════════


def trend_revenue(
    current: int, monthly_change: float, horizon: int = SIM_MONTHS
) -> list[int]:
    """단순 등비 추세로 매출 예측선을 만든다.

    Args:
        current:        현재 월 매출(원)
        monthly_change: 월 변화율. -0.02 = 매달 2% 감소

    실제 서비스에서는 core/forecast.py(POS 기반 예측)로 대체된다.
    계절성·요일 효과가 없으므로 데모·테스트용이다.
    """
    return [int(current * (1 + monthly_change) ** m) for m in range(1, horizon + 1)]


def flat_revenue(current: int, horizon: int = SIM_MONTHS) -> list[int]:
    """현 수준 유지 가정. 간편 모드(POS 없음)에서 쓴다."""
    return [current] * horizon
