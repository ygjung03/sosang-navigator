"""
월별 현금흐름 시뮬레이션 엔진.

성격이 다른 지원(융자·보조금·채무조정)을 비교하기 위한 '공통 화폐'.
모든 조합을 같은 기준점(t=0 = 현재)에서 36개월 시뮬레이션하고, 두 축을 산출한다.

    축 1. 생존 개월수      - 잔고가 0 이하가 되는 첫 시점 (클수록 좋음)
    축 2. 30개월 후 순부채 - 그 시점의 총 부채 잔액 (작을수록 좋음)


핵심 계산식 (기획서 5.1)
------------------------
    balance_at[n] = balance_at[n-1]
                    + 영업 현금흐름   (매출 - 고정비 - 변동비)
                    - 상환액          (기존 부채 + 새로 받은 정책자금)
                    + 유입액          (융자 실행금 · 보조금)

이 한 줄을 36번 반복하는 것이 전부다.
나머지 코드는 전부 "상환액·유입액 배열을 어떻게 채울 것인가"를 다룬다.

나눗셈("보유현금 / 월 순손실")을 쓰지 않는 이유는 매출 하락 추세에서
월 적자가 매달 커지기 때문이다. 흑자·적자를 같은 식으로 처리한다.


═══════════════════════════════════════════════════════════
좌표계 — 이 파일에서 가장 헷갈리는 부분
═══════════════════════════════════════════════════════════
※ 이 섹션은 코드를 읽는 데 필요하므로 ADR로 옮기지 않고 여기 둔다.

두 가지 세는 법이 공존한다. 돈이 나가는 것은 '달 동안' 일어나고,
잔액은 '시점'에 찍히기 때문이다.

    시점 좌표 (0-based)   0 ─────── 1 ─────── 2 ─────── 3 ───────
                          지금      1달 뒤    2달 뒤    3달 뒤

    기간 좌표 (1-based)      [1개월차] [2개월차] [3개월차]

이름으로 구분한다.

    ~_at      저량(stock). 시점의 '상태'.   길이 horizon + 1
              debt_at[i]    = i시점에 남아 있는 빚
              balance_at[i] = i시점의 현금 잔고
              balance_at[0] = 지금 이 순간 (그래서 한 칸 더 길다)

    ~_during  유량(flow). 기간에 '일어난 일'. 길이 horizon
              payment_during[i] = (i+1)개월차에 나간 돈
              inflow_during[i]  = (i+1)개월차에 들어온 돈

    시점 스칼라에도 _at 을 붙인다: start_at, grace_end_at, payout_at
    기간 변수는 month 하나뿐이다 (1-based, 몇 개월차인지)

따라서 payment_during[23]은 24개월차, payment_during[24]는 25개월차다.
`~_at[month - 1]`은 "이번 달 시작 시점", `~_at[month]`는 "이번 달 말"이다.


명명 규칙
---------
코드 식별자는 영어, JSON 키는 한글 그대로 (loader.py 참조). 주석은 한글.


═══════════════════════════════════════════════════════════
가정 (ASSUMPTIONS) - 임의 상수를 만들지 않기 위해 명시적으로 관리
═══════════════════════════════════════════════════════════

A1. 상환 방식은 '원금균등분할'로 계산한다.
    공고문은 "균등분할상환"이라고만 하고 원리금균등인지 원금균등인지
    명시하지 않는다. 원금균등은 매달 갚는 '원금'이 고정(원금 / 상환개월)이고
    이자는 남은 빚에 붙어 점점 줄어든다. 초기 상환 부담을 크게(보수적으로)
    잡으므로 채택했다. -> 문의목록 A-7 확인 대상.

    ※ 거치기간에도 이자는 납부한다. '거치'는 원금 상환을 미루는 것이지
      돈을 안 내는 것이 아니다. 그래서 거치가 길수록 총 이자가 늘어
      "생존 개월수는 늘지만 순부채도 는다"는 2축 트레이드오프가 생긴다.

    ※ 월 금리 = 연 금리 / 12. 이자를 매달 갚아 원금에 얹히지 않으므로
      복리가 붙지 않는다. 7,000만원 4.45%면 매달 259,583원, 12개월 모으면
      3,115,000원 = 7,000만 x 4.45%로 정확히 일치한다.

B1. 기존 부채는 사용자가 '총 잔액'과 '월 상환액'만 입력한다.
    금리를 모르면 월 상환액 전액이 원금 상환이라고 가정한다.
    이자를 임의로 추정해 넣으면 근거 없는 상수가 되기 때문이다.
    -> 부채가 실제보다 빨리 줄어 순부채 축이 낙관적으로 나오지만,
       모든 조합에 동일하게 적용되므로 조합 간 상대 비교는 유지된다.
    사용자가 금리를 입력하면(existing_debt_rate) 정확히 계산한다.

C1. 비용감면(바우처)은 총액 1회성이므로 지급 시점 한 달의 고정비를
    그만큼 줄이는 것으로 처리한다. 실제로는 사용기한 내 분산 소진되나,
    금액이 작아 2축에 미치는 영향이 미미하다.

D1. 상환연장은 잔여 상환기간을 (대상 잔액 / 현재 월 상환액)으로 역산한 뒤,
    거기에 연장 개월을 더한 기간으로 다시 분할한다.
    금리는 (기존 금리 + 공고상 가산금리)를 적용한다.

E1. 변동비율은 매출과 무관하게 고정이다.
    "매출이 절반이 되면 재료비도 절반"을 가정하는 것인데, 실제로는
    최소 발주량·폐기율 때문에 그만큼 줄지 않는다.
    -> 하락 국면에서 영업 현금흐름이 실제보다 낙관적으로 나온다.
       모든 조합에 동일 적용되므로 상대 비교는 유지된다.

F1. 생존이 끝나면 그 시점 상태에서 계산을 멈춘다 (terminal state).
    잔고가 0 이하가 되는 첫 달에 생존을 확정하고, 그 이후로는
      balance_at      0으로 고정 (파산 후 현금은 없다)
      debt_at         종료 시점 값 유지 (못 갚았으므로 빚이 그대로 남는다)
      payment_during  0 (안 냈으므로 나간 돈이 0원이다)
    -> 이전에는 죽은 뒤에도 36개월을 계속 계산해, 잔고 3.6억인 달에 5억을
       갚은 것으로 처리하는 일이 있었다. 그 결과 일찍 죽는 조합일수록
       부채가 작게 나와 파레토 앞줄로 밀려 올라가는 편향이 있었다.
    -> **이것은 실제 부채 예측값이 아니다.** 현실에서는 연체이자·기한이익
       상실·지연배상금으로 부채가 증가한다. 그러나 연체이자율과
       지연배상금률에 근거가 없어(소진공 운영규정 제14조도 "따로 정하는
       지연배상금률"이라고만 한다) 반영하지 않는다.
       따라서 조기 파산 조합의 부채는 실제보다 **과소평가**된다.
       파산 조합끼리는 동일하게 과소평가되므로 그들 사이의 상대 비교는 유지된다.
       TODO(inquiry): 연체이자율 근거를 확보하면 재검토할 것.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .loader import Candidate

# 기존 부채 금리를 모를 때 쓰는 값 (가정 B1). 0%로 두면 월 상환액 전액이
# 원금 상환이 되어 부채가 가장 빨리 줄어드는 낙관적 가정이 된다.
UNKNOWN_RATE = 0.0

# 소진공 대리대출의 상환 구조 (FAQ '약정, 실행 및 상환' 65번)
#   "대리대출은 거치기간 종료 후 상환기간 동안 대출금액의 70%는 3개월마다
#    원금균등분할상환하고, 30%는 상환기간 만료 시 일시상환을 원칙으로 합니다."
# 직접대출은 거치 종료 후 매월 원금균등분할이라 기존 '균등분할'을 그대로 쓴다.
AGENCY_INSTALLMENT_RATIO = 0.7  # 분할상환 비중
AGENCY_BALLOON_RATIO = 0.3      # 만기 일시상환 비중
AGENCY_PAYMENT_CYCLE = 3        # 상환 주기(개월)

# 상환연장에서만 쓰는 별도 기준. 0%를 쓰면 가산금리만 남아 연 0.2%가 되므로
# 그대로 쓸 수 없다. 소진공 정책자금 평균 금리대를 잠정 적용한다.
# TODO(inquiry): 확정 근거 없음. 문의목록에 추가 필요.
UNKNOWN_RATE_FOR_EXTENSION = 0.045

SIM_MONTHS = 36  # 시뮬레이션·그래프 기간 (기획서 확정값)
# 부채 축의 측정 시점은 상수가 아니라 **각 조합의 운영 종료 시점**이다.
# 종료 = min(파산 시점, SIM_MONTHS). 아래 동결 로직 덕분에 debt_at[SIM_MONTHS]를
# 그대로 읽으면 자동으로 종료 시점 값이 나온다 (죽은 뒤로는 변하지 않으므로).
#
# TODO(adr): 이전에는 NET_DEBT_AT = 30 이었다. 근거는 "대부분 2년 거치이므로
#   거치 종료 후 6개월을 본다"였는데, 6이 임의값이고 거치 구조가 다른 상품
#   (대환 거치 0, 만기일시 24개월)에는 같은 30개월이 전혀 다른 국면을 보는
#   문제가 있었다. 평균 부채잔액·누적 원리금 상환액·순부채를 모두 실측 비교한
#   끝에 "종료 시점 부채잔액"으로 정했다. 상세는 문제해결 기록 참조.


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
    """보유 현금(원). 지금 당장 쓸 수 있는 돈. balance_at[0]의 출발점."""

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

    high_rate_debt: int | None = None
    """기존 부채 중 **연 7% 이상 고금리** 부분(원). 대환대출의 대상이자 자격 요건.

    **None(모름)과 0(없음)은 다르다.**
      None -> 사용자가 입력하지 않았다. 기존 부채 전액을 대환 대상으로 본다.
      0    -> 고금리 부채가 없다고 확인했다. 대환 자격이 없다.
    `or`로 두 값을 뭉치면 0이 falsy라 "없음"이 "모름"으로 둔갑해,
    6% 부채밖에 없는 사람에게 대환대출이 추천된다.

    기존 부채를 단일 금리로 뭉뚱그리면 "3,000만원 중 2,000만원만 7%"인 상황을
    표현할 수 없다. 대환대출 자격이 '7% 이상 대출 보유'이고 전환 대상도
    그 부분에 한정되므로, 얼마가 고금리인지를 따로 받는다.
    0이면 대환 자격이 없다. existing_debt를 넘을 수 없다.
    """

    overdue_days: int = 0
    """연체 **일수**. 채무조정·상환연장 자격 판정에 쓰인다(엔진 계산에는 미사용).

    **일 단위인 이유**: 제도의 경계값이 전부 일 단위이고, 그중 10일은
    월 단위로 근사할 수 없다.
        90일 이상        부실차주
        10~89일         부실우려차주 (조건 없음)
        10일 미만        부실우려차주 (폐업·6개월 이상 휴업·세금체납 공공기록 등이 있어야)
        30일 전후        부실우려차주 이자조정 금리가 갈림 (9% / 3.9~4.7%)
        30일 이하        상환연장 자격 (초과하면 배제, 가산금리도 0.2 -> 0.4%p)
    연체 5일과 20일은 개월로 보면 둘 다 0개월이지만 부실우려차주 자격이 정반대다.
    """

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
        if self.high_rate_debt is not None and self.high_rate_debt > self.existing_debt:
            raise ValueError(
                f"high_rate_debt({self.high_rate_debt:,})는 "
                f"existing_debt({self.existing_debt:,})를 넘을 수 없습니다"
            )

    def operating_cashflow(self, month: int) -> float:
        """N개월차의 영업 현금흐름. 흑자면 양수, 적자면 음수.

        Args:
            month: 몇 개월차인지 (1-based, 기간 좌표). **달력 월이 아니다.**
                   t=0(현재)으로부터의 경과 개월이며, 1 = 다음 달.

        흑자·적자에 분기가 없다. 결과가 양수면 잔고가 늘고 음수면 준다.
        """
        revenue = self.monthly_revenue[month - 1]  # 배열은 0-based라 -1
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
    start_at: int = 0,
    horizon: int = SIM_MONTHS,
) -> tuple[list[float], list[float]]:
    """대출 **하나**의 월별 상환액과 남은 빚을 계산한다.

    이 함수는 다른 빚을 보지 않는다. 사용자의 기존 부채나 다른 융자는
    호출한 쪽(simulate)에서 각각 계산해 합산한다.

    Args:
        principal:    대출 원금(원). 실제로 빌리는 금액.
        annual_rate:  연 금리(소수). 0.0445 = 4.45%.
        grace_months: 거치 개월. 원금을 안 갚고 이자만 내는 기간.
                      만기일시 방식에서는 '만기'로 해석한다.
        repay_months: 상환 개월. 거치가 끝난 뒤 원금을 나눠 갚는 기간.
        method:       "균등분할"  원금균등, 매월 (가정 A1). 소진공 직접대출·서울시.
                      "만기일시"  거치 종료 시점에 원금 전액.
                      "대리분기"  소진공 대리대출. 원금의 70%를 3개월마다
                                 원금균등분할하고 30%는 상환기간 만료 시 일시상환.
                                 (소진공 FAQ 65번)
        start_at:     대출이 실행되는 **시점**(0-based). 0 = 지금.
                      미래 자격 후보는 N개월 뒤이므로 N을 넣는다.
        horizon:      시뮬레이션 총 개월수.

    Returns:
        (payment_during, debt_at)
            payment_during[i] : (i+1)개월차에 나가는 돈 (이자 + 원금). 길이 horizon
            debt_at[i]        : i시점에 남아 있는 빚.               길이 horizon + 1
    """
    payment_during = [0.0] * horizon  # 유량: 월별 상환액
    debt_at = [0.0] * (horizon + 1)  # 저량: 시점별 남은 빚

    monthly_rate = annual_rate / 12  # 월 금리 (복리 아님 - 가정 A1 참조)
    grace_end_at = start_at + grace_months  # 거치가 끝나는 시점

    for month in range(1, horizon + 1):  # month = 몇 개월차 (1-based, 기간 좌표)
        # ① 아직 대출을 받기 전이면 이 대출의 빚은 0
        if month <= start_at:
            debt_at[month] = 0.0
            continue

        # ② 대출이 실행되는 순간: 그 '시점'에 원금이 꽂힌다.
        #    N개월차의 시작 = (N-1)시점 이므로 month-1 을 쓴다.
        #    start_at=0 -> month=1 -> debt_at[0] = 원금 (지금 실행)
        #    start_at=3 -> month=4 -> debt_at[3] = 원금 (3개월 뒤 실행)
        if month == start_at + 1:
            debt_at[month - 1] = principal

        opening_debt = debt_at[month - 1]  # 이번 달 시작 시점의 빚
        if opening_debt <= 0:  # 이미 다 갚았으면 계산할 것이 없다
            debt_at[month] = 0.0
            continue

        interest = opening_debt * monthly_rate  # 이자는 '남은 빚'에 붙는다

        if method == "만기일시":
            # '2년만기 일시상환' = 24개월차에 원금 전액 (절벽이 아니라 낭떠러지)
            # grace_months를 만기로 해석한다.
            if month < grace_end_at:
                payment_during[month - 1] = interest
                debt_at[month] = opening_debt
            else:
                payment_during[month - 1] = interest + opening_debt
                debt_at[month] = 0.0

        elif month <= grace_end_at:
            # 거치기간: 원금은 그대로 두고 이자만 낸다.
            # ※ 만기일시가 '<' 인데 여기가 '<=' 인 것은 의도된 차이다.
            #    만기일시의 24개월차는 '갚는 달', 균등분할의 24개월차는 '마지막 거치 달'.
            payment_during[month - 1] = interest
            debt_at[month] = opening_debt

        elif method == "대리분기":
            # 소진공 대리대출 (FAQ 65번).
            #   원금의 70%  -> 거치 종료 후 3개월마다 원금균등분할
            #   원금의 30%  -> 상환기간 만료 시 일시상환
            #
            # 분기 상환이므로 상환 달에는 3개월치가 한 번에 나간다.
            # 매월 균등으로 계산하면 그 달의 잔고 하락 폭을 과소평가한다.
            elapsed = month - grace_end_at        # 거치 종료 후 몇 번째 달인가 (1-based)
            repay_end_at = grace_end_at + repay_months

            if month >= repay_end_at:
                # 상환기간 만료: 남은 것을 전부 갚는다.
                # 30% 일시상환분과 분할분의 반올림 잔여가 여기 모인다.
                payment_during[month - 1] = interest + opening_debt
                debt_at[month] = 0.0
            elif elapsed % AGENCY_PAYMENT_CYCLE == 0:
                # 3개월마다 돌아오는 상환 달.
                # 분할 대상은 원금의 70%이고, 상환기간 동안 몇 번 갚는지로 나눈다.
                installments = max(1, repay_months // AGENCY_PAYMENT_CYCLE)
                principal_part = principal * AGENCY_INSTALLMENT_RATIO / installments
                principal_part = min(principal_part, opening_debt)
                payment_during[month - 1] = interest + principal_part
                debt_at[month] = opening_debt - principal_part
            else:
                # 상환 달이 아니면 이자만 낸다.
                payment_during[month - 1] = interest
                debt_at[month] = opening_debt

        else:
            # 상환기간: 원금을 균등하게 나눠 갚는다 (가정 A1).
            # 나누는 기준은 '남은 빚'이 아니라 '처음 빌린 원금'이다.
            # 7,000만 / 36개월 = 매달 194만원 고정.
            principal_part = principal / repay_months if repay_months else opening_debt
            # 마지막 달에 반올림 오차로 남은 빚보다 많이 갚는 것을 방지
            principal_part = min(principal_part, opening_debt)
            payment_during[month - 1] = interest + principal_part
            debt_at[month] = opening_debt - principal_part

    return payment_during, debt_at


def balloon_after(
    principal: float,
    grace_months: int,
    repay_months: int,
    method: str,
    start_at: int = 0,
    horizon: int = SIM_MONTHS,
) -> tuple[float, int]:
    """**시뮬레이션 창 밖**에서 한 번에 갚아야 하는 원금과 그 시점.

    36개월 안에서는 보이지 않지만 사용자가 반드시 알아야 하는 금액이다.
    예를 들어 일반경영안정자금 7,000만원(대리분기)은 60개월차에 원금의 30%인
    2,100만원이 일시상환으로 돌아온다. 36개월 시뮬레이션에는 흔적조차 없다.

    랭킹 축이 아니라 **정보**다. 두 축은 창 안의 현금흐름만 보므로
    창 밖의 목돈을 잡지 못한다. LLM 프롬프트로 넘겨 경고에 쓴다.

    Returns:
        (금액, 개월차). 창 안에서 다 갚으면 (0.0, 0).
    """
    if method == "만기일시":
        due_at = start_at + grace_months
        amount = principal
    elif method == "대리분기":
        due_at = start_at + grace_months + repay_months
        amount = principal * AGENCY_BALLOON_RATIO
    else:
        return 0.0, 0  # 균등분할은 만기에 몰리는 금액이 없다

    if due_at <= horizon:
        return 0.0, 0  # 창 안에서 이미 반영됐다
    return amount, due_at


def existing_debt_schedule(
    profile: UserProfile, horizon: int = SIM_MONTHS
) -> tuple[list[float], list[float]]:
    """사용자가 이미 지고 있는 빚의 상환 스케줄 (가정 B1).

    Returns:
        (payment_during, debt_at) — loan_schedule과 같은 형식
    """
    payment_during = [0.0] * horizon
    debt_at = [0.0] * (horizon + 1)
    debt_at[0] = profile.existing_debt  # 지금 시점의 빚

    # 빚이 없거나 상환액을 모르면 계산할 것이 없다
    if profile.existing_debt <= 0 or profile.existing_debt_payment <= 0:
        return payment_during, debt_at

    # 금리를 모르면 0으로 가정한다 -> 월 상환액 전액이 원금 상환이 된다 (가정 B1).
    # `or`를 쓰지 않는 이유: 0.0(무이자라고 확인한 경우)도 falsy라 "모름"과
    # 구분되지 않는다. 여기서는 두 경우의 결과가 같지만, 의도를 코드에 남긴다.
    rate = profile.existing_debt_rate
    monthly_rate = (rate if rate is not None else UNKNOWN_RATE) / 12

    for month in range(1, horizon + 1):
        opening_debt = debt_at[month - 1]
        if opening_debt <= 0:
            debt_at[month] = 0.0
            continue

        interest = opening_debt * monthly_rate
        principal_part = min(profile.existing_debt_payment - interest, opening_debt)

        if principal_part <= 0:
            # 이자가 월 상환액보다 큰 경우: 갚는데도 빚이 늘어난다.
            # 고금리 다중채무 상황에서 실제로 발생한다.
            payment_during[month - 1] = profile.existing_debt_payment
            debt_at[month] = opening_debt + (interest - profile.existing_debt_payment)
            continue

        payment_during[month - 1] = interest + principal_part
        debt_at[month] = opening_debt - principal_part

    return payment_during, debt_at


# ══════════════════════════════════════════════════════════
# 시뮬레이션 결과
# ══════════════════════════════════════════════════════════


@dataclass
class SimResult:
    """36개월 시뮬레이션 결과."""

    balance_at: list[float]
    """저량 — 시점별 **현금 잔고**. 길이 horizon+1. balance_at[0] = 보유 현금.

    **0 밑으로 내려가지 않는다.** 생존이 끝난 뒤로는 0으로 고정된다.
    "얼마나 모자랐는가"는 `shortfall_at`을 보라.
    """

    debt_at: list[float]
    """저량 — 시점별 **총 부채 잔액**(기존 + 신규). 길이 horizon+1.

    생존이 끝난 뒤로는 종료 시점 값으로 고정된다 (가정 F1).
    """

    shortfall_at: list[float]
    """저량 — 시점별 **부족분**. 잔고가 음수가 될 값을 그대로 담는다. 길이 horizon+1.

    `balance_at`은 0에서 잘리므로 "얼마가 모자랐는가"를 알 수 없다.
    필요액 역산(`combos.required_amount`)이 이 배열의 최저점을 쓴다.
    양수 구간에서는 `balance_at`과 같다.

    **파산 이후에도 얼리지 않는다.** 다른 배열과 다른 점이다.
    이 배열의 목적이 "목표를 채우려면 얼마가 모자라는가"이므로, 죽은 뒤로
    고정하면 목표를 늘려도 부족분이 그대로여서 역산이 무너진다.
    (실측: 동결했을 때 6개월 목표와 12개월 목표의 필요액이 43,853원으로 동일)
    실제로 일어난 일이 아니라 **역산용 가상 궤적**이다.
    """

    payment_during: list[float]
    """유량 — 월별 **실제로 나간** 총 상환액. 길이 horizon. [23]이 24개월차.

    생존이 끝난 뒤로는 0이다. 못 갚았으므로 나간 돈이 0원이라는 뜻이다.
    (상환 '스케줄'이 아니라 실제 현금흐름이다)
    """

    inflow_during: list[float]
    """유량 — 월별 현금 유입(융자 실행금·보조금). 길이 horizon."""

    survival_months: int
    """축 1 — 잔고가 0 이하가 되는 첫 시점. 끝까지 버티면 horizon."""

    debt_at_exit: float
    """축 2 — **운영 종료 시점의 부채 잔액**.

    종료 = min(파산 시점, horizon).
    36개월까지 살아남은 조합은 36개월 시점 부채,
    그 전에 파산한 조합은 파산 시점 부채를 뜻한다.

    ※ "36개월 후 부채"가 아니다. 파산한 사업의 36개월 후를 예측한 값이 아니라,
      각 조합의 시뮬레이션이 끝나는 시점에 남아 있던 부채다 (terminal state).
    """

    balloon_after_horizon: float = 0.0
    """유량 — **시뮬레이션 창(36개월) 이후** 한 번에 갚아야 하는 원금 합계.

    축이 아니라 정보다. 두 축은 창 안의 현금흐름만 보므로 창 밖의 목돈을
    잡지 못한다. 조합에 든 융자들의 만기 일시상환분을 합산한다.

    - 대리분기: 원금의 30%가 상환기간 만료 시
      (일반경영안정 7,000만 -> 60개월차에 2,100만원)
    - 만기일시: 원금 전액이 만기에. 만기가 36개월 안이면 이미 반영되므로 0.
    - 균등분할: 만기에 몰리는 금액이 없으므로 0.
    """

    balloon_due_at: int = 0
    """balloon_after_horizon 이 발생하는 개월차. 없으면 0."""

    total_payment: float = 0.0
    """유량 합계 — 생존 기간 동안 **실제로 나간** 원리금 총액(기존 부채 + 신규 융자).

    **축이 아니라 정보다.** 최소화 목표로 쓸 수 없다 — 일찍 죽을수록 낼 것이
    없어 작아지므로, 무지원이나 조기 파산 조합이 최소값을 차지한다.
    (실측: 대환대출 5개월 생존 시 56만원, 무지원 3개월 생존 시 150만원)

    두 축이 못 잡는 것을 잡는다. 금리가 다른 두 융자라도 **거치 기간에 죽으면
    신규 융자 원금이 한 푼도 줄지 않아 종료 시점 부채가 동일**하게 나온다.
    이자 부담 차이는 현금으로 빠져나가는데, 종료 시점 잔고는 어차피 0이라
    소멸한다. 그 차이가 남는 곳이 여기다.
    (실측: 일반경영안정 4.45% -> 1,100만 / 신용취약 5.45% -> 1,145만.
     두 축은 생존 18개월 · 종료부채 5,385만으로 완전히 동점)

    LLM 프롬프트에 생존 개월수와 **함께** 넘겨야 한다. 단독으로 보면
    "적게 냈다"가 좋아 보이지만, 일찍 죽어서일 수도 있고 만기일시라
    원금을 미뤄둔 것일 수도 있다.
    """

    survives_full_term: bool = False
    """36개월 내내 잔고가 0 아래로 안 내려갔는지."""

    forgiven: float = 0.0
    """채무조정으로 감면된 원금 총액(원). 상환연장은 감면이 없으므로 0."""

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
    inflow_during = [0.0] * horizon  # 정책에서 들어오는 현금
    new_payment_during = [0.0] * horizon  # 새로 생기는 상환액
    new_debt_at = [0.0] * (horizon + 1)  # 새로 생기는 부채 잔액
    cost_saving_during = [0.0] * horizon  # 비용감면으로 '절약되는' 금액

    forgiven = 0.0
    balloon_total = 0.0   # 창 밖 만기 일시상환 합계
    balloon_due_at = 0  # 감면된 원금 누적
    notes: list[str] = []  # 사람이 읽는 적용 내역

    # 기존 부채 스케줄. 대환·채무조정·상환연장이 이 배열을 변형한다.
    old_payment_during, old_debt_at = existing_debt_schedule(profile, horizon)
    old_payment_during = list(old_payment_during)
    old_debt_at = list(old_debt_at)

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
            payout_at = int(cand.get("지급시점_개월", 0) or 0)  # 언제 들어오는지
            if payout_at < horizon:
                inflow_during[payout_at] += amount

            if cand.is_loan:
                # 융자는 갚아야 하므로 상환 스케줄을 만들어 더한다
                sched_payment, sched_debt = loan_schedule(
                    principal=amount,
                    annual_rate=float(cand.get("금리_연", 0.0)),
                    grace_months=int(cand.get("거치_개월", 0)),
                    repay_months=int(cand.get("상환_개월", 0)),
                    method=str(cand.get("상환방식", "균등분할")),
                    start_at=payout_at,
                    horizon=horizon,
                )
                new_payment_during = [
                    a + b for a, b in zip(new_payment_during, sched_payment)
                ]
                new_debt_at = [a + b for a, b in zip(new_debt_at, sched_debt)]

                # 창 밖에서 한 번에 갚아야 하는 원금 (36개월 안에는 안 보인다)
                bal_amount, bal_at = balloon_after(
                    principal=amount,
                    grace_months=int(cand.get("거치_개월", 0)),
                    repay_months=int(cand.get("상환_개월", 0)),
                    method=str(cand.get("상환방식", "균등분할")),
                    start_at=payout_at,
                    horizon=horizon,
                )
                if bal_amount > 0:
                    balloon_total += bal_amount
                    balloon_due_at = max(balloon_due_at, bal_at)
                    notes.append(
                        f"⚠ {cand}: {bal_at}개월차에 원금 {bal_amount:,.0f}원이 "
                        f"일시상환으로 남는다 (시뮬레이션 {horizon}개월 밖)"
                    )
                notes.append(
                    f"융자 {cand} {amount:,}원 @ {float(cand.get('금리_연', 0)):.2%}"
                )
            else:
                # 보조금은 유출이 없다
                notes.append(f"보조금 {cand} {amount:,}원")

        # ── 2) 대환전환 — 현금 유입 없이 기존 빚을 저금리로 옮긴다 ──
        elif effect == "대환전환":
            # 전환 대상은 '고금리 부채'로 한정된다. 저금리 대출은 대환 대상이 아니다.
            # high_rate_debt를 입력하지 않았으면(0) 기존 부채 전체를 대상으로 본다.
            convertible = profile.high_rate_debt or remaining_existing
            converted = min(amount, convertible, remaining_existing)
            if converted <= 0:
                notes.append(f"대환 {cand}: 전환할 고금리 부채 없음 -> 효과 0")
                continue

            share = converted / remaining_existing  # 기존 부채 중 전환 비율
            # 전환된 만큼 기존 스케줄에서 덜어낸다
            old_payment_during = [x * (1 - share) for x in old_payment_during]
            old_debt_at = [x * (1 - share) for x in old_debt_at]
            remaining_existing -= converted

            # 새 조건으로 스케줄을 만들어 붙인다
            # (빚이 옮겨간 것이지 돈이 들어온 게 아니므로 inflow는 건드리지 않는다)
            sched_payment, sched_debt = loan_schedule(
                principal=converted,
                annual_rate=float(cand.get("전환후_금리_연", 0.0)),
                grace_months=int(cand.get("거치_개월", 0)),
                repay_months=int(cand.get("상환_개월", 0)),
                method=str(cand.get("상환방식", "균등분할")),
                start_at=0,
                horizon=horizon,
            )
            new_payment_during = [
                a + b for a, b in zip(new_payment_during, sched_payment)
            ]
            new_debt_at = [a + b for a, b in zip(new_debt_at, sched_debt)]
            notes.append(
                f"대환 {cand} {converted:,.0f}원 "
                f"-> {float(cand.get('전환후_금리_연', 0)):.2%}"
            )

        # ── 3) 부채감면 — 원금이 깎이고 새 조건으로 재조정된다 ──
        elif effect == "부채감면":
            forgive_rate = float(cand.get("원금감면율_추정", 0.0))
            # 채무조정은 신청 즉시 효과가 나지 않는다. 확정까지 몇 달이 걸리고,
            # 그동안은 기존 조건대로 계속 갚아야 한다.
            start_at = int(cand.get("적용시점_개월", 0) or 0)
            if start_at >= horizon:
                notes.append(f"채무조정 {cand}: 적용 시점이 시뮬레이션 밖 -> 효과 0")
                continue

            # 감면 대상은 '지금' 잔액이 아니라 '확정 시점' 잔액이다.
            # 그때까지 갚은 만큼은 이미 줄어 있다.
            before = old_debt_at[start_at]
            if before <= 0:
                notes.append(f"채무조정 {cand}: 조정할 부채 없음 -> 효과 0")
                continue

            cut = before * forgive_rate  # 감면액
            after = before - cut  # 조정 후 잔액
            forgiven += cut

            # 확정 시점 **이후**의 기존 스케줄만 대체한다.
            # 그 이전은 원래 조건대로 갚은 것이므로 남겨둔다.
            for i in range(start_at, horizon):
                old_payment_during[i] = 0.0
            for i in range(start_at, horizon + 1):
                old_debt_at[i] = 0.0
            remaining_existing = 0.0

            sched_payment, sched_debt = loan_schedule(
                principal=after,
                annual_rate=0.0,  # 채무조정은 이자 감면이 원칙
                grace_months=int(cand.get("조정후_거치_개월_기본값", 0)),
                repay_months=int(cand.get("조정후_상환_개월_기본값", 0)),
                method="균등분할",
                start_at=start_at,
                horizon=horizon,
            )
            new_payment_during = [
                a + b for a, b in zip(new_payment_during, sched_payment)
            ]
            new_debt_at = [a + b for a, b in zip(new_debt_at, sched_debt)]
            notes.append(
                f"채무조정 {cand} {before:,.0f} -> {after:,.0f}원 "
                f"(감면 {cut:,.0f}원, 추정치)"
            )

        # ── 이자조정 — 원금과 대상 채무는 그대로 두고 금리만 낮춘다 ──
        #
        # 대환전환과 다르다: 대환은 빚이 새 대출로 옮겨가지만(고금리 부채로 한정,
        # 새 거치·상환기간 적용) 이자조정은 **같은 빚의 금리만** 바뀐다.
        # 부채감면과도 다르다: 원금이 줄지 않는다.
        # 새출발기금 부실우려차주가 유일한 사례다.
        elif effect == "이자조정":
            if remaining_existing <= 0:
                notes.append(f"이자조정 {cand}: 조정할 기존 부채 없음 -> 효과 0")
                continue

            # 연체 30일 전후로 금리 상한이 갈린다 (새출발기금 FAQ Q27).
            if (profile.overdue_days or 0) <= 30:
                cap = float(cand.get("조정후_금리_상한_연체30일이전", 0.09))
            else:
                cap = float(cand.get("조정후_금리_상한_연체30일이후", 0.047))

            # min(채무조정 전 금리, 상한). 기존 금리를 모르면 상한을 그대로 쓴다.
            before = profile.existing_debt_rate
            new_rate = cap if before is None else min(before, cap)

            start_at = int(cand.get("적용시점_개월", 0) or 0)
            if start_at >= horizon:
                notes.append(f"이자조정 {cand}: 적용 시점이 시뮬레이션 밖 -> 효과 0")
                continue

            # 조정 시점 이전은 기존 스케줄을 그대로 두고, 이후를 새 금리로 다시 만든다.
            target = remaining_existing
            sched_payment, sched_debt = loan_schedule(
                principal=old_debt_at[start_at],
                annual_rate=new_rate,
                grace_months=int(cand.get("조정후_거치_개월_기본값", 0)),
                repay_months=int(cand.get("조정후_상환_개월_기본값", 0)),
                method="균등분할",
                start_at=start_at,
                horizon=horizon,
            )
            # 조정 시점 이후의 기존 스케줄을 덜어내고 새 것으로 갈아끼운다
            for i in range(start_at, horizon):
                old_payment_during[i] = 0.0
            for i in range(start_at, horizon + 1):
                old_debt_at[i] = 0.0
            new_payment_during = [
                a + b for a, b in zip(new_payment_during, sched_payment)
            ]
            new_debt_at = [a + b for a, b in zip(new_debt_at, sched_debt)]
            remaining_existing = 0.0

            notes.append(
                f"이자조정 {cand}: 대상 {target:,.0f}원 · "
                f"금리 {(before if before is not None else cap):.2%} -> {new_rate:.2%} "
                f"({start_at}개월 뒤 적용)"
            )

        # ── 4) 상환연장 — 원금 감면 없이 기간만 늘려 월 상환액을 낮춘다 (가정 D1) ──
        #
        # 이름 주의: 소진공 「2026 상환연장 신청 안내」가 "이자납부 기간을 '상환유예'하는
        # 제도가 아니라 원금 상환기간을 연장하는 '상환연장' 제도"라고 명시하고 있다.
        # 제도 안내와 용어를 맞추기 위해 현금효과 이름을 상환유예 -> 상환연장으로 바꿨다.
        elif effect == "상환연장":
            # 대상은 소진공 정책자금 직접대출 잔액. 기존 부채를 넘을 수 없다.
            target = min(profile.policy_loan_balance, remaining_existing)
            if target <= 0 or profile.existing_debt_payment <= 0:
                notes.append(f"상환연장 {cand}: 대상 정책자금 대출 없음 -> 효과 0")
                continue

            # 상환연장도 신청 즉시가 아니라 약정 체결 후에 효과가 난다.
            start_at = int(cand.get("적용시점_개월", 0) or 0)
            if start_at >= horizon:
                notes.append(f"상환연장 {cand}: 적용 시점이 시뮬레이션 밖 -> 효과 0")
                continue

            share = target / remaining_existing  # 기존 부채 중 유예 대상 비율
            target_payment = profile.existing_debt_payment * share  # 대상분 월 상환액
            remaining_term = math.ceil(target / target_payment)  # 남은 상환 개월(역산)
            extension = int(cand.get("연장_개월_최대", 0))  # 연장 개월
            new_term = remaining_term + extension  # 연장 후 총 상환 개월

            # 연장하면 금리가 가산된다 (공고: 약정금리 + 0.2%p, 단기연체 시 +0.4%p)
            # 연체가 있으면 가산금리가 두 배로 붙는다 (0.2%p -> 0.4%p).
            # 자격 필터가 30일 초과를 이미 걸러내므로 여기 오는 연체는 단기연체다.
            if (profile.overdue_days or 0) > 0:
                surcharge = float(cand.get("금리_가산_연_단기연체", 0.0))
            else:
                surcharge = float(cand.get("금리_가산_연", 0.0))
            # ⚠ 여기서는 "모름"을 0%로 두면 안 된다. 가산금리만 남아 연 0.2%가
            #   되어, 존재하지 않는 초저금리 상품처럼 보인다. 금리를 모르면
            #   가정 B1의 0%가 아니라 별도 기준을 써야 한다. TODO(inquiry): B-20
            base_rate = profile.existing_debt_rate
            if base_rate is None:
                base_rate = UNKNOWN_RATE_FOR_EXTENSION
                notes.append(
                    f"상환연장 {cand}: 기존 금리를 몰라 연 "
                    f"{UNKNOWN_RATE_FOR_EXTENSION:.2%}로 가정했다 (확인 필요)"
                )
            new_rate = base_rate + surcharge

            # 약정 시점 **이후**의 대상분만 기존 스케줄에서 덜어낸다.
            # 그 이전은 원래 금액대로 갚은 것이다
            # (소진공 안내: "약정 전까지 기존 대출에서 발생한 이자는 약정직전 수납 필요").
            for i in range(start_at, horizon):
                old_payment_during[i] *= 1 - share
            for i in range(start_at, horizon + 1):
                old_debt_at[i] *= 1 - share
            remaining_existing -= target

            # 늘어난 기간으로 새 스케줄을 만든다 (거치 없이 바로 분할 상환)
            sched_payment, sched_debt = loan_schedule(
                principal=target,
                annual_rate=new_rate,
                grace_months=0,
                repay_months=new_term,
                method="균등분할",
                start_at=start_at,
                horizon=horizon,
            )
            new_payment_during = [
                a + b for a, b in zip(new_payment_during, sched_payment)
            ]
            new_debt_at = [a + b for a, b in zip(new_debt_at, sched_debt)]
            notes.append(
                f"상환연장 {cand} 대상 {target:,.0f}원 · "
                f"월 상환 {target_payment:,.0f} -> {sched_payment[0]:,.0f}원 "
                f"({remaining_term}개월 -> {new_term}개월, 금리 +{surcharge:.1%})"
            )

        # ── 5) 비용감면 — 유입이 아니라 그 달의 고정비가 줄어든다 (가정 C1) ──
        elif effect == "비용감면":
            payout_at = int(cand.get("지급시점_개월", 0) or 0)
            if payout_at < horizon:
                cost_saving_during[payout_at] += amount
            notes.append(f"비용감면 {cand} {amount:,}원")

        # ── 6) 없음 — 순위 대상이 아니지만 방어적으로 처리 ──
        elif effect == "없음":
            notes.append(f"현금효과 없음: {cand}")

        else:
            raise ValueError(f"알 수 없는 현금효과: {effect!r} ({cand.candidate_id})")

    # ══════════ 2단계: 월별 누적 차감 ══════════
    balance_at = [float(profile.cash_on_hand)] + [0.0] * horizon
    shortfall_at = [float(profile.cash_on_hand)] + [0.0] * horizon
    debt_at = [float(profile.existing_debt)] + [0.0] * horizon
    payment_during = [0.0] * horizon

    survival_months = horizon  # 끝까지 버티면 horizon
    survives = True  # 아직 잔고가 0 아래로 안 갔는지

    for month in range(1, horizon + 1):
        # 영업 현금흐름 = 매출 - 고정비 - 변동비.
        # 비용감면은 고정비를 깎으므로(빼기 앞의 항이 작아지므로) 결과에는 '더해진다'.
        cost_saving = cost_saving_during[month - 1]  # 이번 달 절약액
        operating = profile.operating_cashflow(month) + cost_saving
        # 이번 달 총 상환액
        total_payment = old_payment_during[month - 1] + new_payment_during[month - 1]

        # ── 생존이 이미 끝났으면 아무 일도 일어나지 않는다 (가정 F1) ──
        if not survives:
            payment_during[month - 1] = 0.0      # 못 갚았으므로 나간 돈 0
            balance_at[month] = 0.0              # 잔고는 0에서 멈춘다
            debt_at[month] = debt_at[month - 1]  # 부채는 종료 시점 값 유지
            # shortfall_at은 **얼리지 않는다.** 이 배열의 목적이 "이 목표를 채우려면
            # 얼마가 모자라는가"를 재는 것이라, 죽은 뒤로 고정하면 목표를 늘려도
            # 부족분이 그대로여서 역산이 무너진다(6개월 목표와 12개월 목표의
            # 필요액이 같아진다). 파산 이후에도 원래 조건대로 계속 계산한다.
            shortfall_at[month] = (
                shortfall_at[month - 1]
                + operating
                - total_payment
                + inflow_during[month - 1]
            )
            continue

        raw = (
            balance_at[month - 1]
            + operating
            - total_payment
            + inflow_during[month - 1]
        )
        # shortfall은 자르지 않는다. 역산이 "얼마가 모자랐는가"를 알아야 한다.
        shortfall_at[month] = raw

        if raw <= 0:
            # 이 달에 파산한다. 상환액을 낼 돈이 없으므로 **이 달은 갚지 못한다.**
            # 잔고 42만원인 달에 122만원을 갚은 것으로 처리하면 안 된다.
            payment_during[month - 1] = 0.0
            balance_at[month] = 0.0
            debt_at[month] = debt_at[month - 1]
        else:
            payment_during[month - 1] = total_payment
            balance_at[month] = raw
            debt_at[month] = old_debt_at[month] + new_debt_at[month]

        # 잔고가 0 이하가 되는 '첫' 시점만 기록한다.
        # month개월차에 0이 됐다면 (month-1)개월까지 버틴 것.
        #
        # TODO(forecast): 계절성이 들어오면 "0을 찍었다가 다음 달 매출로 회복"하는
        #   경우가 생길 수 있다. 현재 178개 조합 실측에서는 0건이었다(매출 추세가
        #   단조 감소이고 상환액은 거치 종료로 오히려 늘어나므로 구조적으로 회복이
        #   어렵다). forecast.py가 계절성을 반영하면 이 판정을 재검토할 것.
        if raw <= 0:
            survival_months = month - 1
            survives = False

    return SimResult(
        balance_at=balance_at,
        shortfall_at=shortfall_at,
        debt_at=debt_at,
        payment_during=payment_during,
        inflow_during=inflow_during,
        survival_months=survival_months,
        debt_at_exit=debt_at[horizon],  # 동결 덕분에 종료 시점 값과 같다
        balloon_after_horizon=balloon_total,
        balloon_due_at=balloon_due_at,
        total_payment=sum(payment_during),
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
