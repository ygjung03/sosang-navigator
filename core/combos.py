"""
조합 생성 — 자격 필터 · 문법 제약 · 필요액 역산.

파이프라인에서 이 모듈의 자리:

    정책 후보 18개
        │
        │  ① eligible_candidates()   자격 필터 (국면·업종·신용·매출요건 등)
        ▼
    이 사용자가 받을 수 있는 후보들
        │
        │  ② build_combos()          조합 문법 제약 (융자1·채무조정1·보조금2)
        ▼
    유효 조합 수십~수백 개
        │
        │  ③ evaluate()              각 조합을 36개월 시뮬레이션
        ▼
    조합별 2축 값  →  metrics.py(함정) → optimizer.py(순위)


설계 원칙 (기획서 5.5)
----------------------
- **하드 제약은 코드가 배제한다.** 자격·중복·기한 위반 조합은 생성 단계에서 제외.
  LLM에게 "이건 자격이 안 돼요"를 판단시키지 않는다.
- **정량으로 고를 수 없으면 고르지 않는다.** 상환조건은 후보로 전개되어 있으므로
  이 모듈은 어느 조건이 나은지 판단하지 않는다. 전부 조합으로 만든다.
- **목표 충족 조합을 모두 남긴다.** 먼저 찾은 것이 최선이라는 보장이 없으므로
  조기 종료하지 않는다 (기획서 5.2.1).


명명 규칙: 코드 식별자는 영어, JSON 키는 한글. 시점은 _at, 기간은 _during.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Sequence

from .cashflow import SIM_MONTHS, Applied, SimResult, UserProfile, simulate
from .loader import Candidate, PolicyDB

# 조합 문법 기본값. JSON의 _조합문법에 값이 있으면 그쪽이 우선한다.
DEFAULT_MAX_LOAN = 1  # 융자는 조합당 최대 1개
DEFAULT_MAX_RESTRUCTURE = 1  # 채무조정도 1개
DEFAULT_MAX_GRANT = 2  # 보조금은 2개까지

# 필요액 역산 설정
NEED_ITERATIONS = 6  # 고정점 반복 횟수 (보통 2~3회에 수렴)
NEED_TOLERANCE = 10_000  # 1만원 이내면 수렴으로 본다


# ══════════════════════════════════════════════════════════
# 사용자 속성 — 자격 판정에만 쓰이고 현금흐름에는 안 들어가는 값들
# ══════════════════════════════════════════════════════════


@dataclass
class UserContext:
    """자격 필터가 보는 사용자 정보.

    UserProfile(재무)과 분리한 이유: 이쪽은 시뮬레이션에 전혀 안 쓰이고
    '받을 수 있느냐'만 판정한다. 섞으면 어느 필드가 계산에 영향을 주는지
    구분이 안 된다.
    """

    phase: str
    """국면. "일시적자금난" / "개선여지" / "회복불가" (phase.py가 판단)."""

    region: str = "서울"
    """사업장 소재 지역. 서울시 자금의 자격 요건."""

    industry_code: str | None = None
    """한국표준산업분류 코드. 융자제외업종 판정용. 모르면 None(통과)."""

    credit_score: int | None = None
    """개인신용점수(NICE/NCB). 모르면 None -> 점수 제한이 있는 자금은 통과시킨다.

    소상공인이 자기 점수를 모르는 경우가 많아 '모르면 건너뛰기'로 받는다.
    통과시키는 쪽을 택한 이유: 자격이 있는데 못 보여주는 것보다
    "확인이 필요합니다"를 붙여 보여주는 편이 낫다.
    """

    restart_status: str = "안함"
    """재창업 상태. "안함" / "준비중" / "이미함" (기획서 4.4)."""

    is_closed: bool = False
    """폐업했거나 폐업 예정인지. 재기 계열 정책의 자격."""

    revenue_drop_rate: float = 0.0
    """최근 매출 감소율(소수). 0.15 = 15% 감소. 매출 요건 판정에 쓴다."""

    revenue_growth_rate: float = 0.0
    """매출 증가율(소수). 재도전특별자금 도약형처럼 '증가'가 요건인 경우."""

    annual_revenue: int | None = None
    """연 매출액(원). 경영안정 바우처 등 매출 상한이 있는 정책의 판정."""

    include_loans: bool = True
    """대출을 조합에 포함할지. False면 융자를 하드 제약으로 배제 (기획서 5.2.1)."""

    policy_loan_count_5y: int = 0
    """최근 5년 정책자금 지원 횟수. 3회 이상이면 융자 제한(공고 융자제한 ⑦)."""


# ══════════════════════════════════════════════════════════
# ① 자격 필터
# ══════════════════════════════════════════════════════════


def _passes_phase(cand: Candidate, ctx: UserContext) -> bool:
    """국면 필터. 기획서 4.3에 따라 느슨하게 — 명백히 안 맞는 것만 제외."""
    return ctx.phase in cand.phases


def _passes_region(cand: Candidate, ctx: UserContext) -> bool:
    """지역 요건. 자격요건에 지역이 명시된 정책만 검사한다."""
    required = cand.eligibility.get("지역")
    return required is None or required == ctx.region


def _passes_credit(cand: Candidate, ctx: UserContext) -> bool:
    """신용점수 상한. 점수를 모르면(None) 통과시킨다."""
    ceiling = cand.eligibility.get("신용점수_상한")
    if ceiling is None or ctx.credit_score is None:
        return True
    return ctx.credit_score <= ceiling


def _passes_restart(cand: Candidate, ctx: UserContext) -> bool:
    """재창업 상태. 후보 레벨에 명시된 것이 정책 레벨보다 우선한다.

    재도전특별자금처럼 후보(일반형/희망형/도약형)마다 요건이 다른 경우가 있다.
    """
    allowed = cand.raw.get("재창업상태") or cand.eligibility.get("재창업상태")
    return allowed is None or ctx.restart_status in allowed


def _passes_revenue_requirement(cand: Candidate, ctx: UserContext) -> bool:
    """매출 요건. 감소율 기준과 증가율 기준 두 방향이 있다.

    긴급경영안정자금은 '15% 이상 감소', 재도전 도약형은 '5% 이상 증가'.
    """
    # 후보에 매출 요건이 있으면 그쪽이 우선 (재도전 도약형)
    required = cand.raw.get("매출_요건")
    if required is None:
        required = cand.eligibility.get("매출_요건", False)
    if not required:
        return True

    threshold = cand.raw.get("매출_기준율") or cand.eligibility.get("매출_기준율")
    direction = cand.raw.get("매출_방향") or cand.eligibility.get("매출_방향")
    if threshold is None:
        return True  # 기준율을 모르면 판정하지 않고 통과

    if direction == "감소율_이상":
        return ctx.revenue_drop_rate >= threshold
    if direction == "증가율_이상":
        return ctx.revenue_growth_rate >= threshold
    return True


def _passes_annual_revenue(cand: Candidate, ctx: UserContext) -> bool:
    """연매출 상·하한. 경영안정 바우처(1억 400만원 미만) 등."""
    ceiling = cand.eligibility.get("연매출_상한")
    floor = cand.eligibility.get("연매출_하한")
    if ceiling is None and floor is None:
        return True
    if ctx.annual_revenue is None:
        return True  # 모르면 통과 (확인 필요 라벨을 붙여 안내)
    if ceiling is not None and ctx.annual_revenue >= ceiling:
        return False
    if floor is not None and ctx.annual_revenue < floor:
        return False
    return True


def _passes_closure(cand: Candidate, ctx: UserContext) -> bool:
    """폐업 여부 요건.

    JSON의 '폐업여부'는 자유 서술이라(예: "영업중", "폐업 또는 폐업예정",
    "영업중·폐업·휴업 모두 가능") 키워드로 판정한다.

        "모두 가능"이 들어 있으면      -> 제한 없음
        "폐업"이 들어 있으면            -> 폐업(예정)자만
        그 외에 "영업중"이 들어 있으면  -> 영업 중인 사람만
    """
    requirement = cand.closure_requirement
    if not requirement:
        return True
    # 제한이 없다고 명시한 경우가 먼저다.
    # 서울형 다시서기처럼 "제한 없음 (... 폐업 ... 재창업자는 영업중)"처럼
    # 설명 안에 '폐업'이라는 단어가 들어가는 경우가 있어 순서가 중요하다.
    if "제한 없음" in requirement or "모두 가능" in requirement:
        return True
    if "폐업" in requirement:
        # 폐업(예정)이 전제인 정책. 영업을 계속할 사람에게는 해당하지 않는다.
        return ctx.is_closed
    if "영업중" in requirement:
        return not ctx.is_closed
    return True


def _passes_debt_requirement(cand: Candidate, profile: UserProfile) -> bool:
    """부채 관련 요건.

    - 채무조정: 조정할 빚이 있어야 한다
    - 대환대출: 고금리(7% 이상) 부채가 있어야 한다
    - 상환유예: 정책자금 직접대출 잔액이 있어야 한다
    """
    if cand.method == "채무조정" and profile.existing_debt <= 0:
        return False

    if cand.effect == "대환전환":
        # 고금리 부채를 입력했으면 그 값으로, 안 했으면 기존 부채로 판정
        convertible = profile.high_rate_debt or profile.existing_debt
        if convertible <= 0:
            return False

    if cand.effect == "상환유예" and profile.policy_loan_balance <= 0:
        return False

    return True


def _passes_loan_limit(cand: Candidate, ctx: UserContext, db: PolicyDB) -> bool:
    """융자 횟수 제한. 최근 5년 3회 이상이면 제한(공고 융자제한 ⑦).

    특별경영안정자금 계열은 횟수에서 제외되는데, 현재 데이터에
    해당 플래그가 없어 MVP에서는 일괄 적용한다.
    """
    if not cand.is_loan:
        return True
    return ctx.policy_loan_count_5y < 3


def eligible_candidates(
    db: PolicyDB, profile: UserProfile, ctx: UserContext
) -> list[Candidate]:
    """이 사용자가 신청 자격이 있는 후보만 골라낸다.

    통과 기준을 느슨하게 잡은 항목이 있다(신용점수·연매출을 모르면 통과).
    자격이 있는데 안 보여주는 것보다, 보여주고 "확인이 필요합니다"를
    붙이는 편이 낫다는 판단이다.
    """
    result: list[Candidate] = []
    for cand in db.candidates:
        # 대출 제외 옵션 (기획서 5.2.1 — 자격 필터와 같은 층위의 하드 제약)
        if not ctx.include_loans and cand.is_loan:
            continue
        if not _passes_phase(cand, ctx):
            continue
        if not _passes_region(cand, ctx):
            continue
        if not _passes_credit(cand, ctx):
            continue
        if not _passes_restart(cand, ctx):
            continue
        if not _passes_revenue_requirement(cand, ctx):
            continue
        if not _passes_annual_revenue(cand, ctx):
            continue
        if not _passes_closure(cand, ctx):
            continue
        if not _passes_debt_requirement(cand, profile):
            continue
        if not _passes_loan_limit(cand, ctx, db):
            continue
        result.append(cand)
    return result


# ══════════════════════════════════════════════════════════
# ② 조합 생성
# ══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Combo:
    """하나의 정책 조합. 후보들의 묶음."""

    items: tuple[Applied, ...]

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        return tuple(item.candidate for item in self.items)

    @property
    def label(self) -> str:
        """화면·로그용 이름. '대환대출 + 긴급경영안정자금' 형태."""
        return " + ".join(str(c) for c in self.candidates)

    @property
    def total_inflow(self) -> int:
        """조합 전체의 현금 유입 합계(원).

        기획서 5.2.1 — 희망 금액 판정은 개별 정책이 아니라 조합 전체로 한다.
        대환·채무조정·상환유예는 유입이 없으므로 제외된다.
        """
        return sum(
            item.effective_amount
            for item in self.items
            if item.candidate.effect == "신규유입"
        )

    def has_loan(self) -> bool:
        return any(c.is_loan for c in self.candidates)


def _grammar_limits(db: PolicyDB) -> dict[str, int]:
    """JSON의 _조합문법에서 지원방법별 최대 개수를 읽는다."""
    rules = db.combo_rules
    return {
        "융자": int(rules.get("융자", {}).get("최대", DEFAULT_MAX_LOAN)),
        "채무조정": int(
            rules.get("채무조정", {}).get("최대", DEFAULT_MAX_RESTRUCTURE)
        ),
        "보조금": int(rules.get("보조금", {}).get("최대", DEFAULT_MAX_GRANT)),
    }


def _cross_source_allowed(db: PolicyDB, method: str) -> bool:
    """재원 간 중복 허용 여부 (소진공 + 서울시 동시 수혜 등).

    확인 전까지 보수적으로 false. 문의목록 A-1 / B-10 답변 후
    JSON 플래그만 바꾸면 조합이 자동으로 확장된다.
    """
    return bool(db.combo_rules.get(method, {}).get("재원간_중복허용", False))


def _restructure_loan_exclusive(db: PolicyDB) -> bool:
    """채무조정 이행 중 신규 융자가 배타인지 (문의목록 A-3 / C-14).

    "확인필요" 문자열이면 아직 모르는 상태 -> 배타로 보지 않고 조합을 만든다.
    확정되면 True/False 불리언으로 바뀐다.
    """
    value = db.combo_rules.get("교차제약", {}).get("채무조정_융자_배타")
    return value is True


def _pick_by_method(
    pool: Sequence[Candidate], method: str, limit: int, allow_cross_source: bool
) -> list[tuple[Candidate, ...]]:
    """한 지원방법 안에서 가능한 선택지를 모두 만든다.

    Returns:
        빈 튜플(선택 안 함)을 포함한 모든 조합.
        예: 융자 3개 중 최대 1개 -> [(), (A,), (B,), (C,)]
    """
    items = [c for c in pool if c.method == method]
    options: list[tuple[Candidate, ...]] = [()]  # 아무것도 안 고르는 경우

    for size in range(1, limit + 1):
        for group in combinations(items, size):
            # 재원 간 중복이 금지면 서로 다른 재원이 섞인 조합은 만들지 않는다
            if not allow_cross_source and len({c.source for c in group}) > 1:
                continue
            options.append(group)
    return options


def default_loan_amounts(
    db: PolicyDB,
    pool: Sequence[Candidate],
    profile: UserProfile,
    target_months: int,
    horizon: int = SIM_MONTHS,
) -> dict[str, int]:
    """융자 후보별 디폴트 금액을 필요액으로 역산한다 (기획서 5.4).

    지정하지 않으면 Applied.effective_amount가 **정책 한도**를 쓰는데,
    서울시 성장기반자금처럼 한도가 5억인 자금이 있어 소상공인에게
    비현실적인 조합이 상위를 차지한다.

    기획서 5.4는 디폴트를 '필요액 중 융자분', 슬라이더 상한을 '정책 한도'로
    정해 두었다. 여기서 만드는 것은 **디폴트**이고, 상한은 화면에서 정책
    한도를 그대로 쓴다. 사용자가 한도까지 올려보는 것이 오히려 핵심 데모
    ("다 빌리면 함정, 줄이면 안전")이므로 상한을 좁히지 않는다.
    """
    result: dict[str, int] = {}
    for cand in pool:
        if not cand.is_loan:
            continue  # 보조금·채무조정은 금액이 정책에 고정
        result[cand.candidate_id] = required_loan_amount(
            profile, cand, target_months, horizon
        )
    return result


def build_combos(
    db: PolicyDB, pool: Sequence[Candidate], amounts: dict[str, int] | None = None
) -> list[Combo]:
    """자격을 통과한 후보들로 유효 조합을 모두 만든다.

    Args:
        pool:    eligible_candidates()의 결과
        amounts: 후보id -> 적용 금액. 지정하지 않은 후보는 정책 한도를 쓴다.
                 융자는 default_loan_amounts()로 필요액을 넣어주는 것을 권장한다.
                 (안 넣으면 한도를 다 빌리는 비현실적 조합이 만들어진다)

    조합 폭발은 알고리즘이 아니라 도메인 문법으로 막는다 (기획서 10장-10).
    융자1·채무조정1·보조금2 제약만으로 수십 개 수준에 머문다.
    """
    amounts = amounts or {}
    limits = _grammar_limits(db)

    loan_options = _pick_by_method(
        pool, "융자", limits["융자"], _cross_source_allowed(db, "융자")
    )
    restructure_options = _pick_by_method(
        pool, "채무조정", limits["채무조정"], _cross_source_allowed(db, "채무조정")
    )
    grant_options = _pick_by_method(
        pool, "보조금", limits["보조금"], _cross_source_allowed(db, "보조금")
    )

    exclusive = _restructure_loan_exclusive(db)

    combos: list[Combo] = []
    for loans in loan_options:
        for restructures in restructure_options:
            # 채무조정 ↔ 신규 융자가 배타면 둘이 함께 있는 조합은 만들지 않는다
            if exclusive and loans and restructures:
                continue
            for grants in grant_options:
                picked = loans + restructures + grants
                if not picked:  # 아무것도 안 고른 조합은 제외
                    continue
                items = tuple(
                    Applied(candidate=c, amount=amounts.get(c.candidate_id))
                    for c in picked
                )
                combos.append(Combo(items=items))
    return combos


# ══════════════════════════════════════════════════════════
# ③ 필요액 역산 (기획서 5.2.1)
# ══════════════════════════════════════════════════════════


def required_amount(
    profile: UserProfile, target_months: int, horizon: int = SIM_MONTHS
) -> int:
    """목표 개월수를 버티는 데 필요한 최소 현금(원).

    `목표 개월수 × 월 적자폭` 곱셈을 쓰지 않는다. 그것은 5.1에서 버린
    나눗셈("보유현금 / 월 순손실")의 거울상이며, 매출 하락 추세에서
    적자가 매달 커지기 때문에 근사가 무너진다.

    대신 무지원 시뮬레이션의 **구간 최저점**을 쓴다. 보조금은 상환이 없어
    유입이 곡선을 그대로 평행이동시키므로, 최저점이 0이 되면 구간 내내
    0 이상임이 보장된다.

    생존 개월수의 정의가 "잔고가 0이 되는 **첫** 시점"이므로 최저점을
    기준으로 삼는 것이 맞다. 3개월차 -1,000만 / 6개월차 -300만이면
    필요액은 1,000만원이다. 300만만 넣으면 3개월차에서 멈춘다.
    """
    baseline = simulate(profile, [], horizon=horizon)
    window = baseline.balance_at[: min(target_months, horizon) + 1]
    lowest = min(window)
    # 생존 판정이 `balance <= 0`이므로 최저점을 정확히 0으로 맞추면
    # 그 달에 생존이 끝나 목표를 1개월 못 채운다. 여유 1원을 더한다.
    return max(0, int(-lowest) + 1) if lowest <= 0 else 0


def required_loan_amount(
    profile: UserProfile,
    cand: Candidate,
    target_months: int,
    horizon: int = SIM_MONTHS,
) -> int:
    """융자로 목표 개월수를 채우려면 얼마를 빌려야 하는지 (고정점 반복).

    대출은 거치기간에도 이자가 나가 곡선이 평행이동하지 않고 조금 더
    내려간다. 그래서 최저점만큼 빌려도 이자 때문에 목표에 미달한다.
    부족분을 더해 다시 시뮬레이션하는 것을 몇 번 반복하면 수렴한다.

        1차: 최저점 -1,000만  -> 1,000만원 대출 -> 거치 이자로 -19만원 미달
        2차: 1,019만원        -> 다시 -0.4만원 미달
        3차: 1,024만원        -> 통과

    정책 한도를 넘으면 한도에서 멈춘다. 한도로도 목표에 미달하면
    그 사실이 호출한 쪽에 그대로 드러난다(생존 개월수가 목표보다 작음).

    ※ 대환전환은 예외다. 유입이 없어 '필요액'이 정의되지 않으므로
      전환 가능한 고금리 부채 금액을 그대로 쓴다.
    """
    ceiling = int(cand.get("금액_한도", 0) or 0)
    if ceiling <= 0:
        return 0

    # 대환전환은 '필요액'이라는 개념이 성립하지 않는다.
    # 현금 유입이 0이라 금액을 아무리 올려도 목표 개월수에 도달하지 못하고,
    # 아래 반복문이 매번 한도까지 올라가 멈춘다. 결과는 같지만 의도가 안 보인다.
    # 대환의 기준은 "얼마가 필요한가"가 아니라 "얼마를 갈아탈 수 있는가"이므로
    # 보유한 고금리 부채와 정책 한도 중 작은 값을 바로 돌려준다.
    if cand.effect == "대환전환":
        convertible = profile.high_rate_debt or profile.existing_debt
        return min(ceiling, int(convertible))

    amount = min(required_amount(profile, target_months, horizon), ceiling)

    for _ in range(NEED_ITERATIONS):
        result = simulate(profile, [Applied(cand, amount)], horizon=horizon)
        window = result.balance_at[: min(target_months, horizon) + 1]
        lowest = min(window)
        if lowest > 0:  # 목표 달성 (0은 생존 종료이므로 미달)
            break
        shortfall = int(-lowest) + 1
        if shortfall < NEED_TOLERANCE:  # 1만원 이내면 수렴으로 본다
            amount = min(amount + shortfall, ceiling)
            break
        amount = min(amount + shortfall, ceiling)
        if amount >= ceiling:  # 한도에 걸리면 더 올릴 수 없다
            break

    return amount


# ══════════════════════════════════════════════════════════
# ④ 평가 — 조합별 2축 산출
# ══════════════════════════════════════════════════════════


@dataclass
class Evaluated:
    """조합 하나의 시뮬레이션 결과."""

    combo: Combo
    result: SimResult

    @property
    def survival_months(self) -> int:
        """축 1 — 클수록 좋음."""
        return self.result.survival_months

    @property
    def net_debt_at_30(self) -> float:
        """축 2 — 작을수록 좋음."""
        return self.result.net_debt_at_30

    @property
    def label(self) -> str:
        return self.combo.label


def evaluate(
    profile: UserProfile, combos: Iterable[Combo], horizon: int = SIM_MONTHS
) -> list[Evaluated]:
    """모든 조합을 시뮬레이션해 2축 값을 구한다.

    36칸 배열 계산이라 조합 수백 개도 부담이 없다.
    """
    return [
        Evaluated(combo=combo, result=simulate(profile, combo.items, horizon=horizon))
        for combo in combos
    ]


def filter_by_goal(
    evaluated: Sequence[Evaluated],
    target_months: int | None = None,
    target_inflow: int | None = None,
) -> tuple[list[Evaluated], bool]:
    """목표(생존 개월수·희망 금액)를 충족하는 조합을 **모두** 남긴다.

    기획서 5.2.1 — 먼저 찾은 조합이 최선이라는 보장이 없고, 어차피 2축
    순위를 매기려면 전부 시뮬레이션해야 하므로 조기 종료하지 않는다.

    Returns:
        (충족 조합 목록, 충족하는 것이 있었는지)
        하나도 없으면 (전체 목록, False)를 돌려준다. 호출한 쪽에서
        금액을 상향한 버전을 후보에 추가할지 판단한다.
    """
    passed = []
    for ev in evaluated:
        if target_months is not None and ev.survival_months < target_months:
            continue
        if target_inflow is not None and ev.combo.total_inflow < target_inflow:
            continue
        passed.append(ev)

    if passed:
        return passed, True
    return list(evaluated), False


# ══════════════════════════════════════════════════════════
# 요약 출력 (디버깅용)
# ══════════════════════════════════════════════════════════


def summarize(evaluated: Sequence[Evaluated], top: int = 10) -> str:
    """생존 개월수 기준 상위 몇 개를 표로. 순위 산정은 optimizer.py가 한다."""
    MAN = 10_000
    rows = sorted(
        evaluated, key=lambda e: (-e.survival_months, e.net_debt_at_30)
    )[:top]
    lines = [f"{'생존':>6} {'순부채(만원)':>12}  조합"]
    for ev in rows:
        lines.append(
            f"{ev.result.survival_label:>6} {ev.net_debt_at_30 / MAN:>12,.0f}  {ev.label}"
        )
    return "\n".join(lines)
