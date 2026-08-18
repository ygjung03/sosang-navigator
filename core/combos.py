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
TODO(adr): 아래 원칙들의 채택 근거(왜 조기 종료를 버렸는지 등)는
           docs/adr/ 로 옮길 대상.
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
# 1차식 외삽이라 보통 0~1회면 끝난다. 부동소수점 오차로 한 걸음 더 필요한
# 경우를 대비한 상한이며, 수렴 판정이 아니라 무한루프 방어용이다.
NEED_ITERATIONS = 4

# 목표 시점의 잔고를 이만큼 양수로 만든다. 생존 판정이 `balance <= 0`이라
# 정확히 0으로 맞추면 그 달에 생존이 끝나 목표를 1개월 못 채운다.
# 1원은 부동소수점 오차에 묻힐 수 있어 1,000원으로 둔다.
NEED_MARGIN = 1_000


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

    tax_delinquent: bool = False
    """국세·지방세를 **체납 중**인지.

    같은 사실이 제도에 따라 반대로 작용하므로 필드 하나로 세 곳을 판정한다.

        소진공 융자 6종   자격을 뺏는다 — 공고 제2026-478호 4쪽 융자제한 ①
                        "세금을 체납중인 소상공인"
        상환연장         자격을 뺏는다 — 소진공 FAQ 80번 "세금체납... 지원이 불가"
        새출발기금
        부실우려차주      자격을 **준다** — 연체 10일 미만이어도 체납이면 대상.
                        체납 등재 자체가 상환 곤란의 신호로 취급된다

    **금액 기준은 없다.** 세 곳 모두 "체납 중인지" 여부만 본다.

    ※ 새출발기금만 원문이 "체납으로 **공공기록정보가 등재된 자**"인데, 등재 여부는
      한국신용정보원 내부 상태라 소상공인이 알기 어렵다. 입력은 "체납 중인지"로 받고
      그쪽은 느슨하게 판정한다 — 체납했으나 등재되지 않은 경우 자격을 과대 인정할 수
      있으나, 자격을 잘못 뺏는 것보다 낫다.

    ※ 소진공 융자제한 ①의 예외("압류·매각 유예 등을 받으면 직접대출에 한해 포함")는
      구현하지 않았다. 유예 여부를 받으려면 필드가 하나 더 필요한데 드문 경우다.
      LLM이 대화로 안내한다.
    """

    store_area_pyeong: float | None = None
    """점포 전용면적(평). 희망리턴패키지 점포철거비 금액 계산에 쓰인다.

    지급액이 정액이 아니라 **1평(3.3㎡)당 20만원 × 전용면적, 최대 600만원**이다
    (공고문 확인). 30평 미만이면 600만원보다 적게 나온다.
    모르면 금액을 계산할 수 없으므로 후보에서 제외한다.
    """

    business_years: float | None = None
    """업력(년). 사업을 시작한 뒤 지난 햇수.

    긴급경영안정자금의 **융자방식 판정**에만 쓰인다. 연매출 1억 400만원 미만
    **AND** 업력 7년 미만이면 직접대출(매월 원금균등), 아니면 대리대출
    (70%를 3개월마다 + 30%를 만기에 일시상환)이다. 현금흐름이 달라진다.

    **연매출 조건이 걸릴 때만 묻는다.** 연매출이 1억 400만원 이상이면 업력과
    무관하게 대리대출이므로 물을 이유가 없다(app.py 조건부 질문).
    모르면 직접대출 후보를 제외하고 대리대출로 보낸다 — 분기 상환이 매월보다
    잔고를 깊게 파므로 보수적이다.
    """


    revenue_drop_rate: float = 0.0
    """매출 감소율(소수). 0.15 = 15% 감소. 매출 요건 판정과 국면 판정에 쓴다.

    **분기 비교와 반기 비교 중 큰 값**을 넣는다.
        분기 = 직전 3개월 합계 vs 그 전 3개월 합계
        반기 = 직전 6개월 합계 vs 그 전 6개월 합계

    제도가 그렇게 정하기 때문이다. 서울시 긴급자영업자금은 "신청일 기준
    직전 분기·반기 대비 매출액이 이전 분기·반기 대비 20% 이상 급감"으로,
    **둘 중 하나만 충족하면** 자격이 생긴다(B-21).
    소진공 긴급경영안정자금도 "직전 매출과 직전전 매출(연도/반기/분기/월별)
    비교"로 단위를 정하지 않아, 신청자에게 유리한 단위를 고를 수 있다.

    POS 12개월이면 분기·반기 둘 다 계산할 수 있다. 연 단위나 전년 동기 대비는
    24개월이 있어야 하므로 쓰지 않는다.
    """

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

    JSON의 '폐업여부'가 자유 서술이라 키워드로 판정한다. 실제 값 예시:

        "영업중"
        "영업중 (신청일 기준 휴업·폐업 아님, 국세청 신고 기준)"
        "영업중 (휴·폐업자 제외)"
        "폐업 또는 폐업예정"
        "영업중·폐업·휴업 모두 가능"
        "제한 없음 (성실실패자·성실상환자는 폐업/채무종결 이후, 재창업자는 영업중)"

    **검사 순서가 결과를 좌우한다.** 대부분의 문자열에 '폐업'이라는 글자가
    들어가기 때문이다("휴업·폐업 아님", "휴·폐업자 제외"처럼 부정문에도 들어간다).
    넓은 조건부터 좁은 조건 순으로 본다.

        ① "제한 없음" / "모두 가능"  -> 무조건 통과
        ② "영업중"                   -> 영업 중인 사람만
        ③ "폐업"                     -> 폐업(예정)자만

    ※ 한계: 자유 서술 파싱이라 새 정책의 표현이 다르면 오판할 수 있다.
      JSON에 불리언 필드(영업중_가능 / 폐업자_가능)를 두는 것이 근본 해결책이나,
      기존 데이터를 전부 손봐야 하므로 MVP에서는 키워드 방식을 유지한다.
    """
    requirement = cand.closure_requirement
    if not requirement:
        return True
    # ① 제한이 없다고 명시한 경우
    if "제한 없음" in requirement or "모두 가능" in requirement:
        return True
    # ② '영업중'이 먼저다. "영업중 (휴업·폐업 아님)"처럼 부정문 안에
    #    '폐업'이 들어가는 경우가 많아 ③을 먼저 보면 정반대로 판정된다.
    if "영업중" in requirement:
        return not ctx.is_closed
    # ③ 폐업(예정)이 전제인 정책
    if "폐업" in requirement:
        return ctx.is_closed
    return True


def _passes_debt_requirement(
    cand: Candidate, profile: UserProfile, ctx: UserContext
) -> bool:
    """부채 관련 요건.

    - 채무조정: 조정할 빚이 있어야 한다
    - 대환대출: 고금리(7% 이상) 부채가 있어야 한다
    - 상환연장: 정책자금 직접대출 잔액이 있어야 한다
    """
    if cand.method == "채무조정" and profile.existing_debt <= 0:
        return False

    if cand.effect == "대환전환":
        # high_rate_debt는 None(모름)과 0(없음)이 다르다.
        #   None -> 입력하지 않았다. 기존 부채 전액을 대상으로 보고 통과시킨다.
        #   0    -> 고금리 부채가 없다고 확인했다. 자격 없음.
        # `or`로 뭉치면 0이 falsy라 "없음"이 "모름"으로 바뀌어,
        # 6% 부채밖에 없는 사람에게 대환대출이 추천된다.
        if profile.high_rate_debt is not None:
            if profile.high_rate_debt <= 0:
                return False
        elif profile.existing_debt <= 0:
            return False

        # 정책이 요구하는 기존 금리 하한을 검사한다. 금리를 모르면 통과시킨다
        # (자격이 있는데 안 보여주는 것보다 보여주고 "확인 필요"를 붙이는 편이 낫다).
        floor = cand.get("대상_기존금리_하한")
        if floor is not None and profile.high_rate_debt is None:
            rate = profile.existing_debt_rate
            if rate is not None and rate < float(floor):
                return False

    if cand.effect == "상환연장" and profile.policy_loan_balance <= 0:
        return False

    # 채무 총액 상한 (새출발기금). 총채무 15억원을 넘으면 지원 대상이 아니다.
    #   FAQ Q26 — "지원 한도(총채무 15억원, 담보 10억원, 무담보 5억원) 이상 대출을
    #   보유하였거나 ... 지원 대상에 해당하지 않으며"
    # ⚠ 담보 10억 / 무담보 5억은 판정하지 않는다. 담보·무담보를 나눠 받지 않으므로
    #   구분할 수 없다. 총액만 걸어도 명백한 오추천은 막힌다.
    ceiling = cand.policy_raw.get("자격요건", {}).get("대상채무_총액상한")
    if ceiling is not None and profile.existing_debt > int(ceiling):
        return False

    # 긴급경영안정자금의 융자방식 판정.
    # 연매출 1억 400만원 미만 AND 업력 7년 미만이면 직접대출, 아니면 대리대출이다.
    # 후보를 둘로 나눠 두었으므로 자격 필터가 하나만 통과시킨다.
    rev_cap = cand.get("자격_연매출_상한")
    if rev_cap is not None:
        if ctx.annual_revenue is None or ctx.annual_revenue >= int(rev_cap):
            return False  # 연매출이 기준 이상 -> 대리대출 후보로 간다
        yr_cap = cand.get("자격_업력_상한_년")
        if yr_cap is not None:
            if ctx.business_years is None or ctx.business_years >= float(yr_cap):
                return False  # 업력을 모르거나 기준 이상 -> 대리대출로 (보수적)

    # 세금 체납 중이면 소진공 융자를 받을 수 없다
    # (공고 제2026-478호 4쪽 융자제한 소상공인 ① "세금을 체납중인 소상공인").
    # 예외("압류·매각 유예 시 직접대출에 한해 포함")는 미구현이다.
    if cand.is_loan and cand.get("배제_세금체납") and ctx.tax_delinquent:
        return False

    # 면적에 비례하는 보조금(점포철거비)은 면적을 모르면 금액을 계산할 수 없다.
    # 근거 없는 금액을 넣느니 후보에서 뺀다.
    if cand.get("단가_평당") is not None and (
        ctx.store_area_pyeong is None or ctx.store_area_pyeong <= 0
    ):
        return False

    return True


def _passes_extension_requirement(
    cand: Candidate, profile: UserProfile, ctx: UserContext
) -> bool:
    """상환연장제도의 배제 요건 (소진공 FAQ 80번).

        정상상환 중이거나 단기연체(30일 이하)  -> 가능
        연체 30일 초과                       -> 불가
        세금 체납                            -> 불가
        휴·폐업                              -> 불가 (자격요건.폐업여부로 이미 걸림)
        채무조정 이용 중                      -> 불가 (조합문법 채무조정.최대=1로 이미 걸림)

    ※ "타 금융기관의 연체사실이 있는 경우 연체 해소 후 이용 가능"도 원문에 있으나,
      우리 모델의 overdue_days는 대출을 구분하지 않으므로 함께 걸린다.

    """
    if cand.effect != "상환연장":
        return True

    ceiling = cand.get("연체일수_상한")
    if ceiling is not None and (profile.overdue_days or 0) > int(ceiling):
        return False

    if cand.get("배제_세금체납") and ctx.tax_delinquent:
        return False

    return True


def _passes_overdue_requirement(
    cand: Candidate, profile: UserProfile, ctx: UserContext
) -> bool:
    """연체 상태 요건 (새출발기금 부실차주 / 부실우려차주).

    두 후보는 **연체 일수로 갈린다. 겹치지 않는다.**
        부실차주      90일 이상
        부실우려차주   10~89일
                     또는 10일 미만이면서 폐업·6개월 이상 휴업·
                          세금 체납 공공기록 등재 등에 해당
                     (우리가 판정하는 대체 조건은 **폐업·세금체납** 둘이다.
                      "6개월 이상 휴업"은 휴업 개월 필드가 없어 판정하지 못한다)

    ※ 대체 조건은 **10일 미만일 때만** 적용된다. 90일 이상 연체 + 폐업이면
      부실차주에만 해당하고 부실우려차주가 아니다.
      (이전 구현은 폐업을 무조건 OR로 붙여 둘 다 통과시켰다)

    ※ "6개월 이상 휴업"은 UserContext에 휴업 개월 필드가 없어 판정하지 못한다.
      원문에 '등'이 붙어 열거가 완결적이지 않은 점도 감안해 MVP에서는 생략한다.

    TODO(adr): 이 판정이 없던 동안 부실차주가 **연체 0일인 사람에게도**
      추천되고 있었다. JSON에 `연체일수_하한` 필드가 있었는데 읽는 코드가
      없었다. 부실우려차주를 제거한 기간에는 후보가 하나뿐이라 드러나지 않았고,
      복원하면서 둘이 동시에 나와 발견됐다.
    """
    days = profile.overdue_days or 0

    # ① 상한 초과는 대체 조건으로도 구제되지 않는다.
    #    90일 이상 연체 + 폐업이면 부실차주에만 해당하고 부실우려차주가 아니다.
    ceiling = cand.get("연체일수_상한")
    if ceiling is not None and days > int(ceiling):
        return False

    # ② 하한 미만이면 대체 조건을 본다.
    #    부실우려차주는 10일 미만이라도 폐업·세금체납 등이면 자격이 있다.
    floor = cand.get("연체일수_하한")
    if floor is not None and days < int(floor):
        alt_below = cand.get("대체조건_적용_연체일수_미만")
        if alt_below is None or days >= int(alt_below):
            return False  # 대체 조건이 적용되는 구간이 아니다
        alts = cand.get("자격_대체조건") or []
        ok = ("폐업" in alts and ctx.is_closed) or (
            "세금체납" in alts and ctx.tax_delinquent
        )
        if not ok:
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
        if not _passes_debt_requirement(cand, profile, ctx):
            continue
        if not _passes_overdue_requirement(cand, profile, ctx):
            continue
        if not _passes_extension_requirement(cand, profile, ctx):
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
        대환·채무조정·상환연장은 유입이 없으므로 제외된다.
        """
        return sum(
            item.effective_amount
            for item in self.items
            if item.candidate.effect == "신규유입"
        )

    def has_loan(self) -> bool:
        return any(c.is_loan for c in self.candidates)


# TODO(measure): 융자 한도 초과 빈도를 재측정하고 "융자 최대 1개"를 확장할지 판단할 것.
#
#   왜 미뤘나
#     JSON `_조합문법.융자._MVP제약` 참조. 실제 규칙은 '융자방식별 1개'(대리 1 + 직접 1,
#     A-2 확인)이지만 MVP는 1개만 만든다. 확장하려면 B-9(서울시 자금끼리 중복) 답변과
#     융자방식 판정(loader가 후보에 `융자방식`을 넘겨야 함)이 선행되어야 한다.
#
#   시점
#     국면별 목표 개월수를 확정한 뒤, app.py 배포 전.
#     목표 개월수가 길어지면 한도 미달이 급증해 융자 구조가 바뀔 수 있고,
#     그러면 조합 수 -> 그룹 수 -> 레이어 구조까지 연쇄로 다시 재야 한다.
#     배포 후에 터지면 화면까지 되돌아가야 하므로 그 전에 끝낸다.
#
#   방법 (2026-08-12 1차 실측이 세 가지를 틀렸다. 반복하지 말 것)
#     1. 조합 단위로 잴 것. 융자 단독으로 재면 보조금·채무조정이 필요액을 줄여주는
#        효과가 빠져 미달이 과대평가된다.
#     2. 국면별 목표를 짝지을 것. 전 케이스에 12~36개월을 다 돌리면, 회복불가 국면에
#        36개월 목표를 걸고 "한도로도 못 채운다"고 세는 무의미한 결과가 나온다.
#     3. 두 축을 다 볼 것. 생존 개월수만으로 성패를 가르면 5억을 빌려 36개월을 버티는
#        조합이 '성공'으로 잡힌다. 순부채 축에서는 최악이다.
#
#   판단 기준
#     "융자 2개로 해결 가능한 미달"이 20%를 넘으면 확장한다.
#     한도가 남는데도 상환구조(만기일시·거치 종료) 때문에 무너지는 경우는 제외하고 셀 것.
#     융자를 둘로 나눠도 그 절벽은 사라지지 않는다(오히려 절벽이 둘이 된다).
#
#   1차 실측(2026-08-12) 참고값: 융자후보×목표 135건 중 미달 43건(31.9%).
#   다만 위 세 결함 때문에 위로 부풀려진 숫자이며, 그대로 인용하지 말 것.
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


def _restructure_loan_exclusive(db: PolicyDB) -> tuple[bool, frozenset[str]]:
    """채무조정 이행 중 신규 융자가 배타인지, 그리고 **어느 재원에** 적용되는지.

    Returns:
        (배타 여부, 적용 재원 집합)
        적용 재원이 비어 있으면 모든 재원에 적용한다.

    "확인필요" 문자열이면 아직 모르는 상태 -> 배타로 보지 않고 조합을 만든다.

    재원을 함께 돌려주는 이유: A-3 답변("채무조정 중 신규 정책자금 불가")은
    **소진공 기준**이다. 서울시 재기지원자금은 지원대상에 "새출발기금 채권매각기업"을
    명시하고 있어(서울시 공고 2026-3호 붙임3) 재원별로 다를 수 있다.
    배타를 모든 융자에 걸면 서울시 자금까지 잘못 배제한다.
    확정 전까지 JSON의 `_적용범위`에 적힌 재원에만 적용한다 (문의목록 B-10 / C-14).
    """
    rules = db.combo_rules.get("교차제약", {})
    enabled = rules.get("채무조정_융자_배타") is True
    scope = frozenset(rules.get("_적용범위") or ())
    return enabled, scope


def _pick_by_method(
    pool: Sequence[Candidate], method: str, limit: int, allow_cross_source: bool
) -> list[tuple[Candidate, ...]]:
    """한 지원방법 안에서 가능한 선택지를 모두 만든다.

    Returns:
        빈 튜플(선택 안 함)을 포함한 모든 조합.
        예: 융자 3개 중 최대 1개 -> [(), (A,), (B,), (C,)]
    """
    # is_extra(부가 추천 대상)는 조합에 넣지 않는다. 현금효과가 0이라 좌표가
    # 바뀌지 않으면서 조합 수만 배로 늘어난다. extra_candidates()가 따로 다룬다.
    items = [c for c in pool if c.method == method and not c.is_extra]
    options: list[tuple[Candidate, ...]] = [()]  # 아무것도 안 고르는 경우

    for size in range(1, limit + 1):
        for group in combinations(items, size):
            # 재원 간 중복이 금지면 서로 다른 재원이 섞인 조합은 만들지 않는다
            if not allow_cross_source and len({c.source for c in group}) > 1:
                continue
            options.append(group)
    return options


def grant_amounts(
    db: PolicyDB,
    pool: Sequence[Candidate],
    ctx: UserContext,
) -> dict[str, int]:
    """보조금 중 **금액이 사용자 입력에 따라 달라지는** 후보의 금액을 계산한다.

    대부분의 보조금은 `지급액`이 정액이라 이 함수가 필요 없다.
    점포철거비만 면적에 비례한다.

        점포철거비 = min(단가_평당 x 전용면적, 지급액_상한)
                  = min(200,000 x 평수, 6,000,000)

    30평 이상이면 상한 600만원, 그 미만이면 그보다 적다.
    면적을 모르면 이 dict에 넣지 않는다(자격 필터가 후보를 제외한다).

    반환값은 `build_combos(amounts=...)`에 융자 금액과 함께 합쳐서 넘긴다.
    """
    result: dict[str, int] = {}
    for cand in pool:
        unit = cand.get("단가_평당")
        if unit is None:
            continue  # 정액 보조금은 effective_amount가 `지급액`을 읽는다
        if ctx.store_area_pyeong is None or ctx.store_area_pyeong <= 0:
            continue  # 면적을 모르면 계산 불가. 자격 필터가 제외한다
        ceiling = cand.get("지급액_상한")
        amount = int(unit) * float(ctx.store_area_pyeong)
        if ceiling is not None:
            amount = min(amount, float(ceiling))
        result[cand.candidate_id] = int(amount)
    return result


def default_loan_amounts(
    db: PolicyDB,
    pool: Sequence[Candidate],
    profile: UserProfile,
    target_months: int,
    horizon: int = SIM_MONTHS,
) -> dict[str, int]:
    """융자 후보별 디폴트 금액을 필요액으로 역산한다 (기획서 5.4).

    ⚠ 이 함수는 융자 후보를 **하나씩** 역산한다. 같은 조합에 보조금이나 채무조정이
      함께 들어가는 것을 모르므로 융자 금액이 과다하게 나온다.
      조합을 만든 뒤 반드시 `rebalance_to_months()`로 다시 조정할 것.
      (금액 입력 경로의 `rebalance_to_inflow()`와 대칭이다)

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
        # 실제 신청 가능한 금액으로 맞춘다(100만원 단위 올림 · 최소 금액 적용)
        result[cand.candidate_id] = snap_amount(
            cand, required_loan_amount(profile, cand, target_months, horizon)
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
    
    ※ `is_extra` 후보(사업정리 컨설팅·법률자문 등 `_순위제외`)는 조합에 넣지 않는다.
      현금효과가 0이라 좌표가 바뀌지 않으면서 조합 수만 배로 늘어난다.
      이들은 자격 필터를 통과한 채로 남아 있다가 LLM 프롬프트에 **부가 추천**으로
      따로 전달된다(`extra_candidates()` 참조).
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

    exclusive, exclusive_sources = _restructure_loan_exclusive(db)

    combos: list[Combo] = []
    for loans in loan_options:
        # 배타 규칙이 걸리는 재원의 융자가 이 조합에 있는지
        blocked_loan = exclusive and any(
            not exclusive_sources or c.source in exclusive_sources for c in loans
        )
        for restructures in restructure_options:
            # 채무조정 ↔ 신규 융자가 배타면 둘이 함께 있는 조합은 만들지 않는다.
            # 단 배타는 `_적용범위`에 적힌 재원에만 걸린다(현재 소진공). 서울시 융자는
            # 재기지원자금 지원대상에 채무조정 이력자가 명시되어 있어 배제하지 않는다.
            if blocked_loan and restructures:
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
    TODO(adr): 이 결정의 배경은 기획서 10장-26. ADR로 옮길 대상.

    대신 무지원 시뮬레이션의 **구간 최저점**을 쓴다. 보조금은 상환이 없어
    유입이 곡선을 그대로 평행이동시키므로, 최저점이 0이 되면 구간 내내
    0 이상임이 보장된다.

    생존 개월수의 정의가 "잔고가 0이 되는 **첫** 시점"이므로 최저점을
    기준으로 삼는 것이 맞다. 3개월차 -1,000만 / 6개월차 -300만이면
    필요액은 1,000만원이다. 300만만 넣으면 3개월차에서 멈춘다.
    """
    baseline = simulate(profile, [], horizon=horizon)
    # balance_at은 0에서 잘리므로 "얼마가 모자랐는가"를 알 수 없다.
    # shortfall_at은 음수를 그대로 담는다.
    window = baseline.shortfall_at[: min(target_months, horizon) + 1]
    lowest = min(window)
    # 생존 판정이 `balance <= 0`이므로 최저점을 정확히 0으로 맞추면
    # 그 달에 생존이 끝나 목표를 1개월 못 채운다. 여유 1원을 더한다.
    return max(0, int(-lowest) + 1) if lowest <= 0 else 0


def snap_amount(cand: Candidate, amount: int, unit: int = 1_000_000) -> int:
    """계산된 금액을 **실제로 신청할 수 있는 값**으로 맞춘다.

    역산은 `28,841,810원` 같은 값을 낸다. 소진공 융자는 모두 100만원 단위로
    신청하므로(A-42 확인) 그대로 보여줄 수 없다.

        올림      2,884만 -> 2,900만.
                 **내림하면 목표 개월수에 미달한다.** 반드시 올린다.
        최소      필요액이 `금액_하한`보다 작으면 하한으로 올려 빌린다.
                 (제외가 아니다. 신용취약자금은 최소 1,000만원이므로
                  300만이 필요해도 1,000만을 빌린다)
        한도      한도를 넘으면 한도까지만. 미달 사실은 생존 개월수에 드러난다.
        불가      하한이 한도를 넘으면 0을 돌려준다. 신청 자체가 불가능하다.

    단위는 재원마다 다르다. 소진공은 100만원, 서울시는 만원이다(둘 다 확인됨).
    `unit` 인자는 후보에 `신청금액_단위`가 **없을 때의 폴백**이며, 현재 모든
    융자 후보가 단위를 갖고 있어 도달하지 않는다.
    """
    if amount <= 0:
        return 0
    step = int(cand.get("신청금액_단위") or unit)
    ceiling = int(cand.get("금액_한도", 0) or 0)
    floor = cand.get("금액_하한")

    amount = -(-amount // step) * step  # 올림
    if floor is not None and amount < int(floor):
        amount = int(floor)
    if ceiling:
        if floor is not None and int(floor) > ceiling:
            return 0  # 최소 금액이 한도를 넘는다
        amount = min(amount, ceiling)
    return amount


def required_loan_amount(
    profile: UserProfile,
    cand: Candidate,
    target_months: int,
    horizon: int = SIM_MONTHS,
    base: Sequence[Applied] = (),
) -> int:
    """융자로 목표 개월수를 채우려면 얼마를 빌려야 하는지 (1차식 외삽).

    목표 시점까지의 잔고 최저점은 대출 원금의 **1차 함수**다. 원금은 t=0에
    한 번 들어오고, 이자와 원금 상환은 원금에 비례해 나가기 때문이다.
    그래서 점 두 개로 기울기를 구하면 해가 바로 나온다.

        A0 = 무지원 최저점만큼           -> 최저점 L0 (아직 음수)
        A1 = A0 + 부족분                 -> 최저점 L1 (여전히 조금 음수)
        기울기 = (L1 - L0) / (A1 - A0)   -> 원금 1원당 잔고 상승분 (예: 0.84)
        A  = A1 + (MARGIN - L1) / 기울기 -> 잔고를 MARGIN으로 만드는 원금

    TODO(adr): 이전 구현(부족분을 더해 반복)을 버린 이유. 두 가지 결함이 있었다.
      1) NEED_TOLERANCE(1만원) 이내면 수렴으로 보고 검증 없이 종료했다.
         부족분을 더해도 이자가 같이 늘어 여전히 미달인 채로 빠져나갔다.
      2) 부족분이 매 회차 약 1/6로 줄어드는 기하급수 수렴이라 0에 점근할 뿐
         양수가 되지 않는데, 종료 조건이 `lowest > 0`이었다. 반복을 아무리
         돌려도 목표 달성 판정이 나지 않았다.
      경계에서 생존 개월수가 1개월씩 깎여, 목표 충족 조합 비율이 실측에서
      35% vs 71%로 갈릴 만큼 영향이 컸다.

    ※ 대환전환은 예외다. 유입이 없어 '필요액'이 정의되지 않으므로
      전환 가능한 고금리 부채 금액을 그대로 쓴다.

    Args:
        base: **이 융자와 함께 적용될 다른 항목들**(보조금·채무조정).
            비워두면 융자 단독으로 역산한다.
            채우면 그 효과를 먼저 반영한 뒤 부족분만 융자로 채운다 — 보조금이
            200만원 들어오거나 채무조정으로 원금이 감면되면 그만큼 덜 빌린다.
            `combos.rebalance_to_months()`가 조합마다 이 인자를 채워 호출한다.
    """
    ceiling = int(cand.get("금액_한도", 0) or 0)
    if ceiling <= 0:
        return 0

    # 대환전환은 '얼마가 필요한가'가 아니라 '얼마를 갈아탈 수 있는가'가 기준이다.
    if cand.effect == "대환전환":
        # None(모름)이면 기존 부채 전액, 0(없음)이면 0. `or`로 뭉치지 않는다.
        convertible = (
            profile.high_rate_debt
            if profile.high_rate_debt is not None
            else profile.existing_debt
        )
        return min(ceiling, int(convertible))

    def lowest_with(amount: int) -> float:
        """그 금액을 빌렸을 때 목표 구간의 잔고 최저점."""
        result = simulate(
            profile, [*base, Applied(cand, amount)], horizon=horizon
        )
        # balance_at이 아니라 shortfall_at을 본다 (위와 같은 이유)
        return min(result.shortfall_at[: min(target_months, horizon) + 1])

    # 융자 없이 base(보조금·채무조정)만 적용했을 때 이미 목표를 채우면 빌릴 필요가 없다
    if base:
        without = simulate(profile, list(base), horizon=horizon)
        if min(without.shortfall_at[: min(target_months, horizon) + 1]) >= NEED_MARGIN:
            return 0
        a0 = min(
            int(NEED_MARGIN - min(without.shortfall_at[: min(target_months, horizon) + 1])),
            ceiling,
        )
    else:
        a0 = min(required_amount(profile, target_months, horizon), ceiling)
    if a0 <= 0:
        return 0  # 무지원으로도 목표를 채운다
    l0 = lowest_with(a0)
    if l0 >= NEED_MARGIN:
        return a0

    a1 = min(a0 + int(NEED_MARGIN - l0), ceiling)
    if a1 >= ceiling:
        return ceiling  # 한도로도 모자란다. 미달 사실이 생존 개월수에 그대로 드러난다.
    l1 = lowest_with(a1)

    slope = (l1 - l0) / (a1 - a0)
    if slope <= 0:
        # 원금을 늘려도 잔고가 안 오른다(이자가 유입을 상쇄). 한도까지 가도 소용없다.
        return ceiling

    amount = a1
    for _ in range(NEED_ITERATIONS):
        low = lowest_with(amount)
        if low >= NEED_MARGIN:
            break
        amount = min(amount + int((NEED_MARGIN - low) / slope) + 1, ceiling)
        if amount >= ceiling:
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
    def debt_at_exit(self) -> float:
        """축 2 — 작을수록 좋음."""
        return self.result.debt_at_exit

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


def fixed_amounts(
    profile: UserProfile,
    pool: Sequence[Candidate],
    horizon: int = SIM_MONTHS,
) -> dict[str, int]:
    """**금액을 제도가 정하는** 후보들의 금액.

    사용자가 목표 금액이나 목표 개월수를 무엇으로 주든 이 값은 달라지지 않는다.

        대환전환   갈아탈 수 있는 고금리 부채 잔액 (한도 이내)
        그 외      `Applied.effective_amount`가 JSON에서 읽는다
                   (보조금 정액 · 채무조정은 금액 개념 없음)

    금액 입력 경로에서 `build_combos(amounts=...)`에 융자 금액을 0으로 넘길 때,
    대환대출까지 0이 되면 갈아탈 것이 없어져 효과가 사라진다. 이 함수가 채운다.
    """
    result: dict[str, int] = {}
    for cand in pool:
        if cand.effect == "대환전환":
            result[cand.candidate_id] = required_loan_amount(
                profile, cand, target_months=horizon, horizon=horizon
            )
    return result


def rebalance_to_inflow(
    combos: Sequence[Combo],
    target_inflow: int,
    unit: int = 1_000_000,
) -> list[Combo]:
    """**금액 입력 경로** — 목표 금액에 맞춰 조합별 융자액을 다시 정한다.

    사용자가 "3,000만원이 필요하다"고 하면 그 3,000만원을 **어떻게 조달할지**가
    문제다. 보조금은 공짜 돈이므로 받을 수 있는 만큼 먼저 받고, 부족분만 빌린다.

        보조금 200만 + 융자 2,800만 = 3,000만
        보조금 없음  + 융자 3,000만 = 3,000만

    → **모든 조합의 총유입이 목표 금액으로 같아진다.**
      그러면 조합 간 차이가 순수하게 '상환 구조'(금리·거치·분할)에서만 나온다.
      목표 개월수 경로에서는 후보마다 필요액이 달라(2,818~3,440만) 그 편차가
      부채 축에 섞이는데, 금액 경로에서는 그 문제가 없다.

    Args:
        target_inflow: 사용자가 입력한 목표 금액(원)
        unit: 신청 금액 단위. 소진공은 100만원(A-42 확인).
            부족분을 이 단위로 **올림**한다 — 내림하면 목표에 미달한다.

    Returns:
        융자 금액이 조정된 새 Combo 목록. 융자가 없는 조합은 그대로 둔다.

    제약 처리
        - 부족분이 후보의 `금액_하한`보다 작으면 하한으로 **올려** 빌린다.
          (제외가 아니다. 최소 1,000만원인 자금에 300만이 필요하면 1,000만을 빌린다)
        - 하한이 한도를 넘으면 그 조합을 버린다.
        - 부족분이 한도를 넘으면 한도까지만 빌린다(목표 미달로 남는다).
        - 보조금만으로 목표를 채우면 융자는 0원이 되는데, 그 조합은 융자를
          빼는 것이 맞으므로 버린다(같은 조합이 보조금 단독으로 이미 존재한다).
    """
    result: list[Combo] = []
    for combo in combos:
        # **`is_loan`이 아니라 `신규유입`으로 고른다.** 대환대출은 융자지만 새 돈이
        # 들어오지 않는다(기존 고금리 부채를 갈아타는 것이다). 여기에 목표 금액을
        # 꽂으면 "3,000만원을 빌렸는데 유입은 0원"인 조합이 만들어진다.
        # 목표 금액을 채우는 데 쓸 수 있는 것은 신규유입뿐이다.
        funds = [it for it in combo.items if it.candidate.effect == "신규유입"]
        loans = [it for it in funds if it.candidate.is_loan]
        if not loans:
            # 융자가 없으면 조정할 것이 없다. 보조금 정액과 대환·채무조정처럼
            # 제도가 금액을 정하는 항목만 남은 조합이다.
            result.append(combo)
            continue
        if len(loans) > 1:
            # 융자 2개는 배분 최적해가 없다(JSON `_MVP제약` 참조).
            # 현재 조합 문법이 융자 1개로 제한하므로 여기 오지 않는다.
            raise NotImplementedError("융자 2개 이상의 금액 배분은 미구현")

        loan = loans[0]
        # 융자를 뺀 나머지 유입(보조금)을 먼저 센다
        granted = sum(
            it.effective_amount for it in funds if it is not loan
        )
        short = target_inflow - granted
        if short <= 0:
            continue  # 보조금만으로 충분하다. 융자 없는 조합이 따로 있다

        amount = snap_amount(loan.candidate, short, unit)
        if amount <= 0:
            continue  # 최소 금액이 한도를 넘는다. 신청 자체가 불가능

        items = tuple(
            Applied(it.candidate, amount) if it is loan else it
            for it in combo.items
        )
        result.append(Combo(items=items))
    return result


def resolve_by_both(
    db: PolicyDB,
    pool: Sequence[Candidate],
    profile: UserProfile,
    ctx: UserContext,
    target_inflow: int,
    target_months: int,
    unit: int = 1_000_000,
    horizon: int = SIM_MONTHS,
) -> tuple[list[Combo], bool, int | None]:
    """**금액과 목표 개월수를 둘 다 입력했을 때.**

    사용자가 정한 금액을 존중하되, 그 금액으로 목표를 못 채우면 알려준다.

        1단계  금액 기준으로 조합을 만든다 (`rebalance_to_inflow`)
        2단계  그중 목표 개월수를 채우는 조합이 있으면 -> 그대로 쓴다
        3단계  하나도 없으면 -> 목표 개월수 기준으로 다시 만들고,
               "목표를 채우려면 얼마가 필요한지"를 함께 돌려준다

    3단계에서 금액을 올려버리는 것이 아니라 **두 가지를 다 보여주기 위한 정보**다.
    사용자가 금액을 정한 데는 이유가 있다(신용 한도, 상환 부담 등).
    화면과 LLM 프롬프트에서 이렇게 쓴다.

        "3,000만원으로는 12개월까지입니다.
         18개월을 채우려면 3,800만원이 필요합니다."

    Returns:
        (조합 목록, 금액으로 목표 충족 여부, 목표를 채우는 데 필요한 금액)

        충족이면 (금액 기준 조합, True, target_inflow)
        미달이면 (목표 개월수 기준 조합, False, 필요 금액 또는 None)
        None은 목표 개월수 기준으로도 못 채운다는 뜻이다(한도 초과 등).
    """
    grants = grant_amounts(db, pool, ctx)
    zero = {c.candidate_id: 0 for c in pool if c.is_loan}
    # 대환전환은 갈아탈 금액이 따로 정해진다. 0으로 두면 효과가 사라진다.
    base = build_combos(
        db, pool, {**zero, **fixed_amounts(profile, pool, horizon), **grants}
    )

    # ── 1~2단계: 사용자가 정한 금액으로 ──
    by_inflow = rebalance_to_inflow(base, target_inflow, unit)
    if any(
        simulate(profile, c.items, horizon=horizon).survival_months >= target_months
        for c in by_inflow
    ):
        return by_inflow, True, target_inflow

    # ── 3단계: 목표 개월수 기준으로 다시. 필요 금액을 함께 구한다 ──
    amounts = {
        c.candidate_id: required_loan_amount(profile, c, target_months, horizon)
        for c in pool
        if c.is_loan
    }
    amounts.update(grants)
    by_months = build_combos(db, pool, amounts)

    need = min(
        (
            c.total_inflow
            for c in by_months
            if simulate(profile, c.items, horizon=horizon).survival_months
            >= target_months
        ),
        default=None,
    )
    return by_months, False, need


def rebalance_to_months(
    combos: Sequence[Combo],
    profile: UserProfile,
    target_months: int,
    horizon: int = SIM_MONTHS,
) -> list[Combo]:
    """**목표 개월수 경로** — 조합별로 융자액을 다시 역산한다.

    `default_loan_amounts()`는 융자 후보를 **하나씩** 역산한다. 조합은 그 뒤에
    만들어지므로, 같은 조합에 보조금이나 채무조정이 함께 들어가도 융자 금액이
    줄지 않는다.

        후보 단위   "이 융자 단독으로 18개월 버티려면 2,884만원"
        조합        "이 융자 + 새출발기금(원금 60% 감면)"
                   -> 훨씬 적게 빌려도 되는데 2,884만원을 그대로 쓴다

    과다 차입이 부채 축에 그대로 반영된다. 보조금은 200~600만원이라 영향이 작지만
    채무조정은 원금 감면이 수천만원이라 크다.

    이 함수가 조합마다 **융자 외 항목을 먼저 적용한 상태에서** 필요액을 다시 구한다.
    금액 입력 경로의 `rebalance_to_inflow()`와 대칭이다.

    Returns:
        융자 금액이 조정된 새 Combo 목록. 융자가 없는 조합은 그대로 둔다.
        보조금·채무조정만으로 목표를 채워 융자가 0원이 되면 그 조합은 버린다
        (융자 없는 같은 조합이 이미 따로 존재한다).
    """
    result: list[Combo] = []
    for combo in combos:
        # 역산 대상은 **신규유입 융자**뿐이다. 대환전환은 갈아탈 금액이 기존 부채로
        # 정해져 있고, 채무조정·비용감면도 제도가 금액을 정한다.
        loans = [
            it for it in combo.items
            if it.candidate.is_loan and it.candidate.effect == "신규유입"
        ]
        if not loans:
            result.append(combo)
            continue
        if len(loans) > 1:
            raise NotImplementedError("융자 2개 이상의 금액 배분은 미구현")

        loan = loans[0]
        others = tuple(it for it in combo.items if it is not loan)
        amount = snap_amount(
            loan.candidate,
            required_loan_amount(
                profile, loan.candidate, target_months, horizon, base=others
            ),
        )
        if amount <= 0:
            continue  # 나머지만으로 목표를 채운다. 융자 없는 조합이 따로 있다

        result.append(Combo(items=tuple(
            Applied(it.candidate, amount) if it is loan else it
            for it in combo.items
        )))
    return result


def extra_candidates(pool: Sequence[Candidate]) -> list[Candidate]:
    """조합에는 넣지 않지만 **LLM에 안내할** 후보들.

    현금효과가 0이라 2축에 반영되지 않는 것들이다(사업정리 컨설팅, 법률자문,
    재기사업화 등 `_순위제외`). 조합에 넣으면 좌표는 그대로인데 조합 수만
    배로 늘어나므로 계산에서 빼되, **자격 필터는 이미 통과한 상태**이므로
    국면과 자격에 맞는 것만 남아 있다.

    LLM이 최종 3개를 고른 뒤 "이런 것도 함께 신청해 보세요"로 덧붙이게 한다.
    """
    return [c for c in pool if c.is_extra]


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


def summarize(evaluated: Sequence[Evaluated], top: int = 5) -> str:
    """조합 목록을 **두 축 각각으로** 훑어본다. 디버깅용.

    한 가지 정렬로 줄세우지 않는다. `sorted(key=(-생존, 순부채))`처럼 쓰면
    생존을 1순위 축으로 삼는 것이 되어, "두 축은 동등하게 취급한다"는
    설계 원칙(기획서 5.1)이 정렬 순서를 통해 깨진다.

    실제 후보 선정은 optimizer.py가 축별 상위 N의 합집합으로 한다.
    여기서는 두 축을 나란히 보여주기만 한다.
    """
    MAN = 10_000
    by_survival = sorted(evaluated, key=lambda e: -e.survival_months)[:top]
    by_debt = sorted(evaluated, key=lambda e: e.debt_at_exit)[:top]

    lines = [f"전체 {len(evaluated)}개 조합 · 축별 상위 {top}개 (순위가 아니라 축별 관점)"]

    lines.append(f"\n[축 1] 생존 개월수 — 클수록 좋음")
    for ev in by_survival:
        lines.append(
            f"  {ev.result.survival_label:>7} · 부채 {ev.debt_at_exit / MAN:>7,.0f}만  {ev.label}"
        )

    lines.append(f"\n[축 2] 30개월 후 순부채 — 작을수록 좋음")
    for ev in by_debt:
        lines.append(
            f"  {ev.debt_at_exit / MAN:>7,.0f}만 · 생존 {ev.result.survival_label:>7}  {ev.label}"
        )

    # 두 축에 모두 든 조합은 어느 쪽으로 봐도 상위다
    both = [e for e in by_survival if e in by_debt]
    if both:
        lines.append(f"\n[두 축 모두 상위] {len(both)}개")
        for ev in both:
            lines.append(f"  {ev.label}")

    return "\n".join(lines)
