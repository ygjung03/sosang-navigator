"""
후보 선정 — 조합을 LLM에 넘길 5~10개로 좁힌다.

파이프라인에서 이 모듈의 자리:

    combos.evaluate()   조합 수십~수백 개 + 2축 값
    metrics.detect_all()  조합별 함정
        │
        │  ① group_by_outcome()   (생존, 순부채) 좌표가 같으면 한 그룹
        │  ② select()             비지배 정렬로 층을 벗겨 5개 이상 모을 때까지
        │  ③ prune_members()      그룹 안의 명백한 열등만 제거
        ▼
    후보 그룹 3~8개  →  LLM 에이전트가 최종 3개 선정


코드는 순위를 매기지 않는다
---------------------------
두 축(생존 개월수·30개월 순부채)은 동등하게 취급하므로, 어느 축을 우선할지
코드가 미리 정하지 않는다(기획서 5.1). 이 모듈이 하는 일은 **줄세우기가 아니라
좁히기**다. 각 후보에 두 축 값·레이어·함정을 붙여 넘기고, 최종 선정은
LLM이 국면 맥락으로 한다.


TODO(adr): 아래 "왜 축별 상위 5 합집합을 쓰지 않는가"는 결정의 역사이므로
           docs/adr/ 로 옮기고 여기에는 채택한 방식과 링크만 남긴다.

왜 축별 상위 5 합집합을 쓰지 않는가
-----------------------------------
기획서 5.2는 "두 축 각각 독립 순위 → 축별 상위 5의 합집합(5~10개)"으로
정했으나, 구현 후 실측하니 그 범위가 지켜지지 않았다.

  · 생존 개월수는 **이산값**이라 동점이 대량으로 발생한다. 개선여지 국면에서
    5위 경계가 12개월인데 그 값에 동점이 14개 있어, 모두 넣으면 18개가 된다.
  · 동점을 다른 축으로 자르면 그 축에 사실상 가중치를 주는 셈이라
    "두 축 동등" 원칙이 정렬 순서를 통해 깨진다.
  · 1차 파레토 프론티어만 뽑으면 반대로 1개까지 줄어든다. 만기일시상환 조합이
    두 축 모두에서 최고이기 때문인데, 24개월차 낭떠러지가 30개월 단일 시점
    측정에 잡히지 않는 2축의 맹점이다.

채택한 방식: **좌표 그룹핑 + 비지배 정렬(non-dominated sorting)**

  좌표가 같으면(생존·순부채가 동일) 2축 관점에서 성능이 같으므로 한 그룹으로
  묶는다. 그 뒤 1차 프론티어를 뽑고 제거, 남은 것에서 2차 프론티어를 뽑는 식으로
  필요한 개수가 모일 때까지 층을 벗겨낸다. 다목적 최적화의 표준 기법이며
  임의 상수가 들어가지 않는다.

  6개 케이스로 검증한 결과 후보 그룹이 3~8개로 안정되었다.


명명 규칙: 코드 식별자는 영어, JSON 키는 한글. 시점은 _at, 기간은 _during.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .cashflow import month_display
from .combos import Evaluated
from .metrics import TrapReport

# 후보를 이만큼 모을 때까지 층을 벗긴다. 기획서 5.2의 확정값(5~10개)에서 왔다.
MIN_CANDIDATES = 5

# 부채를 그룹핑할 때 쓰는 단위(원). 만원 단위로 같으면 같은 좌표로 본다.
# 1원 차이로 그룹이 갈리는 것을 막기 위한 반올림이며, 판정 결과를 좌우하지 않는다.
DEBT_BUCKET = 10_000

MAN = 10_000  # 만원. 출력 포맷용


# ══════════════════════════════════════════════════════════
# 후보 그룹
# ══════════════════════════════════════════════════════════


@dataclass
class CandidateGroup:
    """2축 성능이 동일한 조합들의 묶음.

    좌표(생존 개월수, 종료 시점 부채)가 같으면 2축으로는 구별할 수 없으므로
    코드가 대표를 고르지 않는다. 그룹째로 LLM에 넘기고 그 안에서
    고르게 한다 — 신청 부담, 개인 사정 등 코드가 모르는 정보가 판단 재료다.
    """

    survival_months: int
    """축 1 — 이 그룹의 생존 개월수. 클수록 좋음."""

    debt_at_exit: float
    """축 2 — 이 그룹의 **운영 종료 시점 부채 잔액**. 작을수록 좋음.

    종료 = min(파산 시점, 36개월). "36개월 후 부채"가 아니다.
    """

    members: list[Evaluated] = field(default_factory=list)
    """이 성능을 내는 조합들. prune 후에는 명백한 열등이 빠진 상태."""

    dropped: list[Evaluated] = field(default_factory=list)
    """명백한 열등으로 제외된 조합들. 개수만 화면에 안내하는 용도."""

    layer: int = 0
    """비지배 정렬 레이어. 1이 최상위 프론티어."""

    survival_rank: int = 0
    """생존 축에서 몇 번째 값인지(1-based). 동점 그룹은 같은 순위."""

    debt_rank: int = 0
    """순부채 축에서 몇 번째 값인지(1-based)."""

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def has_trap(self) -> bool:
        """남은 멤버 중 하나라도 함정이 있는지."""
        return any(getattr(m, "_trap_flag", False) for m in self.members)

    def label(self, index: int = 0) -> str:
        return self.members[index].label if self.members else ""


# ══════════════════════════════════════════════════════════
# ① 좌표 그룹핑
# ══════════════════════════════════════════════════════════


def group_by_outcome(evaluated: Sequence[Evaluated]) -> list[CandidateGroup]:
    """(생존 개월수, 순부채)가 같은 조합을 한 그룹으로 묶는다.

    같은 좌표라는 것은 2축으로는 구별이 불가능하다는 뜻이다.
    """
    buckets: dict[tuple[int, int], CandidateGroup] = {}
    for ev in evaluated:
        key = (ev.survival_months, round(ev.debt_at_exit / DEBT_BUCKET))
        group = buckets.get(key)
        if group is None:
            group = CandidateGroup(
                survival_months=ev.survival_months,
                debt_at_exit=key[1] * DEBT_BUCKET,
            )
            buckets[key] = group
        group.members.append(ev)
    return list(buckets.values())


# ══════════════════════════════════════════════════════════
# ② 비지배 정렬
# ══════════════════════════════════════════════════════════


def _dominates(a: CandidateGroup, b: CandidateGroup) -> bool:
    """a가 b를 지배하는가.

    두 축 모두에서 나쁘지 않고, 적어도 한 축에서 낫다면 지배한다.
    생존은 클수록, 순부채는 작을수록 좋다.
    """
    not_worse = (
        a.survival_months >= b.survival_months
        and a.debt_at_exit <= b.debt_at_exit
    )
    strictly_better = (
        a.survival_months > b.survival_months
        or a.debt_at_exit < b.debt_at_exit
    )
    return not_worse and strictly_better


def non_dominated_sort(
    groups: Sequence[CandidateGroup], need: int = MIN_CANDIDATES
) -> list[CandidateGroup]:
    """비지배 정렬로 층을 벗겨 need개 이상 모을 때까지 뽑는다.

        1차 프론티어 = 아무에게도 지배당하지 않는 그룹  -> 뽑고 제거
        2차 프론티어 = 남은 것 중 비지배                -> 뽑고 제거
        ...

    파레토 1차만 쓰면 후보가 1개까지 줄어들 수 있고(두 축 모두에서 최고인
    조합이 존재하는 경우), 축별 상위 N은 동점 때문에 범위를 못 지킨다.
    층을 벗기면 임의 상수 없이 원하는 규모를 얻는다.

    한 층을 통째로 넣으므로 need를 조금 넘길 수 있다. 잘라내면 그 자리에
    임의 기준이 필요해지므로 넘치는 쪽을 택한다.
    """
    rest = list(groups)
    selected: list[CandidateGroup] = []
    layer = 0

    while rest and len(selected) < need:
        layer += 1
        front = [a for a in rest if not any(_dominates(b, a) for b in rest)]
        if not front:  # 방어: 순환 지배는 없지만 빈 층이면 무한 루프가 된다
            break
        for group in front:
            group.layer = layer
        selected.extend(front)
        rest = [g for g in rest if g not in front]

    return selected


# ══════════════════════════════════════════════════════════
# ③ 그룹 내 명백한 열등 제거
# ══════════════════════════════════════════════════════════


def prune_members(
    group: CandidateGroup, reports: dict[str, TrapReport]
) -> CandidateGroup:
    """성능이 같은 멤버 중 명백히 열등한 것만 제거한다.

    기준 두 개 모두 불리언 판정과 개수 세기여서 임의 상수가 없다.

      ① 함정이 없는 멤버가 있으면, 함정이 있는 멤버를 뺀다.
         "같은 결과인데 위험만 더 있다"는 명백한 열등이다.
         ※ 기획서 5.3의 "함정이어도 제거하지 않는다"는 **성능이 다른** 조합에
           대한 원칙이므로 충돌하지 않는다. 여기서 빼는 것은 성능이 완전히
           같은 대안 중 위험한 쪽이다.
         ※ 멤버 전부에 함정이 있으면 전부 남긴다. 다 빼면 그룹이 사라지고,
           회복불가 국면에서는 융자 조합이 통째로 안 보이게 된다.

      ② 그래도 여럿이면 구성 정책 수가 가장 적은 것만 남긴다.
         같은 결과를 얻는 데 2개를 신청하는 것과 3개를 신청하는 것은 다르다.
         신청 부담은 소상공인에게 실질적 비용이며, "자격이 있어도 신청까지
         이르지 못한다"는 이 서비스의 문제의식과도 맞는다.
    """
    if len(group.members) <= 1:
        return group

    original = list(group.members)

    # ① 함정 없는 것 우선
    clean = [m for m in original if not reports[m.label].has_trap]
    pool = clean if clean else original

    # ② 구성 정책 수가 가장 적은 것
    fewest = min(len(m.combo.items) for m in pool)
    kept = [m for m in pool if len(m.combo.items) == fewest]

    group.members = kept
    group.dropped = [m for m in original if m not in kept]
    return group


# ══════════════════════════════════════════════════════════
# 통합
# ══════════════════════════════════════════════════════════


def _assign_ranks(groups: Sequence[CandidateGroup]) -> None:
    """축별 순위를 매긴다. 동점 그룹은 같은 순위(경쟁 순위)."""
    for attr, key, reverse in (
        ("survival_rank", lambda g: g.survival_months, True),
        ("debt_rank", lambda g: g.debt_at_exit, False),
    ):
        ordered = sorted(groups, key=key, reverse=reverse)
        rank, prev = 0, object()
        for i, g in enumerate(ordered, start=1):
            value = key(g)
            if value != prev:
                rank, prev = i, value
            setattr(g, attr, rank)


def select(
    evaluated: Sequence[Evaluated],
    reports: dict[str, TrapReport],
    need: int = MIN_CANDIDATES,
) -> list[CandidateGroup]:
    """조합 전체에서 LLM에 넘길 후보 그룹을 고른다.

    Args:
        evaluated: combos.evaluate() 결과
        reports:   metrics.detect_all() 결과 (조합 label -> TrapReport)
        need:      최소 그룹 수

    Returns:
        후보 그룹 목록. 레이어 오름차순으로 정렬되어 있다.
        **순위가 아니다** — 레이어 1 안에서도 어느 것이 나은지는 국면에 달렸다.
    """
    if not evaluated:
        return []

    groups = group_by_outcome(evaluated)
    selected = non_dominated_sort(groups, need=need)
    for group in selected:
        prune_members(group, reports)
        # 화면·프롬프트에서 함정 아이콘을 붙이기 위한 표식
        for member in group.members:
            setattr(member, "_trap_flag", reports[member.label].has_trap)
    _assign_ranks(selected)

    return sorted(
        selected, key=lambda g: (g.layer, -g.survival_months, g.debt_at_exit)
    )


# ══════════════════════════════════════════════════════════
# LLM 전달용 직렬화
# ══════════════════════════════════════════════════════════


def prerequisites(items: Sequence[Any]) -> list[str]:
    """신청 전에 해야 할 일 — **최종 추천에 뽑힌 조합에 대해서만** 표시한다.

    `_확인필요_자격_지금가능: true`인 후보의 선행요건이다.
    "재기사업화에 선정되었는가" 같은 것은 지금 바꿀 수 없어 자격 판정으로 끝나지만,
    "신용관리 교육 이수"는 사용자가 지금 하면 되는 일이라 자격에서 빼지 않고
    **안내로 남긴다.**

    `apply_links()` 와 같은 자리에서 쓴다 — LLM이 최종 3개를 고른 뒤
    `app.py` 가 그 조합에 대해 호출해 화면에 띄운다.
    후보 그룹 전체에 붙이면 프롬프트만 길어지고 순위 판단에 도움이 되지 않는다.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for item in items:
        cand = item.candidate
        if not cand.get("_확인필요_자격_지금가능"):
            continue
        note = cand.get("_선행요건_안내") or cand.get("_확인필요_자격")
        if note and note not in seen:
            seen.add(note)
            lines.append(f"{cand} — {note}")
    return lines


def apply_links(items: Sequence[Any]) -> list[str]:
    """신청처 — 정책별 신청 링크와 문의처.

    **LLM 프롬프트에는 넣지 않는다.** 링크는 순위를 매기는 데 도움이 되지 않고,
    후보 그룹 5~8개에 모두 붙으면 프롬프트만 길어진다.
    LLM이 최종 3개를 고른 뒤 `app.py`가 그 조합에 대해 호출해 화면에 표시한다.

    같은 정책의 후보가 여럿이면 한 번만 나온다.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for item in items:
        cand = item.candidate
        if cand.policy_id in seen:
            continue
        seen.add(cand.policy_id)
        link = cand.policy_raw.get("신청링크")
        contact = cand.policy_raw.get("문의처")
        if not (link or contact):
            continue
        parts = [cand.policy_name]
        if link:
            parts.append(link)
        if contact:
            parts.append(f"문의 {contact}")
        lines.append(" · ".join(parts))
    return lines


def _meets_revenue_rule(cand: Any, context: Any) -> bool:
    """후보의 매출 요건을 충족하는지. 택1요건 질문을 걸러내는 데만 쓴다.

    `combos._passes_revenue_requirement()` 와 달리 `매출_요건_택1대체` 를 무시하고
    **실제 매출 수치만** 본다. 그 플래그는 "미달해도 후보에서 빼지 말라"는 뜻이지
    "요건을 충족했다"는 뜻이 아니기 때문이다.
    """
    if context is None:
        return False
    rate = cand.get("매출_기준율") or cand.eligibility.get("매출_기준율")
    if rate is None:
        return True
    windows = cand.raw.get("매출_비교단위") or cand.eligibility.get("매출_비교단위")
    by_window = getattr(context, "revenue_drop_by_window", None) or {}
    if windows and by_window:
        rates = [by_window[w] for w in windows if w in by_window]
        if rates:
            return max(rates) >= float(rate)
    return getattr(context, "revenue_drop_rate", 0.0) >= float(rate)


def _caveats(items: Sequence[Any], context: Any = None) -> list[str]:
    """조합에 붙일 주의 문구. 성격이 다른 세 종류를 모은다.

    같은 문구가 여러 후보에 붙어 있을 수 있으므로 중복을 제거한다
    (예: 서울시 융자 여러 후보가 같은 변동금리 안내를 갖는다).

    ① `안내문구` — 제도가 정한 사실
        "실제 대출 금액은 은행 심사 결과에 따라 달라질 수 있습니다."
        우리 시뮬레이션은 "역산한 금액이 그대로 실행된다"를 전제하는데,
        소진공 통화(2026-08-13)에서 "한도 금액으로 신청하면 은행에서
        알아서 조절한다"는 답을 받았다. 전제가 깨질 수 있음을 알려야 한다.

    ② 변동금리 — 우리 계산의 한계
        `금리_고정여부: false`인 후보는 실제로 금리가 바뀐다.
        소진공 정책자금은 기준금리가 분기마다 고시된다(1·4·7·10월 10일).
        변동을 예측할 수 없어 현재 금리로 36개월을 고정 계산한다.

    ③ `_계산전제` — 후보별 사정
        예: 새출발기금 부실차주의 감면 대상(순부채·신용대출만) 미반영.
    """
    notes: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        if text and text not in seen:
            seen.add(text)
            notes.append(text)

    for item in items:
        cand = item.candidate
        for line in cand.get("안내문구") or ():
            add(line)
        if cand.is_loan and cand.get("금리_고정여부") is False:
            rate = cand.get("금리_연")
            add(
                f"변동금리입니다. 현재 금리 {rate:.2%}로 계산했으나 "
                f"기준금리가 분기마다 바뀌어 실제 상환액이 달라질 수 있습니다."
                if rate is not None
                else "변동금리입니다. 현재 금리로 계산했으나 실제 상환액이 달라질 수 있습니다."
            )
        add(cand.get("_계산전제"))
        # `_확인필요_자격`·`_확인필요_자격_매출미달시` 는 **여기서 내지 않는다.**
        # 되묻기를 코드가 ⑥ 단계에서 하므로(app.py), 여기까지 온 후보는 셋 중 하나다.
        #   '예' 로 답해 확인이 끝났다       -> 경고할 대상이 아니다
        #   '아니오/모름' + 지금가능         -> 남기고 prerequisites() 가 따로 안내한다
        #   '아니오/모름' + 지금불가         -> 풀에서 빠져 여기 오지 않는다
        # 이 줄을 되살리면 확인이 끝난 사용자에게 "해당 없으면 제외합니다"가 다시 뜬다.
        # 프롬프트도 같은 함수를 쓰므로 LLM 이 그것을 before_apply 에 옮겨 적는다(실측).
    return notes


# _confirm_tag() 는 지웠다. 자격 확인 문구를 [확인 N] 딱지로 줄여 프롬프트 길이를
# 아끼던 함수인데, `_caveats()` 가 그 문구를 더 이상 내지 않아 줄일 것이 없다.


def _situation_lines(context: Any, profile: Any = None) -> list[str]:
    """사용자 상황 요약. **프롬프트에 이미 있는 값만** 쓰고 새로 계산하지 않는다."""
    out: list[str] = []
    drop = getattr(context, "revenue_drop_rate", 0.0) or 0.0
    if drop > 0:
        out.append(f"매출 {drop:.0%} 감소")
    annual = getattr(context, "annual_revenue", None)
    if annual:
        out.append(f"연매출 {annual / MAN:,.0f}만원")
    if profile is not None:
        if getattr(profile, "existing_debt", 0):
            out.append(f"기존 부채 {profile.existing_debt / MAN:,.0f}만원")
        if getattr(profile, "overdue_days", 0):
            out.append(f"연체 {profile.overdue_days}일")
    if getattr(context, "is_closed", False):
        out.append("폐업(예정)")
    restart = getattr(context, "restart_status", None)
    if restart and restart != "안함":
        out.append(f"재창업 {restart}")
    return out


def _is_bullet(items: Sequence[Any]) -> bool:
    """만기 일시상환 융자가 들어 있는가.

    `SimResult.balloon_due_at` 으로는 판별할 수 없다 — 그 필드는 만기가 시뮬레이션 창
    **밖**일 때만 채워진다(창 안이면 0). 데모 B 의 2년만기는 24개월차라 창 안이다.
    """
    return any(
        it.candidate.is_loan and it.candidate.get("상환방식") == "만기일시"
        for it in items
    )


def _repayment_summary(result: Any, report: Any, items: Sequence[Any] = ()) -> list[str]:
    """상환 부담을 판단할 **최소한의 값**. 각 값에 반드시 시점을 붙인다.

    스케줄 36행을 통째로 넣지 않는다 — 프롬프트가 배로 늘고 LLM이 그 숫자를 다 쓸 일도
    없다. 반대로 1회차만 주면 거치 중이라 가장 낮은 값이어서 부담을 과소평가한다.

    넣는 값과 이유:
        첫 달        시작 부담
        거치 종료 후  거치가 있는 조합만. 부담이 실제로 커지는 지점
        가장 큰 달    만기일시·분기상환처럼 특정 달만 튀는 경우를 잡는다
        총액         to_prompt_text 가 이미 낸다

    **없는 값은 "없다"고 적는다.** 그냥 빼면 읽는 쪽이 "값이 없어서 뺐는지, 앞줄과 같아서
    뺐는지" 알 수 없다. 상환이 계속 오르는 구조인지 아닌지가 판단에 영향을 준다 —
    만기 일시상환이 붙으면 마지막에 크게 튀는데, **그게 없다는 것도 정보다.**
    같은 숫자를 반복하지는 않는다(산만해진다).

        월 상환액 — 1개월차 78만원
        거치 종료 후 — 13개월차부터 154만원
        이후 변동 없음                      <- 최고액이 거치 종료 후와 같을 때

        월 상환액 — 1개월차 92만원
        거치 없음, 이후 변동 없음            <- 대환만 있는 조합

    영업이익 대비 비율은 넣지 않는다. 적자면 산출이 안 되고, 흑자여도 어느 시점의
    상환액을 쓰느냐에 따라 달라져 오해를 부른다.

    반올림은 화면과 같은 month_display 를 쓴다(같은 회차를 다르게 말하지 않게).
    """
    payments = result.payment_during
    last = result.survival_months if not result.survives_full_term else len(payments)
    if last <= 0 or not payments:
        return ["상환액 — 없음(갚을 것이 없다)"]
    # 1개월차 상환액은 **기존 부채분과 신규분을 나눠 적는다.** 합계만 주면 "이 조합을
    # 받으면 월 N만원을 갚는다"로 읽히는데, 그 돈의 대부분이 조합과 무관하게 원래 나가던
    # 것일 수 있다. 데모 C 는 100만원이 전액 기존 부채고 신규는 0이다.
    # 반올림은 month_display 한 곳에서 한다 — 각자 반올림하면 덧셈이 안 맞는다.
    cells = month_display(result, 1)
    first = cells.total
    ex = cells.existing_principal + cells.existing_interest
    nw = cells.new_principal + cells.new_interest
    out = [f"월 상환액 — 1개월차 {first:,}만원 (기존 부채 {ex:,} + 신규 {nw:,})"]

    cliff = getattr(report, "cliff_at", None)
    cliff_total = None
    if cliff and cliff <= last:
        cliff_total = month_display(result, cliff).total

    peak_month = max(range(1, last + 1), key=lambda m: payments[m - 1])
    peak_total = month_display(result, peak_month).total
    highest_so_far = max(first, cliff_total or 0)
    rises_later = peak_total > highest_so_far

    # **'거치'라는 말을 쓰지 않는다.** find_cliff_at 이 찾는 것은 신규 상환의 원금이
    # 0에서 양수로 바뀌는 시점, 즉 '원금 상환이 시작되는 시점'이지 거치가 아니다.
    # 그 시점은 성격이 다른 셋에서 나온다 — 융자의 거치 종료, 만기 일시상환, 채무조정
    # 확정 후 조정 스케줄 시작. new_principal_during 이 다섯 갈래를 누적하므로 앞으로
    # 넷째·다섯째가 더 나올 수 있다.
    #
    # 만기 일시상환만 예외로 가른다. 부담이 질적으로 다르기 때문이다 — 분할은 매달,
    # 일시는 한 번에다. '원금 상환 시작'이라고만 쓰면 매달 나가는 것으로 읽힌다.
    bullet = _is_bullet(items)

    if cliff_total is not None:
        out.append(
            f"만기 일시상환 — {cliff}개월차에 원금 전액 포함 {cliff_total:,}만원"
            if bullet else
            f"원금 상환 시작 — {cliff}개월차부터 {cliff_total:,}만원"
        )
        out.append(
            f"가장 큰 달 — {peak_month}개월차 {peak_total:,}만원"
            if rises_later else "이후 변동 없음"
        )
    elif cliff:
        # 원금이 시작되는 시점이 **생존 기간 밖**이다. 예전에는 이 경우가 아래
        # '거치 없음'으로 떨어져 사실과 반대로 나갔다. 라벨에 '2년거치'가 있어도 LLM 은
        # 명시적인 이쪽을 따라 "거치 없이 상환이 시작된다"고 썼다(실측).
        out.append(
            f"만기 일시상환 — {cliff}개월차에 원금 전액. "
            f"생존 {last}개월 안에 그 시점이 오지 않는다"
            if bullet else
            f"원금 상환 시작 — {cliff}개월차. "
            f"생존 {last}개월 안에 그 시점이 오지 않는다"
        )
        # '이후 변동 없음'이라고 쓰면 안 된다 — 그 시점에 실제로 변한다. 생존 기간
        # 안에서만 변동이 없다는 뜻을 밝힌다.
        out.append(
            f"가장 큰 달 — {peak_month}개월차 {peak_total:,}만원"
            if rises_later else "생존 기간 안에는 상환액이 변하지 않는다"
        )
    elif rises_later:
        out.append("신규 상환의 원금이 처음부터 나간다(거치 없음)")
        out.append(f"가장 큰 달 — {peak_month}개월차 {peak_total:,}만원")
    else:
        out.append("신규 상환의 원금이 처음부터 나간다(거치 없음), 이후 변동 없음")
    return out


def to_prompt_text(
    groups: Sequence[CandidateGroup],
    reports: dict[str, TrapReport],
    baseline_survival: int | None = None,
    goal_months: int | None = None,
    goal_satisfied: bool = True,
    extras: Sequence[Any] = (),
    context: Any = None,
    profile: Any = None,
    rejected: Sequence[Any] = (),
) -> str:
    """후보 그룹을 LLM 프롬프트에 넣을 텍스트로 만든다.

    순위(서수)만 주면 간격을 알 수 없으므로 실제 수치를 함께 준다(기획서 5.5).

    Args:
        goal_months:    사용자가 입력한 목표 생존 개월수. 없으면 표시하지 않는다.
        goal_satisfied: `combos.filter_by_goal()`의 두 번째 반환값.
            False면 **목표를 채우는 조합이 하나도 없어** 필터가 무력화되고
            전체가 넘어온 상태다. 아래 후보들은 모두 목표 미달이며,
            LLM이 이 사실을 사용자에게 반드시 알려야 한다.
            (회복불가 국면에서 흔하다)
        extras: `combos.extra_candidates()`가 돌려준 **부가 추천 후보**.
            현금효과가 0이라 순위에 들어가지 않지만 자격은 통과한 것들이다
            (사업정리 컨설팅, 법률자문 등). 최종 3개를 고른 뒤
            "이런 것도 함께 신청해 보세요"로 덧붙이게 한다.
    """
    lines: list[str] = []
    # ── 국면과 사용자 상황 ──
    # 국면은 반드시 넣는다. 아래에서 "어느 축을 중시할지는 국면에 따라 판단할 것"이라
    # 지시하면서 정작 국면을 안 주면 그 임무를 수행할 수 없다.
    # 사용자 상황은 "사용자 말로 설명"의 재료다 — 숫자를 새로 만들지 말고 이것만 쓰게 한다.
    if context is not None:
        phase = getattr(context, "phase", None)
        if phase:
            lines.append(f"[국면] {phase}")
        situation = _situation_lines(context, profile)
        if situation:
            lines.append("[사용자 상황] " + " · ".join(situation))
        lines.append("")

    if baseline_survival is not None:
        lines.append(f"지원을 받지 않으면 생존 {baseline_survival}개월.")
        lines.append("")

    if goal_months is not None:
        if goal_satisfied:
            lines.append(f"목표 {goal_months}개월. 아래 후보는 모두 목표를 채운다.")
        else:
            # 이 문구를 빠뜨리면 "24개월 버틸 수 있습니다"라고 잘못 안내하게 된다.
            lines.append(
                f"⚠ 목표 {goal_months}개월을 채우는 조합이 **하나도 없다.** "
                f"아래는 그중 나은 차선이며 모두 목표에 미달한다. "
                f"사용자에게 이 사실을 반드시 알리고, 목표를 낮추거나 "
                f"고정비 절감 같은 다른 방법을 함께 제안할 것."
            )
        lines.append("")

    # '레이어가 낮을수록'이라고 쓰면 반대로 읽힌다 — 레이어 1이 숫자로는 가장 낮으면서
    # 순위는 최상위다. 번호 기준으로 말한다.
    lines.append(
        f"후보 그룹 {len(groups)}개. **레이어는 비지배(파레토) 정렬의 층**이다. "
        "레이어 1은 어떤 후보에게도 두 축 모두에서 밀리지 않는다. "
        "레이어 **번호가 커질수록** 앞 층에게 한 축 이상에서 밀린 대안이다."
    )
    lines.append(
        "번호가 크다고 나쁜 것은 아니다 — 사용자에게 덜 중요한 축을 양보한 것일 수 있다. "
        "어느 축을 중시할지는 국면에 따라 판단할 것."
    )
    lines.append("")

    # ── 1번 후보 대비 차이를 코드가 계산해 넣는다 ──
    # 후보들이 몰려 있으면(데모 B 는 생존이 전부 같고 부채가 27만원 차이) LLM 이 구별해
    # 설명하려고 직접 뺄셈을 하는데, 그 값은 프롬프트에 없어 숫자 검증이 응답을 통째로
    # 버린다(실측 3회 연속). 차이를 미리 주면 뺄 이유가 없어진다.
    #
    # 기준은 **맨 위 후보**다. 레이어 1에 그룹이 여럿일 수 있어 '레이어 1 대비'는
    # 모호하고, '바로 위 대비'는 사슬이 돼 읽기 어렵다.
    # **부호 대신 말로 쓴다** — '+448만원'은 많다는 뜻인지 적다는 뜻인지 갈리고,
    # LLM 이 방향을 뒤집어 써도 숫자 검증은 못 잡는다.
    # **표시되는 반올림값끼리 뺀다.** 원값으로 빼서 반올림하면 화면의 두 수를 빼도
    # 그 값이 안 나온다.
    # ⚠ 여기의 members[0] 은 **바꾸지 마라.** 화면 카드에서 members[0] 을 대표로 쓰던
    # 버그(LLM 이 대표 아닌 멤버를 고르면 그 pick 이 사라졌다)를 고친 사람이 "여기도
    # [0] 이네" 하고 손대기 쉬운 자리인데, 성격이 다르다.
    # 이건 화면에 보일 조합을 고르는 게 아니라 **비교의 기준점**이다. 모든 후보가 같은
    # 하나를 기준으로 삼아야 '1번 대비'가 성립한다. 어느 것을 기준으로 잡든 상대 관계는
    # 보존되므로 [0] 이어도 무방하고, 기준을 후보마다 다르게 하면 오히려 비교가 깨진다.
    base = groups[0] if groups else None
    base_debt = round(base.debt_at_exit / MAN) if base else 0
    base_total = (
        round(base.members[0].result.total_payment / MAN)
        if base and base.members else 0
    )
    base_member = base.members[0] if base and base.members else None

    def _compare(delta: int, unit: str, more: str, less: str) -> str:
        if delta == 0:
            return "동일"
        return f"{abs(delta):,}{unit} {more if delta > 0 else less}"

    # 생존이 전부 같으면 줄마다 '생존 동일'을 반복하지 않고 목록 위에 한 번만 적는다.
    # 다섯 줄에 같은 말이 반복되면 정보가 아니라 잡음이 되고, 무엇으로 구별해야 하는지도
    # 흐려진다.
    same_survival = len({g.survival_months for g in groups}) == 1 and len(groups) > 1
    if same_survival:
        lines.append(
            f"※ 후보 {len(groups)}개의 생존 개월수가 모두 {groups[0].survival_months}개월로 "
            "같다. 생존 축으로는 구별되지 않으므로 종료 시점 부채와 상환 부담으로 판단할 것."
        )
        lines.append("")

    for group in groups:
        head = (
            f"[레이어 {group.layer}] 생존 {group.survival_months}개월"
            f"(생존 축 {group.survival_rank}위) · "
            f"종료 시점 부채 {round(group.debt_at_exit / MAN):,}만원"
            f"(부채 축 {group.debt_rank}위)"
        )
        if group is base:
            head += "  ← 아래 '1번 대비'의 기준"
        else:
            parts = []
            if not same_survival:
                parts.append("생존 " + _compare(
                    group.survival_months - base.survival_months,
                    "개월", "더 버팀", "덜 버팀",
                ))
            parts.append("부채 " + _compare(
                round(group.debt_at_exit / MAN) - base_debt, "만원", "많음", "적음"
            ))
            head += "  [1번 대비] " + " · ".join(parts)
        lines.append(head)
        for member in group.members:
            report = reports[member.label]
            lines.append(f"  - {member.label}")
            # 총 상환액은 축이 아니라 정보다. 두 축이 동점인 조합들의 이자 부담
            # 차이가 여기서만 드러난다(거치 중 파산하면 종료 시점 부채가 같아짐).
            # 생존 개월수와 함께 읽어야 한다 — 일찍 죽으면 낼 것이 없어 작아진다.
            total_man = round(member.result.total_payment / MAN)
            total_line = (
                f"      36개월간 낸 원리금 총액 {total_man:,}만원 "
                f"(월평균 {member.result.monthly_payment_avg / MAN:,.0f}만원)"
            )
            if member is not base_member:
                total_line += " · 1번 대비 " + _compare(
                    total_man - base_total, "만원", "많음", "적음"
                )
            lines.append(total_line)
            # ── 주의 문구 ──
            # 성격이 다른 셋을 각각 출력한다.
            #   안내문구       제도가 정한 사실 (JSON 정책 레벨)
            #   변동금리       우리가 현재 금리로 고정 계산한다는 한계
            #   _계산전제      계산 전제가 실제와 다를 수 있는 후보의 사정
            # 섞으면 "제도 얘기인지 우리 계산의 한계인지"를 LLM이 구분하지 못한다.
            for note in _caveats(member.combo.items, context):
                lines.append(f"      ※ {note}")


            # 창 밖 만기 일시상환. 두 축이 못 잡는 정보다.
            if member.result.balloon_after_horizon > 0:
                lines.append(
                    f"      ⚠ {member.result.balloon_due_at}개월차에 원금 "
                    f"{member.result.balloon_after_horizon / MAN:,.0f}만원이 "
                    f"일시상환으로 남는다 (36개월 시뮬레이션 밖)"
                )
            # 함정이 없을 때도 **없다고 적는다.** 줄이 그냥 빠지면 "판정에 안 걸렸다"인지
            # "판정을 안 했다"인지 읽는 쪽이 알 수 없다. 함정 4종을 다 통과했다는 것은
            # 그 자체로 이 조합의 강점이라 판단 재료가 된다.
            if report.traps:
                for trap in report.traps:
                    lines.append(f"      [경고] {trap.message}")
            else:
                lines.append("      함정 판정 — 4종 모두 해당 없음")
            for line in _repayment_summary(member.result, report, member.combo.items):
                lines.append(f"      {line}")
        if group.dropped:
            lines.append(
                f"      ※ 같은 성능의 대안 {len(group.dropped)}개가 더 있으나 "
                "함정이 있거나 신청할 정책이 많아 제외했다."
            )
        lines.append("")

    if extras:
        lines.append("")
        lines.append("[함께 안내할 것] 순위에는 들어가지 않지만 자격이 되는 지원")
        for cand in extras:
            note = cand.get("_순위제외근거") or cand.get("조건") or ""
            lines.append(f"  - {cand}" + (f" — {note[:60]}" if note else ""))
        lines.append(
            "  ※ 현금 효과가 없거나 금액을 특정할 수 없어 순위 계산에서 뺐다. "
            "최종 추천과 함께 부가로 안내할 것."
        )

    # [순위에서 밀린 조합] 절은 뺐다(2026-08-26 실측 근거). `rejected` 인자는 호출부
    # 호환을 위해 남겨 두고 쓰지 않는다.
    #
    # ① '상위 5개'에 근거가 없다. A 는 12개 중 5개, B 는 29개 중 5개인데 잘린 것과 남은
    #    것 사이에 경계가 없다 — 임의 상수를 두지 않는다는 원칙과 어긋난다.
    # ② 비용이 25% 늘었다. excluded 가 8~10개로 부풀어 출력이 3,400 토큰까지 갔다.
    # ③ 두 축으로 설명하기 어려운 경우가 있었다. A 에서는 선정(부채 4,231~4,244만원)과
    #    탈락(4,258~4,271)이 27만원 폭에 섞이고 상위 5개 중 2개는 선정 그룹과 좌표가
    #    같아, 남는 근거가 "바우처가 빠져서"뿐이었다. B 에서는 이런 겹침이 없었다.
    #
    # "왜 이건 안 골랐나"(기획서 5.3)는 **선정 그룹 중 안 고른 것들**로 답한다.
    # 후보 그룹이 적은 입력에서는 배제 설명이 빌 수 있다 — planning.md '알려진 한계' 참조.

    # [확인 문구 전문]·[자격 확인 안내] 두 절도 함께 뺐다. `_caveats()` 가 확인 문구를
    # 안 내므로 실을 내용이 없고, 안내 절만 남기면 LLM 이 "자격 확인이 필요하다"를
    # before_apply 에 계속 옮겨 적는다. 되묻기는 코드가 ⑥ 단계에서 끝낸다.

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
# 요약 출력 (디버깅용)
# ══════════════════════════════════════════════════════════


def summarize(groups: Sequence[CandidateGroup]) -> str:
    """후보 그룹을 표로. 순위가 아니라 레이어 순이다.

    `g.label()` 이 members[0] 을 쓰지만 **런타임 경로가 아니다** — tests/test_optimizer.py
    의 print 한 곳에서만 부른다. 화면·프롬프트에는 안 나가고 표 머리글이 그 열을 '대표'
    라고 이미 밝힌다. 화면 카드의 members[0] 버그와는 상관이 없다.
    """
    lines = [
        f"후보 그룹 {len(groups)}개 · 조합 {sum(g.size for g in groups)}개 (순위 아님)",
        f"{'층':>3} {'생존':>7} {'순부채(만원)':>12} {'축순위':>7} {'조합':>4}  대표",
    ]
    for g in groups:
        lines.append(
            f"{g.layer:>3} {g.survival_months:>5}개월 {g.debt_at_exit / MAN:>12,.0f} "
            f"{g.survival_rank:>3}/{g.debt_rank:<3} {g.size:>4}  {g.label()[:44]}"
        )

    return "\n".join(lines)
