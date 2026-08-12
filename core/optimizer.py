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
from typing import Sequence

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


def to_prompt_text(
    groups: Sequence[CandidateGroup],
    reports: dict[str, TrapReport],
    baseline_survival: int | None = None,
) -> str:
    """후보 그룹을 LLM 프롬프트에 넣을 텍스트로 만든다.

    순위(서수)만 주면 간격을 알 수 없으므로 실제 수치를 함께 준다(기획서 5.5).
    """
    lines: list[str] = []
    if baseline_survival is not None:
        lines.append(f"지원을 받지 않으면 생존 {baseline_survival}개월.")
        lines.append("")

    lines.append(
        f"후보 그룹 {len(groups)}개. 레이어 1이 두 축 모두에서 최상위이며, "
        "레이어가 낮을수록 한 축을 양보한 대안이다."
    )
    lines.append("어느 축을 중시할지는 국면에 따라 판단할 것.")
    lines.append("")

    for group in groups:
        lines.append(
            f"[레이어 {group.layer}] 생존 {group.survival_months}개월"
            f"(생존 축 {group.survival_rank}위) · "
            f"종료 시점 부채 {group.debt_at_exit / MAN:,.0f}만원"
            f"(부채 축 {group.debt_rank}위)"
        )
        for member in group.members:
            report = reports[member.label]
            lines.append(f"  - {member.label}")
            # 총 상환액은 축이 아니라 정보다. 두 축이 동점인 조합들의 이자 부담
            # 차이가 여기서만 드러난다(거치 중 파산하면 종료 시점 부채가 같아짐).
            # 생존 개월수와 함께 읽어야 한다 — 일찍 죽으면 낼 것이 없어 작아진다.
            lines.append(
                f"      36개월간 낸 원리금 총액 "
                f"{member.result.total_payment / MAN:,.0f}만원"
            )
            for trap in report.traps:
                lines.append(f"      [경고] {trap.message}")
            if report.burden_ratio_now is not None:
                lines.append(
                    f"      월 상환액 {report.payment_now / MAN:,.0f}만원 / "
                    f"월 영업이익 {report.operating_now / MAN:,.0f}만원 "
                    f"({report.burden_ratio_now:.0%})"
                )
            else:
                lines.append(
                    f"      월 상환액 {report.payment_now / MAN:,.0f}만원 "
                    f"(월 영업이익 {report.operating_now / MAN:,.0f}만원 — 적자라 부담률 산출 불가)"
                )
        if group.dropped:
            lines.append(
                f"      ※ 같은 성능의 대안 {len(group.dropped)}개가 더 있으나 "
                "함정이 있거나 신청할 정책이 많아 제외했다."
            )
        lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
# 요약 출력 (디버깅용)
# ══════════════════════════════════════════════════════════


def summarize(groups: Sequence[CandidateGroup]) -> str:
    """후보 그룹을 표로. 순위가 아니라 레이어 순이다."""
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
