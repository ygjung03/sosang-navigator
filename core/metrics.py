"""
함정 판정 — 시뮬레이션의 부산물로 읽어낸다.

함정은 **정책의 고유 속성이 아니라 "조합 x 상황"의 결과**다 (기획서 5.3).
대출이 항상 나쁜 게 아니라, 겹치거나 국면에 안 맞거나 과다할 때 함정이 된다.
그래서 별도 판정 엔진을 설계하지 않고 현금흐름 곡선의 모양에서 읽는다.


판정하는 것 — 4종
------------------
1. **즉시 악화**       월 상환액 > 월 영업이익.  ※ 영업이익이 흑자일 때만
2. **거치 절벽**       절벽 시점 <= 생존 개월수 < horizon
3. **회복불가 부채증가** 국면 = 회복불가 && 신규 융자 포함
4. **베이스라인 악화**  조합 생존 개월수 < 무지원 생존 개월수

판정하지 않는 것 — 회색지대
---------------------------
상환액이 순이익의 40~60% 같은 구간은 함정/안전을 단정하지 않고 **원본 숫자만**
노출한다(예: "월 상환액 92만원 / 월 영업이익 210만원 (44%)").
임의의 구간 경계를 근거 없이 만들지 않는다.

**경고는 적을수록 신뢰가 높다.** 잘못된 경고 1건이 서비스 전체 신뢰를
무너뜨리므로, 확실히 판정 가능한 것만 경고하고 나머지는 숫자로만 제시한다.

**함정이어도 조합에서 제거하지 않는다.** "받지 마세요"(명령)가 아니라
"이런 위험이 동반된다"(정보)로 제시하고, 판단은 사용자에게 둔다.


═══════════════════════════════════════════════════════════
거치 절벽을 '생존 개월수'로 판정하는 이유
═══════════════════════════════════════════════════════════
TODO(adr): 아래 "대안 두 가지를 검토하고 기각했다" 부분은 결정의 역사이므로
           docs/adr/ 로 옮기고, 여기에는 채택한 기준과 링크만 남긴다.

초안에서는 "절벽 시점의 영업이익 < 그 시점 상환액"으로 판정했다.
그런데 **영업이익이 음수면 상환액이 얼마든(심지어 0이어도) 조건이 항상 참**이 된다.
이 서비스에는 이미 영업 적자인 사용자가 들어온다 — 영업 적자는 국면 점수에서
영업이익 2점과 이자보상배율 2점을 동시에 받아 최소 4점, 즉 개선 여지 이상이 되므로
우리가 다루는 세 국면 중 둘에 해당한다(phase.py 참조).
적자 프로필로 실측했을 때 전체 조합의 100%에 경고가 붙었다.
"경고는 적을수록 신뢰"가 완전히 무너진 것이다.

대안 두 가지를 검토하고 둘 다 기각했다.

  (A) 영업이익이 흑자일 때만 판정
      -> 적자 사용자에게 기능이 아예 죽는다. 그리고 적자 사업자에게도
         절벽은 의미가 있다. 월 -100만 적자여도 3,000만원을 받으면 30개월까지
         버티다가 25개월차 원금 상환으로 27개월에 소진되는 구조라면,
         그 정보는 반드시 알려야 한다.

  (B) "절벽 도달 가능 + 상환액이 X배 이상 급증"
      -> X가 근거 없는 임의 상수가 된다. 게다가 급증률 자체가 위험의 지표가
         아니다. 10만 -> 80만(8배)이어도 월 현금흐름이 +500만이면 문제가 없고,
         10만 -> 30만(3배)이어도 +20만이면 위험하다.

채택한 기준:

    절벽 시점 <= 생존 개월수 < horizon

  · 절벽 전에 죽으면      -> 도달하지 못하므로 경고 대상이 아니다
  · horizon 끝까지 버티면 -> 절벽을 넘겼으므로 경고 대상이 아니다
  · 그 사이에서 죽으면    -> 절벽이 원인일 가능성이 높다

**영업이익도 급증률도 보지 않는다.** 잔고가 답을 준다.
임의 임계값이 하나도 들어가지 않으면서 "지금은 괜찮아 보이지만 그때 무너진다"를
정확히 포착한다.

또한 경고 문구에 "8배 급증" 같은 표현을 쓰지 않는다. 급증은 원인이 아니라
현상이고, 사용자가 알아야 할 것은 "그래서 내 현금이 어떻게 되는가"다.

**감지 방법을 바꿨다(2026-08-27). 채택 기준은 그대로다.**
'절벽 시점 <= 생존 < horizon' 은 손대지 않았다. 바꾼 것은 그 '절벽 시점'을 찾는
방법이다. 옛 방식은 **총 상환액**이 CLIFF_DETECT_RATIO(=1.5)배 뛰는 지점을 찾았는데,
총액에는 기존 부채 상환이 섞여 있어 신규 융자의 원금 개시가 희석됐다. 기존 부채가 크면
같은 정책·같은 거치인데도 감지에 실패해 **함정 2 판정이 시작조차 되지 않았다.**
이제 신규 융자의 원금이 0에서 양수로 바뀌는 시점을 본다 — 거치의 정의 그대로이고
임계값이 없다. 자세한 근거와 상환 방식별 동작은 find_cliff_at 주석에 있다.

실측(데모 A·B·C 전 조합 18개) — **옛 방식은 절벽을 하나도 못 잡았다.**

    판정 동일 9개    원래 절벽이 없는 조합(대환대출·보조금 단독 등)
    달라짐   9개    전부 `옛 None -> 새 값`. 감지가 사실상 작동하지 않았다
      그중 새로 함정 2 로 잡힌 것 1개  [A] 긴급자영업(1년거치 4년 균분), cliff 13
      나머지 8개는 '미도달'(절벽 전에 자금이 마름)이라 경고는 안 나가지만,
      cliff_at 값이 생겨 프롬프트 상환 요약이 "거치 없음"이라 거짓말하던 것이 고쳐진다

**상환 스케줄 표가 원래부터 맞았다.** 표는 신규 원금 열이 0 -> 76 으로 바뀌는 것을
그대로 보여준다(app.py 의 schedule_rows 주석). 새 감지 정의가 그것과 같은 근거이므로,
이제 표·그래프 ▼ 마커·함정 경고·프롬프트 요약이 한 값을 쓴다. 틀렸던 것은 표가 아니라
감지 쪽이었다.


경고의 두 층 (기획서 5.3 보강)
------------------------------
"적자니까 이 정책은 위험합니다"라고 말하지 않는다. 대신 분리한다.

    사업 상태 — 영업 현금흐름이 음수입니다        (phase.py의 몫)
    정책 효과 — 현금 소진 시점이 3 -> 28개월로 연장됩니다
    주의      — 거치 종료 후 상환이 시작되며 잔고가 다시 감소합니다  (여기)

이 모듈은 **정책의 구조적 문제**만 다룬다. 사업 자체의 지속 가능성은
국면 판단이 답할 몫이다.


명명 규칙: 코드 식별자는 영어, JSON 키는 한글. 시점은 _at, 기간은 _during.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from .cashflow import SIM_MONTHS, SimResult, UserProfile, month_display, simulate
from .combos import Combo, Evaluated

# CLIFF_DETECT_RATIO(=1.5) 는 지웠다. 아래 find_cliff_at 주석 참조.

MAN = 10_000  # 만원. 메시지 포맷용


class TrapKind(str, Enum):
    """함정 4종. 문자열 Enum이라 그대로 화면·JSON에 쓸 수 있다."""

    IMMEDIATE = "즉시악화"
    CLIFF = "거치절벽"
    DEBT_IN_RECOVERY_FAIL = "회복불가_부채증가"
    WORSE_THAN_BASELINE = "베이스라인악화"


@dataclass(frozen=True)
class Trap:
    """판정된 함정 하나."""

    kind: TrapKind
    message: str
    """사용자에게 보여줄 문장. 숫자를 포함해 근거가 드러나게 쓴다."""

    at: int | None = None
    """함정이 발생하는 시점(개월차). 시점이 없는 함정은 None."""


@dataclass
class TrapReport:
    """조합 하나의 함정 판정 결과."""

    traps: list[Trap] = field(default_factory=list)

    # ── 회색지대 — 판정하지 않고 숫자만 전달 ──
    payment_now: float = 0.0
    """1개월차 월 상환액(원)."""

    operating_now: float = 0.0
    """1개월차 월 영업이익(원). 음수면 적자."""

    cliff_at: int | None = None
    """원금 상환이 시작되는 시점(개월차). 없으면 None."""

    payment_before_cliff: float = 0.0
    """절벽 직전 월 상환액(원)."""

    payment_after_cliff: float = 0.0
    """절벽 시점 월 상환액(원)."""

    survival_months: int = 0
    """이 조합의 생존 개월수."""

    baseline_survival_months: int = 0
    """무지원 상태의 생존 개월수. 비교 기준."""

    @property
    def has_trap(self) -> bool:
        return bool(self.traps)

    @property
    def icon(self) -> str:
        """접힌 카드에 표시할 아이콘 (기획서 5.2.4)."""
        return "⚠" if self.traps else ""

    @property
    def burden_ratio_now(self) -> float | None:
        """월 상환액 / 월 영업이익. 영업이익이 0 이하면 None.

        회색지대를 판정하지 않고 이 비율을 원본 숫자로 함께 보여준다
        ("월 상환액 92만원 / 월 영업이익 210만원 (44%)").
        적자면 비율이 의미를 잃으므로 None을 돌려주고 화면에서 생략한다.
        """
        if self.operating_now <= 0:
            return None
        return self.payment_now / self.operating_now


# ══════════════════════════════════════════════════════════
# 절벽 시점 탐지
# ══════════════════════════════════════════════════════════


def find_cliff_at(new_principal_during: Sequence[float]) -> int | None:
    """신규 융자의 **원금 상환이 처음 시작되는 시점**(개월차). 없으면 None.

    거치의 정의를 그대로 쓴다 — 거치 중에는 원금이 0이고, 끝나면 양수가 된다.
    임계값이 하나도 없다.

    ※ 이것은 **탐지**이지 판정이 아니다. 위험 여부는 생존 개월수가 판정한다.

    옛 방식(총 상환액이 CLIFF_DETECT_RATIO=1.5 배 뛰는 지점)을 버린 이유
    ------------------------------------------------------------------
    총액에는 **기존 부채 상환이 섞여 있어** 신규 융자의 원금 개시가 희석된다.
    데모 A 의 긴급자영업자금(1년거치)이 그 실증이다.

        회차   총액   기존   신규   총액배율
         12    210    200     10    1.00
         13    309    200    109    1.47   <- 1.5 미만이라 감지 실패

    신규분만 보면 10 -> 109 로 10.9배인데, 기존 상환 200만원이 분모에 섞여 1.47 이
    됐다. 그래서 **함정 2 판정이 시작조차 되지 않았다.** 같은 정책·같은 거치인데
    기존 부채가 작은 데모 B 에서는 1.98 로 잡혔다 — 기존 부채 크기에 따라 결과가
    갈리는 구조였다.
    1.5 라는 값에도 근거가 없었다. "원금이 붙으면 최소 2배 이상 뛴다"는 신규 융자
    단독일 때만 참이고 총액에는 성립하지 않는다.

    상환 방식별로 어떻게 잡히는가
    ---------------------------
        1년거치 4년 균분   0…0 -> 양수      13개월차
        분기 균분상환      0…0 -> 양수(3개월마다 반복)  **첫 번째만** 반환한다.
                          거치가 끝나는 시점이 절벽이지 그 뒤 반복은 아니다.
        만기 일시상환      0…0 -> 마지막에 한 번   그 시점이 맞다(원금 전액이 나간다)
        대환대출(거치 없음) 1개월차부터 양수  -> None. 거치가 없으니 절벽도 없다
        채무조정·보조금만   전부 0            -> None. 신규 융자가 없다

    Returns:
        원금이 처음 0에서 양수로 바뀌는 개월차(1-based). 없으면 None.
    """
    for i in range(1, len(new_principal_during)):
        if new_principal_during[i] > 0 and new_principal_during[i - 1] <= 0:
            return i + 1  # 배열 인덱스 -> 개월차
    return None


# ══════════════════════════════════════════════════════════
# 개별 판정
# ══════════════════════════════════════════════════════════


def _check_immediate(profile: UserProfile, result: SimResult) -> Trap | None:
    """함정 1 — 즉시 악화.

    월 상환액이 현재 월 영업이익보다 크면 지금 당장 갚을 능력이 없다는 뜻이다.

    **영업이익이 흑자일 때만 판정한다.** 적자면 상환액이 얼마든 조건이 항상
    참이 되어 동어반복이 된다. 적자 상태 자체는 국면 판단이 다룰 몫이고,
    적자 사업자의 위험은 함정 2·4가 커버한다.
    """
    operating = profile.operating_cashflow(1)
    if operating <= 0:
        return None  # 적자 — 이 기준으로는 판정하지 않는다

    payment = result.payment_during[0]
    if payment <= operating:
        return None

    return Trap(
        kind=TrapKind.IMMEDIATE,
        message=(
            f"첫 달부터 월 상환액 {payment / MAN:,.0f}만원이 "
            f"월 영업이익 {operating / MAN:,.0f}만원을 넘습니다. "
            f"부족분은 보유 현금으로 메우게 됩니다."
        ),
        at=1,
    )


def _check_cliff(result: SimResult, horizon: int) -> tuple[Trap | None, int | None]:
    """함정 2 — 거치 절벽.

        절벽 시점 <= 생존 개월수 < horizon

    절벽까지는 살아서 도달했는데 horizon을 못 채우고 무너지는 경우다.
    "지금은 괜찮아 보이지만 그때 무너진다"를 정확히 포착한다.

    영업이익도 급증률도 보지 않는다 — 잔고가 답을 준다.
    임의 임계값이 하나도 들어가지 않는다.
    """
    # **감지는 신규 원금분으로, 표시는 총액으로.** 거치가 끝났는지는 신규 융자만 보면
    # 되지만, 사용자가 실제로 내는 돈은 기존 부채를 합한 총액이다.
    cliff_at = find_cliff_at(result.new_principal_during)
    if cliff_at is None or cliff_at > horizon:
        return None, None

    # 절벽에 도달하지 못하면 경고 대상이 아니다 (다른 이유로 먼저 죽는다)
    if result.survival_months < cliff_at:
        return None, cliff_at

    # horizon 끝까지 버티면 절벽을 넘긴 것이다
    if result.survives_full_term:
        return None, cliff_at

    # 표시값은 month_display 한 곳에서만 만든다 — 상환 스케줄 표와 **같은 값**을 써야
    # 나란히 놓았을 때 어긋나지 않는다(각자 반올림하면 원금+이자가 총액과 안 맞는다).
    cells = month_display(result, cliff_at)
    trap = Trap(
        kind=TrapKind.CLIFF,
        message=(
            f"{cliff_at}개월차부터 월 상환액 {cells.total:,}만원 "
            f"(원금 {cells.principal:,} + 이자 {cells.interest:,})이 시작되면서 "
            f"잔고가 다시 줄어, {result.survival_months}개월차에 소진됩니다."
        ),
        at=cliff_at,
    )
    return trap, cliff_at


def _check_debt_in_recovery_fail(combo: Combo, phase: str) -> Trap | None:
    """함정 3 — 회복불가 국면인데 신규 융자가 들어 있다.

    빚을 늘릴 때가 아니라 줄일 때라는 판단이다.
    대환대출은 부채를 늘리지 않고 이자 부담만 낮추므로 제외한다
    (JSON의 함정경고_대상: false).
    """
    if phase != "회복불가":
        return None

    new_loans = [
        c
        for c in combo.candidates
        if c.is_loan and c.effect == "신규유입" and c.get("함정경고_대상", True)
    ]
    if not new_loans:
        return None

    names = ", ".join(str(c) for c in new_loans)
    return Trap(
        kind=TrapKind.DEBT_IN_RECOVERY_FAIL,
        message=(
            f"지금은 빚을 늘리기보다 줄이는 단계입니다. "
            f"이 조합에는 신규 융자({names})가 포함되어 있습니다."
        ),
    )


def _check_worse_than_baseline(
    result: SimResult, baseline: SimResult
) -> Trap | None:
    """함정 4 — 베이스라인 악화.

    지원을 받았는데 무지원보다 더 빨리 무너지는 경우다.
    "받았는데 더 나빠졌다"보다 명백한 함정은 없고, 임의 상수가 전혀 없다.

    주로 만기일시상환이나 거치 없는 대출에서 나타난다.
    """
    if result.survival_months >= baseline.survival_months:
        return None

    return Trap(
        kind=TrapKind.WORSE_THAN_BASELINE,
        message=(
            f"이 조합을 받으면 생존 기간이 오히려 짧아집니다. "
            f"지원 없이 {baseline.survival_months}개월인데 "
            f"{result.survival_months}개월이 됩니다."
        ),
    )


# ══════════════════════════════════════════════════════════
# 통합 판정
# ══════════════════════════════════════════════════════════


def detect_traps(
    profile: UserProfile,
    evaluated: Evaluated,
    phase: str,
    baseline: SimResult | None = None,
    horizon: int = SIM_MONTHS,
) -> TrapReport:
    """조합 하나의 함정 4종을 판정한다.

    Args:
        baseline: 무지원 시뮬레이션 결과. 없으면 여기서 계산한다.
                  여러 조합을 판정할 때는 한 번만 계산해 넘기는 것이 효율적이다.
    """
    if baseline is None:
        baseline = simulate(profile, [], horizon=horizon)

    result = evaluated.result
    report = TrapReport(
        payment_now=result.payment_during[0],
        operating_now=profile.operating_cashflow(1),
        survival_months=result.survival_months,
        baseline_survival_months=baseline.survival_months,
    )

    if trap := _check_immediate(profile, result):
        report.traps.append(trap)

    cliff, cliff_at = _check_cliff(result, horizon)
    report.cliff_at = cliff_at
    if cliff_at is not None and cliff_at >= 2:
        report.payment_before_cliff = result.payment_during[cliff_at - 2]
        report.payment_after_cliff = result.payment_during[cliff_at - 1]
    if cliff:
        report.traps.append(cliff)

    if trap := _check_debt_in_recovery_fail(evaluated.combo, phase):
        report.traps.append(trap)

    if trap := _check_worse_than_baseline(result, baseline):
        report.traps.append(trap)

    return report


def detect_all(
    profile: UserProfile,
    evaluated: Sequence[Evaluated],
    phase: str,
    horizon: int = SIM_MONTHS,
) -> dict[str, TrapReport]:
    """모든 조합의 함정을 판정한다.

    베이스라인은 한 번만 계산해 재사용한다.

    Returns:
        조합 label -> TrapReport
    """
    baseline = simulate(profile, [], horizon=horizon)
    return {
        ev.label: detect_traps(profile, ev, phase, baseline, horizon)
        for ev in evaluated
    }


# ══════════════════════════════════════════════════════════
# 요약 출력 (디버깅용)
# ══════════════════════════════════════════════════════════


def summarize(
    evaluated: Sequence[Evaluated], reports: dict[str, TrapReport], top: int = 5
) -> str:
    """축별 상위 조합에 함정 아이콘을 붙여 보여준다. 디버깅용.

    combos.summarize와 같은 이유로 한 가지 정렬로 줄세우지 않는다.
    두 축은 동등하게 취급하므로(기획서 5.1) 각 축의 관점을 따로 보여준다.
    실제 후보 선정은 optimizer.py가 한다.
    """
    by_survival = sorted(evaluated, key=lambda e: -e.survival_months)[:top]
    by_debt = sorted(evaluated, key=lambda e: e.debt_at_exit)[:top]

    def row(ev: Evaluated) -> str:
        rep = reports.get(ev.label, TrapReport())
        return (
            f"  {ev.result.survival_label:>7} · {ev.debt_at_exit / MAN:>7,.0f}만 "
            f"{rep.icon:>2}  {ev.label}"
        )

    lines = [f"전체 {len(evaluated)}개 조합 · 축별 상위 {top}개 (순위 아님)"]
    lines.append("\n[축 1] 생존 개월수 — 클수록 좋음")
    lines.extend(row(ev) for ev in by_survival)
    lines.append("\n[축 2] 30개월 후 순부채 — 작을수록 좋음")
    lines.extend(row(ev) for ev in by_debt)
    return "\n".join(lines)
