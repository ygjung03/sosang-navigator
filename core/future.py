"""미래 자격 후보 — 지금은 자격이 없지만 N개월 뒤 생기는 정책을 찾는다.

    future_candidates(db, profile, ctx, past, forecast, is_closed) -> {후보id: N}
    delayed(cand, months)                                          -> 후보 사본

기획서 5.2.3 이 정한 설계를 따른다. 예측선(`core/forecast.py`)으로 자격을 다시
판정해 "몇 개월 뒤 신청할 수 있는지"를 계산한다.


무엇을 예측 대상으로 삼는가 — 우리가 정하지 않는다
---------------------------------------------------
정책 데이터의 `미래후보자격` 플래그가 켜진 것만 본다. 매출 요건처럼 POS 추세로
도달 시점을 계산할 수 있는 것만 true 이고, 생계형 여부·고용 형태 같은 **사실 정보**는
false 다(각 정책의 `_미래후보설명` 참조). 지금 켜진 정책은 셋이다.

    긴급경영안정자금      매출 15% 이상 감소
    서울시 긴급자영업자금   매출 20% 이상 급감
    재도전특별자금 도약형   매출 5% 이상 증가


국면은 **신청 시점** 것을 쓴다
------------------------------
정책은 신청하는 시점의 국면으로 자격이 정해진다. 그래서 t=N 의 자격을 볼 때 국면도
t=N 기준으로 다시 판정한다. t=0(지금)의 국면을 고정해 쓰지 않는다 — 지금은 신청하지
않는 시점이기 때문이다.

데모 A 에서 국면이 실제로 흔들린다(t=1 개선여지 · t=2 일시적자금난 · t=3~6 개선여지).
ICR 이 1.0 경계를 오르내려서인데 **문제가 되지 않는다.** 국면을 시계열로 보여주지
않고 후보마다 **자격이 생기는 가장 이른 달 하나**만 쓰기 때문이다. t=2 에 국면이
되돌아온다는 사실은 t=1 에 신청한 사람과 무관하다.


국면 지표 넷이 시점에 따라 어떻게 계산되나
------------------------------------------
    매출감소   실적 뒤에 예측선 N개를 이어 붙여 분기·반기 중 큰 감소율     바뀐다
    영업이익   (그 시점 최근 3개월 평균) x (1 - 변동비율) - 고정비        바뀐다
    ICR       위 영업이익 / 이자비용                                   바뀐다
    연체       현재값 그대로                                          안 바뀐다

하드 규칙은 ① 폐업이 현재값 고정, ② `총점 0 + 생존 < 36` 의 생존이 **그 시점 잔고와
그 이후 매출선으로 다시 시뮬레이션한 값**, ③ 단일 지표 2점은 위 넷에서 파생된다.

**이자비용은 그 시점의 기존 부채 잔액으로 계산한다**(`existing_debt_at[N]`). 현재
부채로 고정하면 상환한 만큼을 빼지 않아 이자를 과대평가하고, ICR 이 과소평가되어
국면이 비관 쪽으로 기운다. 미래 자격은 "지금은 안 되지만 나중에 된다"를 말하는
기능이라 그 치우침이 곧 **자격이 생기는데도 안 생긴다고 말하는 것**이 된다.
규칙을 바꾸는 것이 아니다 — t=0 에서는 두 값이 같고
(`existing_debt_at[0] == existing_debt`), 같은 규칙을 그 시점에 적용할 뿐이다.


시점에 따라 그대로 두는 것
--------------------------
연체일수는 시간이 지나면 늘 수 있는데 우리에게 예측할 재료가 없다. 업력도 시점에
따라 늘지만 지금은 넘기지 않는다 — `UserContext` 가 기준일을 갖고 있어야 해서
구조가 커지고, 업력이 요건인 정책(재도전특별자금)은 회복불가 전용이라 실질적으로
걸리지 않는다. 신용점수는 기획서 5.2.3 이 이미 제외했다(산출 방식이 공개되지 않아
"N개월 뒤 몇 점"을 계산할 근거가 없다).
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

from core.cashflow import SIM_MONTHS, UserProfile, simulate
from core.combos import UserContext, eligible_candidates
from core.loader import Candidate, PolicyDB
from core.phase import PhaseResult, judge

HORIZON_MONTHS = 6
"""몇 개월 앞까지 볼지. 기획서 5.2.3 — "매출 추세 예측은 6개월을 넘어가면 신뢰도가
급격히 낮아지고, 정책 공고도 연·분기 단위로 개정되어 그 이상의 예측은 실효성이 없다"."""

_PHASE_WINDOWS = (("분기", 3), ("반기", 6))
"""국면 판정이 쓰는 비교 단위. 월은 계절성으로 튀고 연도는 POS 24개월이 필요하다
(`core/phase.py` 와 같은 기준)."""


def _window_drops(monthly: Sequence[float]) -> dict[str, float]:
    """비교 단위별 감소율. **감소를 양수로** 담는다.

    `pos.summarize` 와 같은 방식이다 — "직전 N개월 합 vs 그 전 N개월 합".
    2N개월이 없으면 그 단위를 넣지 않는다. 없는 것과 0은 다르다.
    """
    out: dict[str, float] = {}
    for name, months in (("월", 1), ("분기", 3), ("반기", 6), ("연도", 12)):
        if len(monthly) < 2 * months:
            continue
        recent = sum(monthly[-months:])
        before = sum(monthly[-2 * months : -months])
        if before > 0:
            out[name] = -(recent - before) / before
    return out


def _series_at(past: Sequence[float], forecast: Sequence[int], months: int) -> list[float]:
    """t=`months` 시점에서 본 과거 계열 — 실적 뒤에 예측 `months` 개를 이어 붙인다."""
    return list(past) + [float(v) for v in forecast[:months]]


def shift_context(
    ctx: UserContext,
    past: Sequence[float],
    forecast: Sequence[int],
    months: int,
    phase: str,
) -> UserContext:
    """t=`months` 시점의 컨텍스트. 매출에서 나오는 값과 국면만 바꾼다."""
    by_window = _window_drops(_series_at(past, forecast, months))
    drop_rate = max(
        (by_window[w] for w, _ in _PHASE_WINDOWS if w in by_window), default=0.0
    )
    return dataclasses.replace(
        ctx,
        phase=phase,
        revenue_drop_by_window=by_window,
        revenue_drop_rate=drop_rate,
        # 증가율은 감소율의 반대편이다. 감소가 음수(=증가)일 때만 양수가 된다.
        revenue_growth_rate=max(0.0, -drop_rate),
    )


def phase_at(
    profile: UserProfile,
    past: Sequence[float],
    forecast: Sequence[int],
    months: int,
    is_closed: bool,
) -> PhaseResult:
    """t=`months` 시점에 다시 진단하면 나올 국면. 지표별 계산은 모듈 docstring 참조."""
    series = _series_at(past, forecast, months)
    current = sum(series[-3:]) / 3 if len(series) >= 3 else (series[-1] if series else 0.0)
    operating_profit = current * (1 - profile.variable_ratio) - profile.fixed_cost

    by_window = _window_drops(series)
    drop_rate, window_months = 0.0, 3
    for name, span in _PHASE_WINDOWS:
        if by_window.get(name, 0.0) > drop_rate:
            drop_rate, window_months = by_window[name], span

    baseline = simulate(profile, [])
    at = min(months, len(baseline.existing_debt_at) - 1)
    interest_cost = baseline.existing_debt_at[at] * (profile.existing_debt_rate or 0.0) / 12

    cash_left = baseline.balance_at[min(months, len(baseline.balance_at) - 1)]
    # 창 길이를 유지하려고 마지막 예측값으로 채운다. 뒤쪽은 이미 소진 이후라
    # 생존 계산에 닿지 않는다.
    tail = (list(forecast[months:]) + [forecast[-1]] * months) if forecast else []
    ahead = dataclasses.replace(
        profile, cash_on_hand=max(int(cash_left), 0), monthly_revenue=tail[:SIM_MONTHS]
    )
    return judge(
        operating_profit=operating_profit,
        interest_cost=interest_cost,
        revenue_drop_rate=drop_rate,
        revenue_drop_window_months=window_months,
        overdue_days=profile.overdue_days,
        is_closed=is_closed,
        buffer_months=simulate(ahead, []).survival_months,
    )


def is_future_capable(cand: Candidate) -> bool:
    """`미래후보자격` 이 켜진 정책의 후보인가. 후보 -> 정책 순으로 본다."""
    if cand.raw.get("미래후보자격") is True:
        return True
    return cand.policy_raw.get("미래후보자격") is True


def delayed(cand: Candidate, months: int) -> Candidate:
    """`months` 개월 뒤에 받는 것으로 바꾼 후보 사본.

    `지급시점_개월`·`적용시점_개월` 은 원래 **"신청 후 N개월 뒤"** 를 담는 필드다.
    미래 자격도 "N개월 뒤에 신청할 수 있다"이므로 의미가 어긋나지 않는다.
    두 지연은 겹치지 않고 **더해진다** — 신청 전에 심사가 시작될 수 없다.

        미래 자격 후보    2개월 뒤 신청 가능 · 원래 지급시점 3  ->  2 + 3 = 5
        함께 묶인 후보    지금 신청 가능                       ->  자기 값 그대로

    `Candidate` 가 frozen dataclass 라 `dataclasses.replace` 로 만든다. `raw` 를
    **새 dict** 로 갈아끼우므로 원본은 안 건드린다. 깊은 복사는 필요 없다 — 넣는
    값이 스칼라뿐이고 중첩 구조를 수정하지 않는다.

    ※ 정식 방법은 `Applied` 에 지연 필드를 두어 "이 조합에서만 N개월 뒤"를 표현하는
      것이다. 계산 코드(`cashflow.simulate`)에서 시점을 읽는 자리 여섯 곳을 고쳐야
      해서 지금은 사본으로 대신했다(planning.md 5.2.3).
    """
    if months <= 0:
        return cand
    fields: dict[str, int] = {}
    for key in ("지급시점_개월", "적용시점_개월"):
        if cand.get(key) is not None:
            fields[key] = int(cand.get(key) or 0) + months
    if not fields:
        # 시점 필드가 없던 후보(즉시 지급)에는 지급시점을 새로 단다.
        fields["지급시점_개월"] = months
    return dataclasses.replace(cand, raw={**cand.raw, **fields})


def future_candidates(
    db: PolicyDB,
    profile: UserProfile,
    ctx: UserContext,
    past: Sequence[float],
    forecast: Sequence[int],
    is_closed: bool = False,
    max_months: int = HORIZON_MONTHS,
) -> dict[str, int]:
    """지금은 자격이 없지만 N개월 뒤 생기는 후보. `{후보id: 가장 이른 N}`.

    Args:
        past:     과거 월별 매출(원). `PosSummary.monthly_revenue` 의 값.
        forecast: 예측 월별 매출(원). `forecast.ensemble_revenue()` 의 결과.

    **호출을 줄이려고 두 번 거른다.** 시점별로 도는 것은 국면 계산과 자격 필터뿐이다
    (실측 6개 시점에 0.58ms). 비싼 조합·평가는 호출부가 여기서 모은 후보로 **한 번만**
    돌린다 — 시점마다 돌릴 이유가 없다.

        ① 시점 1~6 의 국면을 먼저 계산한다              국면만, 자격 필터 없이
        ② 나타난 국면 중 하나라도 `미래후보자격` 정책의
           적용국면과 겹치는가 -> 안 겹치면 여기서 끝     대부분 여기서 끝난다
        ③ 겹치면 시점별로 자격 필터를 돌린다

    **조기 종료를 넣지 않는다.** 첫 히트에서 멈추면 늦은 시점에만 자격이 생기는 두
    번째 후보를 조용히 놓친다. 0.58ms 를 아끼려고 정확성을 버릴 이유가 없다.

    간편 모드(POS 없음)에서는 호출하지 않는다 — 과거 실적이 있어야 비교 단위별
    감소율을 다시 계산할 수 있다. 호출부가 `summary is not None` 으로 가른다.
    """
    if not forecast or not past or max_months <= 0:
        return {}

    watchable = [c for c in db.candidates if is_future_capable(c)]
    if not watchable:
        return {}

    # ① 시점별 국면. t=0 은 보지 않는다 — 신청하지 않는 시점이다.
    phases = {
        n: phase_at(profile, past, forecast, n, is_closed).phase
        for n in range(1, max_months + 1)
    }

    # ② 사전 필터 — 어느 시점의 국면으로도 안 걸리면 자격 필터를 돌릴 이유가 없다.
    reachable = set(phases.values())
    if not any(set(c.phases) & reachable for c in watchable):
        return {}

    # ③ 시점별 자격 판정. 지금 이미 자격이 있는 후보는 '미래'가 아니다.
    already = {c.candidate_id for c in eligible_candidates(db, profile, ctx)}
    found: dict[str, int] = {}
    for months in range(1, max_months + 1):
        ctx_n = shift_context(ctx, past, forecast, months, phases[months])
        for cand in eligible_candidates(db, profile, ctx_n):
            if cand.candidate_id in already or cand.candidate_id in found:
                continue
            if not is_future_capable(cand):
                continue  # 시점과 무관하게 흔들린 후보는 넣지 않는다
            found[cand.candidate_id] = months
    return found
