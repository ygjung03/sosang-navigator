"""매출 예측 — Prophet · ETS · 선형회귀의 앙상블(월별 중앙값).

    ensemble_revenue(monthly_revenue, horizon) -> list[int]

세 모델을 **로그 공간**에서 적합하고 ETS 에는 감쇠추세를 준다. 왜 그렇게 하는지는
아래 두 절에 적는다. 실측 근거는 planning.md 5.5 의 예측 항목에 있다.

아래 수치는 전부 **POS 12개월** 로 잰 것이다. 앱이 정밀 모드에 12개월을 넘기고
(`app.py` 의 시나리오 생성), planning.md 5.1 이 정밀 모드의 전제를 12개월로 정했다.


로그 공간에서 적합하는 이유
---------------------------
세 모델이 전부 덧셈형이라 원 공간에서는 **같은 방향으로 함께 폭주한다.** 중앙값이
폭주를 잡아준다는 앙상블의 전제가 그때 무너진다. 36개월 뒤 실측:

    데모 B   선형회귀 -380만 · ETS -358만 · Prophet -248만   셋 다 음수
    데모 C   선형회귀 -502만 · ETS -429만 · Prophet -407만   셋 다 음수

매출이 음수라는 것은 그 자체로 틀린 그림이다. 로그 공간에서는 셋이 곱셈형이 되어
**0 에 점근할 뿐 닿지 않는다**(같은 두 계열이 로그 공간에서 150만 · 108만).


ETS 에 감쇠추세를 주는 이유 — 방향을 가리지 않는다
--------------------------------------------------
감쇠의 가정은 "하락이 느려진다"가 아니라 **"추세는 무한히 지속되지 않는다"** 이다.
감쇠는 적합된 추세의 절대값을 줄일 뿐이고 부호를 보지 않는다.

이 서비스를 찾는 사용자는 대체로 매출이 줄고 있어 세 모델이 잡는 추세도 하락 쪽이
많다. 그 국면에서 긴 구간을 외삽할 때 감쇠를 주는 편이 더 정확하다는 것이 M4 의
결과다(Damped OWA 0.907 vs Holt 0.971, 실제 10만 계열).

우리 데이터에서 달라지는 폭은 이렇다. 36개월 뒤가 현재의 몇 배인지:

    반등 약(진단 1점)   로그만 0.32배  ·  로그+감쇠 0.86배
    반등 강(진단 0점)   로그만 0.97배  ·  로그+감쇠 1.37배
    계절성 큰 계열      로그만 0.76배  ·  로그+감쇠 0.60배   <- 감쇠가 손해

'반등 약'은 급락한 뒤 최근 3개월이 642 -> 595 -> 601만원으로 도로 올라온 가게다.
진단은 이 가게를 매출감소 1점으로 보는데, 로그만 쓰면 예측선이 3년 뒤 0.32배
(-68%)까지 내려간다. 감쇠를 주면 0.86배가 되어 **진단과 그래프가 어긋나는 폭이
줄어든다.**

위쪽 폭주는 감쇠가 아니라 중앙값이 막는다. '반등 강'에서 Prophet 은 3년 뒤
24,663만원을 내는데, 선형회귀 761만·ETS 709만 사이에서 중앙값이 그것을 버려
최종 0.97배가 된다.

폭주가 실제로 문제인 쪽은 앙상블이 아니라 **fallback 으로 쓰이는**
`trend_revenue` 다. 그 함수는 최근 3개월 증감률을 한 달치 비율로 바꿔 36개월 등비로
연장하기만 한다. '반등 강'의 3개월 증감률이 +35.1% 라 현재 870만원짜리 가게가
3년 뒤 29,018만원(33배)으로 투영된다. 앙상블이 없을 때만 그 선을 쓴다.

대가는 계절성 계열이다. 진폭 ±32% 인 가게를 계절 저점에서 관측하면(최근 3개월
-30.1%) 세 모델이 그 저점을 추세로 읽는다. 여기서는 감쇠를 켜면 적합 결과가 달라져
오히려 더 낮게 본다 — 실제 수준이 532만원인 가게를 3년 뒤 344만원으로 본다(감쇠가
없으면 501만원). **계절 저점을 추세로 읽는 문제를 감쇠가 풀어주지는 않는다.**

φ 는 우리가 주지 않는다 — `damped_trend=True` 만 주고 statsmodels 가 추정한다.
다만 **12개월에서는 계열 6개의 φ 가 전부 허용 범위의 끝에 놓였다.** 어느 끝인지는
데이터가 정하지만 범위에 막힌 값이다. 아래 `_ets` 주석 참조.


기각한 것
---------
    음수를 0에서 자르기   꼬리가 통째로 0 이 되어 '매출이 사라진 가게'를 계산하게
                        된다(데모 B·C 는 36개월 뒤가 0원이 됐다)
    Prophet 에도 감쇠     Prophet 에는 감쇠 파라미터가 없어 changepoint_prior_scale 을
                        우리가 골라야 한다. 임의 상수를 쓰지 않는다는 원칙에 걸린다
    Prophet 연간 계절성   12개월에서 켜면 **발산한다** — 데모 A 로 재니 36개월 뒤가
                        1.49x10^44 만원이 나왔다. Prophet 자체가 "2년 미만이면
                        추세·계절 분해가 불안정하다"고 경고하는 구간이고 12개월은
                        그 절반이다. 계절성은 중앙값이 대신 걸러낸다
    홀드아웃 · 신뢰도 지표 붙이지 않는다. planning.md 5.1 이 "36개월 현금흐름은 예측이
                        아니라 현재 추세가 유지된다는 가정 하의 시나리오이며 절대값의
                        정확도는 주장하지 않는다"고 정했다. 정확도를 주장하지 않으니
                        정확도 지표를 붙일 이유가 없다. 검토하고 기각한 셋은
                        planning.md 5.5 참조
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

SIM_MONTHS = 36  # core.cashflow 와 같은 값. 순환 import 를 피하려고 여기에 둔다

ENABLED = True
"""앙상블을 쓸지. False 면 `ensemble_revenue` 가 None 을 돌려주고 호출부가
기존 추세선으로 돌아간다. 배포에서 예측이 말썽일 때 코드를 되돌리지 않고 끄려고 둔다."""

MIN_MONTHS = 12
"""적합에 필요한 최소 월 수. 이보다 적으면 None 을 돌려주고 호출부가 추세선을 쓴다.

**예측 이론에서 나온 값이 아니다.** 지수평활이 요구하는 최소치는 3(수준·추세 각 1 +
관측 1)이고, 12 는 planning.md 5.1 이 정한 **정밀 모드의 전제**("POS 12개월 이상
보유")에서 가져왔다. 다른 영역에서 빌려온 값이라는 뜻이다.

그래도 6 같은 임의값보다 이쪽을 택했다. 12개월 미만이면 반기 비교(직전 6개월 vs
그 전 6개월)가 계산되지 않아 **자격·국면 판정이 이미 다른 규칙으로 돈다.** 그 상태에서
예측만 계속 도는 것은 근거가 아니라 우연이다. 한 화면에 서는 셋을 같은 선에서 끊는다.

12개월 미만이 실제로 들어올 수 있는 경로는 **파일 업로드**뿐이다. 데모 시나리오는
항상 12개월치를 만든다."""

FLOOR = 1
"""매출 하한(원). 로그 공간이라 음수는 안 나오지만 0 은 나올 수 있고,
0 을 넘기면 변동비율 환산 등 나눗셈이 걸린다."""

_log = logging.getLogger(__name__)


def _to_log(values: np.ndarray) -> np.ndarray:
    """로그 공간으로. 0 이하인 달(휴업 등)은 1원으로 올려 로그가 발산하지 않게 한다."""
    return np.log(np.maximum(values, 1.0))


def _linear(values: np.ndarray, horizon: int) -> list[float]:
    """로그 매출에 직선을 맞춰 연장한다. 원 공간에서는 등비 추세가 된다."""
    v = _to_log(values)
    slope, intercept = np.polyfit(np.arange(len(v)), v, 1)
    return [float(np.exp(intercept + slope * (len(v) + h))) for h in range(1, horizon + 1)]


def _ets(values: np.ndarray, horizon: int) -> list[float]:
    """감쇠추세 지수평활. φ 는 statsmodels 가 추정한다(우리가 주지 않는다).

    12개월 실측에서는 계열 6개의 φ 가 **전부 허용 범위의 끝에 놓였다.** 범위는
    statsmodels 가 정한 (0.8, 0.995) 다(`holtwinters/model.py`).

        데모 A·B·C        0.995   감쇠를 거의 원하지 않는다
        반등 2종·계절성    0.800   허용된 것보다 더 센 감쇠를 원한다

    어느 끝에 붙는지는 데이터가 정하므로 정보가 없는 값은 아니다. 다만 **범위에 막힌
    값이라 그 계열에서 감쇠 강도는 데이터가 원한 만큼이 아니다.** 관측이 12개뿐이라
    우도가 평평해 끝으로 밀리기도 쉽다. 이력이 길어지면 다시 볼 항목이다.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    fitted = ExponentialSmoothing(
        _to_log(values), trend="add", seasonal=None, damped_trend=True
    ).fit()
    return [float(np.exp(o)) for o in fitted.forecast(horizon)]


def _prophet(index: pd.PeriodIndex, values: np.ndarray, horizon: int) -> list[float]:
    """Prophet. 연간 계절성은 끈다(위 '기각한 것' 참조)."""
    from prophet import Prophet

    frame = pd.DataFrame({"ds": index.to_timestamp(), "y": _to_log(values)})
    model = Prophet(
        yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False
    )
    model.fit(frame)
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    return [float(np.exp(o)) for o in model.predict(future)["yhat"].iloc[-horizon:]]


MODEL_NAMES: tuple[str, ...] = ("linear", "ets", "prophet")
"""앙상블을 이루는 모델. `ensemble_detail` 이 이 중 실제로 적합된 것을 함께 돌려준다."""

RETRY_ON_PARTIAL = 1
"""세 모델이 다 돌지 않았을 때 다시 시도하는 횟수.

**`ImportError` 로만 실패했으면 다시 시도하지 않는다.** 설치되지 않은 패키지는 다시
불러도 생기지 않고, 그때 재시도는 요청마다 적합을 두 번 하게 만들 뿐이다. 되살릴 수
있는 것은 메모리·경합 같은 일시적 실패다."""


def _fit_all(
    monthly_revenue: pd.Series, values: np.ndarray, horizon: int
) -> tuple[list[list[float]], list[str], list[Exception]]:
    """세 모델을 한 번씩 적합한다. (예측선들, 성공한 이름들, 실패 예외들)."""
    paths: list[list[float]] = []
    used: list[str] = []
    failures: list[Exception] = []
    for name, fit in (
        ("linear", lambda: _linear(values, horizon)),
        ("ets", lambda: _ets(values, horizon)),
        ("prophet", lambda: _prophet(monthly_revenue.index, values, horizon)),
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                paths.append(fit())
            used.append(name)
        except Exception as exc:  # 모델 하나의 실패가 예측 전체를 죽이지 않게 한다
            _log.warning("매출 예측 모델 %s 실패: %s: %s", name, type(exc).__name__, exc)
            failures.append(exc)
    return paths, used, failures


def ensemble_detail(
    monthly_revenue: pd.Series, horizon: int = SIM_MONTHS
) -> tuple[list[int] | None, tuple[str, ...]]:
    """`ensemble_revenue` 의 본체. 예측선과 **실제로 적합된 모델 이름**을 함께 돌려준다.

    이름을 함께 내는 이유 — 하나가 빠져도 남은 것으로 값이 나오는데, 어디에도 표시가
    없으면 **두 모델짜리 앙상블이 도는 것을 모르고 지나간다.** 실제로 prophet 이
    설치되지 않은 채 한동안 그렇게 돌았고 결과가 그럴듯해 알아채지 못했다(2026-09-01).
    `len(models) == len(MODEL_NAMES)` 가 "세 모델이 다 돌았다" 플래그다.

    Returns:
        (예측선, 적합된 모델 이름들). 예측선이 None 이면 이름도 비어 있다.
    """
    if not ENABLED:
        return None, ()
    values = np.asarray(monthly_revenue, dtype=float)
    if len(values) < MIN_MONTHS:
        return None, ()

    paths, used, failures = _fit_all(monthly_revenue, values, horizon)
    retries = 0
    while (
        len(used) < len(MODEL_NAMES)
        and retries < RETRY_ON_PARTIAL
        and not all(isinstance(exc, ImportError) for exc in failures)
    ):
        retries += 1
        _log.warning(
            "매출 예측 %d/%d 모델만 적합 — 다시 시도 (%d/%d)",
            len(used), len(MODEL_NAMES), retries, RETRY_ON_PARTIAL,
        )
        paths, used, failures = _fit_all(monthly_revenue, values, horizon)

    if len(used) < len(MODEL_NAMES):
        _log.warning(
            "매출 예측 앙상블 %d/%d 모델로 계산 (적합: %s)",
            len(used), len(MODEL_NAMES), ", ".join(used) or "없음",
        )
    if not paths:
        return None, ()
    line = [max(int(np.median([p[h] for p in paths])), FLOOR) for h in range(horizon)]
    return line, tuple(used)


def ensemble_revenue(
    monthly_revenue: pd.Series, horizon: int = SIM_MONTHS
) -> list[int] | None:
    """월별 매출에서 `horizon` 개월 예측선을 만든다. 못 만들면 None.

    Args:
        monthly_revenue: 월별 매출(원). 인덱스는 Period('M'). `PosSummary` 의 필드를
                         그대로 넘기면 된다.

    Returns:
        길이 `horizon` 의 월 매출(원) 목록. 다음 달이 [0] 이다.
        None 이면 호출부가 기존 추세선을 쓴다 — ENABLED 가 False · 데이터가 짧음 ·
        세 모델이 모두 실패, 셋 중 하나다.

    **모델 하나가 실패해도 남은 것으로 낸다.** 셋 중 둘이면 중앙값이 평균이 되고
    하나면 그 모델의 값이 된다. Prophet 은 환경에 따라 초기화가 실패하는데
    (CLAUDE.md 의 cmdstanpy 항목), 그때 예측 전체가 죽는 것보다 낫다.
    몇 개가 돌았는지 알아야 하면 `ensemble_detail` 을 쓴다.
    """
    return ensemble_detail(monthly_revenue, horizon)[0]
