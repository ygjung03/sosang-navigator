"""매출 예측 — Prophet · ETS · 선형회귀의 앙상블(월별 중앙값).

    ensemble_revenue(monthly_revenue, horizon) -> list[int]

세 모델을 **로그 공간**에서 적합하고 ETS 에는 감쇠추세를 준다. 왜 그렇게 하는지는
아래 두 절에 적는다. 실측 근거는 planning.md 5.5 의 예측 항목에 있다.


로그 공간에서 적합하는 이유
---------------------------
세 모델이 전부 덧셈형이라 원 공간에서는 **같은 방향으로 함께 폭주한다.** 중앙값이
폭주를 잡아준다는 앙상블의 전제가 그때 무너진다. 데모 A 실측: 36개월 뒤 선형회귀
-709만 · ETS -193만 · Prophet -176만으로 셋 다 음수라 중앙값도 음수였다.
매출이 음수라는 것은 그 자체로 틀린 그림이다(데모 C 는 36개월 누적 -1.42억이 됐다).

로그 공간에서는 셋이 곱셈형이 되어 0 에 점근할 뿐 뚫지 않는다. 임계값을 두거나
0 에서 자르는 것과 다르다 — 자르는 방식은 앞 구간이 과대평가된 채 남아 오히려
낙관적이 된다(데모 A 36개월 누적: 자름 0.53억 > 로그 0.39억).


ETS 에 감쇠추세를 주는 이유 — 방향을 가리지 않는다
--------------------------------------------------
감쇠의 가정은 "하락이 느려진다"가 아니라 **"추세는 무한히 지속되지 않는다"** 이다.
상승도 하락도 똑같이 눌린다. 그래서 **급락 후 반등 중인 사용자**를 지키는 장치가
된다. 반등 계열 3종(진단 매출감소 0점) 실측, 36개월 뒤:

    현재 458만 → 감쇠 없음 6,412만(14배) · 감쇠 610만
    현재 610만 → 감쇠 없음   388만(-36%) · 감쇠 582만
    현재 916만 → 감쇠 없음 4,343만(4.7배) · 감쇠 1,067만

감쇠가 없으면 진단이 "매출감소 0점"인 사용자에게 3년 뒤 14배를 그린다. 반대로
'반등 강'에서는 진단 0점인데 그래프만 -36%로 내려가 화면끼리 어긋난다. 감쇠는
양쪽을 다 막는다. 반등을 "곧 사그라든다"고 말하는 쪽은 감쇠가 아니라 선형회귀이고
(72만), 중앙값이 그것을 이미 버린다.

φ 는 우리가 주지 않는다 — `damped_trend=True` 만 주고 statsmodels 가 추정한다.


기각한 것
---------
    음수를 0에서 자르기   앞 구간이 과대평가된 채 남아 오히려 낙관적이 된다
    Prophet 에도 감쇠     Prophet 에는 감쇠 파라미터가 없어 changepoint_prior_scale 을
                        우리가 골라야 한다(임의 상수). 빼고 재보니 결과가 사실상
                        같았다(A 154만 동일 · C 15.7만 동일 · B 116 vs 120만)
    Prophet 연간 계절성   24개월은 Prophet 자체가 "2년 미만이면 분해가 불안정하다"고
                        경고하는 구간이고, 켜서 잰 계열에서 실제로 출렁였다
    홀드아웃 가중치 학습   검증 가능한 구간은 2~3개월인데 쓰는 값은 30개월 뒤다.
                        M5 우승·2위·3위가 전부 동일 가중이었다(planning.md 참조)
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

MIN_MONTHS = 6
"""적합에 필요한 최소 월 수. 이보다 적으면 None 을 돌려준다.

6개월은 ETS 가 추세를 추정하는 데 필요한 최소치(수준·추세 각 1 + 여유)에서 왔다.
이보다 짧은 POS 는 추세랄 것이 없어 예측보다 현 수준 유지가 정직하다."""

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
    """감쇠추세 지수평활. φ 는 statsmodels 가 추정한다(우리가 주지 않는다)."""
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
    """
    if not ENABLED:
        return None
    values = np.asarray(monthly_revenue, dtype=float)
    if len(values) < MIN_MONTHS:
        return None

    paths: list[list[float]] = []
    for name, fit in (
        ("linear", lambda: _linear(values, horizon)),
        ("ets", lambda: _ets(values, horizon)),
        ("prophet", lambda: _prophet(monthly_revenue.index, values, horizon)),
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                paths.append(fit())
        except Exception as exc:  # 모델 하나의 실패가 예측 전체를 죽이지 않게 한다
            _log.warning("매출 예측 모델 %s 실패: %s: %s", name, type(exc).__name__, exc)

    if not paths:
        return None
    return [
        max(int(np.median([p[h] for p in paths])), FLOOR) for h in range(horizon)
    ]
