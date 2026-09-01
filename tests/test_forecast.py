"""매출 예측(앙상블) 검증.

`python -m tests.test_forecast` 로 실행. **PYTHONUTF8=1 을 붙이지 마라** —
한국어 Windows 에서 Prophet 초기화가 실패해 검사 하나가 무의미해진다
(CLAUDE.md 의 cmdstanpy 항목 참조). 다른 테스트와 실행법이 다른 유일한 파일이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import forecast

MAN = 10_000


def series(values: list[float]) -> pd.Series:
    """월별 매출 Series. 인덱스는 PosSummary.monthly_revenue 와 같은 Period('M')."""
    index = pd.period_range("2024-09", periods=len(values), freq="M")
    return pd.Series([v * MAN for v in values], index=index)


# 길이는 12개월로 맞춘다. 앱이 정밀 모드에 넘기는 길이이고(app.py 의 시나리오 생성),
# forecast.MIN_MONTHS 도 같은 값이다. 24개월로 재면 앱과 다른 것을 검사하게 된다.
def decline_12() -> pd.Series:
    """12개월 완만한 하락. 노이즈를 조금 섞어 직선이 되지 않게 한다."""
    rng = np.random.default_rng(7)
    base = [800 * (0.97**m) for m in range(12)]
    return series([b * float(rng.normal(1.0, 0.04)) for b in base])


def rebound_12() -> pd.Series:
    """급락 후 최근 4개월 반등. 진단은 매출감소가 작다고 보는데 예측선만 급락하면
    화면끼리 어긋난다. 감쇠가 그 폭을 좁히는 대상이다."""
    base = [800 * (0.93**m) for m in range(8)]
    base += [base[-1] * (1.07 ** (m + 1)) for m in range(4)]
    return series(base)


def test_no_negative_and_length() -> None:
    """36개월 전부 양수이고 길이가 맞는다. 로그 공간을 쓰는 이유가 이것이다."""
    for name, data in (("하락", decline_12()), ("반등", rebound_12())):
        out = forecast.ensemble_revenue(data)
        assert out is not None, name
        assert len(out) == forecast.SIM_MONTHS, (name, len(out))
        assert all(v >= forecast.FLOOR for v in out), (name, min(out))
        print(f"  {name} 12개월 → 36개월 예측: "
              f"1개월 {out[0] / MAN:,.0f}만 · 36개월 {out[-1] / MAN:,.0f}만 · 전부 양수")


def test_rebound_is_not_extrapolated_to_absurdity() -> None:
    """반등 계열에서 36개월 뒤가 현재의 3배를 넘지 않는다.

    **이 경계가 지키는 대상은 앙상블이지 `trend_revenue` 가 아니다.** 같은 성격의
    계열에서 `trend_revenue` 는 33배를 내는데, 그건 fallback 경로라 여기서 안 잡힌다.
    앙상블 실측은 반등 계열에서 최대 1.37배였으므로 3배는 여유를 둔 자리다. 제도
    근거가 있는 숫자가 아니라, 폭주와 정상 예측을 가르려고 둔 값이다.
    """
    data = rebound_12()
    current = float(data.iloc[-3:].mean())
    out = forecast.ensemble_revenue(data)
    assert out is not None
    ratio = out[-1] / current
    assert ratio <= 3.0, f"36개월 뒤가 현재의 {ratio:.1f}배"
    print(f"  반등 계열: 현재 {current / MAN:,.0f}만 → 36개월 {out[-1] / MAN:,.0f}만 "
          f"({ratio:.2f}배)")


def test_short_history_returns_none() -> None:
    """MIN_MONTHS 미만이면 예측하지 않는다. 호출부가 추세선으로 돌아간다."""
    n = forecast.MIN_MONTHS
    assert forecast.ensemble_revenue(series([800] * (n - 1))) is None
    assert forecast.ensemble_revenue(series([800] * n)) is not None
    print(f"  {n - 1}개월 → None · {n}개월(MIN_MONTHS) → 예측")


def test_disabled_switch() -> None:
    """ENABLED=False 면 None. 배포에서 코드를 되돌리지 않고 끌 수 있어야 한다."""
    forecast.ENABLED = False
    try:
        assert forecast.ensemble_revenue(decline_12()) is None
    finally:
        forecast.ENABLED = True
    print("  ENABLED=False → None")


def test_survives_model_failure() -> None:
    """모델 하나가 죽어도 남은 둘로 예측한다.

    Prophet 은 환경에 따라 초기화가 실패한다. 그때 예측 전체가 죽으면 안 된다.
    """
    original = forecast._prophet
    forecast._prophet = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("일부러"))
    try:
        out = forecast.ensemble_revenue(decline_12())
    finally:
        forecast._prophet = original
    assert out is not None and len(out) == forecast.SIM_MONTHS
    print("  Prophet 실패 → 남은 두 모델로 예측")


def test_all_three_models_run_here() -> None:
    """세 모델이 **전부** 도는가. 하나라도 빠지면 실패로 알린다.

    빠져도 앙상블은 값을 내므로 위 검사들은 전부 통과한다. 그래서 prophet 이 설치되지
    않은 채 두 모델로 도는 상태를 한동안 모르고 지나갔다(2026-09-01). 흔한 원인 둘:
        · prophet 미설치 — requirements.txt 대로 설치했는지
        · PYTHONUTF8=1 로 실행 — 이 파일만은 붙이지 않는다(CLAUDE.md 의 cmdstanpy 항목)

    옛 이름은 `test_prophet_actually_runs_here` 였고 실패해도 경고만 찍었다. 경고는
    지나쳐 읽힌다.
    """
    line, models = forecast.ensemble_detail(decline_12())
    assert line is not None, "예측선이 안 나왔다"
    missing = [m for m in forecast.MODEL_NAMES if m not in models]
    assert not missing, f"적합되지 않은 모델: {', '.join(missing)}"
    print(f"  세 모델 전부 적합: {', '.join(models)}")


def test_app_path_uses_the_ensemble() -> None:
    """앱이 정밀 모드에서 실제로 앙상블을 쓰는가.

    `test_app` 은 예측을 꺼 두고 배선만 본다(그 파일은 PYTHONUTF8=1 로 돌아
    Prophet 이 빠진 두 모델을 검사하게 된다). 앱 경로가 앙상블에 닿는지는
    **세 모델이 다 도는 여기서** 확인한다.
    """
    import app
    from core import pos
    from data.synthetic_pos import generate

    summary = pos.summarize(pos.load(generate("A", months=12)))
    on, from_ensemble, all_models = app.revenue_line(True, summary, 0)
    assert on == ensemble_revenue_of(summary), "앱 경로가 앙상블 결과와 다르다"
    # 앱까지 오는 플래그 둘. 화면을 가르지는 않지만 여기서 값이 맞는지는 확인한다.
    assert from_ensemble, "앙상블에서 나왔는데 플래그가 False 다"
    assert all_models, "세 모델이 다 돌았는데 플래그가 False 다"

    forecast.ENABLED = False
    try:
        off = app.build_monthly_revenue(True, summary, 0)
    finally:
        forecast.ENABLED = True
    assert off != on, "예측을 꺼도 같은 값이라 앙상블을 쓰는지 알 수 없다"
    print(f"  앱 경로 확인: 앙상블 36개월 {on[-1] / MAN:,.0f}만 · "
          f"끄면 추세선 {off[-1] / MAN:,.0f}만")


def ensemble_revenue_of(summary) -> list[int] | None:
    """위 검사가 비교 대상으로 쓰는 값. 호출을 한 줄로 읽히게 하려고 뺐다."""
    return forecast.ensemble_revenue(summary.monthly_revenue)


def main() -> None:
    print("=" * 62)
    print("매출 예측 검증 - 앙상블")
    print("=" * 62)
    for fn in (
        test_no_negative_and_length,
        test_rebound_is_not_extrapolated_to_absurdity,
        test_short_history_returns_none,
        test_disabled_switch,
        test_survives_model_failure,
        test_app_path_uses_the_ensemble,
        test_all_three_models_run_here,
    ):
        fn()
    print("\n" + "=" * 62)
    print("전부 통과")
    print("=" * 62)


if __name__ == "__main__":
    main()
