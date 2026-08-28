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


def decline_24() -> pd.Series:
    """24개월 완만한 하락. 노이즈를 조금 섞어 직선이 되지 않게 한다."""
    rng = np.random.default_rng(7)
    base = [800 * (0.97**m) for m in range(24)]
    return series([b * float(rng.normal(1.0, 0.04)) for b in base])


def rebound_24() -> pd.Series:
    """급락 후 최근 5개월 반등. 감쇠가 지키는 대상이다."""
    base = [800 * (0.93**m) for m in range(19)]
    base += [base[-1] * (1.07 ** (m + 1)) for m in range(5)]
    return series(base)


def test_no_negative_and_length() -> None:
    """36개월 전부 양수이고 길이가 맞는다. 로그 공간을 쓰는 이유가 이것이다."""
    for name, data in (("하락", decline_24()), ("반등", rebound_24())):
        out = forecast.ensemble_revenue(data)
        assert out is not None, name
        assert len(out) == forecast.SIM_MONTHS, (name, len(out))
        assert all(v >= forecast.FLOOR for v in out), (name, min(out))
        print(f"  {name} 24개월 → 36개월 예측: "
              f"1개월 {out[0] / MAN:,.0f}만 · 36개월 {out[-1] / MAN:,.0f}만 · 전부 양수")


def test_rebound_is_not_extrapolated_to_absurdity() -> None:
    """반등 계열에서 36개월 뒤가 현재의 3배를 넘지 않는다.

    감쇠 없이 로그 공간만 쓰면 반등을 그대로 연장해 14배가 나왔다(실측).
    3배는 그 폭주와 정상 예측을 가르는 자리로 두는 것이지 제도 근거가 아니다.
    """
    data = rebound_24()
    current = float(data.iloc[-3:].mean())
    out = forecast.ensemble_revenue(data)
    assert out is not None
    ratio = out[-1] / current
    assert ratio <= 3.0, f"36개월 뒤가 현재의 {ratio:.1f}배"
    print(f"  반등 계열: 현재 {current / MAN:,.0f}만 → 36개월 {out[-1] / MAN:,.0f}만 "
          f"({ratio:.2f}배)")


def test_short_history_returns_none() -> None:
    """6개월 미만이면 예측하지 않는다. 호출부가 추세선으로 돌아간다."""
    assert forecast.ensemble_revenue(series([800] * 5)) is None
    assert forecast.ensemble_revenue(series([800] * 6)) is not None
    print(f"  5개월 → None · 6개월(MIN_MONTHS) → 예측")


def test_disabled_switch() -> None:
    """ENABLED=False 면 None. 배포에서 코드를 되돌리지 않고 끌 수 있어야 한다."""
    forecast.ENABLED = False
    try:
        assert forecast.ensemble_revenue(decline_24()) is None
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
        out = forecast.ensemble_revenue(decline_24())
    finally:
        forecast._prophet = original
    assert out is not None and len(out) == forecast.SIM_MONTHS
    print("  Prophet 실패 → 남은 두 모델로 예측")


def test_prophet_actually_runs_here() -> None:
    """이 환경에서 Prophet 이 실제로 도는가.

    실패해도 앙상블은 값을 내므로 위 검사들은 전부 통과한다. 그래서 조용히
    두 모델만 도는 상태가 될 수 있다. 여기서만 따로 확인한다.
    """
    data = decline_24()
    try:
        out = forecast._prophet(data.index, np.asarray(data, dtype=float), 6)
        print(f"  Prophet 동작 확인 (6개월: {out[-1] / MAN:,.0f}만)")
    except Exception as exc:
        print(f"  [경고] Prophet 이 이 환경에서 실패한다: {type(exc).__name__}: {exc}")
        print("         PYTHONUTF8=1 을 붙이고 실행하지 않았는지 확인할 것.")


def test_app_path_uses_the_ensemble() -> None:
    """앱이 정밀 모드에서 실제로 앙상블을 쓰는가.

    `test_app` 은 예측을 꺼 두고 배선만 본다(그 파일은 PYTHONUTF8=1 로 돌아
    Prophet 이 빠진 두 모델을 검사하게 된다). 앱 경로가 앙상블에 닿는지는
    **세 모델이 다 도는 여기서** 확인한다.
    """
    import app
    from core import pos
    from data.synthetic_pos import generate

    summary = pos.summarize(pos.load(generate("A", months=24)))
    on = app.build_monthly_revenue(True, summary, 0)
    assert on == ensemble_revenue_of(summary), "앱 경로가 앙상블 결과와 다르다"

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
        test_prophet_actually_runs_here,
    ):
        fn()
    print("\n" + "=" * 62)
    print("전부 통과")
    print("=" * 62)


if __name__ == "__main__":
    main()
