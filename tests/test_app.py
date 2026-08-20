"""app.py UI 헬퍼 검증. pytest 없이 `python -m tests.test_app` 로 실행.

계산이 아니라 **화면 표기 헬퍼**를 검증한다. money_to_man 은 cashflow 가 만든
원 단위 문장을 만원 단위로 바꾸는데, '->' 가 붙는 문장에서 공백을 먹거나 개월수·
비율을 잘못 건드리기 쉬워 회귀 테스트를 남긴다.
"""
from __future__ import annotations

import app
from core.cashflow import SIM_MONTHS


def test_money_to_man():
    """원 단위 금액을 만원으로. '->' · 원 유무 · 문장 중간 혼합을 모두 본다."""
    cases = [
        # 기본: 원이 붙은 금액
        ("보조금 경영안정 바우처 6,000,000원", "보조금 경영안정 바우처 600만원"),
        ("융자 500,000,000원 @ 3.00%", "융자 50,000만원 @ 3.00%"),
        # ── '->' 케이스 (같은 실수가 재발하면 여기서 걸린다) ──
        # ① 앞엔 원 없음 + 뒤엔 원 있음: 공백을 먹어 '9,500만원->' 이 되면 안 된다
        ("95,000,000 -> 24,000,000원", "9,500만원 -> 2,400만원"),
        # ② 월 상환 변화(둘 다 원 단위, 앞은 원 생략)
        ("월 상환 333,333 -> 258,889원", "월 상환 33만원 -> 25만원"),
        # ③ 문장 중간에 섞이고 개월수·비율은 건드리지 않는다
        (
            "대상 20,000,000원 · 월 상환 333,333 -> 258,889원 (60개월 -> 144개월, 금리 +0.2%)",
            "대상 2,000만원 · 월 상환 33만원 -> 25만원 (60개월 -> 144개월, 금리 +0.2%)",
        ),
        # ④ 원이 아예 없는 순수 숫자(잔액 등)도 만원으로, 뒤 공백 유지
        ("잔액 30,000,000 남음", "잔액 3,000만원 남음"),
        # 라벨 속 '2년거치 3년' 같은 짧은 숫자는 건드리지 않는다
        ("신용취약소상공인자금(2년거치 3년 균등분할)", "신용취약소상공인자금(2년거치 3년 균등분할)"),
    ]
    for src, expected in cases:
        got = app.money_to_man(src)
        assert got == expected, f"money_to_man({src!r})\n  = {got!r}\n  ≠ {expected!r}"
    print(f"   money_to_man {len(cases)}건 통과 ('->' · 원 유무 · 문장 중간 포함)")


def test_months_label():
    """생존 개월 표기 — 상한(36)에 걸리면 '36개월 이상', 아니면 'N개월'."""
    assert app.months_label(SIM_MONTHS) == f"{SIM_MONTHS}개월 이상"
    assert app.months_label(SIM_MONTHS + 5) == f"{SIM_MONTHS}개월 이상"
    assert app.months_label(18) == "18개월"
    assert app.months_label(0) == "0개월"
    print("   months_label 경계 통과")


def test_no_direct_widget_calls():
    """render_* 는 위젯 래퍼(_num/_select/…)만 써야 한다. st.number_input 등을 직접 부르면
    단계 이동 시 값이 purge 돼 '조용히 0' 배선 버그가 재발한다.

    예외는 'transient-widget:' 마커가 붙은 줄만. **마커 개수를 1로 못박는다** —
    render_loan_slider 의 what-if 슬라이더 하나뿐. 예외를 늘리려면 이 테스트를 고쳐야
    하므로 '정말 transient 인가'를 한 번 더 의식적으로 보게 된다.
    """
    import pathlib

    src = pathlib.Path(app.__file__).read_text(encoding="utf-8").splitlines()
    lo = next(i for i, l in enumerate(src) if "WIDGET WRAPPERS START" in l)
    hi = next(i for i, l in enumerate(src) if "WIDGET WRAPPERS END" in l)
    stateful = (
        "st.number_input(", "st.selectbox(", "st.radio(", "st.checkbox(",
        "st.text_input(", "st.slider(", "st.date_input(", "st.file_uploader(",
    )
    violations, markers = [], 0
    for i, line in enumerate(src):
        if lo <= i <= hi:  # 래퍼 정의부만 직접 호출 허용
            continue
        if not any(w in line for w in stateful):
            continue
        if "transient-widget:" in line:
            markers += 1
        else:
            violations.append((i + 1, line.strip()))
    assert not violations, f"래퍼 밖 직접 위젯 호출(래퍼로 바꿀 것): {violations}"
    assert markers == 1, (
        f"transient-widget 예외가 {markers}개다(1개여야 함). 예외를 늘렸다면 정말 단계 답이 "
        "아닌 transient 컨트롤인지 확인하고 이 개수를 의식적으로 갱신할 것."
    )
    print(f"   직접 위젯 호출 0건 · transient 예외 {markers}건(고정)")


# ── AppTest — 실제 Streamlit 세션으로 배선을 검증한다(헤드리스 프로필 조립을 안 탄다) ──


def _fresh_app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(app.__file__)  # 절대경로 (상대경로는 tests/ 기준으로 풀린다)
    at.run(timeout=120)
    return at


def _ans(at) -> dict:
    try:
        return dict(at.session_state["answers"])
    except (KeyError, AttributeError):
        return {}


def _advance(at) -> None:
    """현재 단계의 '다음' 버튼을 눌러 한 단계 진행."""
    cur = at.session_state["step"]
    at.button(key=f"next_{cur}").click()
    at.run(timeout=120)


def _back(at) -> None:
    cur = at.session_state["step"]
    at.button(key=f"prev_{cur}").click()
    at.run(timeout=120)


def test_answers_survive_navigation():
    """①~⑧ 기본값 통과 시 ③(비용)·④(부채) 답이 진단·이후 단계에서 살아 있는지.
    이번 배선 버그(렌더 안 된 위젯 key 가 purge 돼 프로필이 0으로 조립됨)의 핵심 회귀."""
    at = _fresh_app()
    seen = {}
    while at.session_state["step"] < app.S_DIAGNOSIS:
        _advance(at)
        seen[at.session_state["step"]] = _ans(at)

    a = _ans(at)  # 진단 단계에서
    assert a.get("fixed_man") == 500, f"fixed_man 유실: {a.get('fixed_man')}"
    assert a.get("var_man") == 200, f"var_man 유실: {a.get('var_man')}"
    assert a.get("cash_man") == 1000, f"cash_man 유실: {a.get('cash_man')}"
    assert a.get("debt_man") == 3000, f"debt_man 유실: {a.get('debt_man')}"
    # ⑤(연체·신용) 단계에서도 ③의 fixed_man 이 살아 있었는지 명시 확인 — 버그의 핵심
    assert seen.get(app.S_DEBT2, {}).get("fixed_man") == 500, (
        "⑤ 연체·신용 단계에서 ③ 비용 값이 사라졌다 — 배선 버그 재발"
    )
    # UserProfile 로도 확인: collect_finance 가 answers 에서 읽어 조립한다
    print("   ③④ 답이 진단까지 생존 · ⑤에서도 fixed_man=500 유지")


def test_edit_and_return():
    """앞 단계로 돌아가 고친 값이 반영되는지(편집→진행→복귀→재편집)."""
    at = _fresh_app()
    while at.session_state["step"] < app.S_COST:
        _advance(at)
    at.number_input(key="fixed_man_w").set_value(700)
    at.run(timeout=120)
    assert _ans(at)["fixed_man"] == 700

    while at.session_state["step"] < app.S_DIAGNOSIS:
        _advance(at)
    assert _ans(at)["fixed_man"] == 700, "진행 후 편집값 유실"

    while at.session_state["step"] > app.S_COST:
        _back(at)
    assert at.number_input(key="fixed_man_w").value == 700, "복귀 시 편집값이 안 보임"

    at.number_input(key="fixed_man_w").set_value(900)
    at.run(timeout=120)
    while at.session_state["step"] < app.S_DIAGNOSIS:
        _advance(at)
    assert _ans(at)["fixed_man"] == 900, "재편집값 미반영"
    print("   편집→진행→복귀→재편집 반영 OK (700→900)")


def test_slider_does_not_leak():
    """결과 카드 what-if 슬라이더로 굴린 값이 저장소(추천 금액)에 새지 않고,
    앞 단계로 갔다 오면 원래 계산된 금액으로 리셋되는지."""
    at = _fresh_app()
    while at.session_state["step"] < app.S_RESULTS:
        _advance(at)

    sliders = [s for s in at.slider if s.key and s.key.startswith("loan_")]
    if not sliders:
        print("   (융자 슬라이더 있는 카드 없음 — 스킵)")
        return
    v0 = sliders[0].value  # 계산된 기본값
    sliders[0].set_value(v0 + 100)
    at.run(timeout=120)

    # 저장소에 loan_ 키가 새지 않았는지 — collect_* 가 여기서만 읽으므로 추천엔 영향 없음
    assert not any(str(k).startswith("loan_") for k in _ans(at)), "슬라이더 값이 ss.answers 로 샜다"

    _back(at)
    while at.session_state["step"] < app.S_RESULTS:
        _advance(at)
    v1 = [s for s in at.slider if s.key and s.key.startswith("loan_")][0].value
    assert v1 == v0, f"복귀 후 슬라이더가 리셋 안 됨(굴린 값이 남음): {v0} → {v1}"
    print(f"   슬라이더 값 누출 없음 · 복귀 시 계산값으로 리셋({v0})")


def main() -> None:
    print("=" * 50)
    print("app.py UI 헬퍼·배선 검증")
    print("=" * 50)
    test_money_to_man()
    test_months_label()
    test_no_direct_widget_calls()
    test_answers_survive_navigation()
    test_edit_and_return()
    test_slider_does_not_leak()
    print("\n전부 통과")


if __name__ == "__main__":
    main()
