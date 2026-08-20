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


def main() -> None:
    print("=" * 50)
    print("app.py UI 헬퍼 검증")
    print("=" * 50)
    test_money_to_man()
    test_months_label()
    print("\n전부 통과")


if __name__ == "__main__":
    main()
