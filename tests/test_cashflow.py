"""현금흐름 엔진 검증.

pytest 없이도 `python -m tests.test_cashflow`로 돌려볼 수 있게 만들었다.
기획서의 데모 시나리오 3개(A 무부채·B 거치절벽·C 연체회복불가)를 염두에 둔 케이스.
"""

from __future__ import annotations

from core import loader
from core.cashflow import (
    NET_DEBT_MONTH,
    SIM_MONTHS,
    Applied,
    UserProfile,
    simulate,
    추세매출,
)

db = loader.load()
만 = 10_000


def _프로필_분식집() -> UserProfile:
    """기획서 예시 1 — 충무로 분식집. 매출 하락, 기존 대출 있음.

    ※ 기획서 예시 1의 수치는 그대로 쓰면 앞뒤가 안 맞는다.
      매출 1,440만 / 고정비 720만 / 변동비율 32%로 계산하면
      영업 현금흐름이 +259만원(흑자)인데 기획서는 "월 순이익 −85만원"이라 적혀 있다.
      여기서는 순이익 −85만원 쪽을 기준으로 삼아 고정비를 역산했다.
      (1,440 − 1,440×0.32 − 고정비 = −85  →  고정비 ≈ 1,064만원)
      → 기획서 예시 수치를 이 값으로 정정할 것.
    """
    return UserProfile(
        보유현금=800 * 만,
        월매출=추세매출(1_440 * 만, -0.015),  # 월 1.5% 하락
        고정비=1_064 * 만,
        변동비율=0.32,
        기존부채_잔액=4_000 * 만,
        기존부채_월상환=65 * 만,
        기존부채_금리=0.07,  # 대환 대상(7% 이상)
    )


def _출력(제목: str, r) -> None:
    print(f"\n── {제목}")
    print(f"   생존 {r.생존표기} · {NET_DEBT_MONTH}개월 순부채 {r.순부채_30/만:,.0f}만원")
    for 줄 in r.적용내역:
        print(f"   · {줄}")


def test_베이스라인():
    """무지원 = 현재 버퍼. 조합을 안 넣으면 나온다."""
    p = _프로필_분식집()
    r = simulate(p)
    _출력("무지원 (현재 버퍼)", r)
    assert r.생존개월수 < SIM_MONTHS, "적자 상태인데 36개월을 버티면 안 됨"
    assert len(r.잔고) == SIM_MONTHS + 1
    return r


def test_거치절벽():
    """일반경영안정 7,000만원 — 25개월차에 상환이 시작되는지."""
    p = _프로필_분식집()
    c = db.find("fund_general__2y3y")
    r = simulate(p, [Applied(c, 7_000 * 만)])
    _출력("일반경영안정 7,000만원 (2년거치 3년)", r)

    거치중 = r.월상환[23]  # 24개월차
    상환후 = r.월상환[24]  # 25개월차
    print(f"   24개월차 상환 {거치중/만:,.0f}만원 → 25개월차 {상환후/만:,.0f}만원")
    assert 상환후 > 거치중 * 1.5, "거치 종료 시점에 상환액이 급증해야 함(절벽)"
    return r


def test_만기일시():
    """서울시 성장기반자금 2년만기 일시상환 — 24개월차 낭떠러지."""
    p = _프로필_분식집()
    c = db.find("seoul_growth_base__2y_bullet")
    r = simulate(p, [Applied(c, 5_000 * 만)])
    _출력("서울시 성장기반 5,000만원 (2년만기 일시상환)", r)

    직전 = r.월상환[22]
    만기 = r.월상환[23]  # 24개월차에 원금 전액
    print(f"   23개월차 {직전/만:,.0f}만원 → 24개월차 {만기/만:,.0f}만원")
    assert 만기 > 4_000 * 만, "만기에 원금 전액이 나가야 함"
    # 부채잔액은 (기존 + 신규) 합계이므로, 만기 전후 감소폭으로 확인한다
    감소 = r.부채잔액[23] - r.부채잔액[24]
    print(f"   부채잔액 {r.부채잔액[23]/만:,.0f} → {r.부채잔액[24]/만:,.0f}만원")
    assert 감소 > 4_900 * 만, "만기 상환으로 신규 융자 잔액이 사라져야 함"
    return r


def test_대환전환():
    """대환은 유입이 없고 월 상환액만 낮춘다."""
    p = _프로필_분식집()
    base = simulate(p)
    c = db.find("fund_refinance__2y8y")
    r = simulate(p, [Applied(c, 4_000 * 만)])
    _출력("대환대출 4,000만원 (7% → 4.5%, 2년거치 8년)", r)

    assert sum(r.유입) == 0, "대환은 현금 유입이 없어야 함"
    print(f"   월 상환 {base.월상환[0]/만:,.1f}만원 → {r.월상환[0]/만:,.1f}만원")
    assert r.월상환[0] < base.월상환[0], "월 상환 부담이 줄어야 함"
    assert r.생존개월수 > base.생존개월수, "생존 개월수가 늘어야 함"
    return r


def test_채무조정():
    """새출발기금 부실차주 — 원금 감면 + 월 상환액 급감."""
    p = UserProfile(
        보유현금=300 * 만,
        월매출=추세매출(1_100 * 만, -0.03),
        고정비=950 * 만,
        변동비율=0.32,
        기존부채_잔액=9_500 * 만,
        기존부채_월상환=165 * 만,
        연체개월=4,
    )
    base = simulate(p)
    _출력("무지원 (연체 4개월)", base)

    c = db.find("new_start__default")
    r = simulate(p, [Applied(c)])
    _출력("새출발기금 부실차주 (원금 80% 감면 추정)", r)

    assert r.순부채_30 < base.순부채_30, "순부채가 줄어야 함"
    assert r.감면액 > 0
    assert sum(r.유입) == 0, "채무조정은 현금 유입이 없어야 함"
    return r


def test_상환유예():
    """소진공 상환연장제도 — 원금 감면 없이 월 상환액만 낮춤."""
    p = _프로필_분식집()
    p.정책자금_직접대출_잔액 = 3_000 * 만
    base = simulate(p)
    c = db.find("repay_extension__max5y")
    r = simulate(p, [Applied(c)])
    _출력("상환연장제도 (대상 3,000만원, 최대 5년 연장)", r)

    print(f"   월 상환 {base.월상환[0]/만:,.1f}만원 → {r.월상환[0]/만:,.1f}만원")
    assert r.월상환[0] < base.월상환[0], "월 상환액이 줄어야 함"
    assert r.감면액 == 0, "상환유예는 원금 감면이 없어야 함"
    return r


def test_비용감면():
    """경영안정 바우처 25만원 — 해당 월 고정비 감소."""
    p = _프로필_분식집()
    base = simulate(p)
    c = db.find("voucher_mgmt__default")
    r = simulate(p, [Applied(c)])
    _출력("경영안정 바우처 25만원", r)
    assert r.잔고[1] > base.잔고[1], "첫 달 잔고가 바우처만큼 늘어야 함"
    return r


def test_조합():
    """대환 + 긴급경영안정 — 기획서 예시 1의 1순위 조합."""
    p = _프로필_분식집()
    r = simulate(
        p,
        [
            Applied(db.find("fund_refinance__2y8y"), 4_000 * 만),
            Applied(db.find("fund_emergency__2y3y"), 3_000 * 만),
        ],
    )
    _출력("대환 4,000만 + 긴급경영안정 3,000만", r)
    assert sum(r.유입) == 3_000 * 만, "융자 유입만 잡혀야 함(대환은 0)"
    return r


def main() -> None:
    print("=" * 62)
    print("현금흐름 엔진 검증 — 현금효과 6종 + 조합")
    print("=" * 62)
    for fn in (
        test_베이스라인,
        test_거치절벽,
        test_만기일시,
        test_대환전환,
        test_채무조정,
        test_상환유예,
        test_비용감면,
        test_조합,
    ):
        fn()
    print("\n" + "=" * 62)
    print("전부 통과")
    print("=" * 62)


if __name__ == "__main__":
    main()
