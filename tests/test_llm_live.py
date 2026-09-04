"""실제 LLM 을 부르는 검증. **기본으로는 돌지 않는다.**

    SOSANG_LLM_LIVE=1 python -m tests.test_llm_live

프롬프트가 제대로 나가는지, JSON 이 파싱되는지, 숫자 검증에 걸리지 않는지는 실제로
불러 봐야 안다. 다만 호출마다 돈이 나가고(약 $0.03) 응답이 매번 달라 결과가 흔들리므로
`tests/test_app.py` 에서는 부르지 않는다 — 거기는 고정 응답으로 돌린다.

**언제 돌리나** — 시스템 프롬프트나 `to_prompt_text()` 를 고친 뒤, 그리고 배포 전.
"""
from __future__ import annotations

import os
import sys

import app
from core import agent, optimizer
from tests.test_app import _fresh_app, _to_results, _advance, _back

IN_RATE, OUT_RATE = 2.00, 10.00   # $/MTok (claude-sonnet-5)
ENV_FLAG = "SOSANG_LLM_LIVE"

BANNED = (
    "레이어", "그룹", "후보", "파레토", "비지배", "좌표", "함정", "4종",
    "1번", "2번", "3번", "4번", "5번",
)
"""사용자 문장에 나오면 안 되는 내부 표기.

시스템 프롬프트의 '절대 규칙' 과 **같은 목록이어야 한다.** 한쪽만 늘리면 검사가
프롬프트를 못 따라간다.

'그룹'·'후보'·'함정'·'4종'은 2026-09-04 실측에서 실제로 샜다. 그때까지 이 목록은
('레이어', '1번', '2번', '3번')뿐이라 넷 다 못 잡았다.
'함정'과 '후보'는 일상어라 자연스러워 보이는데, 그래서 오히려 눈에 안 띈다."""

WARNINGS: list[str] = []
"""단언이 아니라 경고로 남길 것들. 지금은 그룹 겹침뿐이다.

**나중에 코드로 거르기로 한다면**(재현성 30회에서 빈도를 보고 정한다) 규정은
"코드가 LLM 을 뒤집는다"가 아니라 **"같은 좌표 중복을 코드가 거른다"** 다 — 두 축이
같은 조합을 둘 보여주는 것은 정보가 아니기 때문이다.
다만 **버린 자리를 무엇으로 채울지가 남는다.** 안 뽑힌 그룹 중 위엣것으로 채우면 LLM 이
고르지 않은 것을 3위로 올리는 셈이라 근거가 없다. 채우지 않고 2개만 보여주는 쪽도
선택지다. 그때 정한다."""


def run_live() -> None:
    """A·B·C 를 한 번씩 실제 호출해 응답이 쓸 만한지 본다."""
    real_select = agent.select_final_three   # _fresh_app 이 스텁을 끼우므로 되돌린다
    seen: dict = {}

    def spy(prompt_text, labels, client=None):
        seen["prompt"] = prompt_text
        advice = real_select(prompt_text, labels, client)
        seen["advice"] = advice
        return advice

    def spy_prompt(groups, *a, _real=optimizer.to_prompt_text, **k):
        seen["groups"] = groups
        return _real(groups, *a, **k)

    optimizer.to_prompt_text = spy_prompt
    at = _fresh_app()          # 여기서 스텁이 끼워진다
    agent.select_final_three = spy   # 그 뒤에 실물로 되돌린다
    while at.session_state["step"] < app.S_REVENUE:
        _advance(at)

    total = 0.0
    for scenario in ("A", "B", "C"):
        seen.clear()
        while at.session_state["step"] > app.S_REVENUE:
            _back(at)
        at.selectbox(key="scenario_key_w").set_value(scenario)
        at.run(timeout=300)
        at.session_state["_advice"] = None   # 캐시를 비워 매번 새로 부른다
        _to_results(at)

        advice = seen.get("advice")
        groups = seen.get("groups") or []
        assert advice is not None, f"[{scenario}] select_final_three 가 안 불렸다"
        usage = advice.usage or {}
        cost = (usage.get("input_tokens", 0) / 1e6 * IN_RATE
                + usage.get("output_tokens", 0) / 1e6 * OUT_RATE)
        total += cost

        assert advice.ok, (
            f"[{scenario}] 폐기됨 — {advice.failure} {advice.detail}\n"
            f"  걸린 숫자: {advice.bad_numbers}"
        )
        assert len(advice.picks) == 3, f"[{scenario}] picks {len(advice.picks)}개"

        gid = {m.label: i for i, g in enumerate(groups, 1) for m in g.members}
        picked = [gid.get(p.label, 0) for p in advice.picks]
        assert 0 not in picked, f"[{scenario}] 후보에 없는 label 을 골랐다: {picked}"

        # ── 그룹 분산은 **경고**로만 남긴다 ──
        # "서로 다른 그룹에서 고르라"는 지시가 확률적으로 무시된다. 단언으로 두면
        # 게이트가 확률적으로 실패해 신호가 못 된다 — 코드가 깨진 것인지 LLM 이 다르게
        # 답한 것인지 구분이 안 되기 때문이다. 횟수를 세어 재현성 30회에서 쓴다.
        if len(set(picked)) != 3:
            dup = sorted({g for g in picked if picked.count(g) > 1})
            # 축 최상위가 화면에서 사라졌는지 — 이쪽은 심각도가 다르다.
            best_survival = max(range(len(groups)), key=lambda i: groups[i].survival_months) + 1
            best_debt = min(range(len(groups)), key=lambda i: groups[i].debt_at_exit) + 1
            lost = [
                name for name, top in (("생존", best_survival), ("부채", best_debt))
                if top not in picked
            ]
            WARNINGS.append(
                f"[{scenario}] 그룹 겹침 {picked} (겹친 그룹 {dup})"
                + (f" · 축 최상위 누락: {', '.join(lost)}" if lost else " · 축 최상위는 살아 있다")
            )
            print(f"  ⚠ {WARNINGS[-1]}")

        # 내부 표기가 사용자 문장에 새지 않았는지
        texts = [advice.phase_reading or ""]
        for p in advice.picks:
            texts += [p.why or "", p.trap_note or "", p.before_apply or ""] + list(p.steps or [])
        texts += [i.get("reason", "") for i in (advice.excluded or [])]
        leaked = sorted({w for w in BANNED for t in texts if w in t})
        assert not leaked, f"[{scenario}] 내부 표기가 샜다: {leaked}"

        print(f"[{scenario}] picks 그룹={picked} · excluded={len(advice.excluded or [])} "
              f"· 출력 {usage.get('output_tokens', 0)} 토큰 · ${cost:.4f}")

    print(f"\n3회 합계 ${total:.4f} · 호출당 평균 ${total / 3:.4f} "
          f"· 시연 100회 ${total / 3 * 100:.2f}")
    print(f"그룹 겹침 {len(WARNINGS)}/3")
    for line in WARNINGS:
        print(f"  ⚠ {line}")


def main() -> None:
    print("=" * 58)
    print("실제 LLM 호출 검증")
    print("=" * 58)
    if os.environ.get(ENV_FLAG) != "1":
        print(f"건너뜀 — 실제 API 를 호출하므로 기본으로는 돌지 않는다(호출당 약 $0.03).")
        print(f"돌리려면:  {ENV_FLAG}=1 python -m tests.test_llm_live")
        return
    if not agent._api_key():
        print("ANTHROPIC_API_KEY 가 없다. .env 나 환경변수에 넣고 다시 돌려라.")
        sys.exit(1)
    run_live()
    print("\n전부 통과")


if __name__ == "__main__":
    main()
