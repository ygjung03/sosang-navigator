"""LLM 연결 — 최종 3개 선정과 설명.

**LLM은 계산하지 않는다.** 코드가 자격을 거르고, 조합을 만들고, 36개월을 시뮬레이션하고,
함정을 판정하고, 비지배 정렬로 후보를 좁힌 뒤다. 여기서 하는 일은 기획서가 정한 것들이다.

    국면 맥락으로 어느 축을 중시할지 판단   (planning 174~179)
    최종 3개 선정                          (planning 200)
    함정을 사용자 말로 서술                 (planning 200)
    자격은 되지만 추천에서 뺀 이유           (planning 897·979)
    실행 순서 제안                          (planning 1586, `_연계정보` 가 있을 때만)

`_확인필요_자격` 되묻기는 **코드가 한다**(app.py 의 ⑥ 단계). 질문 문구가 조합과 무관하게
완결돼 있어 LLM 이 다시 쓸 필요가 없고, 호출을 1회로 유지할 수 있으며, LLM 이 실패해도
자격 확인은 그대로 작동한다. 그래서 응답 스키마에 되묻기 필드가 없다.

■ 숫자 환각을 기계적으로 막는다
지시만으로는 막히지 않는다. 응답의 숫자를 뽑아 **프롬프트에 없는 값이 하나라도 있으면
응답을 통째로 버린다.** 필드만 버리면 설명이 군데군데 빈 카드가 나오는데, 사용자는 왜
비었는지 알 수 없고 화면은 정상인 척한다. 통째로 폴백하면 배너가 상황을 말하고 재시도
버튼도 뜬다. 조합 이름(label)을 잘못 고른 경우와 같은 취급이다.
"숫자에 근거가 있다"가 이 프로젝트의 강점이라, 화면에 나가는 문장의 숫자는 전부 코드가
만든 것이어야 한다.

■ 실패하면 조용히 넘어가지 않는다
키 없음·API 오류·타임아웃·파싱 실패·검증 실패를 구분해 돌려준다. 화면이 그 이유를 배너로
알리고, 재시도가 의미 있는 경우에만 버튼을 띄운다(app.py).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2000

TEMPERATURE = 0.0
"""선정은 창작이 아니라 판단이다. 재현성이 심사에서 중요해 온도를 0으로 둔다
(기획서 1003행). **0이어도 완전 결정적이지는 않아** 재현성은 따로 실측한다."""

TIMEOUT_SECONDS = 60.0


class Failure(str, Enum):
    """실패 유형. 화면이 이 값으로 문구와 재시도 버튼 유무를 정한다."""

    NO_KEY = "no_key"          # 재시도해도 안 된다 -> 버튼 없음
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    PARSE = "parse"            # JSON 이 아니거나 스키마가 안 맞음
    VERIFY = "verify"          # 없는 조합을 고르거나 없는 숫자를 썼다 -> 통째로 폐기


@dataclass
class Pick:
    """최종 3개 중 하나."""

    label: str
    axis_focus: str = ""
    why: str = ""
    trap_note: str | None = None
    steps: list[str] = field(default_factory=list)
    before_apply: str | None = None


@dataclass
class Advice:
    """LLM 응답. `ok` 가 False 면 `failure` 와 `detail` 을 화면이 쓴다."""

    ok: bool
    picks: list[Pick] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)
    phase_reading: str = ""
    failure: Failure | None = None
    detail: str = ""

    bad_numbers: list[str] = field(default_factory=list)
    """검증에 걸린 숫자와 그 자리. 재현성 실측에서 **얼마나 자주, 어떤 숫자가** 걸리는지
    보려고 남긴다. 자주 걸리면 폴백이 잦다는 뜻이라 프롬프트를 고쳐야 한다."""

    usage: dict = field(default_factory=dict)
    """입력·출력 토큰. 비용 추정을 실제 값으로 갱신하려고 남긴다."""


SYSTEM_PROMPT = """당신은 서울 소상공인에게 정책 조합을 추천하는 상담사다.

계산은 이미 끝났다. 아래 후보는 자격 필터·36개월 현금흐름 시뮬레이션·함정 판정·비지배
정렬을 거친 결과다.

■ 후보가 어떻게 정리돼 있는가

두 축으로 평가했다. **생존 개월수**(지원을 받으면 몇 개월 버티는가)와 **종료 시점
부채**(그때 빚이 얼마 남는가)다. 코드는 두 축에 가중치를 두지 않는다 — 어느 쪽이 더
중요한지는 사람의 사정에 달렸고, 그 판단이 당신 몫이다.

**레이어는 비지배(파레토) 정렬의 층이다.**
  · 레이어 1 — 어떤 후보에게도 **두 축 모두에서** 밀리지 않는다. 그 층을 걷어내고 다시
    같은 기준으로 고른 것이 레이어 2이고, 이런 식으로 층이 쌓인다.
  · 레이어 **번호가 커질수록** 앞 층에게 한 축 이상에서 밀린 대안이다.
  · **번호가 크다고 나쁜 것은 아니다.** 한 축을 양보한 대신 다른 축을 얻은 조합이라,
    사용자에게 덜 중요한 축을 양보한 것이라면 오히려 그쪽이 낫다.
    예를 들어 회복불가 국면에서는 몇 달 더 버티는 것보다 빚을 줄여 재기 기반을 남기는
    쪽이 나을 수 있고, 그런 조합은 생존 축에서 밀려 아래 레이어에 있을 수 있다.
  · 괄호 안의 '생존 축 N위 · 부채 축 N위'는 전체 후보 중 그 축만 보고 매긴 순위다.
  · 같은 레이어 안에 여러 줄이 있으면 두 축 성능이 같고 상환 조건만 다른 것들이다.

'순위에서 밀린 조합'은 자격은 되지만 최종 후보에 못 든 것들이다. 왜 추천하지 않았는지
설명할 때 쓴다.

■ 당신이 할 일 다섯 가지

1. 국면 맥락으로 **어느 축을 중시할지** 판단한다.
2. 후보 중 **최종 3개**를 고른다. 레이어 1부터 고를 필요는 없다 — 위 기준으로 판단한다.
3. 함정 경고([경고] 줄)를 **사용자의 말로** 풀어 쓴다. '함정 판정 — 4종 모두 해당 없음'
   이라고 적힌 조합은 걸린 함정이 없다는 뜻이다.
4. 자격은 되지만 **추천에서 뺀 이유**를 쓴다.
5. 실행 순서가 실제로 중요한 조합이면 순서를 제안한다.

■ 절대 규칙 — 숫자를 새로 만들지 마라

프롬프트에 있는 숫자만 써라. 더하거나 빼거나 비율을 구하지 마라. 없는 숫자가 하나라도
있으면 응답 전체가 버려진다.

조합 이름(label)은 프롬프트에 있는 것을 **한 글자도 바꾸지 말고** 그대로 옮겨라.

■ 출력

JSON 하나만 출력한다. 설명이나 코드블록 표시를 붙이지 마라.

{
  "phase_reading": "국면을 읽고 어느 축을 왜 중시했는지 2~3문장",
  "picks": [
    {
      "label": "<프롬프트의 조합 이름 원문 그대로>",
      "axis_focus": "생존 | 부채 | 균형",
      "why": "이 사용자에게 왜 이 조합인지 2~3문장",
      "trap_note": "함정을 사용자 말로. 함정이 없으면 null",
      "steps": ["먼저 …", "그다음 …"],
      "before_apply": "신청 전에 해야 할 일. 없으면 null"
    }
  ],
  "excluded": [{"label": "...", "reason": "자격은 되지만 뺀 이유 한 문장"}]
}

picks 는 정확히 3개다. 후보가 3개보다 적으면 있는 만큼만 고른다.
steps 는 순서가 실제로 중요한 조합에서만 채우고, 아니면 빈 배열로 둔다."""


# ══════════════════════════════════════════════════════════
# 숫자 검증
# ══════════════════════════════════════════════════════════

_NUMBER = re.compile(r"[\d][\d,]*(?:\.\d+)?")


def _numbers_in(text: str) -> set[str]:
    """텍스트의 숫자를 **정규화해서** 뽑는다.

    같은 값의 다른 표기를 같게 본다 — '3,657만원'·'3657'·'36,570,000원' 은 모두 3657.
    쉼표를 지우고, 뒤에 '만'이 붙지 않은 큰 수(10만 이상)는 원 단위로 보아 만원으로
    환산한다. 소수점 이하는 버린다 — 프롬프트에도 같은 규칙을 적용하므로 대칭이다.
    """
    out: set[str] = set()
    for match in _NUMBER.finditer(text):
        raw = match.group().replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if value >= 100_000 and text[match.end(): match.end() + 1] != "만":
            value = value / 10_000  # 원 단위로 쓴 큰 수 -> 만원
        out.add(str(int(value)))
    return out


def _unknown_numbers(text: str | None, allowed: set[str]) -> list[str]:
    """텍스트에서 **프롬프트에 없는** 숫자만 골라 돌려준다. 없으면 빈 목록."""
    if not text:
        return []
    return sorted(_numbers_in(text) - allowed)


# ══════════════════════════════════════════════════════════
# 호출
# ══════════════════════════════════════════════════════════


def _api_key() -> str | None:
    """환경변수 우선, 없으면 st.secrets.

    로컬은 .env(gitignore 됨), 배포는 Streamlit Cloud 의 secrets UI 를 쓴다.
    st.secrets 접근은 Streamlit 밖(테스트·스크립트)에서 예외가 나므로 감싼다.
    """
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def _extract_json(text: str) -> dict:
    """응답에서 JSON 을 꺼낸다. 코드블록으로 감싸 오는 경우까지 받아 준다."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("JSON 을 찾지 못했다")
    return json.loads(text[start: end + 1])


def select_final_three(
    prompt_text: str, labels: Sequence[str], client: Any = None
) -> Advice:
    """최종 3개를 고른다. 실패해도 예외를 던지지 않고 `ok=False` 인 Advice 를 돌려준다.

    Args:
        prompt_text: `optimizer.to_prompt_text()` 가 만든 후보 설명.
        labels:      고를 수 있는 조합 이름 전부. 여기 없는 것을 고르면 응답을 폐기한다.
        client:      테스트에서 주입한다. 없으면 anthropic 클라이언트를 만든다.
    """
    key = _api_key()
    if client is None and not key:
        return Advice(
            ok=False, failure=Failure.NO_KEY, detail="ANTHROPIC_API_KEY 가 없습니다."
        )

    try:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=key, timeout=TIMEOUT_SECONDS)
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt_text}],
        )
    except Exception as error:  # 네트워크·인증·한도 등
        name = type(error).__name__.lower()
        kind = Failure.TIMEOUT if "timeout" in name else Failure.API_ERROR
        return Advice(ok=False, failure=kind, detail=f"{type(error).__name__}: {error}")

    usage: dict = {}
    try:
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    except Exception:
        pass

    try:
        raw = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", "") == "text"
        )
        data = _extract_json(raw)
    except Exception as error:
        return Advice(
            ok=False, failure=Failure.PARSE,
            detail=f"{type(error).__name__}: {error}", usage=usage,
        )

    return build_advice(data, prompt_text, labels, usage)


def build_advice(
    data: dict, prompt_text: str, labels: Sequence[str], usage: dict | None = None
) -> Advice:
    """파싱된 JSON 을 검증해 Advice 로. 검증 규칙이 한곳에 모여 테스트하기 쉽다."""
    usage = usage or {}
    allowed = _numbers_in(prompt_text)
    known = set(labels)

    picks_raw = data.get("picks")
    if not isinstance(picks_raw, list) or not picks_raw:
        return Advice(
            ok=False, failure=Failure.PARSE, detail="picks 가 비어 있다.", usage=usage
        )

    # 숫자 검증은 **모아서 한 번에** 판정한다. 하나라도 걸리면 응답 전체를 버린다.
    bad: list[str] = []

    def check(where: str, text: str | None) -> None:
        for number in _unknown_numbers(text, allowed):
            bad.append(f"{where}: {number}")

    picks: list[Pick] = []
    for index, item in enumerate(picks_raw):
        if not isinstance(item, dict):
            return Advice(
                ok=False, failure=Failure.PARSE,
                detail=f"picks[{index}] 형식 오류", usage=usage,
            )
        label = str(item.get("label", "")).strip()
        if label not in known:
            return Advice(
                ok=False, failure=Failure.VERIFY,
                detail=f"프롬프트에 없는 조합을 골랐다: {label[:60]}", usage=usage,
            )
        why = str(item.get("why") or "").strip()
        trap_note = str(item.get("trap_note") or "").strip() or None
        before = str(item.get("before_apply") or "").strip() or None
        check(f"picks[{index}].why", why)
        check(f"picks[{index}].trap_note", trap_note)
        check(f"picks[{index}].before_apply", before)
        steps = [
            str(s) for s in (item.get("steps") or []) if isinstance(s, (str, int, float))
        ]
        for step_index, step in enumerate(steps):
            check(f"picks[{index}].steps[{step_index}]", step)
        picks.append(
            Pick(
                label=label,
                axis_focus=str(item.get("axis_focus", "")).strip(),
                why=why,
                trap_note=trap_note,
                steps=steps,
                before_apply=before,
            )
        )

    phase_reading = str(data.get("phase_reading") or "").strip()
    check("phase_reading", phase_reading)

    excluded: list[dict] = []
    for item in data.get("excluded") or []:
        if not isinstance(item, dict):
            continue
        reason = str(item.get("reason") or "").strip()
        check("excluded.reason", reason)
        excluded.append({"label": str(item.get("label", "")).strip(), "reason": reason})

    if bad:
        return Advice(
            ok=False, failure=Failure.VERIFY, bad_numbers=bad, usage=usage,
            detail="프롬프트에 없는 숫자를 썼다 — " + ", ".join(bad[:5]),
        )

    return Advice(
        ok=True, picks=picks, excluded=excluded,
        phase_reading=phase_reading, usage=usage,
    )
