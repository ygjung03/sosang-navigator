"""LLM 연결 — 최종 3개 선정과 설명.

**LLM은 계산하지 않는다.** 코드가 자격을 거르고, 조합을 만들고, 36개월을 시뮬레이션하고,
함정을 판정하고, 비지배 정렬로 후보를 좁힌 뒤다. 여기서 하는 일은 기획서가 정한 것들이다.

    국면 맥락으로 어느 축을 중시할지 판단   (기획서 4.3 국면은 LLM 판단의 무게중심)
    최종 3개 선정                          (기획서 5. AI 역할 분담)
    함정을 사용자 말로 서술                 (기획서 5. AI 역할 분담)
    자격은 되지만 추천에서 뺀 이유           (기획서 5.3 자격은 되지만 추천에서 제외한 이유)
    실행 순서 제안                          (기획서 10. 정책 간 선후관계, `_연계정보` 가 있을 때만)

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
from pathlib import Path
from typing import Any, Sequence

MODEL = "claude-sonnet-5"
"""$2/$10 per MTok. 이 작업은 판단과 서술이라 소넷 급이면 충분하다.
(Sonnet 3.5 는 2025-10-28 에 은퇴해 쓸 수 없다.)

**모델을 바꿀 때 고치는 곳은 이 파일의 상수 넷뿐이다** — MODEL · MAX_TOKENS ·
THINKING · TEMPERATURE. 프롬프트는 손대지 않는다.
재현성이 부족하면 claude-sonnet-4-6($3/$15)이 대안이다. temperature 를 받고
thinking 이 기본 꺼짐이라 재현성만 놓고 보면 유리하다."""

MAX_TOKENS = 8000
"""잘림을 먼저 없애고 실제 output_tokens 를 보고 낮춘다. 잘리면 파싱 실패로 폴백된다."""

THINKING: dict | None = {"type": "disabled"}
"""끈다. 코드가 계산·판정·정렬을 끝냈고 LLM 은 고르고 서술만 한다.
켜면 사고 토큰이 max_tokens 를 먹어 JSON 이 잘리고 출력 비용도 늘어난다.
None 이면 인자를 넘기지 않는다(모델 기본값)."""

TEMPERATURE: float | None = None
"""Sonnet 5 는 샘플링 파라미터(temperature·top_p·top_k)를 받지 않고 400 을 낸다.
기획서 5.5 의 재현성 확보 장치 중 '온도 낮춤'이 빠지므로 30회 실측으로 확인한다
(데모 프리셋 고정은 그대로 유효하다).
temperature 를 받는 모델(claude-sonnet-4-6 · claude-haiku-4-5)로 바꾸면 0.0 을 넣는다."""

TIMEOUT_SECONDS = 60.0

RESPONSE_ENCODING = {"Accept-Encoding": "gzip, deflate"}
"""응답 압축에서 brotli 를 뺀다. Brotli 1.1.0 이하의 Decompressor.process() 는 httpx2 가
넘기는 output_buffer_limit 인자를 받지 않아 TypeError 가 나고, 그게 SDK 안에서
APIConnectionError 로 감싸여 나온다 — 표면 메시지가 원인과 전혀 다르다.
requirements.txt 에 Brotli>=1.2.0 을 못박았지만, 남의 환경에서 옛 버전이 깔려도
brotli 를 안 받으면 걸리지 않는다(1.1.0 에서 이 헤더로 성공하는 것을 실측했다).
gzip 은 그대로 쓰므로 전송량 손해는 작다."""


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
    """검증에 걸린 **자리 · 숫자 · 그 문장**. `"picks[2].trap_note: 171 | <원문>"` 꼴이다.
    재현성 실측에서 얼마나 자주, 어떤 숫자가 걸리는지 보려고 남긴다. 자주 걸리면
    폴백이 잦다는 뜻이라 프롬프트를 고쳐야 한다.

    **재현성 30회 때 함께 잴 것**(9회 표본으로는 갈리지 않아 미룬다).
    지금까지 폐기 3건 중 둘은 phase_reading 이었고 성격이 달라 보인다.
      171   목표 18 로 바꾼 뒤 picks[2] 의 trap_note·before_apply 두 곳에 같은 값이
            걸렸다. 한 숫자를 잘못 옮긴 것으로 보이나 그때는 문장을 안 남겨 못 봤다.
            (그래서 문장을 함께 담게 고쳤다 — 2026-08-31)
      6600  프롬프트의 6,660·6,710 을 반올림한 값으로 보인다 -> 파생 계산
      100   '100%'·'100만원' 같은 관용 표현일 수 있다
    관용 표현이면 숫자 금지를 되살려도 안 잡히고 검증기 쪽을 손봐야 한다.
      · 폐기가 난 **필드와 숫자를 전부** 기록한다. phase_reading 에 몰리는지,
        다른 필드에도 나는지. 서술이 긴 필드라 숫자가 나올 기회가 많은 것뿐일 수 있다.
      · 걸린 숫자가 **어떤 문장에** 쓰였는지 원문을 남긴다. 파생 계산인지 관용
        표현인지는 문장을 봐야 갈린다."""

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

■ 당신이 할 일 다섯 가지

1. 국면 맥락으로 **어느 축을 중시할지** 판단한다.
2. 후보 중 **최종 3개**를 고른다. 레이어 1부터 고를 필요는 없다 — 위 기준으로 판단한다.
   **후보가 3개 이상이면 반드시 3개를 채운다.** 함정이 있는 조합도 3위에 넣을 수 있다 —
   함정은 제외 사유가 아니라 trap_note 로 알릴 사항이다. 판단은 사용자가 한다.
   2개만 고르고 나머지를 excluded 로 보내지 마라.

   **3개는 서로 다른 그룹에서 고른다. 한 그룹에서 둘 이상 뽑지 마라** — 같은 그룹은
   두 축(생존·부채)이 완전히 같아 사용자에게 선택지가 하나로 보이고, 다른 그룹의 축
   최상위가 화면에서 사라진다.
   같은 그룹에 조합이 여럿이면 국면에 비추어 나은 하나만 고르고, 나머지는 excluded 에
   왜 그쪽이 아닌지 쓴다.
3. 경고([경고] 줄)를 **사용자의 말로** 풀어 쓴다. '함정 판정 — 4종 모두 해당 없음'
   이라고 적힌 조합은 걸린 것이 없다는 뜻이다. **그 표기를 그대로 옮기지 마라** —
   '함정'도 '4종'도 우리 분류 이름이라 사용자는 본 적이 없다.
   trap_note 를 "주의할 점이 없습니다"처럼 쓰거나, 굳이 쓸 말이 없으면 null 로 둔다.
4. 자격은 되지만 **추천에서 뺀 이유**를 쓴다. 고르지 않은 조합을 **전부** 넣는다.
   이유는 국면과 사용자 상황에 비추어 "이 사람에게는 왜 아닌지"로 쓴다. 두 갈래가 있고
   각각 판단 재료가 다르다.
   - 최종 3개에 없는 후보 그룹 → 두 축(생존 개월수·종료 시점 부채)의 차이가 이 사람의
     국면에서 어떤 뜻인지
   - 고른 것과 **두 축이 같은 다른 조합** → 두 축으로는 구별되지 않으므로 상환 구조
     (거치 후 급증·균등 상환·만기 일시상환 등)가 이 사람의 현금흐름에 어떻게 작용하는지
5. 실행 순서가 실제로 중요한 조합이면 순서를 제안한다.

■ 절대 규칙 — 사용자가 읽는 문장에서 무엇을 가리킬 수 있는가

사용자는 화면에서 **추천 3개와 그 설명만** 본다. 우리가 후보를 어떻게 정리했는지는
모른다. 그러니 사용자가 읽는 문장(phase_reading·why·trap_note·before_apply·steps·
excluded.reason)에서 **무엇을 가리킬 때는 아래 것만** 쓴다.

  · 조합 이름 — 프롬프트에 적힌 그대로
  · 정책·제도 이름 — 프롬프트에 적힌 그대로 (경영안정 바우처 · 새출발기금 등)
  · **네가 정한 순위** — 1위 · 2위 · 3위
  · 금액 · 개월수 · 금리 · 상환 조건 (거치 · 분할 · 만기 일시상환 등)
  · 국면 이름과 [사용자 상황] 에 적힌 값 (매출 감소율 · 부채 · 연체 일수 · 폐업 여부 등)
  · 두 축의 이름 — 생존 개월수 · 종료 시점 부채
  · [경고] 와 계산 전제 안내에 적힌 내용

**조합끼리 비교하는 것은 권장한다.** 왜 그 순위인지는 비교로만 설명된다. 다만 부르는
이름은 언제나 '조합'이다 — 화면에 카드로 뜨는 것이 조합이라 사용자가 아는 말이다.

    "1위 조합보다 총 상환액이 448만원 적습니다"
    "종료 시점 부채가 가장 적은 조합보다 8만원 많지만"

쓰기 전에 그 후보를 네가 몇 위로 골랐는지 확인하라. 고르지 않은 조합이면 순위가 없으므로
특성으로만 가리킨다.

**이 목록에 없는 말로 후보를 가리키지 마라.** 특히 우리가 후보를 정리·분류하려고 쓴
표기 — 레이어 · 그룹 · 후보 · 파레토 · 좌표 · 축 N위 · N번 대비 · **함정** · **4종** —
는 정보로만 읽고 문장에 옮기지 마라. 목록에 없으면 위에 적히지 않은 말이라도 쓰지 않는다.

'함정'과 '4종'은 화면에 없는 말이다. 사용자가 보는 것은 경고 상자뿐이다.

    "함정으로 분류된 항목이 없습니다"   ← 어딘가에 분류 체계가 있다는 뜻이 된다
    "주의할 점이 없습니다"              ← 이렇게 쓴다

'후보'도 마찬가지다. 사용자는 후보 목록을 본 적이 없다.
개수를 다른 말로 바꾸지 말고 아예 쓰지 마라.
'모두'라고 쓸 때는 '아래 조합'처럼 범위를 함께 밝혀라 — 안 밝히면
"아무 지원이나 받아도 된다"로 읽힌다.

    "후보 전체에서 가장 적은 조합입니다"  ← "종료 시점 부채가 가장 적은 조합입니다"
    "모든 후보가 목표 18개월을 채우고"    ← "아래 조합은 모두 목표 18개월을 채우고"
    "세 후보 그룹 모두 충족하므로"        ← "아래 조합이 모두 목표를 충족하므로"

※ 위 제한은 **후보를 가리키는 말**에 대한 것이다. 설명을 잇는 보통의 말(매출 회복 ·
  신청 절차 · 예비자금 등)은 자유롭게 써도 된다.

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

picks 는 **정확히 3개**다. 후보가 3개보다 적을 때만 있는 만큼 고른다.
후보가 3개 이상인데 2개만 넣으면 잘못된 응답이다.
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


_YEARS = re.compile(r"(\d+)\s*년")


def _year_months(labels: Sequence[str]) -> set[str]:
    """조합 이름의 'N년'을 개월로 환산한 값. 허용 목록에 더한다.

    '1년거치 4년 균분상환'을 LLM 이 '12개월 거치'로 풀어 쓰면 프롬프트에 없는 숫자가 되어
    응답이 통째로 버려졌다(실측: A 는 48, B 는 12). 사용자에게 친절한 표기라 막을 이유가
    없다. 반면 뺄셈 같은 **파생 계산은 그대로 막는다** — 그쪽은 optimizer 가 차이값을
    프롬프트에 실어 해결했다('1번 대비 448만원 적음').

    **조합 이름에서만 찾는다.** 프롬프트 전체를 훑으면 자격 문구의 '3년 내' 같은 것까지
    걸려 허용 목록이 넓어지고, 그만큼 환각이 통과할 자리가 생긴다.
    연->개월(x12) 하나만 연다. 만원<->원 같은 다른 변환은 지금 문제가 아니고, 넓게 열면
    검증이 무력해진다. 필요해지면 그때 추가한다.
    """
    out: set[str] = set()
    for label in labels:
        for hit in _YEARS.finditer(label):
            out.add(str(int(hit.group(1)) * 12))
    return out


# ══════════════════════════════════════════════════════════
# 호출
# ══════════════════════════════════════════════════════════


_dotenv_loaded = False


def _load_dotenv_once() -> None:
    """프로젝트 루트의 .env 를 환경변수로 올린다. 한 번만.

    이게 없어서 .env 에 키를 넣어도 앱이 못 읽었다. 이미 환경에 있는 값은 덮지 않는다
    (배포 환경의 설정이 로컬 파일에 밀리면 안 된다).
    import 시점이 아니라 첫 호출에 읽는다 — 모듈을 불러오는 것만으로 파일을 건드리지 않는다.
    """
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _dotenv_loaded = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return  # 배포(Streamlit Cloud)는 secrets 를 쓰므로 없어도 된다
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


def _api_key() -> str | None:
    """환경변수 우선, 없으면 st.secrets.

    로컬은 .env(gitignore 됨), 배포는 Streamlit Cloud 의 secrets UI 를 쓴다.
    st.secrets 접근은 Streamlit 밖(테스트·스크립트)에서 예외가 나므로 감싼다.
    """
    _load_dotenv_once()
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

            client = anthropic.Anthropic(
                api_key=key, timeout=TIMEOUT_SECONDS, default_headers=RESPONSE_ENCODING
            )
        # 모델마다 받는 인자가 다르다. None 인 상수는 아예 안 넘긴다 —
        # Sonnet 5 에 temperature 를 넘기면 400 이다.
        extra: dict = {}
        if THINKING is not None:
            extra["thinking"] = THINKING
        if TEMPERATURE is not None:
            extra["temperature"] = TEMPERATURE
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt_text}],
            **extra,
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
    allowed = _numbers_in(prompt_text) | _year_months(labels)
    known = set(labels)

    picks_raw = data.get("picks")
    if not isinstance(picks_raw, list) or not picks_raw:
        return Advice(
            ok=False, failure=Failure.PARSE, detail="picks 가 비어 있다.", usage=usage
        )

    # 숫자 검증은 **모아서 한 번에** 판정한다. 하나라도 걸리면 응답 전체를 버린다.
    bad: list[str] = []

    def check(where: str, text: str | None) -> None:
        # **문장까지 함께 남긴다.** 자리와 숫자만으로는 재현성 실측에서 "왜 그 숫자가
        # 나왔나"를 되짚을 수 없다. 실제로 목표 18 로 바꾼 뒤 picks[2] 의 두 필드에
        # 같은 값(171)이 걸렸는데, 어느 값을 잘못 옮긴 것인지 문장이 없어 못 봤다.
        # 30회 실측에서 반복되는 패턴을 찾으려면 문장이 있어야 한다.
        for number in _unknown_numbers(text, allowed):
            bad.append(f"{where}: {number} | {(text or '').strip()[:120]}")

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
