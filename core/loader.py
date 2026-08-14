"""
정책 데이터 로더.

policies_all.json은 '정책 → 후보' 2계층 구조다.
- 정책(policy): 자격 판정의 단위 (업종·국면·신용점수 등)
- 후보(candidate): 시뮬레이터가 소비하는 원자적 계산 단위
                   같은 자금이라도 상환조건이 다르면 별개 후보

조합 생성은 정책이 아니라 후보 단위로 이루어지므로,
이 모듈은 정책 정보를 후보에 상속시켜 평평한 리스트로 펼친다.

명명 규칙
---------
코드 식별자는 영어, JSON 키는 한글 그대로 쓴다.
policies_all.json은 공고문 용어로 만든 데이터라 번역하면
기획서 <-> JSON <-> 코드 대응이 한 겹 깨진다.
주석과 docstring은 한글을 쓴다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "policies_all.json"

# 현금효과 6종. 시뮬레이터의 계산 분기와 1:1로 대응한다.
CashEffect = Literal[
    "신규유입", "대환전환", "이자조정", "부채감면", "상환연장", "비용감면", "없음"
]

# 지원방법 4종. 조합 문법 제약의 기준.
SupportMethod = Literal["융자", "보조금", "채무조정", "컨설팅"]

PHASES = ("일시적자금난", "개선여지", "회복불가")
METHODS = ("융자", "보조금", "채무조정")


@dataclass(frozen=True)
class Candidate:
    """시뮬레이터가 소비하는 원자적 단위."""

    candidate_id: str
    policy_id: str
    policy_name: str
    source: str  # 재원 (소진공 / 서울시 / 캠코·신복위)
    method: str  # 지원방법
    effect: str  # 현금효과
    phases: tuple[str, ...]  # 적용국면
    label: str = ""

    is_extra: bool = False
    """**부가 추천 대상**인지 (`_순위제외`).

    현금효과가 0이라 2축에 반영되지 않는 후보다(사업정리 컨설팅, 법률자문 등).
    조합에 넣으면 좌표가 바뀌지 않으면서 조합 수만 배로 늘어나므로 제외한다.

    다만 **자격 필터는 통과시킨다.** 국면·자격에 맞는 것만 골라 LLM 프롬프트에
    별도 목록으로 넘겨 최종 추천과 함께 안내한다.
    (이전에는 로더가 아예 넘기지 않아 자격 필터에 도달조차 못 했다)
    """
    # 원본 후보 dict. 금액·금리·거치 등 지원방법별 필드는 여기서 꺼내 쓴다.
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    # 정책 레벨 원본 (자격요건 등)
    policy_raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def get(self, key: str, default: Any = None) -> Any:
        """후보 -> 정책 순으로 JSON 필드를 찾는다."""
        if key in self.raw:
            return self.raw[key]
        return self.policy_raw.get(key, default)

    @property
    def eligibility(self) -> dict[str, Any]:
        """자격요건 원본."""
        return self.policy_raw.get("자격요건", {})

    @property
    def is_loan(self) -> bool:
        return self.method == "융자"

    @property
    def closure_requirement(self) -> str | None:
        """자격요건의 '폐업여부' 문자열. 없으면 None.

        값이 자유 서술이라(예: "영업중", "폐업 또는 폐업예정",
        "영업중·폐업·휴업 모두 가능") 정확한 파싱 대신
        combos.py에서 키워드로 판정한다.
        """
        return self.eligibility.get("폐업여부")

    def __str__(self) -> str:
        return f"{self.policy_name}({self.label})" if self.label else self.policy_name


@dataclass
class PolicyDB:
    raw: dict[str, Any]
    candidates: list[Candidate]

    # ── 메타 접근자 ──
    @property
    def combo_rules(self) -> dict[str, Any]:
        """조합 문법 (융자1·채무조정1·보조금2 등 하드 제약)."""
        return self.raw["_조합문법"]

    @property
    def loan_rules(self) -> dict[str, Any]:
        return self.raw["공통규칙_융자"]

    @property
    def base_rate(self) -> float:
        """정책자금 기준금리 (분기 변동)."""
        return self.raw["공통규칙_융자"]["기준금리"]

    @property
    def deadline_filter_enabled(self) -> bool:
        """MVP에서는 False. 신청기간은 기록만 하고 필터에 쓰지 않는다."""
        return bool(self.raw.get("_신청기한필터", {}).get("MVP_적용", False))

    # ── 조회 ──
    def by_phase(self, phase: str) -> list[Candidate]:
        if phase not in PHASES:
            raise ValueError(f"알 수 없는 국면: {phase!r} (가능: {PHASES})")
        return [c for c in self.candidates if phase in c.phases]

    def by_method(self, method: str) -> list[Candidate]:
        return [c for c in self.candidates if c.method == method]

    def find(self, candidate_id: str) -> Candidate:
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                return c
        raise KeyError(candidate_id)


def _is_excluded(policy: dict, cand: dict) -> bool:
    """순위 대상에서 제외된 후보인지.

    정책 레벨 또는 후보 레벨에 _순위제외가 있으면 제외한다.
    (컨설팅, 용도가 묶인 사업화자금 등 2축에 반영되지 않는 것들)
    """
    return bool(policy.get("_순위제외") or cand.get("_순위제외"))


def load(path: str | Path = DATA_PATH, include_excluded: bool = True) -> PolicyDB:
    """정책 JSON을 읽어 후보 단위로 펼친다.

    Args:
        path: policies_all.json 경로
        include_excluded: `_순위제외` 후보(부가 추천 대상)를 포함할지.
            **기본값이 True다.** 이들도 자격 필터를 통과해야 LLM에 안내할 수 있다.
            조합에서 빼는 것은 `combos.build_combos()`가 `is_extra`를 보고 한다.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    candidates: list[Candidate] = []
    for policy in raw["policies"]:
        for cand in policy["후보"]:
            excluded = _is_excluded(policy, cand)
            if not include_excluded and excluded:
                continue
            candidates.append(
                Candidate(
                    is_extra=excluded,
                    candidate_id=cand["후보id"],
                    policy_id=policy["id"],
                    policy_name=policy["name"],
                    source=policy["재원"],
                    # 지원방법은 후보에 있으면 후보 것이 우선
                    # (희망리턴처럼 한 정책 안에 보조금·컨설팅이 섞이는 경우)
                    method=cand.get("지원방법", policy["지원방법"]),
                    effect=cand["현금효과"],
                    # 적용국면도 후보가 우선한다.
                    # 정책 레벨 태그는 제도 전체의 범위라 후보보다 넓을 수 있다.
                    # (예: 희망리턴패키지는 폐업 트랙과 경영개선 트랙이 섞여
                    #      정책 태그가 [회복불가, 개선여지]인데, 점포철거비는
                    #      폐업 확정이 전제라 [회복불가]만 해당한다)
                    phases=tuple(cand.get("적용국면") or policy["적용국면"]),
                    label=cand.get("라벨", ""),
                    raw=cand,
                    policy_raw=policy,
                )
            )

    return PolicyDB(raw=raw, candidates=candidates)


def summary(db: PolicyDB) -> str:
    """국면 x 지원방법 후보 수를 표로."""
    lines = [f"{'국면':<12} {'융자':>4} {'보조금':>5} {'채무조정':>6} {'합계':>5}"]
    for phase in PHASES:
        pool = db.by_phase(phase)
        counts = {m: sum(1 for c in pool if c.method == m) for m in METHODS}
        lines.append(
            f"{phase:<12} {counts['융자']:>4} {counts['보조금']:>5} "
            f"{counts['채무조정']:>6} {len(pool):>5}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    from collections import Counter

    db = load()
    print(f"정책 {len(db.raw['policies'])}개 / 순위 대상 후보 {len(db.candidates)}개")
    print(f"기준금리 {db.base_rate:.2%} / 기한필터 적용: {db.deadline_filter_enabled}")
    print()
    print(summary(db))
    print()
    print("현금효과별 후보 수:")
    for effect, n in Counter(c.effect for c in db.candidates).most_common():
        print(f"  {effect:<8} {n}")
