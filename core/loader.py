"""
정책 데이터 로더.

policies_all.json은 '정책 → 후보' 2계층 구조다.
- 정책(policy): 자격 판정의 단위 (업종·국면·신용점수 등)
- 후보(candidate): 시뮬레이터가 소비하는 원자적 계산 단위
                   같은 자금이라도 상환조건이 다르면 별개 후보

조합 생성은 정책이 아니라 후보 단위로 이루어지므로,
이 모듈은 정책 정보를 후보에 상속시켜 평평한 리스트로 펼친다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "policies_all.json"

# 현금효과 6종. 시뮬레이터의 계산 분기와 1:1로 대응한다.
CashEffect = Literal["신규유입", "대환전환", "부채감면", "상환유예", "비용감면", "없음"]

# 지원방법 4종. 조합 문법 제약의 기준.
SupportMethod = Literal["융자", "보조금", "채무조정", "컨설팅"]

PHASES = ("일시적자금난", "개선여지", "회복불가")


@dataclass(frozen=True)
class Candidate:
    """시뮬레이터가 소비하는 원자적 단위."""

    후보id: str
    정책id: str
    정책명: str
    재원: str
    지원방법: str
    현금효과: str
    적용국면: tuple[str, ...]
    라벨: str = ""
    # 원본 후보 dict. 금액·금리·거치 등 지원방법별 필드는 여기서 꺼내 쓴다.
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    # 정책 레벨 원본(자격요건 등)
    policy_raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def get(self, key: str, default: Any = None) -> Any:
        """후보 → 정책 순으로 필드를 찾는다."""
        if key in self.raw:
            return self.raw[key]
        return self.policy_raw.get(key, default)

    @property
    def 자격요건(self) -> dict[str, Any]:
        return self.policy_raw.get("자격요건", {})


@dataclass
class PolicyDB:
    raw: dict[str, Any]
    candidates: list[Candidate]

    # ── 메타 접근자 ──
    @property
    def 조합문법(self) -> dict[str, Any]:
        return self.raw["_조합문법"]

    @property
    def 공통규칙_융자(self) -> dict[str, Any]:
        return self.raw["공통규칙_융자"]

    @property
    def 기준금리(self) -> float:
        return self.raw["공통규칙_융자"]["기준금리"]

    @property
    def 기한필터_적용(self) -> bool:
        """MVP에서는 False. 신청기간은 기록만 하고 필터에 쓰지 않는다."""
        return bool(self.raw.get("_신청기한필터", {}).get("MVP_적용", False))

    # ── 조회 ──
    def by_phase(self, phase: str) -> list[Candidate]:
        if phase not in PHASES:
            raise ValueError(f"알 수 없는 국면: {phase!r} (가능: {PHASES})")
        return [c for c in self.candidates if phase in c.적용국면]

    def by_method(self, method: str) -> list[Candidate]:
        return [c for c in self.candidates if c.지원방법 == method]

    def find(self, 후보id: str) -> Candidate:
        for c in self.candidates:
            if c.후보id == 후보id:
                return c
        raise KeyError(후보id)


def _is_excluded(policy: dict, cand: dict) -> bool:
    """순위 대상에서 제외된 후보인지.

    정책 레벨 또는 후보 레벨에 _순위제외가 있으면 제외한다.
    (컨설팅·용도제한 보조금 등 2축에 반영되지 않는 것들)
    """
    return bool(policy.get("_순위제외") or cand.get("_순위제외"))


def load(path: str | Path = DATA_PATH, include_excluded: bool = False) -> PolicyDB:
    """정책 JSON을 읽어 후보 단위로 펼친다.

    Args:
        path: policies_all.json 경로
        include_excluded: True면 _순위제외 후보(부가 추천 대상)도 포함
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    candidates: list[Candidate] = []
    for policy in raw["policies"]:
        for cand in policy["후보"]:
            if not include_excluded and _is_excluded(policy, cand):
                continue
            candidates.append(
                Candidate(
                    후보id=cand["후보id"],
                    정책id=policy["id"],
                    정책명=policy["name"],
                    재원=policy["재원"],
                    # 지원방법은 후보에 있으면 후보 것이 우선
                    # (희망리턴처럼 한 정책 안에 보조금·컨설팅이 섞이는 경우)
                    지원방법=cand.get("지원방법", policy["지원방법"]),
                    현금효과=cand["현금효과"],
                    적용국면=tuple(policy["적용국면"]),
                    라벨=cand.get("라벨", ""),
                    raw=cand,
                    policy_raw=policy,
                )
            )

    return PolicyDB(raw=raw, candidates=candidates)


def summary(db: PolicyDB) -> str:
    """국면 × 지원방법 후보 수를 표로."""
    lines = [f"{'국면':<12} {'융자':>4} {'보조금':>5} {'채무조정':>6} {'합계':>5}"]
    for phase in PHASES:
        pool = db.by_phase(phase)
        counts = {m: sum(1 for c in pool if c.지원방법 == m)
                  for m in ("융자", "보조금", "채무조정")}
        lines.append(
            f"{phase:<12} {counts['융자']:>4} {counts['보조금']:>5} "
            f"{counts['채무조정']:>6} {len(pool):>5}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    db = load()
    print(f"정책 {len(db.raw['policies'])}개 / 순위 대상 후보 {len(db.candidates)}개")
    print(f"기준금리 {db.기준금리:.2%} / 기한필터 적용: {db.기한필터_적용}")
    print()
    print(summary(db))
    print()
    print("현금효과별 후보 수:")
    from collections import Counter
    for effect, n in Counter(c.현금효과 for c in db.candidates).most_common():
        print(f"  {effect:<8} {n}")
