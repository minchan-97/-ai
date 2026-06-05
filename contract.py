"""
contract.py — 계약서 조항 분할 및 두 관점 분석 (Logic AI Contract)
======================================================================

도메인 특화 부분만 격리. v4 코어(임베딩 비교, 안전 차단)는 그대로 사용.

핵심 함수:
  split_clauses(text)         — 한국어 계약서를 조항 단위로 분할
  CLAUSE_PROMPTS              — '일반 해석' vs '위험 검토' 두 system prompt
  build_clause_query(clause)  — 조항 1개에 대한 분석 질의 텍스트 생성

설계 원칙:
  - 정규식 기반 분할(외부 라이브러리 없음, 의존성 0)
  - 분할 실패 시 통째로 1개 조항으로 처리 (안전 폴백)
  - 분할 규칙은 한국어 계약서 통상 양식 기준
  - 시스템 프롬프트는 변호사 자문 톤이 아니라 '같이 검토하는 보조' 톤

정직한 한계 (UI에도 명시):
  - 조항 분할은 휴리스틱. 비표준 양식 계약서는 잘못 분할될 수 있음.
  - 두 관점은 LLM의 self-consistency 신호일 뿐, 실제 법적 위험 판정이 아님.
  - 변호사 대체 아님. 사용자가 본인 책임으로 검토하는 보조 도구.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional


# ----------------------------------------------------------------
# 조항 분할
# ----------------------------------------------------------------

# 한국어 계약서에서 자주 나오는 조항 시작 패턴
# - "제1조", "제 1 조", "제1조(제목)", "제1조 [제목]"
# - 영문 계약서 일부 흡수: "Article 1.", "Section 1."
# - 번호만: "1.", "1)" (덜 신뢰)
_CLAUSE_PATTERNS = [
    re.compile(r"^\s*제\s*\d+\s*조[\s\(\[【]", re.MULTILINE),
    re.compile(r"^\s*Article\s+\d+[\.\s]", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Section\s+\d+[\.\s]", re.MULTILINE | re.IGNORECASE),
]


@dataclass
class Clause:
    index: int           # 1부터 시작
    header: str          # "제1조(목적)" 같은 헤더
    body: str            # 본문
    raw: str             # 전체 (header + body)


def _find_clause_starts(text: str) -> List[int]:
    """모든 조항 시작 위치를 찾아 정렬해 반환."""
    positions = set()
    for pat in _CLAUSE_PATTERNS:
        for m in pat.finditer(text):
            positions.add(m.start())
    return sorted(positions)


def _extract_header(chunk: str) -> tuple[str, str]:
    """청크에서 헤더 한 줄과 나머지 본문 분리."""
    # 첫 줄을 헤더로 (괄호 안 제목까지 포함)
    lines = chunk.split("\n", 1)
    header = lines[0].strip()
    body = lines[1].strip() if len(lines) > 1 else ""
    # 헤더에 ")" 또는 "]"가 있으면 거기까지를 헤더로
    # (예: "제1조(목적) 본 계약은..." → 헤더 "제1조(목적)", 본문 "본 계약은...")
    for close, open_ in [(")", "("), ("]", "["), ("】", "【")]:
        if close in header and open_ in header:
            idx = header.find(close)
            if idx > 0 and idx < len(header) - 1:
                rest = header[idx+1:].strip()
                header = header[:idx+1].strip()
                if rest:
                    body = rest + ("\n" + body if body else "")
                break
    return header, body


def split_clauses(text: str, min_clause_chars: int = 30) -> List[Clause]:
    """계약서 텍스트를 조항 단위로 분할.

    분할 실패 시(조항 시작 패턴이 1개 이하)면 전체를 한 조항으로 반환.
    너무 짧은 조항(min_clause_chars 미만)은 이전 조항에 병합.
    """
    text = text.strip()
    if not text:
        return []

    starts = _find_clause_starts(text)
    if len(starts) <= 1:
        # 분할 실패 — 통째로 1개
        return [Clause(index=1, header="(전체)", body=text, raw=text)]

    # 첫 패턴 매치 이전의 머리말(전문, 당사자 표시 등)은 별도로 보존하지 않고 무시
    # — 실제 계약서에서 보통 의미 있는 위험 분석 대상은 조항 본문이므로
    clauses: List[Clause] = []
    boundaries = starts + [len(text)]
    for i in range(len(boundaries) - 1):
        chunk = text[boundaries[i]:boundaries[i+1]].strip()
        if not chunk:
            continue
        header, body = _extract_header(chunk)
        clause = Clause(index=len(clauses)+1, header=header, body=body, raw=chunk)
        # 너무 짧으면 이전과 병합
        if clauses and len(chunk) < min_clause_chars:
            prev = clauses[-1]
            clauses[-1] = Clause(
                index=prev.index,
                header=prev.header,
                body=prev.body + "\n" + chunk,
                raw=prev.raw + "\n" + chunk,
            )
        else:
            clauses.append(clause)

    return clauses


# ----------------------------------------------------------------
# 두 관점 system prompt
# ----------------------------------------------------------------

CLAUSE_PROMPTS = {
    "neutral": (
        "당신은 계약서 조항을 일반적인 시각에서 설명하는 보조자입니다. "
        "주어진 조항이 무엇을 규정하는지, 양 당사자의 권리·의무가 무엇인지 "
        "한국어로 객관적이고 간결하게 설명하세요. "
        "추측하지 말고 조항 본문에 적힌 것만 다루세요. "
        "법적 자문이 아닌 정보 제공임을 잊지 마세요."
    ),
    "risk": (
        "당신은 계약서 조항을 검토하며 잠재적 위험을 식별하는 보조자입니다. "
        "주어진 조항에서 사용자(계약 일방)에게 불리하게 작용할 수 있는 부분, "
        "모호한 표현, 일방적 의무·책임, 면책·제한·해지 관련 위험을 한국어로 "
        "간결하게 짚어 주세요. 추측을 최소화하고 조항 본문에서 근거를 들어 "
        "설명하세요. 법적 자문이 아닌 검토 보조임을 잊지 마세요."
    ),
}


def build_clause_query(clause: Clause) -> str:
    """조항 1개를 LLM 질의 텍스트로 변환."""
    header = clause.header if clause.header else f"제{clause.index}조"
    return f"[조항 {header}]\n{clause.body}"


# ----------------------------------------------------------------
# 자체 검증
# ----------------------------------------------------------------

if __name__ == "__main__":
    sample = """
계약서

본 계약은 갑과 을 사이에 체결된다.

제1조 (목적) 본 계약은 갑이 을에게 제공하는 용역의 조건을 정함을 목적으로 한다.

제2조 (계약기간)
본 계약의 기간은 2025년 1월 1일부터 2025년 12월 31일까지로 한다.
다만 갑은 언제든지 30일 전 통보로 계약을 해지할 수 있다.

제3조 (대금지급) 을은 매월 말일까지 용역 대금을 청구하고, 갑은 청구일로부터 60일 이내에 지급한다.

제4조 (지적재산권) 본 계약 수행 과정에서 발생한 모든 지적재산권은 갑에게 귀속된다.
"""
    clauses = split_clauses(sample)
    print(f"=== 분할 결과: {len(clauses)}개 조항 ===\n")
    for c in clauses:
        print(f"--- 조항 {c.index} ---")
        print(f"헤더: {c.header}")
        print(f"본문: {c.body[:80]}{'...' if len(c.body) > 80 else ''}")
        print()

    # 비표준 양식 폴백 테스트
    print("=== 비표준 양식 (조항 시작 패턴 없음) ===")
    flat_text = "이것은 평문 계약서입니다. 갑과 을은 다음에 합의한다. 1년간 매월 100만원을 지급한다."
    clauses2 = split_clauses(flat_text)
    print(f"분할 수: {len(clauses2)} (1개로 폴백되어야 함)")
    assert len(clauses2) == 1
    print(f"본문 일부: {clauses2[0].body[:60]}")
