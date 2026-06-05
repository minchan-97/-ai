"""
contract_app.py — Logic AI Contract Reviewer
============================================

계약서 조항별 두 관점(중립 설명 vs 위험 검토) 일관성 점검 도구.

이 도구가 무엇이고 무엇이 아닌가:
  - LLM consistency 체크 도구의 계약서 도메인 변주
  - 사용자가 본인 계약서를 직접 검토할 때의 보조용
  - 변호사 자문 대체가 아님
  - 법적 효력 판정이 아님
  - 책임은 사용자에게 있음

v4 코어와 공유:
  - OpenAI 임베딩 mismatch 계산 (코어 함수를 contract_app.py에 inline 복제)
  - safety 누적 차단 (선택)
  - cloud 위상 분석은 본 앱에서 제외 (조항별로 N회 호출은 비용 과다)
"""
from __future__ import annotations
import os, json, time, sqlite3, hashlib
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv

from contract import (
    split_clauses, CLAUSE_PROMPTS, build_clause_query, Clause,
)

# Safety 모듈 (선택)
try:
    from safety import (
        SafetyConfig, SafetyState,
        record_verdict as safety_record,
        request_release as safety_request_release,
        status as safety_status,
        reset_state as safety_reset,
    )
    SAFETY_AVAILABLE = True
except Exception as _e:
    SAFETY_AVAILABLE = False


load_dotenv()
st.set_page_config(page_title="Logic AI Contract Reviewer", layout="wide")

DB_PATH = "logic_ai_contract.db"
CHAT_URL_DEFAULT = "https://api.openai.com/v1/chat/completions"
EMBEDDING_URL_DEFAULT = "https://api.openai.com/v1/embeddings"
EMBEDDING_MODEL_DEFAULT = "text-embedding-3-small"


# =========================================================
# DB
# =========================================================
@st.cache_resource
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clause_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            doc_hash TEXT,
            clause_index INTEGER,
            clause_header TEXT,
            neutral_text TEXT,
            risk_text TEXT,
            mismatch REAL,
            threshold REAL,
            verdict TEXT,
            user_note TEXT
        )
    """)
    conn.commit()
    return conn


conn = get_db()


def save_review(doc_hash, idx, header, neutral, risk, mismatch, threshold, verdict, note=""):
    conn.execute(
        """INSERT INTO clause_reviews
           (doc_hash, clause_index, clause_header, neutral_text, risk_text,
            mismatch, threshold, verdict, user_note)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (doc_hash, idx, header, neutral, risk, mismatch, threshold, verdict, note),
    )
    conn.commit()


# =========================================================
# API helpers (v4 코어와 동일 패턴 - 자기완결성 위해 inline)
# =========================================================
def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


def call_chat(prompt: str, system: str, model: str, api_key: str,
              url: str = CHAT_URL_DEFAULT, temperature: float = 0.3) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    res = requests.post(url, headers=headers, json=body, timeout=90)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"].strip()


@lru_cache(maxsize=512)
def _cached_emb(key: str, text: str, model: str, url: str, api_key: str) -> Tuple[float, ...]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    res = requests.post(url, headers=headers,
                         json={"model": model, "input": text}, timeout=60)
    res.raise_for_status()
    return tuple(res.json()["data"][0]["embedding"])


def get_embedding(text: str, api_key: str, model: str, url: str) -> np.ndarray:
    if not text.strip():
        raise ValueError("빈 텍스트")
    h = hashlib.sha256(f"{model}::{text}".encode()).hexdigest()
    return np.array(_cached_emb(h, text, model, url, api_key), dtype=float)


def compute_mismatch(a: str, b: str, api_key: str, model: str, url: str) -> float:
    va = get_embedding(a, api_key, model, url)
    vb = get_embedding(b, api_key, model, url)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 100.0
    cs = float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))
    return (1.0 - cs) / 2.0 * 100.0


# =========================================================
# Session state
# =========================================================
if "threshold" not in st.session_state:
    st.session_state.threshold = 20.0  # 계약서 도메인은 v4 일반보다 약간 관대 (전문 용어 다양성)
if "reviews" not in st.session_state:
    st.session_state.reviews = []  # 마지막 분석의 조항별 결과
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if SAFETY_AVAILABLE:
    if "safety_state" not in st.session_state:
        st.session_state.safety_state = SafetyState(
            # 계약서는 보통 한 번에 N개 조항을 분석하므로 window를 더 크게
        )
    if "safety_cfg" not in st.session_state:
        st.session_state.safety_cfg = SafetyConfig(window_size=20, min_samples=10)


# =========================================================
# Header
# =========================================================
st.title("📑 Logic AI Contract Reviewer")
st.caption("계약서 조항별 두 관점 일관성 점검 (중립 설명 vs 위험 검토)")

with st.expander("⚠️ 이 도구가 무엇이고 무엇이 아닌가 (꼭 읽어주세요)", expanded=False):
    st.markdown("""
**무엇인가**
- 계약서 조항을 LLM에게 두 가지 관점(중립 설명 / 위험 검토)으로 분석시킨 뒤
  두 설명의 의미 거리를 OpenAI 임베딩으로 측정하는 일관성 점검 도구입니다.
- 두 관점이 크게 갈리는 조항이 **사용자가 직접 더 면밀히 살펴봐야 할 조항**
  이라는 신호로 활용됩니다.

**무엇이 아닌가**
- 변호사 자문 대체 아닙니다.
- 법적 효력 판정 도구 아닙니다.
- 조항의 적법성·유효성을 판정하지 않습니다.
- 모든 위험을 자동 탐지하지 않습니다.
- 결과의 책임은 사용자에게 있습니다.

**한계**
- 조항 분할은 휴리스틱입니다. 비표준 양식 계약서는 잘못 분할될 수 있습니다.
- LLM 두 응답이 비슷해도 둘 다 같이 빠뜨릴 수 있습니다(공통 환각).
- 두 응답이 다르다고 반드시 문제 있는 조항은 아닙니다. 검토 우선순위 신호일 뿐입니다.
- 임계값은 데이터로 검증되지 않은 휴리스틱입니다.

**중요한 의무 고지**
- 계약 체결 전 반드시 변호사·법무 전문가의 자문을 받으십시오.
- 본 도구의 결과를 법적 판단의 단독 근거로 사용하지 마십시오.
    """)


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("⚙️ 설정")
    st.session_state.threshold = st.slider(
        "조항별 임계값 τ (%)",
        1.0, 80.0, float(st.session_state.threshold), 1.0,
        help="이 값 초과면 INCONSISTENT(검토 필요) 표시",
    )

    st.divider()
    st.header("🔌 API")
    api_key = st.text_input("OpenAI API Key", value=get_secret("OPENAI_API_KEY"),
                             type="password")
    chat_model = st.text_input("Chat model",
                                value=get_secret("OPENAI_MODEL", "gpt-4o-mini"))
    emb_model = st.text_input("Embedding model",
                               value=get_secret("OPENAI_EMBEDDING_MODEL", EMBEDDING_MODEL_DEFAULT))

    st.divider()
    st.header("📝 시스템 프롬프트")
    neutral_sys = st.text_area("중립 설명", value=CLAUSE_PROMPTS["neutral"], height=150)
    risk_sys = st.text_area("위험 검토", value=CLAUSE_PROMPTS["risk"], height=150)

    # Safety (선택)
    if SAFETY_AVAILABLE:
        st.divider()
        st.markdown("### 🚨 누적 안전 차단 (선택)")
        safety_mode = st.checkbox(
            "INCONSISTENT 누적 감시",
            value=False,
            help="여러 계약서를 분석하다가 INCONSISTENT 비율이 높아지면 자동 잠금."
                 " 검토 피로 누적 방지용.",
        )
    else:
        safety_mode = False


# =========================================================
# Input
# =========================================================
st.subheader("1) 계약서 입력")

input_mode = st.radio("입력 방법", ["텍스트 붙여넣기", "파일 업로드 (.txt)"], horizontal=True)

if input_mode == "텍스트 붙여넣기":
    doc_text = st.text_area(
        "계약서 전체 텍스트",
        value=st.session_state.doc_text,
        height=250,
        placeholder="제1조 (목적) 본 계약은...\n제2조 (계약기간)...",
    )
else:
    uploaded = st.file_uploader("텍스트 파일", type=["txt", "md"])
    doc_text = ""
    if uploaded:
        doc_text = uploaded.read().decode("utf-8", errors="ignore")
        st.success(f"파일 로드 완료 ({len(doc_text):,} 자)")
        st.text_area("미리보기", value=doc_text[:500] + ("..." if len(doc_text) > 500 else ""),
                      height=120, disabled=True)

st.session_state.doc_text = doc_text


# =========================================================
# Preview clauses
# =========================================================
clauses: List[Clause] = []
if doc_text.strip():
    clauses = split_clauses(doc_text)
    st.subheader(f"2) 조항 분할 결과: {len(clauses)}개")
    if len(clauses) == 1 and clauses[0].header == "(전체)":
        st.warning(
            "조항 시작 패턴('제N조' 등)을 찾지 못해 전체를 1개 조항으로 처리합니다. "
            "분석 품질이 떨어질 수 있으니 가능하면 조항 헤더를 명시한 텍스트를 사용하세요."
        )
    with st.expander("조항 미리보기", expanded=False):
        for c in clauses:
            st.markdown(f"**[{c.index}] {c.header}**")
            preview = c.body[:200] + ("..." if len(c.body) > 200 else "")
            st.text(preview)


# =========================================================
# Run
# =========================================================
st.subheader("3) 분석 실행")

cost_estimate = ""
if clauses:
    # 조항당 chat 2회 + embed 2회. gpt-4o-mini + text-embedding-3-small 기준 대략.
    n = len(clauses)
    cost_estimate = (f"예상 호출: chat {2*n}회 + embedding {2*n}회. "
                     f"gpt-4o-mini 기준 약 {n*0.5:.1f}원 내외.")
    st.caption(cost_estimate)

run = st.button("📊 모든 조항 분석", type="primary", disabled=not clauses)

if run:
    # Safety 가드
    if safety_mode and SAFETY_AVAILABLE and st.session_state.safety_state.locked:
        st.error(f"🚨 안전 잠금 상태입니다. 사유: {st.session_state.safety_state.locked_reason}")
        st.stop()
    if not api_key:
        st.error("OpenAI API Key가 필요합니다.")
        st.stop()

    doc_hash = hashlib.sha256(doc_text.encode()).hexdigest()[:16]
    st.session_state.reviews = []

    progress = st.progress(0)
    status_box = st.empty()

    chat_url = get_secret("OPENAI_CHAT_URL", CHAT_URL_DEFAULT)
    emb_url = get_secret("OPENAI_EMBEDDING_URL", EMBEDDING_URL_DEFAULT)

    for i, c in enumerate(clauses):
        status_box.info(f"조항 {c.index}/{len(clauses)} 분석 중: {c.header}")
        query = build_clause_query(c)
        try:
            neutral = call_chat(query, neutral_sys, chat_model, api_key, chat_url)
            risk = call_chat(query, risk_sys, chat_model, api_key, chat_url)
            mm = compute_mismatch(neutral, risk, api_key, emb_model, emb_url)
            verdict = "PASS" if mm <= st.session_state.threshold else "INCONSISTENT"
        except Exception as e:
            neutral = f"[ERROR] {e}"
            risk = ""
            mm = float("nan")
            verdict = "ERROR"

        st.session_state.reviews.append({
            "clause": c,
            "neutral": neutral,
            "risk": risk,
            "mismatch": mm,
            "verdict": verdict,
        })
        save_review(doc_hash, c.index, c.header, neutral, risk,
                     mm if not (isinstance(mm, float) and np.isnan(mm)) else 0.0,
                     st.session_state.threshold, verdict)

        # Safety 기록
        if safety_mode and SAFETY_AVAILABLE and verdict in ("PASS", "INCONSISTENT"):
            ev = safety_record(st.session_state.safety_state, verdict, st.session_state.safety_cfg)
            if ev.get("just_triggered"):
                status_box.error("🚨 누적 안전 차단 발동. 더 이상 분석을 진행하지 않습니다.")
                break

        progress.progress((i + 1) / len(clauses))

    status_box.success(f"분석 완료: {len(st.session_state.reviews)}개 조항")


# =========================================================
# Results
# =========================================================
if st.session_state.reviews:
    st.divider()
    st.subheader("4) 분석 결과")

    # 요약 통계
    inconsistent = [r for r in st.session_state.reviews if r["verdict"] == "INCONSISTENT"]
    passed = [r for r in st.session_state.reviews if r["verdict"] == "PASS"]
    errored = [r for r in st.session_state.reviews if r["verdict"] == "ERROR"]

    c1, c2, c3 = st.columns(3)
    c1.metric("⚠️ 검토 우선", len(inconsistent))
    c2.metric("✅ 일관", len(passed))
    c3.metric("❌ 오류", len(errored))

    if inconsistent:
        st.warning(
            f"**{len(inconsistent)}개 조항**의 두 관점 분석이 크게 갈렸습니다. "
            "이 조항들을 우선적으로 직접 검토하세요. "
            "**(이는 문제 있는 조항이라는 단정이 아니라 검토 우선순위 신호입니다)**"
        )

    # 정렬: INCONSISTENT > PASS > ERROR, mismatch 높은 순
    def sort_key(r):
        order = {"INCONSISTENT": 0, "PASS": 1, "ERROR": 2}
        m = r["mismatch"] if not (isinstance(r["mismatch"], float) and np.isnan(r["mismatch"])) else 0
        return (order.get(r["verdict"], 99), -m)

    reviews_sorted = sorted(st.session_state.reviews, key=sort_key)

    for r in reviews_sorted:
        c = r["clause"]
        mm = r["mismatch"]
        mm_str = f"{mm:.1f}%" if not (isinstance(mm, float) and np.isnan(mm)) else "—"
        if r["verdict"] == "INCONSISTENT":
            icon = "⚠️"
        elif r["verdict"] == "PASS":
            icon = "✅"
        else:
            icon = "❌"
        with st.expander(f"{icon} [조항 {c.index}] {c.header} — mismatch {mm_str}",
                          expanded=(r["verdict"] == "INCONSISTENT")):
            st.caption("**원문**")
            st.text(c.body)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📘 중립 설명**")
                st.write(r["neutral"])
            with col2:
                st.markdown("**🔍 위험 검토**")
                st.write(r["risk"])
            if r["verdict"] == "INCONSISTENT":
                st.info(
                    "두 관점이 충분히 갈립니다. 이 조항이 모호하거나, "
                    "당사자에 따라 해석이 달라질 여지가 있을 수 있습니다. "
                    "원문과 두 분석을 비교하며 본인 입장에서 다시 읽어 보세요."
                )

# =========================================================
# Safety lock panel
# =========================================================
if safety_mode and SAFETY_AVAILABLE and st.session_state.safety_state.locked:
    st.divider()
    st.markdown("## 🚨 안전 잠금 관리")
    s = st.session_state.safety_state
    cfg = st.session_state.safety_cfg
    st.error(f"잠금 사유: {s.locked_reason}")
    st.caption(f"필요 승인: {cfg.required_signatures}명 ({', '.join(cfg.authorized_signers)} 중)")
    st.caption(f"현재 서명: {s.pending_signatures or '없음'}")
    signer = st.text_input("관리자 ID", key="signer_in")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("서명 제출", use_container_width=True):
            if signer:
                r = safety_request_release(s, signer.strip(), cfg)
                if r.get("released"):
                    st.success(r["msg"]); time.sleep(0.5); st.rerun()
                elif r.get("ok"):
                    st.info(r["msg"]); st.rerun()
                else:
                    st.error(r["msg"])
    with c2:
        if st.button("강제 초기화", use_container_width=True):
            safety_reset(s); st.warning("초기화됨"); time.sleep(0.5); st.rerun()
