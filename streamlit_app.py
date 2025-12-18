import os
import re
import json
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

# --- OpenAI (latest official SDK style) ---
from openai import OpenAI


# =========================
# Config
# =========================
st.set_page_config(page_title="유의어 & 품사변화 챗봇", layout="wide")

SYSTEM_CHAT_PROMPT = """You are a helpful English learning assistant for Korean high-school students.
1) Be concise and clear.
2) If user asks for synonyms/derivations, prefer high-frequency, exam-appropriate vocabulary.
3) Avoid extremely basic middle-school words unless explicitly requested.
"""

SYSTEM_ANALYSIS_PROMPT = """You are an English linguistics assistant.
Given an English sentence and a list of extracted keywords (content words),
return JSON ONLY (no markdown) with:
[
  {
    "keyword": "...",
    "pos": "NOUN|VERB|ADJ|ADV|OTHER",
    "synonyms": ["...", "...", ...],          # 5~8 items, context-appropriate
    "derivations": {                          # provide common POS variations if meaningful
      "noun": ["..."],
      "verb": ["..."],
      "adjective": ["..."],
      "adverb": ["..."]
    }
  },
  ...
]
Rules:
- Keep synonyms context-appropriate for the sentence meaning.
- Derivations should be real, common English words (not forced).
- If unknown/none, use empty list.
- Output must be valid JSON.
"""


# =========================
# Helpers: API Key + Client
# =========================
def get_api_key() -> str:
    # Streamlit Cloud: st.secrets -> env fallback
    key = None
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        key = st.secrets["OPENAI_API_KEY"]
    if not key:
        key = os.getenv("OPENAI_API_KEY")
    return (key or "").strip()


@st.cache_resource
def get_openai_client() -> OpenAI:
    key = get_api_key()
    if not key:
        # No API key: fail early with a clear message.
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되어 있지 않습니다. "
            ".streamlit/secrets.toml에 OPENAI_API_KEY를 설정하세요."
        )
    # Either set env or pass directly; passing directly is explicit and safe.
    return OpenAI(api_key=key)


# =========================
# Helpers: Keyword extraction (lightweight, no extra deps)
# =========================
STOPWORDS = {
    "the","a","an","and","or","but","if","then","so","because","as","while","when",
    "to","of","in","on","at","for","with","by","from","into","over","under","about",
    "is","are","was","were","be","been","being","am",
    "do","does","did","doing",
    "have","has","had","having",
    "will","would","can","could","may","might","should","must",
    "this","that","these","those","it","they","we","you","i","he","she","them","us",
    "my","your","his","her","their","our",
    "not","no","yes",
}

def normalize_token(t: str) -> str:
    t = re.sub(r"[^A-Za-z'-]", "", t).lower()
    return t.strip("-'")

def extract_keywords_simple(text: str, max_keywords: int = 10) -> List[str]:
    # Simple heuristic: take content-like words, remove stopwords, unique preserve order
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    out = []
    seen = set()
    for tok in tokens:
        w = normalize_token(tok)
        if not w or w in STOPWORDS:
            continue
        if len(w) <= 2:
            continue
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= max_keywords:
            break
    return out


# =========================
# Helpers: OpenAI Calls (Responses API preferred)
# =========================
def call_llm_chat(client: OpenAI, messages: List[Dict[str, str]]) -> str:
    """
    messages: [{"role":"system|user|assistant","content":"..."}]
    """
    # Prefer Responses API (newer). Fallback to Chat Completions if needed.
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=messages,
            temperature=0.4,
        )
        # Responses API returns output_text aggregated
        return (resp.output_text or "").strip()
    except Exception:
        # Fallback (older style)
        cc = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
        )
        return (cc.choices[0].message.content or "").strip()


def call_llm_json(client: OpenAI, sentence: str, keywords: List[str], max_syn: int = 8) -> List[Dict[str, Any]]:
    user_payload = {
        "sentence": sentence,
        "keywords": keywords,
        "max_synonyms_each": max_syn,
    }

    messages = [
        {"role": "system", "content": SYSTEM_ANALYSIS_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    text = call_llm_chat(client, messages)

    # Strict JSON parsing with a couple of safe fallbacks
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Attempt to extract JSON array substring if the model added extra text
    m = re.search(r"(\[\s*\{.*\}\s*\])", text, flags=re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    raise ValueError("분석 결과 JSON 파싱에 실패했습니다. (모델 출력 형식 오류)")


def to_table_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for it in items:
        kw = (it.get("keyword") or "").strip()
        pos = (it.get("pos") or "OTHER").strip().upper()

        syns = it.get("synonyms") or []
        if not isinstance(syns, list):
            syns = []

        deriv = it.get("derivations") or {}
        if not isinstance(deriv, dict):
            deriv = {}

        def fmt_list(x):
            if not x:
                return ""
            if not isinstance(x, list):
                return ""
            # unique preserve order
            seen = set()
            out = []
            for v in x:
                v = str(v).strip()
                if v and v not in seen:
                    seen.add(v)
                    out.append(v)
            return ", ".join(out)

        rows.append({
            "Keyword": kw,
            "POS": pos,
            "Synonyms": fmt_list(syns) or "(not found)",
            "Deriv(N)": fmt_list(deriv.get("noun")),
            "Deriv(V)": fmt_list(deriv.get("verb")),
            "Deriv(Adj)": fmt_list(deriv.get("adjective")),
            "Deriv(Adv)": fmt_list(deriv.get("adverb")),
        })
    return rows


# =========================
# UI
# =========================
st.title("유의어 & 품사변화 정리 챗봇 (GPT-4o-mini)")

# Initialize client (fail fast with a clear message)
try:
    client = get_openai_client()
except Exception as e:
    st.error(str(e))
    st.stop()

tab1, tab2 = st.tabs(["💬 챗봇", "🧠 문장 분석(유의어/품사변화 표)"])

# ---- Tab 1: Chat ----
with tab1:
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role": "system", "content": SYSTEM_CHAT_PROMPT}]

    # Render history (skip system)
    for m in st.session_state.chat_messages:
        if m["role"] == "system":
            continue
        with st.chat_message(m["role"]):
            st.write(m["content"])

    prompt = st.chat_input("메시지를 입력하세요 (예: 이 문장 쉬운 말로 바꿔줘 / 유의어 알려줘 등)")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("GPT-4o-mini 응답 생성 중..."):
                answer = call_llm_chat(client, st.session_state.chat_messages)
            st.write(answer)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})

# ---- Tab 2: Analysis table ----
with tab2:
    st.caption("영어 문장을 입력하면 핵심 어휘를 뽑고(간단 규칙), GPT가 문맥 유의어 + 품사변화를 표로 정리합니다.")

    default_sentence = "The government implemented a new policy to protect the environment."
    sentence = st.text_area("영어 문장 입력", value=default_sentence, height=100)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        max_keywords = st.slider("키워드 개수", 5, 20, 10)
    with c2:
        max_syn = st.slider("키워드당 유의어", 5, 10, 8)

    if st.button("분석하기", type="primary"):
        s = sentence.strip()
        if not s:
            st.error("문장을 입력하세요.")
            st.stop()

        keywords = extract_keywords_simple(s, max_keywords=max_keywords)
        if not keywords:
            st.warning("추출된 키워드가 없습니다. 문장을 조금 더 길게 입력해보세요.")
            st.stop()

        st.write("추출 키워드:", ", ".join(keywords))

        try:
            with st.spinner("GPT로 유의어/품사변화 분석 중..."):
                items = call_llm_json(client, s, keywords, max_syn=max_syn)
            rows = to_table_rows(items)
            df = pd.DataFrame(rows)
            st.subheader("결과 표")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"분석 중 오류: {e}")
            st.info("모델이 JSON 형식을 어기면 오류가 날 수 있습니다. 다시 시도하거나 문장을 조금 바꿔보세요.")
