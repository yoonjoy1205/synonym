import os
import re
import json
from typing import List, Dict, Any
from datetime import datetime
import uuid
from pathlib import Path

import streamlit as st
import pandas as pd

# --- OpenAI (latest official SDK style) ---
from openai import OpenAI
from openai import RateLimitError, AuthenticationError, BadRequestError


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

# 단어뜻 정리 전용 시스템 프롬프트
SYSTEM_VOCAB_PROMPT = """You are an English vocabulary assistant for Korean learners.
Given a list of English words, return JSON ONLY (no markdown) as an array:
[
    {
        "word": "...",
        "pos": "NOUN|VERB|ADJ|ADV|OTHER",
        "meaning_ko": "간결한 한국어 뜻 (핵심만)",
        "synonyms": ["...", "..."],
        "examples": ["짧은 예문 1", "짧은 예문 2"],
        "derivations": {
            "noun": ["..."],
            "verb": ["..."],
            "adjective": ["..."],
            "adverb": ["..."]
        }
    }
]
Rules:
- Keep meanings short and accurate in Korean.
- Provide 2-4 context-appropriate synonyms if available.
- Provide 1-2 short, simple example sentences.
- If any field is not applicable, use an empty list or empty string.
- Output must be valid JSON array only.
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
# Helpers: Simple Login & Local Storage (Vocabulary)
# =========================
DATA_DIR = Path("data/vocab")

def sanitize_username(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_\-ㄱ-힣]", "_", name.strip())
    return name or "user"

def user_file(username: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{username}.json"

def load_wordbook(username: str) -> List[Dict[str, Any]]:
    fp = user_file(username)
    if fp.exists():
        try:
            data = json.load(fp.open("r", encoding="utf-8")) or []
            # ensure ids on load
            changed = False
            for it in data:
                if "id" not in it:
                    it["id"] = str(uuid.uuid4())
                    changed = True
            if changed:
                save_wordbook(username, data)
            return data
        except Exception:
            return []
    return []

def save_wordbook(username: str, items: List[Dict[str, Any]]):
    fp = user_file(username)
    fp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

def append_wordbook(username: str, entries: List[Dict[str, Any]]):
    if not entries:
        return
    data = load_wordbook(username)
    # ensure ids for new entries
    for it in entries:
        it.setdefault("id", str(uuid.uuid4()))

    # merge + dedupe by (word, pos, source, sentence)
    key_fn = lambda it: (str(it.get("word", "")).lower(), str(it.get("pos", "")).upper(), str(it.get("source", "")), str(it.get("sentence", "")))
    merged: Dict[Any, Dict[str, Any]] = {}
    for it in data + entries:
        k = key_fn(it)
        if k not in merged:
            merged[k] = it
            continue
        # merge fields: prefer newer timestamp, union lists
        base = merged[k]
        try:
            t_new = it.get("timestamp", "") or ""
            t_old = base.get("timestamp", "") or ""
            if t_new > t_old:
                base["timestamp"] = t_new
                base["id"] = it.get("id", base.get("id", str(uuid.uuid4())))
        except Exception:
            pass
        # fill meaning if empty
        if (not base.get("meaning_ko")) and it.get("meaning_ko"):
            base["meaning_ko"] = it.get("meaning_ko")
        # union synonyms/examples
        def union_list(a, b):
            out, seen = [], set()
            for v in (a or []) + (b or []):
                s = str(v).strip()
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
            return out
        base["synonyms"] = union_list(base.get("synonyms"), it.get("synonyms"))
        base["examples"] = union_list(base.get("examples"), it.get("examples"))
        # merge derivations by pos lists
        d0 = base.get("derivations") or {}
        d1 = it.get("derivations") or {}
        for p in ["noun", "verb", "adjective", "adverb"]:
            d0[p] = union_list(d0.get(p), d1.get(p))
        base["derivations"] = d0
        # keep note/tags
        base.setdefault("note", it.get("note", ""))
        base.setdefault("tags", it.get("tags", []))

    data = list(merged.values())
    save_wordbook(username, data)

def get_all_tags(username: str) -> List[str]:
    data = load_wordbook(username)
    tags = []
    seen = set()
    for it in data:
        for t in it.get("tags", []) or []:
            s = str(t).strip()
            if s and s not in seen:
                seen.add(s)
                tags.append(s)
    return tags

def delete_wordbook_entry(username: str, entry_id: str):
    data = load_wordbook(username)
    data = [e for e in data if str(e.get("id")) != str(entry_id)]
    save_wordbook(username, data)

def record_vocab_from_vocab_items(username: str, items: List[Dict[str, Any]]):
    now = datetime.utcnow().isoformat()
    entries = []
    for it in items:
        entries.append({
            "id": str(uuid.uuid4()),
            "word": it.get("word"),
            "pos": it.get("pos"),
            "meaning_ko": it.get("meaning_ko", ""),
            "synonyms": it.get("synonyms", []),
            "examples": it.get("examples", []),
            "derivations": it.get("derivations", {}),
            "tags": [],
            "note": "",
            "source": "vocab",
            "sentence": "",
            "timestamp": now,
        })
    append_wordbook(username, entries)

def record_vocab_from_analysis_items(username: str, items: List[Dict[str, Any]], sentence: str):
    now = datetime.utcnow().isoformat()
    entries = []
    for it in items:
        entries.append({
            "id": str(uuid.uuid4()),
            "word": it.get("keyword"),
            "pos": it.get("pos"),
            "meaning_ko": "",
            "synonyms": it.get("synonyms", []),
            "examples": [],
            "derivations": it.get("derivations", {}),
            "tags": [],
            "note": "",
            "source": "analysis",
            "sentence": sentence,
            "timestamp": now,
        })
    append_wordbook(username, entries)


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
    표준 Chat Completions API만 사용해 간결하게 응답을 생성합니다.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
        )
        return (response.choices[0].message.content or "").strip()
    except RateLimitError:
        st.error("🚫 OpenAI API 쿼터가 부족합니다. (충전된 크레딧 잔액을 확인하세요)")
        st.stop()
    except AuthenticationError:
        st.error("🚫 OpenAI API 키가 유효하지 않습니다. .streamlit/secrets.toml의 OPENAI_API_KEY를 확인하세요.")
        st.stop()
    except BadRequestError as e:
        st.error(f"요청 형식 오류: {e}")
        st.stop()
    except Exception as e:
        st.error(f"오류 발생: {e}")
        st.stop()


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


def call_llm_vocab_json(client: OpenAI, words: List[str]) -> List[Dict[str, Any]]:
    payload = {"words": words}

    messages = [
        {"role": "system", "content": SYSTEM_VOCAB_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    text = call_llm_chat(client, messages)

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    m = re.search(r"(\[\s*\{.*\}\s*\])", text, flags=re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    raise ValueError("단어뜻 JSON 파싱에 실패했습니다. (모델 출력 형식 오류)")


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


def to_vocab_rows(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for it in items:
        word = str(it.get("word", "")).strip()
        pos = str(it.get("pos", "")).strip().upper() or "OTHER"
        meaning_ko = str(it.get("meaning_ko", "")).strip()
        syns = it.get("synonyms") or []
        exs = it.get("examples") or []
        deriv = it.get("derivations") or {}

        def fmt_list(x):
            if not x or not isinstance(x, list):
                return ""
            seen, out = set(), []
            for v in x:
                v = str(v).strip()
                if v and v not in seen:
                    seen.add(v)
                    out.append(v)
            return ", ".join(out)

        rows.append({
            "Word": word,
            "POS": pos,
            "Meaning(KO)": meaning_ko or "-",
            "Synonyms": fmt_list(syns),
            "Examples": fmt_list(exs),
            "Deriv(N)": fmt_list((deriv or {}).get("noun")),
            "Deriv(V)": fmt_list((deriv or {}).get("verb")),
            "Deriv(Adj)": fmt_list((deriv or {}).get("adjective")),
            "Deriv(Adv)": fmt_list((deriv or {}).get("adverb")),
        })
    return rows


# =========================
# UI
# =========================
st.title("🔑유의어 & 품사변화 정리 챗봇")

# --- Login gate ---
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if not st.session_state.user_name:
    with st.form("login_form", clear_on_submit=False):
        name_input = st.text_input("이름으로 로그인", placeholder="예: 홍길동")
        login_btn = st.form_submit_button("로그인")
    if login_btn:
        name = sanitize_username(name_input)
        if not name:
            st.error("이름을 입력하세요.")
            st.stop()
        st.session_state.user_name = name
        st.success(f"{name}님, 환영합니다!")
        st.rerun()
    st.stop()

user_name = st.session_state.user_name
st.caption(f"현재 사용자: {user_name}")

# Initialize client (fail fast with a clear message)
try:
    client = get_openai_client()
except Exception as e:
    st.error(str(e))
    st.stop()

tab1, tab2, tab3 = st.tabs(["💬 챗봇", "🧠 문장 분석(유의어/품사변화 표)", "📒 단어장"])

# ---- Tab 1: Chat ----
with tab1:
    st.subheader("🧾 단어뜻 정리")
    col1, col2 = st.columns([3,1])
    with col1:
        vocab_input = st.text_input("단어들을 쉼표로 구분해 입력 (예: implement, policy, environment)")
    with col2:
        run_vocab = st.button("정리하기")

    if run_vocab:
        words = [w.strip() for w in (vocab_input or "").replace("\n", " ").split(",") if w.strip()]
        if not words:
            st.error("단어를 한 개 이상 입력하세요.")
        else:
            try:
                with st.spinner("단어 뜻 정리 중..."):
                    items = call_llm_vocab_json(client, words)
                vrows = to_vocab_rows(items)
                vdf = pd.DataFrame(vrows)
                st.dataframe(vdf, use_container_width=True)
                # 기록 저장
                record_vocab_from_vocab_items(user_name, items)
            except Exception as e:
                st.error(f"단어뜻 정리 중 오류: {e}")

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

        SAMPLE_SENTENCE = "the government implemented a new policy to protect the environment."  # lowercase 비교용
        SAMPLE_ROWS = [
            {"Keyword": "government", "POS": "NOUN", "Synonyms": "administration, authority, regime, state", "Deriv(N)": "", "Deriv(V)": "", "Deriv(Adj)": "governmental", "Deriv(Adv)": ""},
            {"Keyword": "implement", "POS": "VERB", "Synonyms": "execute, carry out, enforce, apply", "Deriv(N)": "implementation, implement", "Deriv(V)": "", "Deriv(Adj)": "implementable", "Deriv(Adv)": ""},
            {"Keyword": "policy", "POS": "NOUN", "Synonyms": "strategy, guideline, measure, rule", "Deriv(N)": "", "Deriv(V)": "", "Deriv(Adj)": "policy-related, policy-driven", "Deriv(Adv)": ""},
            {"Keyword": "protect", "POS": "VERB", "Synonyms": "safeguard, defend, preserve, secure", "Deriv(N)": "protection, protector", "Deriv(V)": "", "Deriv(Adj)": "protective, protected", "Deriv(Adv)": ""},
            {"Keyword": "environment", "POS": "NOUN", "Synonyms": "surroundings, setting, habitat, ecosystem", "Deriv(N)": "", "Deriv(V)": "", "Deriv(Adj)": "environmental", "Deriv(Adv)": "environmentally"},
        ]

        try:
            with st.spinner("GPT로 유의어/품사변화 분석 중..."):
                items = call_llm_json(client, s, keywords, max_syn=max_syn)
            rows = to_table_rows(items)
            df = pd.DataFrame(rows)
            st.subheader("결과 표")
            st.dataframe(df, use_container_width=True)
            # 기록 저장
            record_vocab_from_analysis_items(user_name, items, s)
        except Exception as e:
            # Fallback: 기본 예시 문장일 때 샘플 표 표시
            if s.lower() == SAMPLE_SENTENCE:
                st.warning("모델 호출에 실패했습니다. 샘플 표를 대신 표시합니다.")
                df = pd.DataFrame(SAMPLE_ROWS)
                st.subheader("결과 표 (샘플)")
                st.dataframe(df, use_container_width=True)
                # 샘플 표도 기록 (간단 변환)
                now = datetime.utcnow().isoformat()
                sample_entries = []
                for r in SAMPLE_ROWS:
                    syns = [v.strip() for v in (r.get("Synonyms") or "").split(",") if v.strip()]
                    sample_entries.append({
                        "word": r.get("Keyword"),
                        "pos": r.get("POS"),
                        "meaning_ko": "",
                        "synonyms": syns,
                        "examples": [],
                        "derivations": {
                            "noun": [v.strip() for v in (r.get("Deriv(N)") or "").split(",") if v.strip()],
                            "verb": [v.strip() for v in (r.get("Deriv(V)") or "").split(",") if v.strip()],
                            "adjective": [v.strip() for v in (r.get("Deriv(Adj)") or "").split(",") if v.strip()],
                            "adverb": [v.strip() for v in (r.get("Deriv(Adv)") or "").split(",") if v.strip()],
                        },
                        "source": "analysis",
                        "sentence": s,
                        "timestamp": now,
                    })
                append_wordbook(user_name, sample_entries)
            else:
                st.error(f"분석 중 오류: {e}")
                st.info("모델이 JSON 형식을 어기면 오류가 날 수 있습니다. 다시 시도하거나 문장을 조금 바꿔보세요.")

# ---- Tab 3: Wordbook ----
with tab3:
    st.caption("로그인 사용자별로 누적된 검색 기록을 단어장으로 제공합니다.")
    data = load_wordbook(user_name)

    # Filters
    q = st.text_input("검색(단어/뜻/예문 포함)", placeholder="예: environment")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        group_by_date = st.checkbox("날짜별 그룹", value=False)
    with c2:
        source_filter = st.multiselect("소스", options=["vocab","analysis"], default=["vocab","analysis"])
    with c3:
        tag_options = get_all_tags(user_name)
        tag_filter = st.multiselect("태그 필터", options=tag_options, default=[])

    view = data
    if q:
        ql = q.lower()
        def match(entry: Dict[str, Any]) -> bool:
            text_blob = json.dumps(entry, ensure_ascii=False).lower()
            return ql in text_blob
        view = [e for e in view if match(e)]
    if source_filter:
        view = [e for e in view if (e.get("source") in source_filter)]
    if tag_filter:
        tagset = set(tag_filter)
        view = [e for e in view if tagset.intersection(set(e.get("tags", [])))]

    # 표로 보기 (요약)
    def to_display_rows(items: List[Dict[str, Any]]):
        rows = []
        for it in items:
            rows.append({
                "ID": it.get("id", ""),
                "Word": it.get("word", ""),
                "POS": (it.get("pos") or "").upper(),
                "Meaning(KO)": it.get("meaning_ko", ""),
                "Synonyms": ", ".join(it.get("synonyms") or []),
                "Tags": ", ".join(it.get("tags") or []),
                "Source": it.get("source", ""),
                "When(UTC)": it.get("timestamp", ""),
            })
        return rows

    disp_rows = to_display_rows(view)
    if not group_by_date:
        st.dataframe(pd.DataFrame(disp_rows), use_container_width=True)
    else:
        # group by YYYY-MM-DD
        from collections import defaultdict
        buckets = defaultdict(list)
        for it in view:
            ts = (it.get("timestamp") or "")[:10]
            buckets[ts].append(it)
        for day in sorted(buckets.keys(), reverse=True):
            with st.expander(f"{day} ({len(buckets[day])})", expanded=False):
                st.dataframe(pd.DataFrame(to_display_rows(buckets[day])), use_container_width=True)

    st.markdown("---")
    st.subheader("✏️ 기록 편집")
    # 선택 및 편집 폼
    if view:
        options = [f"{i+1}. {e.get('word','')} / {e.get('pos','')} / {e.get('source','')} @ {e.get('timestamp','')}" for i, e in enumerate(view)]
        idx = st.selectbox("수정할 항목 선택", options=range(len(view)), format_func=lambda i: options[i])
        sel = view[idx]
        cur_tags = ", ".join(sel.get("tags") or [])
        cur_note = sel.get("note", "")
        tags_in = st.text_input("태그(쉼표로 구분)", value=cur_tags)
        note_in = st.text_area("메모", value=cur_note)
        save_col, del_col = st.columns(2)
        if save_col.button("변경 저장"):
            # update in full data by id if available else by tuple key
            full = load_wordbook(user_name)
            target_id = sel.get("id")
            for e in full:
                if (target_id and e.get("id") == target_id) or (
                    not target_id and e.get("word") == sel.get("word") and e.get("pos") == sel.get("pos") and e.get("timestamp") == sel.get("timestamp")
                ):
                    e["tags"] = [t.strip() for t in tags_in.split(",") if t.strip()]
                    e["note"] = note_in
                    break
            save_wordbook(user_name, full)
            st.success("저장했습니다.")
            st.rerun()
        if del_col.button("선택 항목 삭제"):
            if sel.get("id"):
                delete_wordbook_entry(user_name, sel.get("id"))
            else:
                # fallback match deletion
                full = load_wordbook(user_name)
                full = [e for e in full if not (e.get("word") == sel.get("word") and e.get("pos") == sel.get("pos") and e.get("timestamp") == sel.get("timestamp"))]
                save_wordbook(user_name, full)
            st.success("삭제했습니다.")
            st.rerun()

    st.markdown("---")
    if st.button("🧹 중복 정리 및 병합"):
        # trigger dedupe by appending nothing (save re-run not enough). We'll reload and re-append same data to apply merge logic.
        data_all = load_wordbook(user_name)
        save_wordbook(user_name, [])
        append_wordbook(user_name, data_all)
        st.success("중복 정리 완료")
        st.rerun()

    # 내보내기/초기화
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📤 CSV 내보내기"):
            import io
            import csv
            headers = list(disp_rows[0].keys()) if disp_rows else ["ID","Word","POS","Meaning(KO)","Synonyms","Tags","Source","When(UTC)"]
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=headers)
            writer.writeheader()
            for r in disp_rows:
                writer.writerow(r)
            st.download_button("CSV 다운로드", data=buf.getvalue(), file_name=f"{user_name}_wordbook.csv", mime="text/csv")
        if st.button("📥 JSON 내보내기"):
            st.download_button("JSON 다운로드", data=json.dumps(view, ensure_ascii=False, indent=2), file_name=f"{user_name}_wordbook.json", mime="application/json")
    with c2:
        if st.button("🗑️ 전체 삭제"):
            save_wordbook(user_name, [])
            st.success("단어장 기록을 모두 삭제했습니다.")
            st.rerun()
