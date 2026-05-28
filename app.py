"""
app.py — Streamlit frontend for the RAG Research Assistant
Run: streamlit run app.py
"""

import streamlit as st
import requests
import json
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "eval_scores" not in st.session_state:
    st.session_state.eval_scores = {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return None


def ingest_files(files):
    file_tuples = [("files", (f.name, f.getvalue(), f.type)) for f in files]
    r = requests.post(f"{API_URL}/ingest/upload", files=file_tuples, timeout=60)
    return r.json()


def ingest_urls(urls: list):
    r = requests.post(f"{API_URL}/ingest/urls", json=urls, timeout=60)
    return r.json()


def query_api(question: str, mode: str = "agentic"):
    r = requests.post(
        f"{API_URL}/query",
        json={"question": question, "mode": mode},
        timeout=60,
    )
    return r.json()


def reset_api():
    requests.delete(f"{API_URL}/reset", timeout=10)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📚 RAG Assistant")
    st.caption("Multi-source Research Assistant")

    # Health check
    health = api_health()
    if health:
        status_color = "🟢" if health.get("vector_store_ready") else "🟡"
        st.markdown(f"{status_color} API connected  \n"
                    f"LLM: `{health.get('llm_provider')}`  \n"
                    f"Embeddings: `{health.get('embedding_provider')}`")
    else:
        st.error("🔴 API not reachable — start the FastAPI server first.\n\n`python src/api.py`")

    st.divider()

    # ── Ingest section ────────────────────────────────────────────────────
    st.subheader("📥 Ingest Documents")

    ingest_tab, url_tab = st.tabs(["Upload Files", "From URLs"])

    with ingest_tab:
        uploaded_files = st.file_uploader(
            "PDF, DOCX, TXT",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )
        if st.button("Ingest Files", disabled=not uploaded_files):
            with st.spinner("Processing documents..."):
                result = ingest_files(uploaded_files)
                if result.get("status") == "success":
                    st.success(result["message"])
                    st.session_state.ingested = True
                else:
                    st.error(result.get("detail", "Ingest failed"))

    with url_tab:
        url_input = st.text_area("Enter URLs (one per line)")
        if st.button("Ingest URLs", disabled=not url_input.strip()):
            urls = [u.strip() for u in url_input.strip().splitlines() if u.strip()]
            with st.spinner(f"Fetching {len(urls)} URL(s)..."):
                result = ingest_urls(urls)
                if result.get("status") == "success":
                    st.success(result["message"])
                    st.session_state.ingested = True
                else:
                    st.error(result.get("detail", "Ingest failed"))

    st.divider()

    # ── Settings ──────────────────────────────────────────────────────────
    st.subheader("⚙️ Settings")
    mode = st.radio("RAG Mode", ["agentic", "basic"], index=0,
                    help="Agentic: ReAct agent with query rewriting. Basic: simple retrieve → generate.")
    show_sources = st.checkbox("Show source documents", value=True)
    show_eval = st.checkbox("Auto-evaluate responses", value=False)

    if st.button("🗑️ Reset Chat & Index"):
        reset_api()
        st.session_state.messages = []
        st.session_state.ingested = False
        st.session_state.eval_scores = {}
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

st.title("🔍 RAG Research Assistant")
st.caption("Ask questions about your documents. Powered by LangGraph Agentic RAG.")

# Eval scores banner
if st.session_state.eval_scores:
    scores = st.session_state.eval_scores
    cols = st.columns(len(scores))
    for col, (metric, val) in zip(cols, scores.items()):
        col.metric(metric.replace("_", " ").title(), f"{val:.2f}")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            with st.expander(f"📎 {len(msg['sources'])} source(s)"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {i}** — `{src['source']}`" +
                                (f" (page {src['page']})" if src.get("page") else ""))
                    st.caption(src["content"])

# Input
if not st.session_state.ingested:
    st.info("👆 Upload documents or enter URLs in the sidebar to get started.")

if prompt := st.chat_input("Ask anything about your documents...", disabled=not st.session_state.ingested):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = query_api(prompt, mode=mode)

        answer = result.get("answer", "Sorry, I could not generate an answer.")
        sources = result.get("sources", [])

        st.markdown(answer)

        if sources and show_sources:
            with st.expander(f"📎 {len(sources)} source(s)"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**Source {i}** — `{src['source']}`" +
                                (f" (page {src['page']})" if src.get("page") else ""))
                    st.caption(src["content"])

        # Auto-eval
        if show_eval and sources:
            with st.spinner("Evaluating response quality..."):
                eval_resp = requests.post(f"{API_URL}/evaluate", json={
                    "question": prompt,
                    "answer": answer,
                    "source_texts": [s["content"] for s in sources],
                }, timeout=60)
                if eval_resp.ok:
                    st.session_state.eval_scores = eval_resp.json().get("scores", {})
                    st.rerun()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
