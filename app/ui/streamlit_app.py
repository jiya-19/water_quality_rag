import sys
from pathlib import Path

# ── Ensure project root is in path ───────────────────────────
# Allows running from the project root or ui directory
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from app.core.config import settings
from app.core.logger import setup_logger
from app.services.rag_pipeline import rag_service
from app.services.vector_store import delete_vector_store


# ── Page configuration ────────────────────────────────────────
st.set_page_config(
    page_title="💧 Water Quality Assistant",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CSS Styling ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container (inherits theme background) */
    .main { }

    /* Chat message boxes */
    .user-message {
        background: #1a73e8;
        color: black;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .assistant-message {
        background: white;
        color: black;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
        float: left;
        clear: both;
        border: 1px solid #e0e9f5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
    .clearfix { clear: both; }

    /* WQI badge colours */
    .badge-excellent { background:#00b300; color:white; padding:3px 8px; border-radius:12px; font-size:0.8em; }
    .badge-good      { background:#66cc00; color:white; padding:3px 8px; border-radius:12px; font-size:0.8em; }
    .badge-medium    { background:#ffaa00; color:white; padding:3px 8px; border-radius:12px; font-size:0.8em; }
    .badge-bad       { background:#ff5500; color:white; padding:3px 8px; border-radius:12px; font-size:0.8em; }
    .badge-very-bad  { background:#cc0000; color:white; padding:3px 8px; border-radius:12px; font-size:0.8em; }

    /* Source card (semi-transparent tint that adapts to both light/dark backgrounds) */
    .source-card {
        background: rgba(26, 115, 232, 0.08);
        border-left: 3px solid #1a73e8;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 0.85em;
    }

    /* Header */
    .header-title {
        font-size: 2em;
        font-weight: 700;
        color: #0d47a1;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper: WQI colour badge ──────────────────────────────────
def _wqi_badge(category: str) -> str:
    cls_map = {
        "excellent": "badge-excellent",
        "good": "badge-good",
        "medium": "badge-medium",
        "bad": "badge-bad",
        "very bad": "badge-very-bad",
    }
    cls = cls_map.get(category.lower(), "badge-medium")
    return f'<span class="{cls}">{category}</span>'


# ── Initialise RAG service (once per session) ─────────────────
@st.cache_resource(show_spinner="🔄 Initialising Water Quality RAG system…")
def initialise_rag():
    setup_logger()
    rag_service.initialize()
    return rag_service


# ── Session state init ────────────────────────────────────────
def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0


# ── Sidebar ───────────────────────────────────────────────────
def _render_sidebar():
    with st.sidebar:
        # st.image(
        #     "https://www.who.int/images/default-source/infographics/water-and-sanitation/water-drop.png",
        #     width=60,
        # )
        st.markdown("## 💧 Water Quality Assistant")
        st.markdown("*Powered by Groq + LangChain + FAISS*")
        st.divider()

        # # ── Dataset stats ────────────────────────────────────
        # st.markdown("### 📊 Dataset Overview")
        # try:
        #     df = pd.read_csv(settings.dataset_path)
        #     st.metric("Water Bodies", len(df))
        #     st.metric("Locations", df["Location"].nunique())
        #     avg_wqi = df["Water Quality Index (WQI)"].mean()
        #     st.metric("Avg WQI", f"{avg_wqi:.1f}")

        #     # Category breakdown
        #     st.markdown("**WQI Categories**")
        #     cat_counts = df["Water Quality Category"].value_counts()
        #     for cat, count in cat_counts.items():
        #         pct = 100 * count / len(df)
        #         st.write(f"{_wqi_badge(cat)} {count} ({pct:.0f}%)", unsafe_allow_html=True)
        # except Exception:
        #     st.warning("Dataset not found. Add CSV to app/data/")

        # st.divider()

        # ── WQI Reference ────────────────────────────────────
        st.markdown("### 🗺️ WQI Reference Scale")
        st.markdown("""
        | Score | Category |
        |-------|----------|
        | 90–100 | Excellent |
        | 70–89  | Good |
        | 50–69  | Medium |
        | 25–49  | Bad |
        | 0–24   | Worst |
        """)

        st.divider()

        # ── Model info ───────────────────────────────────────
        st.markdown("### ⚙️ Configuration")
        st.code(f"Model: {settings.groq_model_name}", language=None)
        st.code(f"Embeddings: all-MiniLM-L6-v2", language=None)
        st.code(f"Top-K: {settings.top_k_results}", language=None)

        st.divider()

        # ── Admin: Rebuild index ─────────────────────────────
        st.markdown("### 🔧 Admin")
        if st.button("🔄 Rebuild FAISS Index", use_container_width=True):
            with st.spinner("Rebuilding index…"):
                delete_vector_store()
                st.cache_resource.clear()
                st.success("Index cleared! Refresh the page to rebuild.")

        # ── Clear chat ───────────────────────────────────────
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.rerun()

        # ── Session stats ────────────────────────────────────
        st.caption(f"Queries this session: {st.session_state.total_queries}")


# ── Suggested questions ───────────────────────────────────────
SUGGESTED_QUESTIONS = [
    "What is the WQI of Sabarmati River?",
    "Which water bodies have Excellent water quality?",
    "What are WHO guidelines for safe drinking water pH?",
    "Compare Tapi River and Narmada River water quality",
    "Which water body has the highest pollution?",
    "What is a safe level of nitrate in drinking water?",
]


# ── Main Chat UI ──────────────────────────────────────────────
def _render_chat(service):
    st.markdown(
        '<div class="header-title">💧 Water Quality Assistant</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Ask me anything about water quality, WQI scores, WHO guidelines, "
        "or specific water bodies in the dataset.",
    )

    # ── Suggested questions (only when chat is empty) ─────────
    if not st.session_state.messages:
        st.markdown("#### 💡 Try asking:")
        cols = st.columns(3)
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            with cols[i % 3]:
                if st.button(question, key=f"suggestion_{i}", use_container_width=True):
                    st.session_state["prefill_query"] = question
                    st.rerun()

    st.divider()

    # ── Render message history ────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "💧"):
            st.markdown(msg["content"])

            # Show sources if available (assistant messages only)
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📚 Retrieved Sources ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        if src.get("source") == "dataset":
                            wqi_val = src.get("wqi", "N/A")
                            cat = src.get("wqi_category", "")
                            st.markdown(
                                f'<div class="source-card">'
                                f'📍 <b>{src.get("water_body", "Unknown")}</b> — '
                                f'{src.get("location", "")} | '
                                f'WQI: <b>{wqi_val}</b> {_wqi_badge(cat)}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div class="source-card">'
                                f'📖 <b>WHO Guidelines</b> — {src.get("topic", "")}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

    # ── Chat input ────────────────────────────────────────────
    # Check for a prefilled query from suggestion buttons
    default_value = st.session_state.pop("prefill_query", "")
    user_input = st.chat_input(
        "Ask about water quality, WQI, water bodies, or WHO standards…",
    )

    query = user_input or default_value
    if not query:
        return

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.total_queries += 1

    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant", avatar="💧"):
        with st.spinner("🔍 Searching water quality database…"):
            result = service.generate_response(query)

        answer = result["answer"]
        sources = result.get("sources", [])
        is_on_topic = result.get("is_on_topic", True)

        st.markdown(answer)

        if sources and is_on_topic:
            with st.expander(f"📚 Retrieved Sources ({len(sources)})"):
                for src in sources:
                    if src.get("source") == "dataset":
                        wqi_val = src.get("wqi", "N/A")
                        cat = src.get("wqi_category", "")
                        st.markdown(
                            f'<div class="source-card">'
                            f'📍 <b>{src.get("water_body", "Unknown")}</b> — '
                            f'{src.get("location", "")} | '
                            f'WQI: <b>{wqi_val}</b> {_wqi_badge(cat)}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="source-card">'
                            f'📖 <b>WHO Guidelines</b> — {src.get("topic", "")}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Persist assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    st.rerun()


# ── Entry point ───────────────────────────────────────────────
def main():
    _init_state()
    _render_sidebar()

    service = initialise_rag()
    if not service.is_ready:
        st.error("❌ RAG service failed to initialise. Check your GROQ_API_KEY and dataset path.")
        st.stop()

    _render_chat(service)


if __name__ == "__main__":
    main()
