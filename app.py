import streamlit as st
from src.rag_pipeline import build_rag_chain, get_or_create_vectorstore

st.set_page_config(
    page_title="Energy Assistant",
    page_icon="⚡",
    layout="wide",
)
@st.cache_resource(show_spinner=False)
def load_chain():
    vectorstore, n_docs = get_or_create_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)
    return chain, retriever, n_docs

with st.spinner("Loading knowledge base..."):
    chain, retriever, n_docs = load_chain()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Energy Assistant")
    st.markdown(
        f"**{n_docs} chunks** indexed from Alliander & Enexis reports"
    )
    st.divider()

    st.markdown("### Try asking:")
    examples = [
        "What are Alliander's CO₂ targets?",
        "How much did Enexis invest in 2023?",
        "What is grid congestion?",
        "Compare Alliander and Enexis revenues.",
    ]
    for q in examples:
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q

    st.divider()
    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ── Header ────────────────────────────────────────────────────
st.markdown("## ⚡ LLM Energy Assistant")
st.markdown(
    "Ask anything about Alliander & Enexis annual reports. "
    "Answers are grounded in retrieved document passages."
)
st.divider()


# ── Session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ── Chat history ──────────────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + " · ".join(msg["sources"]))


# ── Input ─────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", None)
user_input = st.chat_input("Ask about energy reports...") or prefill

if user_input:
    # Show user message
    st.session_state["messages"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            # Get answer from chain
            answer = chain.invoke(user_input)

            # Get source documents separately
            source_docs = retriever.invoke(user_input)
            sources = []
            seen = set()
            for doc in source_docs:
                company = doc.metadata.get("company", "")
                page = doc.metadata.get("page", "")
                label = f"{company} p.{page}"
                if label not in seen:
                    sources.append(label)
                    seen.add(label)

        st.markdown(answer)
        if sources:
            st.caption("Sources: " + " · ".join(sources))

    # Save to history
    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })