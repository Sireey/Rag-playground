"""app.py — Streamlit UI for the modular RAG pipeline.

Run with:
    streamlit run app.py
"""

import glob
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Playground",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 RAG Pipeline Playground")
st.caption("Configure, ingest, query, and compare — all from one place.")

# ── Sidebar: Pipeline Configuration ───────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    st.info("Change any setting and click **Build Pipeline** to apply.", icon="💡")

    # ── Parser ────────────────────────────────────────────────────────────
    st.subheader("Parser")
    parser_clean = st.checkbox("Clean text", value=True,
                               help="Remove hyphenated line breaks and page numbers")

    # ── Retriever ─────────────────────────────────────────────────────────
    st.subheader("Retriever")
    retriever_type = st.selectbox(
        "Type",
        options=["chroma", "rerank", "parent_document"],
        format_func={
            "chroma": "Chroma — vector similarity only",
            "rerank": "ReRank — cross-encoder + optional multi-query",
            "parent_document": "Parent-Document — child retrieval, parent context",
        }.get,
        help=(
            "**chroma**: fast baseline, ranks by vocabulary overlap.\n\n"
            "**rerank**: cross-encoder reads query+chunk together for semantic ranking. "
            "Optional multi-query expands recall.\n\n"
            "**parent_document**: indexes small chunks for precision, "
            "returns large parent chunks to the LLM for rich context."
        ),
    )

    collection_name = st.text_input(
        "Collection name",
        value={
            "chroma": "rag_papers",
            "rerank": "rag_papers",
            "parent_document": "rag_papers_pdr",
        }[retriever_type],
        help="ChromaDB collection. Use different names to keep separate indexes.",
    )

    # Settings that vary by retriever type
    if retriever_type in ("chroma", "rerank"):
        st.markdown("**Chunker**")
        chunk_size = st.slider("Chunk size (tokens)", 64, 1024, 512, step=64)
        chunk_overlap = st.slider("Chunk overlap (tokens)", 0, 256, 50, step=10)

        if retriever_type == "chroma":
            k_final = st.slider("k — chunks returned to LLM", 1, 10, 3)

        else:  # rerank
            k_retrieve = st.slider("k_retrieve — candidate pool", 5, 30, 10)
            k_final = st.slider("k_final — chunks returned to LLM", 1, 10, 3)
            multi_query = st.checkbox("Multi-query expansion", value=True,
                                      help="LLM generates 3 query variants to widen recall")
            if multi_query:
                multi_query_variants = st.slider("Variants", 1, 5, 3)
            else:
                multi_query_variants = 3

    else:  # parent_document
        st.markdown("**Child chunker** *(indexed for retrieval)*")
        child_chunk_size = st.slider("Child chunk size (tokens)", 32, 512, 128, step=32)
        child_chunk_overlap = st.slider("Child overlap (tokens)", 0, 64, 20, step=10)

        st.markdown("**Parent chunker** *(returned to LLM)*")
        parent_chunk_size = st.slider("Parent chunk size (tokens)", 128, 1024, 512, step=64)
        parent_chunk_overlap = st.slider("Parent overlap (tokens)", 0, 64, 0, step=10)

        k_final = st.slider("k — parents returned to LLM", 1, 10, 3)

    # ── LLM ───────────────────────────────────────────────────────────────
    st.subheader("LLM")
    llm_model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.1,
                             help="0 = deterministic; higher = more creative")

    # ── Build pipeline button ──────────────────────────────────────────────
    st.divider()
    build_clicked = st.button("🔨 Build Pipeline", use_container_width=True, type="primary")

    # ── Ingest ────────────────────────────────────────────────────────────
    st.subheader("Ingest")
    available_pdfs = sorted(glob.glob("data/raw/*.pdf"))
    pdf_names = [os.path.basename(p) for p in available_pdfs]

    if pdf_names:
        selected_pdfs = st.multiselect("PDFs to index", pdf_names, default=pdf_names)
        selected_paths = [p for p in available_pdfs if os.path.basename(p) in selected_pdfs]
        col1, col2 = st.columns(2)
        ingest_clicked = col1.button("📥 Ingest", use_container_width=True)
        reset_clicked = col2.button("🗑️ Reset + Ingest", use_container_width=True,
                                    help="Wipe the index and rebuild from scratch")
    else:
        st.warning("No PDFs found in data/raw/")
        ingest_clicked = reset_clicked = False
        selected_paths = []


# ── Build config dict from sidebar selections ──────────────────────────────
def build_config() -> dict:
    cfg: dict = {
        "ssl_patch": True,
        "llm": {"model": llm_model, "temperature": temperature},
        "embedder": {"model": "text-embedding-3-small"},
        "parser": {"type": "pymupdf", "clean": parser_clean},
        "retriever": {
            "type": retriever_type,
            "chroma_dir": "./chroma_db",
            "collection_name": collection_name,
        },
    }

    ret = cfg["retriever"]

    if retriever_type == "chroma":
        ret["k_final"] = k_final
        ret["chunker"] = {"type": "recursive_token",
                          "chunk_size": chunk_size,
                          "chunk_overlap": chunk_overlap}

    elif retriever_type == "rerank":
        ret["k_retrieve"] = k_retrieve
        ret["k_final"] = k_final
        ret["multi_query"] = multi_query
        ret["multi_query_variants"] = multi_query_variants
        ret["chunker"] = {"type": "recursive_token",
                          "chunk_size": chunk_size,
                          "chunk_overlap": chunk_overlap}

    else:  # parent_document
        ret["k"] = k_final
        ret["child_chunker"] = {"type": "recursive_token",
                                 "chunk_size": child_chunk_size,
                                 "chunk_overlap": child_chunk_overlap}
        ret["parent_chunker"] = {"type": "recursive_token",
                                  "chunk_size": parent_chunk_size,
                                  "chunk_overlap": parent_chunk_overlap}

    return cfg


# ── Session state ──────────────────────────────────────────────────────────
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── Handle Build ───────────────────────────────────────────────────────────
if build_clicked:
    from rag import RAGFactory
    with st.spinner("Building pipeline…"):
        try:
            st.session_state.pipeline = RAGFactory.from_dict(build_config())
            st.session_state.last_result = None
            st.success(f"Pipeline ready — **{retriever_type}** retriever on `{collection_name}`")
        except Exception as e:
            st.error(f"Failed to build pipeline: {e}")

# ── Handle Ingest ──────────────────────────────────────────────────────────
if (ingest_clicked or reset_clicked) and selected_paths:
    if st.session_state.pipeline is None:
        st.warning("Build the pipeline first.")
    else:
        reset = reset_clicked
        action = "Resetting index and ingesting" if reset else "Ingesting"
        with st.spinner(f"{action} {len(selected_paths)} file(s)…"):
            try:
                n = st.session_state.pipeline.ingest(selected_paths, reset=reset)
                st.success(f"Indexed **{n} chunks** from {len(selected_paths)} file(s).")
            except Exception as e:
                st.error(f"Ingest failed: {e}")

# ── Main area: Query ───────────────────────────────────────────────────────
st.divider()

pipeline_ready = st.session_state.pipeline is not None
if not pipeline_ready:
    st.info("👈 Configure the pipeline in the sidebar, then click **Build Pipeline**.")

question = st.text_input(
    "Ask a question",
    placeholder="e.g. What is the difference between RAG-Sequence and RAG-Token?",
    disabled=not pipeline_ready,
)

run_clicked = st.button("▶ Run", type="primary", disabled=not pipeline_ready or not question)

if run_clicked and question:
    with st.spinner("Retrieving and generating…"):
        try:
            result = st.session_state.pipeline.query(question)
            st.session_state.last_result = result
        except Exception as e:
            st.error(f"Query failed: {e}")

# ── Display results ────────────────────────────────────────────────────────
result = st.session_state.last_result
if result:
    st.subheader("Answer")
    st.markdown(result["result"])

    st.subheader(f"Retrieved context — {len(result['source_documents'])} document(s)")

    for i, doc in enumerate(result["source_documents"], 1):
        src = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        parent_id = doc.metadata.get("parent_id")
        label = f"**[{i}]** `{src}` — page {page}"
        if parent_id:
            label += f" *(parent: {parent_id})*"

        with st.expander(label, expanded=(i == 1)):
            st.markdown(
                f"```\n{doc.content[:1200]}{'…' if len(doc.content) > 1200 else ''}\n```"
            )
            if doc.metadata:
                st.caption("Metadata: " + str({k: v for k, v in doc.metadata.items()
                                               if k not in ("parent_id",)}))
