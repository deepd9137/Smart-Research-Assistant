from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

load_dotenv(Path(__file__).resolve().parent / ".env")
from app.agent import route_and_answer
from app.chunking import chunk_documents
from app.config import get_settings
from app.conversation import transcript_to_messages
from app.embeddings import make_embeddings
from app.pdf_loader import load_many_pdfs
from app.vector_store import build_vector_store, save_vector_store

MAX_PDFS = 5


def _hydrate_streamlit_secrets() -> None:
    import os

    try:
        for key in ("GOOGLE_API_KEY", "TAVILY_API_KEY"):
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except StreamlitSecretNotFoundError:
        pass


def _init_session() -> None:
    defaults: Dict[str, Any] = {
        "messages": [],
        "vectorstore": None,
        "indexed_files": [],
        "embeddings": None,
        "build_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _settings() -> Any:
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
    return get_settings()


def _embeddings():
    if st.session_state.embeddings is None:
        st.session_state.embeddings = make_embeddings(_settings())
    return st.session_state.embeddings


def main() -> None:
    st.set_page_config(
        page_title="Smart Research Assistant",
        page_icon="📚",
        layout="wide",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": None,
        },
    )
    _init_session()
    _hydrate_streamlit_secrets()

    st.title("Smart Research Assistant")
    st.caption("RAG over your PDFs · Gemini · Pinecone · Tavily fallback")

    with st.sidebar:
        st.subheader("1. Upload PDFs")
        st.caption(f"Up to {MAX_PDFS} files (~10 pages each recommended).")
        files = st.file_uploader(
            "PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if files and len(files) > MAX_PDFS:
            st.error(f"Please upload at most {MAX_PDFS} PDFs.")
            files = files[:MAX_PDFS]

        if st.button("Build knowledge base", type="primary"):
            st.session_state.build_error = None
            if not files:
                st.session_state.build_error = "Upload at least one PDF first."
            else:
                with st.spinner("Extracting text, chunking, embedding…"):
                    try:
                        tmp_paths: List[Path] = []
                        for f in files:
                            p = Path(tempfile.gettempdir()) / f"sr_{f.name}"
                            p.write_bytes(f.getvalue())
                            tmp_paths.append(p)
                        pages = load_many_pdfs(tmp_paths, max_files=MAX_PDFS)
                        chunks = chunk_documents(pages, _settings())
                        emb = _embeddings()
                        store = build_vector_store(chunks, emb, _settings())
                        st.session_state.vectorstore = store
                        st.session_state.indexed_files = [f.name for f in files]
                        save_vector_store(store, _settings())
                        st.success(f"Indexed {len(chunks)} chunks from {len(files)} file(s).")
                    except Exception as e:
                        st.session_state.build_error = str(e)
                        st.error(str(e))

        if st.session_state.indexed_files:
            st.info("**Indexed:**\n" + "\n".join(f"- {n}" for n in st.session_state.indexed_files))

        if st.session_state.build_error:
            st.error(st.session_state.build_error)

        st.divider()
        st.subheader("API keys")
        st.caption("Set `GOOGLE_API_KEY` and `TAVILY_API_KEY` in `.env` (see `.env.example`).")

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    def _render_assistant_meta(meta: Dict[str, Any]) -> None:
        if meta.get("confidence") is not None:
            st.progress(float(meta["confidence"]))
            st.caption(f"Retrieval confidence (heuristic): {meta['confidence']:.0%}")
        if meta.get("citations"):
            st.markdown("**References**")
            for line in meta["citations"]:
                st.markdown(f"- {line}")
        if meta.get("excerpts"):
            with st.expander("Source excerpts (retrieved chunks)"):
                for label, ex in meta["excerpts"]:
                    st.markdown(f"**{label}** {ex}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                _render_assistant_meta(msg["meta"])

    user_q = st.chat_input("Ask a question about your documents…")
    if not user_q:
        return
    user_q = user_q.strip()
    if not user_q:
        with st.chat_message("assistant"):
            st.markdown("Please type a question (at least a few words) and I will help.")
        return

    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    settings = _settings()
    hist = transcript_to_messages(
        [m for m in st.session_state.messages[:-1] if m["role"] in ("user", "assistant")]
    )

    meta: Dict[str, Any] = {
        "citations": [],
        "confidence": None,
        "excerpts": [],
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = route_and_answer(
                    user_q,
                    st.session_state.vectorstore,
                    settings,
                    chat_history=hist,
                )
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                    st.error(
                        "Gemini quota reached right now. Please wait 20-60 seconds and try again, "
                        "or switch to a lower-cost model (e.g., `CHAT_MODEL=gemini-2.5-flash-lite`)."
                    )
                else:
                    st.error(err)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Sorry — something went wrong: {e}",
                        "meta": {},
                    }
                )
                return

        st.markdown(result.answer)

        if result.rag_result is not None and result.show_document_provenance:
            rr = result.rag_result
            meta["confidence"] = rr.confidence
            meta["citations"] = [
                f"[{c.index}] {c.document_name}, p. {c.page}" for c in rr.citations
            ]
            meta["excerpts"] = rr.source_excerpts

        _render_assistant_meta(meta)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "meta": meta,
            }
        )


if __name__ == "__main__":
    main()
