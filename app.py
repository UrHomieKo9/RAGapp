"""
Streamlit frontend for interacting with the RAG backend.
"""

from __future__ import annotations

import io
import os
from typing import Dict, List

import requests
import streamlit as st


def get_backend_url() -> str:
    """Resolve backend base URL from env or sidebar input."""
    return st.session_state.get("backend_url") or os.getenv(
        "RAG_BACKEND_URL", "http://localhost:8000"
    )


def post_request(endpoint: str, **kwargs):
    """Helper to post to backend with error handling."""
    url = f"{get_backend_url().rstrip('/')}/{endpoint.lstrip('/')}"
    response = requests.post(url, timeout=120, **kwargs)
    response.raise_for_status()
    return response.json()


def render_sidebar():
    """Sidebar controls."""
    st.sidebar.header("Configuration")
    backend_url = st.sidebar.text_input(
        "Backend URL",
        value=get_backend_url(),
        help="FastAPI server base URL",
    )
    st.session_state["backend_url"] = backend_url
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **How to use**
        1. Upload one or more PDFs.
        2. Wait for the indexing confirmation.
        3. Ask natural-language questions.
        """
    )


def render_chat_history(history: List[Dict[str, str]]):
    """Show chat transcript."""
    st.subheader("Conversation History")
    if not history:
        st.info("No questions asked yet.")
        return
    for turn in history:
        role = "🧑‍💻 User" if turn["role"] == "human" else "🤖 Assistant"
        st.markdown(f"**{role}:** {turn['content']}")


def main():
    st.set_page_config(
        page_title="LangChain RAG UI",
        layout="wide",
        page_icon="🔎",
    )
    st.title("🔎 Retrieval-Augmented Generation Playground")
    st.caption("Upload PDFs, build embeddings, and chat with your documents.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    render_sidebar()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("1. Upload PDFs")
        uploaded_file = st.file_uploader(
            "Select a PDF file",
            type=["pdf"],
            accept_multiple_files=False,
            help="Only PDF files are supported",
        )

        if uploaded_file and st.button("Upload & Index", use_container_width=True):
            try:
                st.session_state.chat_history = []
                file_bytes = uploaded_file.getvalue()
                payload = {
                    "file": (
                        uploaded_file.name,
                        io.BytesIO(file_bytes),
                        "application/pdf",
                    )
                }
                with st.spinner("Indexing document with backend..."):
                    response = post_request("upload", files=payload)
                st.success(
                    f"Indexed successfully! Chunks stored: {response.get('chunks_indexed')}"
                )
            except requests.HTTPError as err:
                st.error(f"Upload failed: {err.response.text}")
            except Exception as err:  # pragma: no cover - UI feedback
                st.error(f"Unexpected error: {err}")

    with col_right:
        st.subheader("2. Ask a Question")
        question = st.text_input(
            "What would you like to know?",
            placeholder="e.g. Summarize key points from the document",
        )
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)

        if ask_clicked:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    with st.spinner("Thinking..."):
                        response = post_request("query", json={"question": question})
                    st.session_state.chat_history = response.get("chat_history", [])
                    st.markdown("### Answer")
                    st.success(response.get("answer"))

                    sources = response.get("sources", [])
                    if sources:
                        st.markdown("### Sources")
                        for idx, source in enumerate(sources, start=1):
                            meta = source.get("metadata", {})
                            page = meta.get("page", "N/A")
                            st.write(f"**{idx}. Page {page}**")
                            st.caption(source.get("page_content"))
                except requests.HTTPError as err:
                    st.error(f"Query failed: {err.response.text}")
                except Exception as err:
                    st.error(f"Unexpected error: {err}")

    render_chat_history(st.session_state.chat_history)


if __name__ == "__main__":
    main()


