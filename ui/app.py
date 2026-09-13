"""Streamlit UI for the Alert_IQ RAG API."""
import os
from typing import Any, Dict

import requests
import streamlit as st


API_URL = os.getenv("RAG_API_URL", "http://localhost:5000")

st.set_page_config(page_title="Alert_IQ RAG", page_icon="🚨", layout="centered")
st.title("🚨 Alert_IQ")
st.caption("Ask questions against the grounded knowledge base")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for index, source in enumerate(message["sources"], 1):
                    name = source.get("source_document", source.get("source", "Unknown source"))
                    chunk_id = source.get("chunk_id", source.get("id", ""))
                    score = source.get("score")
                    label = f"{name}"
                    if chunk_id:
                        label += f" — {chunk_id}"
                    if score is not None:
                        label += f" (score: {score:.3f})"
                    st.write(f"{index}. {label}")

question = st.chat_input("Ask a question about the knowledge base...")

if question:
    question = question.strip()
    if not question:
        st.warning("Please enter a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the knowledge base..."):
                try:
                    response = requests.post(
                        f"{API_URL.rstrip('/')}/query",
                        json={"question": question},
                        timeout=30,
                    )
                    response.raise_for_status()
                    result: Dict[str, Any] = response.json()
                    answer = result.get("answer", "No answer was returned.")
                    sources = result.get("sources", [])
                    status = result.get("status", "unknown")

                    if status != "success":
                        raise RuntimeError(result.get("error", "RAG API returned an error."))

                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources", expanded=True):
                            for index, source in enumerate(sources, 1):
                                name = source.get("source_document", source.get("source", "Unknown source"))
                                chunk_id = source.get("chunk_id", source.get("id", ""))
                                score = source.get("score")
                                metadata = source.get("metadata", {})
                                label = f"**{index}. {name}**"
                                if chunk_id:
                                    label += f" · `{chunk_id}`"
                                if score is not None:
                                    label += f" · score `{score:.3f}`"
                                st.markdown(label)
                                if metadata:
                                    st.caption(str(metadata))
                    else:
                        st.info("No retrieved sources were returned. The backend may have refused the question.")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except requests.RequestException as exc:
                    st.error(f"Could not reach the RAG API: {exc}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "The RAG service is unavailable right now. Please try again.",
                        "sources": [],
                    })
                except (ValueError, RuntimeError) as exc:
                    st.error(f"RAG request failed: {exc}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I couldn't get a reliable answer from the RAG service.",
                        "sources": [],
                    })
