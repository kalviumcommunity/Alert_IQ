"""Streaming Streamlit chat UI for Alert_IQ."""
import json
import os
from typing import Any, Dict, Iterator, List, Tuple

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:5000")

st.set_page_config(page_title="Alert_IQ Streaming RAG", page_icon="🚨")
st.title("🚨 Alert_IQ")
st.caption("Streaming grounded answers with inspectable citations")

if "history" not in st.session_state:
    st.session_state.history = []


def stream_request(question: str) -> Iterator[Tuple[str, Any]]:
    response = requests.post(
        f"{API_URL.rstrip('/')}/query/stream",
        json={"question": question},
        stream=True,
        timeout=60,
    )
    response.raise_for_status()
    if not response.headers.get("Content-Type", "").startswith("text/event-stream"):
        raise RuntimeError("The backend did not return a streaming response.")

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        yield "event", json.loads(raw_line[6:])


for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("sources"):
            with st.expander("Sources", expanded=False):
                for source in item["sources"]:
                    st.markdown(f"**{source['label']} {source['document']}** · `{source['chunk_id']}`")
                    st.caption(source["text"])

question = st.chat_input("Ask a question about the knowledge base...")

if question:
    question = question.strip()
    if not question:
        st.warning("Please enter a question.")
    else:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        answer_placeholder = None
        answer = ""
        sources: List[Dict[str, Any]] = []
        stream_error = None

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            status_box = st.empty()
            try:
                status_box.info("🔎 Retrieving sources and starting stream…")
                for _, event in stream_request(question):
                    event_type = event.get("type")
                    if event_type == "citations":
                        sources = event.get("sources", [])
                        status_box.info("✍️ Answer is streaming…")
                    elif event_type == "token":
                        answer += event.get("text", "")
                        answer_placeholder.markdown(answer)
                    elif event_type == "done":
                        status_box.success("✓ Complete")
                    elif event_type == "error":
                        stream_error = event.get("message", "The stream failed.")
                        status_box.error(stream_error)
                        break
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                stream_error = f"The answer stopped streaming: {exc}"
                status_box.error(stream_error)

            if answer:
                answer_placeholder.markdown(answer)
            if sources:
                with st.expander("Sources", expanded=True):
                    for source in sources:
                        st.markdown(
                            f"**{source['label']} {source['document']}** · "
                            f"`{source['chunk_id']}` · score `{source.get('score', 0):.3f}`"
                        )
                        st.caption(source["text"])

            if stream_error:
                st.warning("Partial output is preserved. Submit the same question again to retry.")

        st.session_state.history.append({
            "role": "assistant",
            "content": answer or "The RAG stream did not return an answer.",
            "sources": sources,
            "incomplete": bool(stream_error),
        })
