# Alert_IQ Chat UI

A small Streamlit frontend for the Alert_IQ RAG backend.

## Run

From the repository root:

```bash
pip install -r ui/requirements.txt
set RAG_API_URL=http://localhost:5000
streamlit run ui/app.py
```

On macOS/Linux, use `export RAG_API_URL=http://localhost:5000` instead.

## Flow

1. The user enters a question in `st.chat_input`.
2. The UI sends `POST /query` with `{ "question": "..." }`.
3. While waiting, Streamlit shows a spinner.
4. A successful response renders the grounded answer and expands the returned sources.
5. API/network failures produce a clear error message and a retry-friendly assistant message.

## Source display

Each source can show its document name, chunk ID, retrieval score, and metadata when the backend returns those fields. This keeps the evidence visible instead of hiding the provenance behind the answer.

## Configuration

`RAG_API_URL` controls the backend URL and is intentionally kept outside the UI code. The default is `http://localhost:5000` for local development.

## Sample interaction

Question:

> Do uploaded documents become searchable without restarting the API?

Answer:

> Yes. Successfully indexed uploads are written to the same vector collection used by the query endpoint, so they can be retrieved without restarting the API.

Sources:

- `runtime_policy.md` — runtime upload policy
- chunk metadata/retrieval score when returned by the backend

Loading state:

> Searching the knowledge base...

Error state:

> Could not reach the RAG API. Please try again.

A committed sample interaction is available in `logs/chat_ui_sample.json`.
