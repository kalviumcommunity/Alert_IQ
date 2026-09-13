# Streaming Responses & Citation Display

## Endpoint

`POST /query/stream` accepts `{ "question": "..." }` and returns Server-Sent Events (`text/event-stream`). The existing non-streaming `/query` endpoint remains unchanged.

## Event contract

- `citations` — emitted after retrieval; contains `[1]`-style labels, document names, chunk IDs, scores, and retrieved source text.
- `token` — emitted progressively as answer text becomes available.
- `done` — marks successful completion.
- `error` — marks an interrupted or failed stream.

## Run the UI

Start the existing Flask backend, then install the UI dependencies:

```bash
pip install -r ui/requirements.txt
streamlit run ui/streamlit_stream.py
```

Set `RAG_API_URL` when the backend is not at `http://localhost:5000`.

## Trust and failure behavior

Citations are displayed before/alongside the answer and can be expanded to inspect the exact retrieved chunk text. If streaming fails after partial output, the partial answer and citations remain visible and the UI tells the user to retry. A failed stream never claims completion.

See `logs/streaming_citation_sample.json` for the committed sample interaction.
