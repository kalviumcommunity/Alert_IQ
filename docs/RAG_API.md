# Alert_IQ RAG API

## Purpose

The API exposes the existing RAG pipeline to a frontend or another service over HTTP. A client sends a question to `POST /query`; the server runs retrieval and grounded answer generation and returns the answer together with structured source metadata.

## Configuration

Runtime configuration is read from environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | LLM API credential | empty |
| `OPENAI_MODEL` | LLM model name | `gemini-2.5-flash` |
| `OPENAI_BASE_URL` | OpenAI-compatible API base URL | existing client default |
| `VECTOR_DB_PERSIST_DIR` | Vector database persistence directory | `data/vector_store` |
| `VECTOR_COLLECTION_NAME` | Vector collection | `alert_iq_knowledge_base` |
| `RAG_API_HOST` | HTTP bind host | `127.0.0.1` |
| `RAG_API_PORT` | HTTP port | `5000` |

No API key is committed to the repository.

## Start

```bash
pip install -r requirements.txt
python -m src.api
```

## Query endpoint

`POST /query`

Request:

```json
{
  "question": "How do I troubleshoot database replica latency?"
}
```

Successful response:

```json
{
  "answer": "Based on the verified Alert_IQ knowledge base: ...",
  "sources": [
    {
      "id": "chunk-db-rb-402-01",
      "rank": 1,
      "score": 0.91,
      "source_document": "runbook_database_replica_lag.md",
      "chunk_index": 1
    }
  ],
  "status": "success",
  "metadata": {
    "query": "How do I troubleshoot database replica latency?",
    "model_name": "gemini-2.5-flash",
    "stage_metrics": {
      "total_latency_ms": 42.1,
      "retrieved_chunk_count": 1,
      "embedding_dimension": 768
    }
  }
}
```

## Error responses

- `400` — malformed JSON object, missing `question`, or blank/non-string question.
- `415` — request does not declare a JSON content type.
- `500` — unexpected RAG/pipeline failure.

Example:

```json
{
  "status": "error",
  "error": "Missing required field: question."
}
```

## Frontend consumption

A frontend can call `/query` with `fetch()`, parse the JSON body, render `answer`, and display `sources` as citations or expandable source details. The `status` field provides a simple success/error indicator while HTTP status codes allow normal error handling.
