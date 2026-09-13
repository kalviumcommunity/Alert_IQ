# Alert_IQ 🚨

Alert_IQ is an intelligent monitoring, alerting, and notification management system with a grounded Retrieval-Augmented Generation (RAG) application. The RAG delivery includes document ingestion, chunking, embeddings, ChromaDB storage, retrieval, grounded generation, citations, conversational context, streaming, caching, and usage monitoring.

## RAG Features

- Upload `.txt`, `.md`, or `.pdf` documents through the backend.
- Ingest, clean, chunk, embed, and index uploaded content.
- Ask questions through the HTTP API or Streamlit chat UI.
- Retrieve relevant chunks and generate grounded answers.
- Display inspectable source citations.
- Preserve conversational follow-up context.
- Stream citations and answer tokens with Server-Sent Events.
- Cache repeated `/query` requests with a TTL cache.
- Log requests, sources, errors, latency, token estimates, and approximate cost.
- Aggregate usage into a summary report.

## Architecture

```text
Document upload
      |
      v
DocumentLoader -> TextCleaner -> Chunker -> Embeddings -> ChromaDB
                                                        |
                                                        v
Question -> Retrieval -> Reranking/grounding -> Answer + Citations
             |                                      |
             +---- cache ---------------------------+
             |
             +---- structured usage/logging

Streamlit UI <---- HTTP/SSE ---- Flask API
```

## Prerequisites

- Python 3.10+
- Git
- An OpenAI-compatible LLM/embedding provider configured through environment variables
- ChromaDB persistence is local by default; no separate vector database server is required for the default setup.
- Node.js is only needed for unrelated parts of the original Alert_IQ project; the RAG service itself is Python-based.

## Setup

Clone the repository and enter it:

```bash
git clone https://github.com/kalviumcommunity/Alert_IQ.git
cd Alert_IQ
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
pip install -r ui/requirements.txt
```

Create the local environment file from the committed template:

```bash
cp .env.example .env
```

On Windows PowerShell, use:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and provide a real API key and provider configuration.

## Configuration

The committed `.env.example` documents the required provider and vector-store settings. Common values are:

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_BASE_URL` | OpenAI-compatible provider endpoint | Google Gemini-compatible endpoint in `.env.example` |
| `OPENAI_API_KEY` | Provider credential | empty/local only |
| `OPENAI_MODEL` | Chat model | `gemini-2.5-flash` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-004` |
| `EMBEDDING_DIMENSION` | Embedding vector size | `768` |
| `VECTOR_DB_PERSIST_DIR` | ChromaDB persistence path | `data/vector_store` |
| `VECTOR_COLLECTION_NAME` | Chroma collection | `alert_iq_knowledge_base` |
| `VECTOR_DISTANCE_METRIC` | Similarity metric | `cosine` |
| `RAG_API_HOST` | Backend bind address | `127.0.0.1` |
| `RAG_API_PORT` | Backend port | `5000` |
| `RAG_API_URL` | Streamlit backend URL | `http://localhost:5000` |
| `RAG_CACHE_TTL_SECONDS` | `/query` cache lifetime | `900` |
| `UPLOAD_DIR` | Runtime upload directory | `uploads` |
| `UPLOAD_MAX_BYTES` | Upload size limit | `5242880` |

See `.env.example` for the provider-compatible starting configuration.

## Secrets Safety

**Never commit `.env` or real API keys.** The repository `.gitignore` excludes `.env` and `.env.local`. Use `.env.example` as a placeholder-only configuration template. For deployment, put secrets in the hosting platform's environment-variable/secret settings rather than source code.

If a secret is ever exposed, rotate/revoke it at the provider immediately.

## Run the Backend

Start the Flask RAG API from the repository root:

```bash
python -m src.api
```

The default backend is available at `http://localhost:5000`.

Check health:

```bash
curl http://localhost:5000/health
```

Expected response:

```json
{"status":"ok","service":"Alert_IQ RAG API"}
```

### API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health check |
| `POST` | `/documents` | Upload, ingest, embed, and index a document |
| `POST` | `/query` | Ask a grounded question |
| `POST` | `/query/stream` | Stream citations and answer tokens as SSE |

Upload example:

```bash
curl -X POST http://localhost:5000/documents \
  -F "file=@data/samples/refund-policy.md"
```

Question example:

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the refund window?"}'
```

## Run the Streamlit UI

With the backend running, open another terminal and run:

```bash
streamlit run ui/streamlit_stream.py
```

The UI uses `RAG_API_URL` to locate the backend and displays streamed answers plus expandable citation/source details.

## End-to-End Demo

A reproducible smoke script is included:

```bash
python scripts/e2e_demo.py
```

It performs the complete delivery flow against the local API:

1. Check `/health`.
2. Upload and index `data/samples/refund-policy.md`.
3. Ask `What is the refund window?`.
4. Print the grounded answer and retrieved source citation.

The repository already contains the same refund-policy content used by the citation/streaming examples, including a source chunk stating that refund requests must be submitted within 14 days.

A successful run should produce output similar to:

```text
[1/3] Backend healthy
[2/3] Indexed: refund-policy.md
[3/3] Grounded answer
Customers can request a refund within 14 days of purchase.

Sources:
[1] refund-policy.md · chunk refund-policy-03 · score 0.91
```

Exact answer wording and scores can vary with the configured model and retrieval state. The important acceptance criteria are: upload succeeds, indexing succeeds, the answer is grounded in the uploaded document, and at least one source citation is returned.

## Tests

Run the test suite with:

```bash
pytest -q
```

The tests cover ingestion, retrieval, grounding, citations, streaming, conversational context, caching, and usage monitoring.

## Observability

The final RAG service includes structured usage logging and a committed sample report under `logs/`. The observability implementation records cache state, latency, approximate token usage, and estimated cost without storing a full secret or API credential. See `docs/OBSERVABILITY.md` for the assumptions and debugging workflow.

## Deployment

The service is intentionally runnable as a conventional Python application. For a hosted deployment:

1. Build/install from `requirements.txt`.
2. Configure the provider API key and model/vector settings as deployment environment variables.
3. Provide persistent storage for `data/vector_store` if the indexed knowledge base must survive restarts.
4. Start the API with `python -m src.api` (or a production WSGI server configured by the hosting platform).
5. Set `RAG_API_URL` for the Streamlit frontend to the deployed API URL.
6. Restrict CORS/network access and upload limits appropriately for the chosen hosting environment.
7. Keep secrets in the hosting provider's secret manager/environment configuration.

For local development, the defaults require only the Python dependencies, a configured provider key, and local disk.

## Final Delivery Marker

The Sprint 2 RAG final-delivery candidate is this branch/PR and its merged commit. After review, the final repository version should be tagged with:

```bash
git tag sprint-2-rag-final
 git push origin sprint-2-rag-final
```

The tag is deliberately created after the final PR is merged so it points at the reviewed `main` commit.

## Project Structure

```text
Alert_IQ/
├── src/                 # RAG, API, ingestion, retrieval, streaming, cache, monitoring
├── ui/                  # Streamlit RAG chat UI
├── tests/               # Automated tests
├── data/samples/        # Reproducible demo documents
├── scripts/             # End-to-end smoke/demo scripts
├── logs/                # Committed sample evidence and reports
├── docs/                # Feature and operational documentation
├── .env.example         # Safe configuration template
├── .gitignore           # Secret/local-data exclusions
├── requirements.txt     # Backend dependencies
└── README.md            # This delivery guide
```

## Next Improvements

Potential next steps include production-grade authentication/authorization, a managed vector database, persistent distributed caching, asynchronous ingestion for large files, richer evaluation/observability dashboards, and a production deployment pipeline with automated CI/CD.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
