# Alert_IQ — Sprint 2 RAG Product Requirements

## 1. Product Summary

Alert_IQ's Sprint 2 RAG application turns a controlled knowledge base into a question-answering system that retrieves relevant document chunks before generating an answer. The product is designed to make answers grounded and verifiable rather than relying on unsupported model memory.

## 2. Problem

Users need to ask questions about operational and policy information stored in documents. A conventional chat model can produce plausible answers without evidence. Alert_IQ therefore needs a retrieval layer that finds relevant source material and exposes those sources alongside the generated answer.

## 3. Target Users

- **Operators / support users:** need fast answers to questions about internal operational knowledge.
- **Developers / maintainers:** need a reproducible ingestion, retrieval, and generation pipeline that can be tested and inspected.
- **Reviewers:** need to trace an answer back to the source document and chunk that supported it.

## 4. Knowledge Source

The RAG knowledge base is document-based. The application supports runtime upload of `.txt`, `.md`, and `.pdf` files. Documents are loaded, cleaned, split into token-aware chunks, enriched with source metadata, embedded, and stored in the persistent ChromaDB collection.

The repository includes `data/samples/refund-policy.md` as a reproducible demonstration document.

## 5. Questions the AI App Should Answer

The application should answer questions whose evidence exists in the indexed knowledge base, such as:

- policy rules and time windows;
- operational procedures;
- facts contained in uploaded documentation;
- follow-up questions that can be resolved using conversational context.

When retrieval is empty or insufficiently relevant, hallucination guardrails should refuse rather than invent an answer.

## 6. RAG Pipeline

```text
Documents
   ↓
Load + Clean
   ↓
Token-aware Chunking + Source Metadata
   ↓
Embeddings
   ↓
ChromaDB Vector Store
   ↓
Query Embedding + Similarity Retrieval
   ↓
Relevant Context / Guardrails
   ↓
Grounded LLM Answer
   ↓
Answer + Citations
```

The delivered application also includes conversational follow-up handling, streaming responses, query caching, and usage monitoring.

## 7. Functional Requirements

### FR1 — Document ingestion
A user must be able to upload a supported document and have it indexed without restarting the service.

### FR2 — Retrieval
A question must retrieve relevant chunks from the indexed knowledge base.

### FR3 — Grounded generation
The answer must be generated using retrieved context rather than unsupported claims.

### FR4 — Citations
The response must expose source document/chunk information so the user can verify the evidence.

### FR5 — Refusal
Weak or missing retrieval must produce a safe refusal instead of a fabricated answer.

### FR6 — Conversation
Follow-up questions must be interpreted using bounded conversation history where appropriate.

### FR7 — Streaming
The UI/API must support progressive citation and answer-token delivery.

### FR8 — Observability
Requests should record cache state, latency, approximate token usage, cost estimates, sources, and errors.

## 8. Non-Functional Requirements

- **Reproducibility:** a new developer can follow the README to configure and run the application.
- **Traceability:** every grounded response exposes source information.
- **Security:** API credentials are supplied through environment variables and are never committed.
- **Testability:** core RAG components have automated tests that do not require production credentials.
- **Maintainability:** ingestion, retrieval, generation, API, UI, caching, and monitoring remain separable components.

## 9. What Good Looks Like

A successful query should follow this contract:

1. The question is accepted.
2. Relevant source chunks are retrieved.
3. The answer is grounded in those chunks.
4. The response includes citations identifying the supporting source.
5. A reviewer can inspect the cited source text.
6. Unsupported questions are refused safely.

## 10. Scope for Sprint 2

### In scope
- document ingestion and chunking;
- embeddings and vector storage;
- retrieval and ranking;
- grounded answer generation;
- citations and refusal guardrails;
- conversational context;
- backend API and Streamlit UI;
- streaming responses;
- caching and usage monitoring;
- reproducible documentation and final delivery evidence.

### Future scope
- production authentication and authorization;
- managed/distributed vector infrastructure;
- asynchronous large-document ingestion;
- distributed cache;
- richer evaluation dashboards;
- automated production CI/CD and deployment infrastructure.
