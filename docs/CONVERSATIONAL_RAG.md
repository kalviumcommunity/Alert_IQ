# Conversational RAG

## Why naive retrieval breaks on follow-ups

A single-turn retriever expects the query to contain the important entities and topic. Follow-ups often contain references such as `it`, `that`, or `what's the next step?` without repeating the incident, service, or runbook name. Embedding that short follow-up by itself can therefore retrieve unrelated context.

## Flow

```text
user question
    ↓
conversation history
    ↓
standalone query rewrite
    ↓
embedding / retrieval
    ↓
verified context
    ↓
grounded answer
    ↓
append user + assistant turn to bounded history
```

`src/conversational_rag.py` exposes `ConversationalRAG` for this flow. Its `run()` method rewrites first, passes the rewritten string directly to retrieval, generates from the retrieved context, and then records the completed turn.

## History and token limits

The implementation reuses `ConversationManager` from `src/history_manager.py`. It always keeps the system prompt and removes the oldest conversation messages when the configured token budget is exceeded. This prevents an ever-growing transcript from consuming the model context window.

For long conversations, a stronger production design is:

- keep a bounded recent window;
- summarize older turns into compact conversation state;
- retain source/chunk IDs for important facts;
- retrieve fresh corpus evidence on every follow-up;
- apply the hallucination guardrail before generation.

## Example

Initial turn:

> User: PostgreSQL replica latency is 920ms. What should I check first?

Follow-up:

> User: Latency dropped to 410ms, but connection pool utilization is 88%. What's the next step?

The second question depends on the first turn's incident context. The conversational layer carries that context into the standalone retrieval query and retrieves the DB-RB-402 pool-utilization guidance.

See `logs/conversational_rag_demo.log` for the complete sample dialogue, rewritten queries, retrieved context, and final answers.
