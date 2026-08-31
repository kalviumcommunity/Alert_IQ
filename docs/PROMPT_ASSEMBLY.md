# Grounded Prompt Assembly & Token Budgeting Guide 📝

This document establishes the prompt engineering architecture, token budgeting formulas, citation marker standards, and anti-hallucination grounding guardrails for the **Alert_IQ RAG System**.

---

## 1. 🧮 Token Budgeting Mathematical Formula

To ensure prompt stability and prevent context window truncation errors, available token capacity is partitioned dynamically:

$$\text{Context Budget} = \text{Total Window Limit} - \text{System Tokens} - \text{Query Tokens} - \text{Metadata Tokens} - \text{Boilerplate Overhead} - \text{Response Reserve}$$

### Example Standard Production Profile (2,048 Token Window):
- **Total Limit**: `2,048` tokens
- **System Instructions**: ~`130` tokens
- **User Query + Metadata**: ~`60` tokens
- **Prompt Framing Boilerplate**: ~`30` tokens
- **Response Generation Reserve**: `400` tokens
- $\rightarrow$ **Allowed Evidence Context Budget**: $\approx \mathbf{1,428}$ **tokens**

Chunks are admitted in rank order ($1, 2, \dots$) until the allocated context tokens exceed the allowed context budget.

---

## 2. 🔖 Source Citation Marker Schema

Each retrieved chunk injected into the prompt is encapsulated with structured delimiters:

```text
--- [SOURCE REF: [1] | Document: runbook_database_lag.md | Chunk ID: runbook_database_lag_md::chunk_000] ---
# Database Replica Latency Triage Runbook (DB-RB-402)
...
```

The system prompt mandates that the model anchor all actionable claims and triage recommendations to these bracketed markers (`[1]`, `[2]`).

---

## 3. 🛡️ Anti-Hallucination Grounding Instructions

The system prompt enforces strict operational guardrails:

```text
STRICT GROUNDING RULES:
1. FACTUAL ADHERENCE: Answer ONLY using information explicitly stated in the provided context chunks.
2. NO SPECULATION: Do NOT extrapolate commands, flags, URLs, or procedures not explicitly documented in the context.
3. CITATION MANDATE: When citing facts or recommending steps, include the source reference marker (e.g. [1], [2]).
4. INSUFFICIENT CONTEXT FALLBACK: If the provided context does not contain sufficient details to confidently answer the question, you MUST explicitly respond with:
   "Insufficient alert context to recommend action. Refer to standard incident escalation policy."
```
