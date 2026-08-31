# Source Citations & Verifiability Guide 🔖

This document defines the citation schema, metadata mapping standards, verifiability cross-checks, and anti-fabrication safeguards implemented in the **Alert_IQ RAG System**.

---

## 1. 📋 Citation Schema & Metadata Mapping

Every claim in a generated triage recommendation must be grounded in an evidence passage and mapped to a structured `CitationReference`:

```text
Claim in Answer:
"...run pg_terminate_backend(pid) for transactions running longer than 30 minutes [1]."

Mapped Footnote Reference:
[1] runbook_database_lag.md (Chunk: runbook_database_lag_md::chunk_000, Sec: 'Immediate Triage Steps')
    Quote: "2. **Terminate Stalled Queries**: Execute SELECT pg_terminate_backend(pid) for transactions running longer than 30 minutes..."
```

### Metadata Mapping Fields:
- **`marker`**: Bracketed index tag (`[1]`, `[2]`).
- **`source_document`**: Original source filename (`runbook_database_lag.md`, `incident_policy.txt`).
- **`chunk_id`**: Deterministic ChromaDB record ID (`runbook_database_lag_md::chunk_000`).
- **`chunk_index`**: Ordinal position of chunk within the original ingested document.
- **`section`**: Heading or section title within the document.
- **`snippet_quote`**: Verifiable text quote matching the stored chunk.
- **`similarity_score`**: Relevance confidence score.

---

## 2. 🔍 Vector Store Verifiability Audit

To verify that an answer is accurate and trustworthy:
1. The citation tracker queries ChromaDB directly using the `chunk_id`.
2. It cross-checks that `source_document` matches the record's metadata.
3. It performs a substring match of `snippet_quote` against the stored vector record text.
4. If any citation fails the audit, the response status is flagged as `STORE_INTEGRITY_MISMATCH`.

---

## 3. 🛡️ Anti-Fabrication Safeguards

1. **Detection of Unregistered Citation Markers**:
   - If an answer mentions markers not in the retrieved candidate pool (e.g. `[3]` when only 2 chunks were retrieved), the system flags `FABRICATION_DETECTED` and scrubs unsupported markers.
2. **Missing-Context Fallback**:
   - For out-of-domain queries where no chunks meet the relevance threshold, the system returns:
     > *"Insufficient alert context to recommend action. Refer to standard incident escalation policy."*
   - Zero citations are attached, preventing the model from hallucinating plausible-sounding but fictitious sources.
