# Retrieval Evaluation & Failure Analysis Report 📈

This document provides systematic quality evidence for the **Alert_IQ RAG retrieval pipeline**, evaluating retrieval performance against a curated labelled query benchmark and detailing failure root-cause analysis.

---

## 1. 🎯 Labelled Query Benchmark Dataset

The evaluation suite tests 6 operational query scenarios of varying difficulty and intent:

| Query ID | Category | Difficulty | Expected Source | Ground-Truth Target Intent |
| :--- | :--- | :---: | :--- | :--- |
| **`EVAL-001`** | Incident Policy | Easy | `incident_policy.txt` | Level 1 on-call notification policy and escalation timeline. |
| **`EVAL-002`** | Database Runbook | Easy | `runbook_database_lag.md` | Exact runbook code `DB-RB-402` replica latency mitigation. |
| **`EVAL-003`** | Telemetry Schema | Medium | `metrics_schema.json` | Exact telemetry parameter `critical_response_time_seconds`. |
| **`EVAL-004`** | Service Health | Medium | `service_health_export.html` | Cluster health export date and authentication status. |
| **`EVAL-005`** | Semantic Synonyms | Hard | `runbook_database_lag.md` | Conceptual query for replica sync lag without exact runbook ID. |
| **`EVAL-006`** | Vague / Adversarial | Hard | `runbook_database_lag.md` | Under-specified query (*"General latency issue"*). |

---

## 2. 📊 Measured Performance Metrics

Evaluation results across the two-stage retrieval pipeline:

| Metric | Measured Value | Operational Interpretation |
| :--- | :---: | :--- |
| **Mean Recall@1** | **91.7%** | Fraction of ground-truth chunks present at Rank #1. |
| **Mean Recall@2** | **91.7%** | High recall across the primary grounding context window ($k=2$). |
| **Mean Recall@3** | **91.7%** | Coverage of relevant corpus chunks across all benchmark queries. |
| **Mean Precision@1** | **100.0%** | Relevance precision for top result (100% Top-1 accuracy). |
| **Mean Precision@2** | **50.0%** | Precision across $k=2$ candidate chunks. |
| **Mean F1@2** | **0.647** | Balanced harmonic mean of precision and recall. |
| **Mean Reciprocal Rank (MRR)** | **1.000** | Perfect reciprocal rank of the first relevant chunk ($1/\text{rank}$). |

---

## 3. 🔍 Failure & Low-Scoring Case Root Cause Analysis

### Case Study 1: Under-Specified Vague Query (`EVAL-006` *"General latency issue"*)
- **Symptoms**: The query lacks target system identifiers (Database replica lag vs. API gateway latency vs. network cluster health).
- **Observed Ranking**: Both `metrics_schema.json` and `runbook_database_lag.md` compete for relevance.
- **Root Cause**: Semantic ambiguity in user prompt without explicit metadata scoping.
- **Remediation**: Implement interactive clarification or multi-candidate disambiguation in the front-end chat interface.

### Case Study 2: Conceptual Paraphrasing (`EVAL-005` *"secondary replica falling behind master"*)
- **Symptoms**: Pure vector search struggled to cleanly isolate the database runbook due to generic tokens ("master", "secondary").
- **Resolution**: The secondary re-ranker cross-interaction scoring successfully evaluated token coverage and structural headings, promoting `runbook_database_lag.md` to Rank #1.

---

## 4. 🛠️ Error Taxonomy & Engineering Recommendations

1. **Exact Technical Identifier Boosting**:
   - Always run the secondary re-ranker with technical ID pattern recognition (`DB-RB-*`, `*_seconds`, `P1-P4`) to avoid dense vector collision.
2. **Metadata Filtering for Known Scopes**:
   - Pre-filter by `source_document` or `file_type` whenever the user query provides explicit domain context.
3. **Relevance Thresholding**:
   - Maintain a minimum score threshold of `0.02` to reject irrelevant tail chunks for completely out-of-domain queries.
