# Model Parameter Configuration Guide for Grounded RAG 🎛️

This document outlines the empirical findings from tuning hyper-parameters (`temperature`, `max_tokens`, `top_p`, and `stop`) for the **Alert_IQ Incident Assistant**, establishing the optimal configuration for factual, cost-bounded, and deterministic operation.

---

## 1. ⚙️ Recommended Grounded Production Configuration

| Parameter | Recommended Value | Primary Purpose & Justification |
| :--- | :--- | :--- |
| **`temperature`** | `0.0` (or `0.1`) | **Eliminate Hallucinations & Ensure Reproducibility**: Sets greedy/near-greedy sampling where the model always picks the highest-probability factual token. Guarantees that two on-call engineers querying the same incident runbook receive identical, predictable triage instructions. |
| **`max_tokens`** | `300` | **Enforce Strict Budget & Prevent Runaway Loops**: Caps generation length to ensure responses remain concise (2–3 bullet points) while protecting the monthly token budget against prompt injection or runaway repetitive generation. |
| **`top_p`** | `0.1` | **Nucleus Sampling Filtering**: Constrains the token selection pool strictly to the top 10% probability mass, discarding low-probability tail tokens that cause creative drift or unsupported speculation. |
| **`stop`** | `["\n\nUser:", "---", "### End"]` | **Boundary Enforcement**: Halts token generation immediately if the model attempts to simulate another dialogue turn or bleed past markdown boundaries. |

---

## 2. 📊 Parameter Deep Dive & Experimental Analysis

### A. Temperature ($T$) Dynamics
- **$T = 0.0$ (Deterministic)**: The probability distribution becomes a delta function on the argmax token. Responses strictly stick to standard runbook terminology (`pg_terminate_backend`, `systemctl restart service`).
- **$T = 0.7$ (Default Creative)**: Introducing temperature introduces slight stylistic variance ("*Here are a few quick tips you might want to try...*"). While conversational, this adds unnecessary token overhead.
- **$T = 1.4$ (High Entropy / Unstable)**: The flattened probability distribution introduces high hallucination risks, recommending non-existent flags or rambling tangents.

---

### B. Length Capping with `max_tokens`
- **Cost Protection**: Without `max_tokens`, an unconstrained model can generate up to 4,096 output tokens if triggered by verbose context.
- **Latency Reduction**: Time-To-First-Token (TTFT) and total generation time scale linearly with output token length. Setting `max_tokens=300` ensures typical response latencies remain under 1.2 seconds.

---

### C. `top_p` vs. `temperature` Synergy
- Combining low temperature ($0.0 - 0.2$) with strict nucleus sampling ($top\_p = 0.1$) acts as a dual-filter:
  1. `top_p` clips the improbable vocabulary tail.
  2. `temperature` sharpens the remaining valid candidate distribution.

---

## 3. 🛡️ Verification & Safety Checklist
- [x] Production system prompts explicitly specify $T = 0.0$.
- [x] `max_tokens` is hard-capped on all API client wrapper methods.
- [x] Stop tokens are registered to prevent multi-turn simulation leaks.
