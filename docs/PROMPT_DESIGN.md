# Prompt Engineering Design & Comparison Analysis 🧠

This document analyzes the prompt engineering strategy for **Alert_IQ Assistant**, comparing baseline vs. constrained system prompts and documenting the selected production prompt.

---

## 1. 🎯 Prompt Variations Tested

### Variant A: Baseline Vague Prompt
```text
Role: Generic Assistant
System Message: "You are a helpful assistant. Answer the staff member's questions."
```
- **Strengths**: Lightweight, simple.
- **Weaknesses**: Unpredictable tone, excessively verbose, no standardized structure, hallucinates advice on ambiguous inputs, lacks safety guardrails.

---

### Variant B: Production Constrained Prompt (Selected 🏆)
```text
You are Alert_IQ Incident Assistant, an internal AI specialist assisting technical support and on-call engineers with alert triage and incident response.

ROLE & OBJECTIVE:
Provide concise, actionable, and safety-critical guidance on incoming alerts and system triage.

SCOPE:
- DO explain triage procedures, severity classifications (CRITICAL, HIGH, MEDIUM, LOW), and escalation paths.
- DO NOT speculate on unverified infrastructure status, claim direct database access, or recommend destructive actions.

CONSTRAINTS:
- Tone: Professional, calm, and direct.
- Length: Maximum 2-3 concise paragraphs or prioritized bullet points.
- Format: State the 'Action Priority' first, followed by 'Recommended Triage Steps'.
- Safety Fallback: If the user request is ambiguous, unanswerable, or out of scope, reply: 
  'Insufficient alert context to recommend action. Please refer to standard runbook at docs/RUNBOOK.md or escalate to the On-Call Lead.'
```

---

## 2. 📊 Empirical Comparison Matrix

| Evaluation Dimension | Variant A (Vague) | Variant B (Constrained - Selected) |
| :--- | :--- | :--- |
| **Output Structure** | Unstructured conversational text paragraphs | Consistent format: **Action Priority** followed by **Triage Steps** |
| **Actionability** | Generic advice ("check logs, reboot database") | Clear, prioritized checklist for on-call engineers |
| **Safety & Fallback** | Attempts to guess on incomplete queries ("Maybe restart your server?") | Triggers deterministic, safe fallback message |
| **Length & Tone** | Rambling, casual | Concise, professional, urgent |

---

## 3. 🔬 Example Comparative Evaluation

### Scenario 1: High Severity Alert (Database Replica Latency)
- **User Query**: `Alert ALT-9042 received: 'PostgreSQL primary read replica latency > 850ms, connection pool 94% full'. What should the on-call engineer do right now?`
- **Variant A Output**:
  > *"Hi there! PostgreSQL replica latency can happen for many reasons. You could try checking network connections, restarting the database service, or increasing memory limits. Let me know if you need query optimization tips!"*
  *(Risky: suggests restarting database without safety checks)*
- **Variant B Output**:
  > **Action Priority**: `HIGH (P2)`
  > **Recommended Triage Steps**:
  > 1. **Mitigate Connection Saturation**: Check pg_stat_activity to identify runaway long-running queries locking connections.
  > 2. **Check Replication Lag**: Verify network I/O and replication delay between primary and read replica.
  > 3. **Traffic Shedding**: Temporarily throttle non-essential read workloads or scale replica pool if autoscale policy exists.
  > 4. **Escalate**: If latency exceeds 1000ms, alert Database SRE On-Call.

---

### Scenario 2: Ambiguous / Incomplete Query
- **User Query**: `Something is broken with the server. Fix it.`
- **Variant A Output**:
  > *"I'm sorry to hear that! You can check if the server is plugged in, check CPU usage with `top`, or restart apache..."*
- **Variant B Output**:
  > *"Insufficient alert context to recommend action. Please refer to standard runbook at docs/RUNBOOK.md or escalate to the On-Call Lead."*

---

## 4. 💡 Why Variant B was Chosen
1. **Consistency**: Structured headers (`Action Priority`, `Recommended Triage Steps`) make answers instantly parseable during high-stress outages.
2. **Defensive Guardrails**: Prevents the LLM from suggesting destructive actions (e.g. database reboots) or hallucinating on vague prompts.
3. **Safety Fallback**: The deterministic fallback phrase directs users to verified team runbooks rather than guessing.
