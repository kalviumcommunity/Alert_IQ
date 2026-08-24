"""
Prompt Template Engine & Isolated Template Store for Alert_IQ
Decouples prompt text and interpolation logic from downstream service implementations.
"""
import re
from typing import Dict, Any, List, Set


class PromptTemplate:
    """
    Reusable prompt template supporting named placeholders and runtime validation.
    """

    def __init__(self, template: str, name: str = "custom_template"):
        self.template = template.strip()
        self.name = name
        self.input_variables = self._extract_placeholders(self.template)

    def _extract_placeholders(self, text: str) -> Set[str]:
        """
        Extracts variable names enclosed in single curly braces {variable}.
        Ignores escaped braces {{like_this}}.
        """
        # Finds words inside single braces not prefixed or suffixed by another brace
        pattern = r"(?<!\{)\{([a-zA-Z0-9_]+)\}(?!\})"
        return set(re.findall(pattern, text))

    def render(self, **kwargs: Any) -> str:
        """
        Renders template with provided keyword arguments.
        Raises ValueError if any required placeholder is missing.
        """
        missing = [var for var in self.input_variables if var not in kwargs]
        if missing:
            raise ValueError(
                f"Cannot render template '{self.name}'. Missing required variables: {missing}. "
                f"Expected: {sorted(list(self.input_variables))}"
            )

        # Protect escaped braces first
        rendered = self.template.replace("{{", "__DOUBLE_LEFT_BRACE__").replace("}}", "__DOUBLE_RIGHT_BRACE__")

        for key, val in kwargs.items():
            str_val = str(val) if val is not None else ""
            rendered = rendered.replace(f"{{{key}}}", str_val)

        # Restore escaped braces as single literal braces
        rendered = rendered.replace("__DOUBLE_LEFT_BRACE__", "{").replace("__DOUBLE_RIGHT_BRACE__", "}")
        return rendered

    def format_messages(self, **kwargs: Any) -> Dict[str, str]:
        """
        Helper returning a standard role/content dict for chat completion.
        """
        return {"content": self.render(**kwargs)}


# ==============================================================================
# Isolated Reusable Templates (Task 4)
# ==============================================================================

# 1. System Prompt Template (Role & Guardrails)
TRIAGE_SYSTEM_TEMPLATE = PromptTemplate(
    name="triage_system_template",
    template="""You are {assistant_name}, an internal AI specialist assisting on-call engineers with {domain} triage.

ROLE & OBJECTIVE:
Provide concise, factual triage instructions based on verified runbook context.

CONSTRAINTS:
- Tone: Professional and direct.
- Length: Max {max_paragraphs} paragraphs.
- Fallback: If context is insufficient, reply: "Insufficient alert context to recommend action. Refer to {fallback_runbook}."
"""
)

# 2. Interactive RAG QA Template (Feature 1: Interactive Chat)
RAG_QA_USER_TEMPLATE = PromptTemplate(
    name="rag_qa_user_template",
    template="""[ALERT METADATA]
Alert ID: {alert_id}
Severity: {severity}
Service : {service_name}

[RETRIEVED CONTEXT]
{context}

[STAFF QUESTION]
{question}

Provide actionable triage steps adhering to the retrieved runbook context."""
)

# 3. Batch Alert Triage Template (Feature 2: Batch / CLI Processing)
BATCH_TRIAGE_USER_TEMPLATE = PromptTemplate(
    name="batch_triage_user_template",
    template="""[BATCH PROCESSING JOB: {batch_id}]
Target SLA: {target_sla}
Total Incidents: {incident_count}

[INCIDENT SUMMARY RECORDS]
{records}

Analyze the above batch records and output prioritized mitigation steps for high-severity incidents."""
)

# 4. Structured JSON Output Template
STRUCTURED_TRIAGE_TEMPLATE = PromptTemplate(
    name="structured_triage_template",
    template="""Analyze the following alert incident and output strict JSON.

Alert ID: {alert_id}
Incident Summary: {incident_summary}
Runbook Source: {runbook_source}

Format strictly as:
{{
  "alert_id": "{alert_id}",
  "severity": "CRITICAL|HIGH|MEDIUM|LOW",
  "answer": "<triage summary>",
  "source": "{runbook_source}",
  "action_steps": ["<step 1>", "<step 2>"]
}}"""
)
