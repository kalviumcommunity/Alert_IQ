"""
Prompt Templates Package for Alert_IQ
Centralized repository of prompt templates isolated from application logic.
"""
from src.templates.prompt_templates import (
    PromptTemplate,
    TRIAGE_SYSTEM_TEMPLATE,
    RAG_QA_USER_TEMPLATE,
    BATCH_TRIAGE_USER_TEMPLATE,
    STRUCTURED_TRIAGE_TEMPLATE
)

__all__ = [
    "PromptTemplate",
    "TRIAGE_SYSTEM_TEMPLATE",
    "RAG_QA_USER_TEMPLATE",
    "BATCH_TRIAGE_USER_TEMPLATE",
    "STRUCTURED_TRIAGE_TEMPLATE"
]
