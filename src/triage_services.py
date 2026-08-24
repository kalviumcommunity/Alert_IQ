"""
Application Services Demonstrating Reusable Prompt Template Consumption
Implements:
1. Interactive RAG Chat Service (Feature 1)
2. Batch Alert CLI Processor Service (Feature 2)
Both consume the isolated templates defined in src.templates.prompt_templates.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.templates.prompt_templates import (
    TRIAGE_SYSTEM_TEMPLATE,
    RAG_QA_USER_TEMPLATE,
    BATCH_TRIAGE_USER_TEMPLATE,
    STRUCTURED_TRIAGE_TEMPLATE
)
from src.chat_client import ChatClient


class InteractiveChatService:
    """
    Feature 1: Interactive On-Call Q&A Assistant.
    Renders system and user prompts using shared prompt templates.
    """

    def __init__(self, client: Optional[ChatClient] = None):
        self.client = client or ChatClient()

    def build_prompts(
        self,
        alert_id: str,
        severity: str,
        service_name: str,
        context: str,
        question: str
    ) -> List[Dict[str, str]]:
        """
        Renders templates dynamically at runtime.
        """
        # Render system template
        system_content = TRIAGE_SYSTEM_TEMPLATE.render(
            assistant_name="Alert_IQ Incident Assistant",
            domain="Infrastructure & Database",
            max_paragraphs=2,
            fallback_runbook="docs/runbooks/master_triage.md"
        )

        # Render user QA template
        user_content = RAG_QA_USER_TEMPLATE.render(
            alert_id=alert_id,
            severity=severity,
            service_name=service_name,
            context=context,
            question=question
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def answer_question(
        self,
        alert_id: str,
        severity: str,
        service_name: str,
        context: str,
        question: str
    ) -> Optional[str]:
        messages = self.build_prompts(alert_id, severity, service_name, context, question)
        return self.client.chat_completion(messages=messages, temperature=0.0)


class BatchTriageService:
    """
    Feature 2: Automated Batch Incident Processor & Summary Generator.
    Reuses the same system template and the batch user template.
    """

    def __init__(self, client: Optional[ChatClient] = None):
        self.client = client or ChatClient()

    def build_batch_prompts(
        self,
        batch_id: str,
        target_sla: str,
        incidents: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Renders batch template dynamically at runtime.
        """
        # Reusing the exact same system template structure
        system_content = TRIAGE_SYSTEM_TEMPLATE.render(
            assistant_name="Alert_IQ Batch Analyzer",
            domain="Automated Telemetry",
            max_paragraphs=3,
            fallback_runbook="docs/runbooks/batch_triage.md"
        )

        formatted_records = "\n".join(
            f"• [{inc.get('severity', 'UNKNOWN')}] {inc.get('alert_id')}: {inc.get('message')} (Service: {inc.get('service')})"
            for inc in incidents
        )

        # Render batch user template
        user_content = BATCH_TRIAGE_USER_TEMPLATE.render(
            batch_id=batch_id,
            target_sla=target_sla,
            incident_count=len(incidents),
            records=formatted_records
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def process_batch(
        self,
        batch_id: str,
        target_sla: str,
        incidents: List[Dict[str, Any]]
    ) -> Optional[str]:
        messages = self.build_batch_prompts(batch_id, target_sla, incidents)
        return self.client.chat_completion(messages=messages, temperature=0.0)


def generate_template_renders_demo() -> str:
    """
    Renders all templates with dynamic sample data and produces a comprehensive demonstration log.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append("📄 Alert_IQ - Prompt Template Engine & Multi-Feature Renders")
    log_lines.append("=" * 80)

    # 1. Feature 1 Demonstration (Interactive Chat)
    chat_service = InteractiveChatService()
    chat_messages = chat_service.build_prompts(
        alert_id="ALT-9042",
        severity="HIGH",
        service_name="payment-gateway-replica",
        context="Runbook Section 4.2: If replica lag > 500ms and active connections > 90%, terminate idle transactions via pg_terminate_backend.",
        question="How should we mitigate the replica lag right now?"
    )

    log_lines.append("\n📌 FEATURE 1: INTERACTIVE RAG CHAT PROMPTS")
    log_lines.append("-" * 80)
    log_lines.append(f"[System Prompt Render]:\n{chat_messages[0]['content']}")
    log_lines.append("-" * 40)
    log_lines.append(f"[User Prompt Render]:\n{chat_messages[1]['content']}")

    # 2. Feature 2 Demonstration (Batch CLI Processor)
    batch_service = BatchTriageService()
    batch_incidents = [
        {"alert_id": "ALT-101", "severity": "CRITICAL", "message": "Auth pod OOMKilled", "service": "auth-service"},
        {"alert_id": "ALT-102", "severity": "HIGH", "message": "Stripe webhook 504 timeout", "service": "billing-api"},
        {"alert_id": "ALT-103", "severity": "LOW", "message": "Log rotation disk warn (81%)", "service": "logging-agent"}
    ]
    batch_messages = batch_service.build_batch_prompts(
        batch_id="BATCH-2026-08-24-001",
        target_sla="15 minutes",
        incidents=batch_incidents
    )

    log_lines.append("\n" + "=" * 80)
    log_lines.append("📌 FEATURE 2: BATCH / CLI TELEMETRY PROCESSOR PROMPTS")
    log_lines.append("-" * 80)
    log_lines.append(f"[System Prompt Render]:\n{batch_messages[0]['content']}")
    log_lines.append("-" * 40)
    log_lines.append(f"[User Batch Prompt Render]:\n{batch_messages[1]['content']}")

    # 3. Structured JSON Template Render
    log_lines.append("\n" + "=" * 80)
    log_lines.append("📌 FEATURE 3: STRUCTURED JSON SCHEMA TEMPLATE RENDER")
    log_lines.append("-" * 80)
    structured_render = STRUCTURED_TRIAGE_TEMPLATE.render(
        alert_id="ALT-4012",
        incident_summary="Host memory exhaustion (>95%) on auth-worker-01",
        runbook_source="docs/runbooks/memory_exhaustion.md"
    )
    log_lines.append(structured_render)
    log_lines.append("=" * 80)

    return "\n".join(log_lines)


def main():
    report = generate_template_renders_demo()
    print(report)


if __name__ == "__main__":
    main()
