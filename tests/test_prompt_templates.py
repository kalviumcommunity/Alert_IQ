"""
Unit tests for PromptTemplate engine and multi-feature template reuse.
"""
import unittest
from unittest.mock import MagicMock
from src.templates.prompt_templates import (
    PromptTemplate,
    TRIAGE_SYSTEM_TEMPLATE,
    RAG_QA_USER_TEMPLATE,
    BATCH_TRIAGE_USER_TEMPLATE
)
from src.triage_services import InteractiveChatService, BatchTriageService


class TestPromptTemplates(unittest.TestCase):

    def test_placeholder_extraction(self):
        tmpl = PromptTemplate("Hello {name}, your alert is {alert_id} with status {status}.")
        self.assertEqual(tmpl.input_variables, {"name", "alert_id", "status"})

    def test_successful_render(self):
        tmpl = PromptTemplate("Alert {alert_id} on {service}: {message}")
        result = tmpl.render(alert_id="ALT-1", service="auth", message="OOM")
        self.assertEqual(result, "Alert ALT-1 on auth: OOM")

    def test_missing_variable_raises_error(self):
        tmpl = PromptTemplate("Hello {name} on {date}")
        with self.assertRaises(ValueError) as ctx:
            tmpl.render(name="Alice")  # Missing date
        self.assertIn("Missing required variables", str(ctx.exception))

    def test_escaped_braces(self):
        tmpl = PromptTemplate("Payload: {{ 'key': '{value}' }}")
        result = tmpl.render(value="123")
        self.assertEqual(result, "Payload: { 'key': '123' }")

    def test_interactive_chat_service_render(self):
        service = InteractiveChatService()
        messages = service.build_prompts(
            alert_id="ALT-9042",
            severity="HIGH",
            service_name="payment-svc",
            context="Runbook context",
            question="What to do?"
        )
        self.assertEqual(len(messages), 2)
        self.assertIn("Alert_IQ Incident Assistant", messages[0]["content"])
        self.assertIn("ALT-9042", messages[1]["content"])
        self.assertIn("Runbook context", messages[1]["content"])

    def test_batch_triage_service_render(self):
        service = BatchTriageService()
        incidents = [
            {"alert_id": "ALT-1", "severity": "CRITICAL", "message": "Crash", "service": "db"},
            {"alert_id": "ALT-2", "severity": "LOW", "message": "Warn", "service": "cache"}
        ]
        messages = service.build_batch_prompts(
            batch_id="BATCH-001",
            target_sla="30m",
            incidents=incidents
        )
        self.assertEqual(len(messages), 2)
        self.assertIn("BATCH-001", messages[1]["content"])
        self.assertIn("ALT-1", messages[1]["content"])
        self.assertIn("ALT-2", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
