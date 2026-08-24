"""
Unit tests for structured JSON parsing, auto-repair, and schema validation.
"""
import unittest
from unittest.mock import MagicMock
from src.structured_output import (
    parse_json_response,
    validate_alert_payload,
    query_structured_alert,
    repair_malformed_json_string
)


class TestStructuredOutput(unittest.TestCase):

    def test_parse_valid_json(self):
        raw = '{"alert_id": "ALT-1", "severity": "HIGH", "answer": "Test", "source": "runbook.md", "action_steps": ["step 1"]}'
        ok, data, msg = parse_json_response(raw)
        self.assertTrue(ok)
        self.assertEqual(data["alert_id"], "ALT-1")

    def test_parse_markdown_wrapped_json(self):
        raw = 'Here is the result:\n```json\n{"alert_id": "ALT-2", "severity": "LOW", "answer": "OK", "source": "r.md", "action_steps": ["step 1"]}\n```\nThanks!'
        ok, data, msg = parse_json_response(raw)
        self.assertTrue(ok)
        self.assertEqual(data["alert_id"], "ALT-2")

    def test_repair_malformed_json(self):
        raw = "{'alert_id': 'ALT-3', 'severity': 'MEDIUM', 'answer': 'Lag', 'source': 'r.md', 'action_steps': ['reboot',],}"
        ok, data, msg = parse_json_response(raw)
        self.assertTrue(ok)
        self.assertEqual(data["alert_id"], "ALT-3")
        self.assertEqual(data["severity"], "MEDIUM")

    def test_validate_alert_payload_success(self):
        data = {
            "alert_id": "ALT-100",
            "severity": "critical",  # Should normalize to CRITICAL
            "answer": "Primary down",
            "source": "docs/db.md",
            "action_steps": ["Failover to replica"]
        }
        is_valid, errors, validated = validate_alert_payload(data)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.assertEqual(validated["severity"], "CRITICAL")

    def test_validate_missing_required_fields(self):
        incomplete_data = {
            "alert_id": "ALT-101",
            "severity": "HIGH"
        }
        is_valid, errors, validated = validate_alert_payload(incomplete_data)
        self.assertFalse(is_valid)
        self.assertTrue(any("answer" in err for err in errors))
        self.assertTrue(any("source" in err for err in errors))
        self.assertTrue(any("action_steps" in err for err in errors))

    def test_validate_invalid_severity(self):
        data = {
            "alert_id": "ALT-102",
            "severity": "SUPER_URGENT",  # Invalid enum
            "answer": "Test",
            "source": "doc.md",
            "action_steps": ["Action"]
        }
        is_valid, errors, validated = validate_alert_payload(data)
        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid severity" in err for err in errors))

    def test_query_structured_alert_mock(self):
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = '{"alert_id": "ALT-5", "severity": "HIGH", "answer": "Auth fail", "source": "auth.md", "action_steps": ["check tokens"]}'
        ok, result, msg = query_structured_alert("Query", mock_client)
        self.assertTrue(ok)
        self.assertEqual(result["alert_id"], "ALT-5")


if __name__ == "__main__":
    unittest.main()
