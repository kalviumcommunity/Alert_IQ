"""
Structured JSON Output Parser & Validation Module for Alert_IQ RAG Pipeline
Prompts LLM for strict JSON schema, robustly parses/repairs malformed JSON, and validates required fields.
"""
import re
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chat_client import ChatClient

logger = logging.getLogger("StructuredOutput")

# Required schema fields and allowed enum values
REQUIRED_FIELDS = ["alert_id", "severity", "answer", "source", "action_steps"]
ALLOWED_SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

# Task 1: System prompt defining strict JSON schema
STRUCTURED_SYSTEM_PROMPT = """You are Alert_IQ Structured Triage Assistant.
You MUST respond with a valid JSON object matching this exact schema. Do not output any markdown formatting, preamble, or commentary outside the JSON object.

JSON Schema:
{
  "alert_id": "string (e.g. ALT-1001)",
  "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
  "answer": "string (clear, direct explanation of the issue)",
  "source": "string (name of runbook or documentation source)",
  "action_steps": ["step 1", "step 2", "step 3"]
}"""


import ast

def repair_malformed_json_string(raw_text: str) -> str:
    """
    Cleans and repairs common LLM JSON formatting issues:
    - Strips markdown code blocks (```json ... ```)
    - Extracts JSON object from conversational text
    - Removes trailing commas before closing braces/brackets
    - Converts single quotes to double quotes for keys and strings
    """
    if not raw_text:
        return ""

    cleaned = raw_text.strip()

    # 1. Remove markdown code fences
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(fence_pattern, cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()

    # 2. Extract substring between first '{' and last '}'
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace : last_brace + 1]

    # 3. Remove trailing commas before '}' or ']'
    cleaned = re.sub(r",\s*([\}\]])", r"\1", cleaned)

    return cleaned


def parse_json_response(raw_text: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Parses a raw LLM text response into a Python dictionary.

    Returns:
        Tuple: (success: bool, parsed_dict: Optional[Dict], status_message: str)
    """
    if not raw_text or not raw_text.strip():
        return False, None, "Empty response received from LLM."

    # First attempt: direct json.loads
    try:
        data = json.loads(raw_text.strip())
        if isinstance(data, dict):
            return True, data, "Parsed directly as valid JSON."
    except json.JSONDecodeError:
        pass

    # Second attempt: repair cleaned JSON
    repaired_text = repair_malformed_json_string(raw_text)
    try:
        data = json.loads(repaired_text)
        if isinstance(data, dict):
            return True, data, "Successfully recovered and parsed markdown-wrapped/cleaned JSON."
    except json.JSONDecodeError:
        pass

    # Third attempt: safe ast.literal_eval for single-quoted Python dict representations
    try:
        val = ast.literal_eval(repaired_text)
        if isinstance(val, dict):
            return True, val, "Successfully recovered malformed single-quoted JSON via AST repair."
    except (ValueError, SyntaxError, MemoryError, TypeError):
        pass

    # Fourth attempt: regex replacement of single quotes with double quotes
    try:
        normalized = re.sub(r"(?<!\\)'", '"', repaired_text)
        normalized = re.sub(r",\s*([\}\]])", r"\1", normalized)
        data = json.loads(normalized)
        if isinstance(data, dict):
            return True, data, "Successfully recovered JSON via quote normalization."
    except json.JSONDecodeError as e:
        error_msg = f"Malformed JSON unrecoverable: {e.msg} at line {e.lineno}, col {e.colno}."
        logger.warning(error_msg)
        return False, None, error_msg

    return False, None, "Failed to parse or recover JSON object."


def validate_alert_payload(data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validates that all required fields are present and correctly typed.

    Returns:
        Tuple: (is_valid: bool, validation_errors: List[str], validated_data: Dict[str, Any])
    """
    errors = []
    validated = dict(data)

    # Check required fields presence
    for field in REQUIRED_FIELDS:
        if field not in validated or validated[field] is None:
            errors.append(f"Missing required field: '{field}'")

    # Validate and normalize severity enum
    if "severity" in validated and validated["severity"]:
        sev = str(validated["severity"]).upper().strip()
        if sev not in ALLOWED_SEVERITIES:
            errors.append(f"Invalid severity '{sev}'. Must be one of {ALLOWED_SEVERITIES}")
        else:
            validated["severity"] = sev

    # Validate action_steps is a list
    if "action_steps" in validated:
        if not isinstance(validated["action_steps"], list):
            errors.append(f"'action_steps' must be a list of strings, got {type(validated['action_steps']).__name__}")
        elif len(validated["action_steps"]) == 0:
            errors.append("'action_steps' list cannot be empty")

    is_valid = len(errors) == 0
    return is_valid, errors, validated


def query_structured_alert(
    user_query: str,
    client: Optional[ChatClient] = None
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Sends structured request using response_format={"type": "json_object"} and validates output.
    """
    if client is None:
        client = ChatClient()

    messages = [
        {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    # Task 1: Request JSON response format
    raw_response = client.chat_completion(
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    if not raw_response:
        return False, None, "LLM request returned no content."

    # Task 2 & 3: Parse and handle malformed JSON
    parse_ok, parsed_obj, parse_msg = parse_json_response(raw_response)
    if not parse_ok or parsed_obj is None:
        return False, None, f"JSON parsing failed: {parse_msg}"

    # Task 4: Validate required fields
    is_valid, validation_errors, validated_data = validate_alert_payload(parsed_obj)
    if not is_valid:
        error_summary = "; ".join(validation_errors)
        return False, validated_data, f"Validation failed: {error_summary}"

    return True, validated_data, "Successfully parsed and validated structured alert payload."


def run_structured_output_demonstration() -> str:
    """
    Demonstrates parsing and validation across normal, markdown-wrapped, malformed, and invalid samples.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    test_cases = [
        {
            "case_id": "case_1_clean_json",
            "title": "Clean Direct JSON Output",
            "raw_input": """{
  "alert_id": "ALT-9042",
  "severity": "HIGH",
  "answer": "PostgreSQL read replica latency exceeded 850ms due to connection saturation.",
  "source": "docs/runbooks/database_replica_lag.md",
  "action_steps": [
    "Check active connections in pg_stat_activity",
    "Terminate idle-in-transaction queries",
    "Scale replica read pool"
  ]
}"""
        },
        {
            "case_id": "case_2_markdown_wrapped",
            "title": "Markdown Code-Fenced JSON (Common LLM Output)",
            "raw_input": """Here is the structured triage breakdown:
```json
{
  "alert_id": "ALT-4012",
  "severity": "CRITICAL",
  "answer": "Host memory utilization at 96.4% on auth-worker-prod-02.",
  "source": "docs/runbooks/memory_exhaustion.md",
  "action_steps": [
    "Identify memory hogs with top -o %MEM",
    "Flush drop_caches buffers",
    "Restart worker pod"
  ]
}
```
Please let me know if you need more details."""
        },
        {
            "case_id": "case_3_malformed_repaired",
            "title": "Malformed JSON (Single Quotes & Trailing Commas) - RECOVERED",
            "raw_input": """{
  'alert_id': 'ALT-2031',
  'severity': 'MEDIUM',
  'answer': 'Redis cache eviction rate elevated above baseline.',
  'source': 'docs/runbooks/redis_cache.md',
  'action_steps': [
    'Inspect maxmemory-policy setting',
    'Increase Redis RAM allocation',
  ],
}"""
        },
        {
            "case_id": "case_4_missing_required_field",
            "title": "Missing Required Field ('source' and 'action_steps' absent) - REJECTED",
            "raw_input": """{
  "alert_id": "ALT-5501",
  "severity": "LOW",
  "answer": "Disk volume /var/log reached 78% capacity."
}"""
        }
    ]

    report = []
    report.append("=" * 80)
    report.append("📦 Alert_IQ - Structured JSON Output Parsing & Validation Demo")
    report.append("=" * 80)

    for tc in test_cases:
        report.append(f"\n🔬 [TEST CASE: {tc['title']}]")
        report.append("Raw Input:")
        report.append(tc["raw_input"])
        report.append("-" * 50)

        # Parse
        parse_ok, parsed_obj, parse_msg = parse_json_response(tc["raw_input"])
        report.append(f"• Parse Status  : {'✅ SUCCESS' if parse_ok else '❌ FAILED'} ({parse_msg})")

        if parse_ok and parsed_obj is not None:
            # Validate
            val_ok, val_errors, validated_data = validate_alert_payload(parsed_obj)
            report.append(f"• Validation    : {'✅ PASSED' if val_ok else '❌ REJECTED'}")
            if not val_ok:
                report.append(f"  - Rejection Errors: {val_errors}")
            report.append(f"• Parsed Dict Object:\n{json.dumps(validated_data, indent=2)}")
        report.append("=" * 80)

    return "\n".join(report)


def main():
    report = run_structured_output_demonstration()
    print(report)


if __name__ == "__main__":
    main()
