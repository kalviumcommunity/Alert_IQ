"""
Prompt Engineering & Comparison Framework for Alert_IQ
Evaluates baseline (vague) vs. constrained production system prompts.
"""
from typing import Dict, Any, List
from src.chat_client import ChatClient

# ==============================================================================
# Prompt Definitions (Task 1 & Task 2)
# ==============================================================================

# Variant A: Baseline Vague Prompt
PROMPT_VARIANT_A = {
    "name": "Variant A (Vague / Unconstrained)",
    "system_message": "You are a helpful assistant. Answer the staff member's questions.",
    "description": "Minimal baseline without role boundaries, scope limits, formatting rules, or safety fallbacks."
}

# Variant B: Production Constrained Prompt
PROMPT_VARIANT_B = {
    "name": "Variant B (Structured / Role, Scope & Constraints)",
    "system_message": (
        "You are Alert_IQ Incident Assistant, an internal AI specialist assisting technical support "
        "and on-call engineers with alert triage and incident response.\n\n"
        "ROLE & OBJECTIVE:\n"
        "Provide concise, actionable, and safety-critical guidance on incoming alerts and system triage.\n\n"
        "SCOPE:\n"
        "- DO explain triage procedures, severity classifications (CRITICAL, HIGH, MEDIUM, LOW), and escalation paths.\n"
        "- DO NOT speculate on unverified infrastructure status, claim direct database access, or recommend destructive actions.\n\n"
        "CONSTRAINTS:\n"
        "- Tone: Professional, calm, and direct.\n"
        "- Length: Maximum 2-3 concise paragraphs or prioritized bullet points.\n"
        "- Format: State the 'Action Priority' first, followed by 'Recommended Triage Steps'.\n"
        "- Safety Fallback: If the user request is ambiguous, unanswerable, or out of scope, reply: "
        "'Insufficient alert context to recommend action. Please refer to standard runbook at docs/RUNBOOK.md "
        "or escalate to the On-Call Lead.'"
    ),
    "description": "Production prompt defining explicit role, scope boundaries, output formatting, length constraints, and a strict safety fallback."
}

# Test Scenarios
TEST_SCENARIOS = [
    {
        "id": "scenario_1_high_severity",
        "title": "High Severity Database Latency Alert",
        "user_query": (
            "Alert ALT-9042 received: 'PostgreSQL primary read replica latency > 850ms, connection pool 94% full'. "
            "What should the on-call engineer do right now?"
        )
    },
    {
        "id": "scenario_2_out_of_scope_ambiguous",
        "title": "Ambiguous / Incomplete Query (Tests Safety Fallback)",
        "user_query": "Something is broken with the server. Fix it."
    }
]


def run_prompt_comparison(client: ChatClient = None) -> List[Dict[str, Any]]:
    """
    Executes test scenarios against both prompt variants and collects comparative results.
    """
    if client is None:
        client = ChatClient()

    results = []

    for scenario in TEST_SCENARIOS:
        scenario_record = {
            "scenario_id": scenario["id"],
            "title": scenario["title"],
            "user_query": scenario["user_query"],
            "runs": {}
        }

        for variant in [PROMPT_VARIANT_A, PROMPT_VARIANT_B]:
            messages = [
                {"role": "system", "content": variant["system_message"]},
                {"role": "user", "content": scenario["user_query"]}
            ]
            response = client.chat_completion(messages=messages)
            scenario_record["runs"][variant["name"]] = {
                "system_message": variant["system_message"],
                "response": response
            }

        results.append(scenario_record)

    return results


def main():
    print("=" * 75)
    print("🔬 Alert_IQ - Prompt Engineering & Behavior Comparison")
    print("=" * 75)

    client = ChatClient()
    comparison_data = run_prompt_comparison(client)

    for record in comparison_data:
        print(f"\n📌 Scenario: {record['title']}")
        print(f"❓ User Query:\n   \"{record['user_query']}\"\n")

        for variant_name, run_info in record["runs"].items():
            print(f"--- 🧪 {variant_name} ---")
            print(f"Response:\n{run_info['response']}\n")
        print("=" * 75)


if __name__ == "__main__":
    main()
