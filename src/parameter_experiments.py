"""
Model Parameter Tuning & Empirical Analysis Framework for Alert_IQ
Evaluates temperature, max_tokens, top_p, and stop sequences for grounded RAG generation.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chat_client import ChatClient
from src.token_counter import count_tokens

# Base benchmark query for parameter experiments
SYSTEM_PROMPT = "You are Alert_IQ Assistant. Provide factual triage steps for server alert ALT-4012 (Memory utilization > 95%)."
USER_PROMPT = "What are the immediate 3 steps to mitigate this memory pressure alert?"


def run_temperature_experiment(client: Optional[ChatClient] = None) -> List[Dict[str, Any]]:
    """
    Task 1: Varies temperature (0.0, 0.7, 1.4) to demonstrate stability vs. creativity.
    """
    if client is None:
        client = ChatClient()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]

    temp_configs = [
        {"temp": 0.0, "label": "Deterministic / Highly Grounded (0.0)", "desc": "Greedy sampling: outputs the single highest-probability token. 100% reproducible."},
        {"temp": 0.7, "label": "Balanced Default (0.7)", "desc": "Standard sampling balance between variety and coherence."},
        {"temp": 1.4, "label": "High Entropy / Creative (1.4)", "desc": "Flattens token probability distribution; produces novel, erratic phrasing."}
    ]

    results = []
    for cfg in temp_configs:
        response = client.chat_completion(messages=messages, temperature=cfg["temp"], max_tokens=150)
        tokens_used = count_tokens(response) if response else 0
        results.append({
            "temperature": cfg["temp"],
            "label": cfg["label"],
            "description": cfg["desc"],
            "response": response,
            "tokens_produced": tokens_used
        })
    return results


def run_max_tokens_experiment(client: Optional[ChatClient] = None) -> List[Dict[str, Any]]:
    """
    Task 2: Sets max_tokens (25, 80, 200) to demonstrate output truncation and budget enforcement.
    """
    if client is None:
        client = ChatClient()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]

    token_caps = [25, 80, 200]
    results = []

    for cap in token_caps:
        response = client.chat_completion(messages=messages, temperature=0.0, max_tokens=cap)
        toks = count_tokens(response) if response else 0
        results.append({
            "max_tokens_cap": cap,
            "response": response,
            "actual_tokens": toks,
            "is_truncated": (toks >= cap - 5)
        })
    return results


def run_additional_parameters_experiment(client: Optional[ChatClient] = None) -> Dict[str, Any]:
    """
    Task 3: Tests top_p (nucleus sampling) and stop sequences.
    """
    if client is None:
        client = ChatClient()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "List the steps numbered 1., 2., 3. Stop after step 2."}
    ]

    # Test top_p: 0.1 (strict nucleus) vs 0.95 (broad nucleus)
    resp_strict_p = client.chat_completion(messages=messages, temperature=0.7, top_p=0.1, max_tokens=150)
    resp_broad_p = client.chat_completion(messages=messages, temperature=0.7, top_p=0.95, max_tokens=150)

    # Test stop sequence: ["3.", "Step 3"] to halt generation before generating Step 3
    resp_stopped = client.chat_completion(messages=messages, temperature=0.0, max_tokens=150, stop=["3.", "Step 3"])

    return {
        "top_p_strict_0_1": resp_strict_p,
        "top_p_broad_0_95": resp_broad_p,
        "stop_sequence_response": resp_stopped
    }


def generate_experiment_report() -> str:
    """
    Executes all experiments and produces a formatted markdown/text report.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    client = ChatClient()

    lines = []
    lines.append("=" * 80)
    lines.append("🧪 Alert_IQ - LLM Parameter Tuning & Grounding Experiments Report")
    lines.append("=" * 80)

    # Task 1: Temperature Experiment
    lines.append("\n📌 TASK 1: TEMPERATURE VARIATION EXPERIMENT")
    lines.append("Prompt: 'What are the immediate 3 steps to mitigate this memory pressure alert?'")
    lines.append("-" * 80)
    temp_results = run_temperature_experiment(client)
    for r in temp_results:
        lines.append(f"\n🔬 [Temperature = {r['temperature']}] — {r['label']}")
        lines.append(f"• Behavior Note : {r['description']}")
        lines.append(f"• Tokens Used   : {r['tokens_produced']}")
        lines.append(f"• Model Output  :\n{r['response']}")

    # Task 2: Max Tokens Experiment
    lines.append("\n" + "=" * 80)
    lines.append("📌 TASK 2: MAX_TOKENS CAPPING EXPERIMENT")
    lines.append("-" * 80)
    cap_results = run_max_tokens_experiment(client)
    for r in cap_results:
        lines.append(f"\n🔬 [max_tokens = {r['max_tokens_cap']}]")
        lines.append(f"• Generated Tokens : {r['actual_tokens']} / {r['max_tokens_cap']} limit")
        lines.append(f"• Truncated Early  : {'Yes (Reached Limit)' if r['is_truncated'] else 'No (Completed naturally)'}")
        lines.append(f"• Model Output     :\n{r['response']}")

    # Task 3: Top-P & Stop Sequences Experiment
    lines.append("\n" + "=" * 80)
    lines.append("📌 TASK 3: TOP_P & STOP SEQUENCES EXPERIMENT")
    lines.append("-" * 80)
    param_results = run_additional_parameters_experiment(client)
    lines.append("\n🔬 [top_p = 0.1 (Strict Nucleus Sampling)]:")
    lines.append(f"{param_results['top_p_strict_0_1']}")
    lines.append("\n🔬 [top_p = 0.95 (Broad Nucleus Sampling)]:")
    lines.append(f"{param_results['top_p_broad_0_95']}")
    lines.append("\n🔬 [Stop Sequence: ['3.', 'Step 3'] (Halted prior to generating 3rd point)]:")
    lines.append(f"{param_results['stop_sequence_response']}")

    lines.append("\n" + "=" * 80)
    lines.append("🎯 SUMMARY OF RECOMMENDED GROUNDED SETTINGS")
    lines.append("=" * 80)
    lines.append("• Temperature : 0.0 - 0.2  (Maximizes factual consistency, prevents creative hallucination)")
    lines.append("• Max Tokens  : 250 - 400  (Provides adequate room for triage steps while bounding costs)")
    lines.append("• Top_P       : 0.1 - 0.2  (Restricts token selection to high-confidence mass)")
    lines.append("• Stop Tokens : ['---', 'User:'] (Prevents prompt bleed and multi-turn boundary leakage)")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    report = generate_experiment_report()
    print(report)


if __name__ == "__main__":
    main()
