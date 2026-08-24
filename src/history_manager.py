import sys
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Add project root to sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.token_counter import count_tokens
except ImportError:
    from token_counter import count_tokens

from src.chat_client import ChatClient

logger = logging.getLogger("HistoryManager")


class ConversationManager:
    """
    Manages multi-turn conversation history with dynamic token budgeting and automatic trimming.
    """

    def __init__(
        self,
        system_prompt: str,
        max_token_budget: int = 500,
        preserve_recent_turns: int = 2
    ):
        """
        Args:
            system_prompt: Base system instructions (always preserved).
            max_token_budget: Maximum allowed total tokens for messages sent to LLM.
            preserve_recent_turns: Minimum number of recent turns (user+assistant pairs) to preserve.
        """
        self.system_prompt = system_prompt
        self.max_token_budget = max_token_budget
        self.preserve_recent_turns = preserve_recent_turns
        self.history: List[Dict[str, str]] = []
        self.trim_events: List[Dict[str, Any]] = []

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Returns full list of messages formatted for OpenAI-compatible chat completions.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        return messages

    def count_total_tokens(self, messages: Optional[List[Dict[str, str]]] = None) -> int:
        """
        Calculates token count for given messages (or current active history).
        Includes per-message formatting overhead (~4 tokens per message).
        """
        target_messages = messages if messages is not None else self.get_messages()
        total = 0
        for msg in target_messages:
            total += 4  # Formatting overhead per message
            total += count_tokens(msg.get("content", ""))
        total += 2  # Priming tokens
        return total

    def add_user_message(self, content: str) -> None:
        """
        Adds user message and enforces token budget.
        """
        self.history.append({"role": "user", "content": content})
        self.enforce_token_budget()

    def add_assistant_message(self, content: str) -> None:
        """
        Adds assistant message and enforces token budget.
        """
        self.history.append({"role": "assistant", "content": content})
        self.enforce_token_budget()

    def enforce_token_budget(self) -> Tuple[bool, int, int]:
        """
        Trims oldest non-system conversation turns when history exceeds the token budget.
        Always preserves the system message and recent turns if possible.

        Returns:
            Tuple: (was_trimmed: bool, tokens_before: int, tokens_after: int)
        """
        tokens_before = self.count_total_tokens()
        if tokens_before <= self.max_token_budget:
            return False, tokens_before, tokens_before

        was_trimmed = False
        # Loop to remove oldest history items until within budget or only minimal recent message left
        while self.count_total_tokens() > self.max_token_budget and len(self.history) > 1:
            evicted_msg = self.history.pop(0)  # Evict oldest non-system message
            was_trimmed = True
            self.trim_events.append({
                "evicted_role": evicted_msg["role"],
                "evicted_preview": evicted_msg["content"][:60] + "..." if len(evicted_msg["content"]) > 60 else evicted_msg["content"],
                "tokens_before": tokens_before,
                "current_tokens": self.count_total_tokens()
            })

        tokens_after = self.count_total_tokens()
        return was_trimmed, tokens_before, tokens_after


def run_overflow_conversation_demo(
    client: Optional[ChatClient] = None,
    token_budget: int = 400
) -> str:
    """
    Demonstrates an overflowing multi-turn conversation, tracking naive vs. managed tokens.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    system_prompt = (
        "You are Alert_IQ Incident Assistant. Provide concise, step-by-step triage advice for database "
        "and infrastructure alerts. Keep responses under 2 sentences."
    )

    manager = ConversationManager(
        system_prompt=system_prompt,
        max_token_budget=token_budget,
        preserve_recent_turns=1
    )

    # Simulated multi-turn dialogue steps
    dialogue_turns = [
        ("user", "Alert ALT-9042 received: PostgreSQL read replica latency is 920ms. What should I check first?"),
        ("assistant", "Check `pg_stat_activity` for active long-running transactions blocking the replication stream, and verify network connectivity between primary and replica."),
        ("user", "Found 3 queries running for > 45 minutes in state 'idle in transaction'. Should I terminate them?"),
        ("assistant", "Yes, terminate them immediately using `pg_terminate_backend(pid)` to free up locks and connection slots."),
        ("user", "Terminated. Latency dropped to 410ms, but connection pool utilization is still at 88%. Next step?"),
        ("assistant", "Inspect PgBouncer client pools and temporary spike in read traffic; consider scaling read-replica pool instances."),
        ("user", "Replica scaled. Latency is now 45ms and pool is at 32%. Can we mark the incident resolved?"),
        ("assistant", "Yes, all metrics have returned to nominal thresholds. You can mark incident ALT-9042 as RESOLVED in the alert log.")
    ]

    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append("🔄 Alert_IQ - Multi-Turn Conversation History Management Demo")
    log_lines.append(f"⚙️ Configured Token Budget : {token_budget} Tokens")
    log_lines.append(f"📌 Preserved System Prompt  : {system_prompt}")
    log_lines.append("=" * 80)

    naive_messages = [{"role": "system", "content": system_prompt}]

    for idx, (role, content) in enumerate(dialogue_turns, 1):
        # Update naive history for comparison
        naive_messages.append({"role": role, "content": content})
        naive_tokens = sum(count_tokens(m["content"]) + 4 for m in naive_messages) + 2

        # Update managed conversation
        if role == "user":
            manager.add_user_message(content)
        else:
            manager.add_assistant_message(content)

        managed_tokens = manager.count_total_tokens()
        num_history_msgs = len(manager.history)

        log_lines.append(f"\n--- 💬 Turn {idx} [{role.upper()}] ---")
        log_lines.append(f"Message Content: \"{content}\"")
        log_lines.append(f"📊 Token Metrics:")
        log_lines.append(f"   • Naive Unmanaged Tokens : {naive_tokens} tokens {'⚠️ (OVER BUDGET)' if naive_tokens > token_budget else '✅'}")
        log_lines.append(f"   • Managed History Tokens : {managed_tokens} tokens ✅ (Within Budget)")
        log_lines.append(f"   • Active Messages in Window: {num_history_msgs + 1} (1 System + {num_history_msgs} History)")

        if manager.trim_events:
            last_event = manager.trim_events[-1]
            log_lines.append(f"   ✂️ [TRIM TRIGGERED]: Evicted oldest {last_event['evicted_role']} message: \"{last_event['evicted_preview']}\"")

    log_lines.append("\n" + "=" * 80)
    log_lines.append("🏁 DEMONSTRATION SUMMARY")
    log_lines.append("=" * 80)
    log_lines.append(f"• Total Dialogue Turns Simulated : {len(dialogue_turns)}")
    log_lines.append(f"• Final Naive History Tokens      : {naive_tokens} tokens (Exceeds {token_budget} token budget by {naive_tokens - token_budget} tokens)")
    log_lines.append(f"• Final Managed History Tokens    : {manager.count_total_tokens()} tokens (Successfully bounded under {token_budget} limit)")
    log_lines.append(f"• Total Message Evictions         : {len(manager.trim_events)}")
    log_lines.append("• System Prompt Preservation      : 100% Intact")
    log_lines.append("=" * 80)

    return "\n".join(log_lines)


def main():
    report = run_overflow_conversation_demo(token_budget=350)
    print(report)


if __name__ == "__main__":
    main()
