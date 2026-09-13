"""Conversational RAG orchestration for follow-up questions.

The module keeps a bounded conversation history, rewrites follow-ups into
standalone retrieval queries, retrieves with that rewritten query, and then
passes the retrieved context to an answer generator.

The rewrite step is deliberately deterministic so the feature can be tested
without API credentials. A production implementation can replace the rewrite
function with an LLM while keeping the same interface and history budget.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

from src.history_manager import ConversationManager


@dataclass
class ConversationalTurn:
    """One user/assistant exchange retained for conversational context."""
    user: str
    assistant: str


@dataclass
class ConversationalRAGResult:
    """Traceable output from a conversational RAG turn."""
    question: str
    rewritten_query: str
    retrieved_context: List[Any]
    answer: str
    history_messages: List[Dict[str, str]] = field(default_factory=list)


class ConversationalRAG:
    """Add history-aware query rewriting on top of the existing RAG stages."""

    def __init__(self, system_prompt: str, max_token_budget: int = 500):
        if max_token_budget <= 0:
            raise ValueError("max_token_budget must be positive")
        self.history = ConversationManager(
            system_prompt=system_prompt,
            max_token_budget=max_token_budget,
            preserve_recent_turns=2,
        )

    def add_turn(self, user: str, assistant: str) -> None:
        """Track both sides of a completed conversation turn."""
        if not user.strip() or not assistant.strip():
            raise ValueError("user and assistant messages must be non-empty")
        self.history.add_user_message(user.strip())
        self.history.add_assistant_message(assistant.strip())

    def messages(self) -> List[Dict[str, str]]:
        """Return the bounded history including the preserved system prompt."""
        return self.history.get_messages()

    def rewrite_follow_up(self, question: str) -> str:
        """Turn a follow-up into a standalone retrieval query.

        This offline rewrite uses the most recent user turn and its assistant
        answer as conversational context. Explicitly standalone questions are
        returned unchanged.
        """
        question = question.strip()
        if not question:
            raise ValueError("question must be non-empty")

        if not self.history.history:
            return question

        last_user = next(
            (m["content"] for m in reversed(self.history.history) if m["role"] == "user"),
            "",
        )
        last_assistant = next(
            (m["content"] for m in reversed(self.history.history) if m["role"] == "assistant"),
            "",
        )

        lowered = question.lower()
        follow_up_markers = (
            "it ", "it?", "that ", "that?", "this ", "this?",
            "they ", "they?", "them ", "them?", "those ", "those?",
            "what about", "how about", "and then", "next step", "why",
        )
        is_follow_up = any(marker in lowered for marker in follow_up_markers)
        if not is_follow_up:
            return question

        # Preserve the original subject plus the new intent. The previous
        # assistant answer is included only as a compact grounding hint.
        subject = last_user.rstrip("?. ")
        answer_hint = last_assistant.split(".")[0].strip()
        if answer_hint:
            return f"{subject}. Follow-up: {question} Context hint: {answer_hint}."
        return f"{subject}. Follow-up: {question}."

    def run(
        self,
        question: str,
        retrieve: Callable[[str], Sequence[Any]],
        generate: Callable[[str, Sequence[Any]], str],
    ) -> ConversationalRAGResult:
        """Rewrite, retrieve, generate, then append the completed turn."""
        rewritten = self.rewrite_follow_up(question)
        context = list(retrieve(rewritten))
        answer = generate(question, context)
        self.add_turn(question, answer)
        return ConversationalRAGResult(
            question=question,
            rewritten_query=rewritten,
            retrieved_context=context,
            answer=answer,
            history_messages=self.messages(),
        )


def run_conversational_demo() -> str:
    """Return a deterministic two-turn dialogue suitable for review/demo."""
    system = "You are Alert_IQ Incident Assistant. Answer only from verified incident context."
    rag = ConversationalRAG(system_prompt=system, max_token_budget=350)

    corpus = {
        "replica": "DB-RB-402: For PostgreSQL replica lag, check replication slots and long-running transactions first; after termination, verify replica latency and connection pool utilization.",
        "pool": "DB-RB-402: If latency remains elevated after transaction cleanup, inspect PgBouncer pool utilization and scale read-replica capacity when pool utilization stays high.",
    }

    def retrieve(query: str) -> Sequence[str]:
        q = query.lower()
        if "pool" in q or "next step" in q or "connection" in q:
            return [corpus["pool"]]
        return [corpus["replica"]]

    def generate(question: str, context: Sequence[str]) -> str:
        return f"Based on verified context: {context[0]}"

    first = rag.run(
        "PostgreSQL replica latency is 920ms. What should I check first?",
        retrieve,
        generate,
    )
    second = rag.run(
        "Latency dropped to 410ms, but connection pool utilization is 88%. What's the next step?",
        retrieve,
        generate,
    )

    lines = [
        "=" * 80,
        "Alert_IQ - Conversational RAG Demonstration",
        "=" * 80,
        "TURN 1",
        f"User: {first.question}",
        f"Rewritten retrieval query: {first.rewritten_query}",
        f"Retrieved context: {first.retrieved_context[0]}",
        f"Assistant: {first.answer}",
        "",
        "TURN 2 - FOLLOW-UP",
        f"User: {second.question}",
        f"Rewritten retrieval query: {second.rewritten_query}",
        f"Retrieved context: {second.retrieved_context[0]}",
        f"Assistant: {second.answer}",
        "",
        f"Bounded history messages: {len(rag.messages())}",
        f"Token budget: {rag.history.max_token_budget}",
        "History is trimmed by ConversationManager when the budget is exceeded.",
        "=" * 80,
    ]
    return "\n".join(lines)
