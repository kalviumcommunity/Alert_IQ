"""Retrieval-strength guardrails that refuse unsupported RAG questions before generation."""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence

MIN_TOP_SCORE = 0.72
MIN_SUPPORTING_CHUNKS = 1
REFUSAL_MESSAGE = "I don't have enough reliable context to answer that."


@dataclass(frozen=True)
class GuardrailDecision:
    """Decision made from retrieval evidence before an answer is generated."""

    allowed: bool
    status: str
    reason: str
    supporting_chunks: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "status": self.status,
            "reason": self.reason,
            "supporting_chunk_count": len(self.supporting_chunks),
        }


def _score(chunk: Any) -> float:
    """Read a retrieval score from a dict-like or object-like chunk."""
    if isinstance(chunk, Mapping):
        value = chunk.get("score", 0.0)
    else:
        value = getattr(chunk, "score", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def retrieval_is_strong(
    chunks: Sequence[Any],
    min_top_score: float = MIN_TOP_SCORE,
    min_supporting_chunks: int = MIN_SUPPORTING_CHUNKS,
) -> bool:
    """Return True only when enough retrieved chunks clear the relevance threshold."""
    if min_top_score < 0:
        raise ValueError("min_top_score must be >= 0")
    if min_supporting_chunks < 1:
        raise ValueError("min_supporting_chunks must be >= 1")

    strong_chunks = [chunk for chunk in chunks if _score(chunk) >= min_top_score]
    return len(strong_chunks) >= min_supporting_chunks


def evaluate_retrieval(
    chunks: Sequence[Any],
    min_top_score: float = MIN_TOP_SCORE,
    min_supporting_chunks: int = MIN_SUPPORTING_CHUNKS,
) -> GuardrailDecision:
    """Explain whether retrieval is strong enough to permit answer generation."""
    strong_chunks = [chunk for chunk in chunks if _score(chunk) >= min_top_score]

    if not chunks:
        return GuardrailDecision(
            allowed=False,
            status="refused_no_context",
            reason="Retrieval returned no chunks.",
        )

    if len(strong_chunks) < min_supporting_chunks:
        top_score = max((_score(chunk) for chunk in chunks), default=0.0)
        return GuardrailDecision(
            allowed=False,
            status="refused_weak_context",
            reason=(
                f"Only {len(strong_chunks)} chunk(s) met the relevance threshold "
                f"of {min_top_score:.2f}; top score was {top_score:.4f}."
            ),
        )

    return GuardrailDecision(
        allowed=True,
        status="answered",
        reason=(
            f"{len(strong_chunks)} chunk(s) met the relevance threshold "
            f"of {min_top_score:.2f}."
        ),
        supporting_chunks=strong_chunks,
    )


def guarded_answer(
    question: str,
    retrieve: Callable[[str, int], Sequence[Any]],
    generate: Callable[[str, Sequence[Any]], str],
    k: int = 4,
    min_top_score: float = MIN_TOP_SCORE,
    min_supporting_chunks: int = MIN_SUPPORTING_CHUNKS,
) -> Dict[str, Any]:
    """Retrieve, apply the guardrail, and generate only when evidence is strong."""
    if not question or not question.strip():
        raise ValueError("question cannot be empty")
    if k < 1:
        raise ValueError("k must be >= 1")

    chunks = list(retrieve(question, k))
    decision = evaluate_retrieval(
        chunks,
        min_top_score=min_top_score,
        min_supporting_chunks=min_supporting_chunks,
    )

    if not decision.allowed:
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "status": decision.status,
            "reason": decision.reason,
        }

    answer = generate(question, decision.supporting_chunks)
    return {
        "answer": answer,
        "sources": decision.supporting_chunks,
        "status": "answered",
        "reason": decision.reason,
    }
