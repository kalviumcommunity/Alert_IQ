"""Tests for retrieval-strength hallucination guardrails."""
from dataclasses import dataclass

import pytest

from src.hallucination_guardrails import (
    MIN_TOP_SCORE,
    REFUSAL_MESSAGE,
    evaluate_retrieval,
    guarded_answer,
    retrieval_is_strong,
)


@dataclass
class Chunk:
    score: float
    source: str = "incident_policy.md"


def test_empty_retrieval_is_weak():
    assert retrieval_is_strong([]) is False
    decision = evaluate_retrieval([])
    assert decision.allowed is False
    assert decision.status == "refused_no_context"


def test_low_score_retrieval_is_refused():
    chunks = [Chunk(score=0.41), Chunk(score=0.68)]
    decision = evaluate_retrieval(chunks)
    assert decision.allowed is False
    assert decision.status == "refused_weak_context"
    assert "0.72" in decision.reason


def test_strong_chunk_allows_generation():
    chunks = [Chunk(score=0.81), Chunk(score=0.55)]
    decision = evaluate_retrieval(chunks)
    assert decision.allowed is True
    assert decision.status == "answered"
    assert len(decision.supporting_chunks) == 1


def test_guarded_answer_refuses_before_generation():
    calls = {"generate": 0}

    def retrieve(_question, _k):
        return [Chunk(score=0.50)]

    def generate(_question, _chunks):
        calls["generate"] += 1
        return "This must never be generated."

    result = guarded_answer("Unsupported question", retrieve, generate)

    assert result["status"] == "refused_weak_context"
    assert result["answer"] == REFUSAL_MESSAGE
    assert result["sources"] == []
    assert calls["generate"] == 0


def test_guarded_answer_preserves_grounded_answer():
    calls = {"generate": 0}
    strong_chunk = Chunk(score=0.91, source="project_submission.md")

    def retrieve(_question, _k):
        return [strong_chunk]

    def generate(question, chunks):
        calls["generate"] += 1
        assert question == "What evidence is required for submission?"
        assert chunks == [strong_chunk]
        return "The submission requires the evidence described in the verified project guide [1]."

    result = guarded_answer(
        "What evidence is required for submission?",
        retrieve,
        generate,
        k=4,
    )

    assert result["status"] == "answered"
    assert result["answer"].endswith("[1].")
    assert result["sources"] == [strong_chunk]
    assert calls["generate"] == 1


def test_threshold_can_be_tuned_from_evaluation():
    chunks = [Chunk(score=0.70)]
    assert retrieval_is_strong(chunks, min_top_score=0.65) is True
    assert retrieval_is_strong(chunks, min_top_score=0.75) is False


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        retrieval_is_strong([], min_top_score=-0.1)
    with pytest.raises(ValueError):
        retrieval_is_strong([], min_supporting_chunks=0)
