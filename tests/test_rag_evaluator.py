from src.rag_evaluator import (
    score_correctness,
    score_grounding,
    score_citations,
    summarize,
    EvaluationResult,
)


def test_correctness_rewards_expected_answer_overlap():
    assert score_correctness("Check replication slots and long-running transactions first.", "Check replication slots and long-running transactions first.") == 1.0


def test_grounding_uses_retrieved_context():
    assert score_grounding("Check replication slots first.", ["Check replication slots and long-running transactions first."]) > 0.5


def test_citations_require_expected_supporting_source():
    assert score_citations(["runbook_database_replica_lag.md"], ["runbook_database_replica_lag.md"]) == 1.0
    assert score_citations(["wrong.md"], ["runbook_database_replica_lag.md"]) == 0.0


def test_empty_expected_sources_require_no_citation():
    assert score_citations([], []) == 1.0
    assert score_citations(["fabricated.md"], []) == 0.0


def test_summary_reports_weak_dimensions_and_failures():
    result = EvaluationResult("E", "q", "a", 0.5, 1.0, 0.0, [], ["expected.md"], "low correctness; citation mismatch")
    summary = summarize([result])
    assert summary["correctness"] == 0.5
    assert summary["grounding"] == 1.0
    assert summary["citation_quality"] == 0.0
    assert summary["overall_quality"] == 0.5
    assert summary["failures"][0]["id"] == "E"
