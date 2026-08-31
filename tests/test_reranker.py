"""
Unit tests for Alert_IQ Two-Stage Retrieval & Contextual Re-Ranking Engine.
"""
import sys
import pytest
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.vector_store import VectorStore
from src.index_corpus import CorpusIndexer
from src.reranker import (
    ReRanker,
    ReRankedChunk,
    TwoStageRetriever,
    run_reranking_demo,
)


class TestReRanker:
    """Test suite covering candidate pool expansion, re-ranking scoring, and before/after comparisons."""

    @pytest.fixture
    def indexed_store(self, tmp_path):
        """Sets up an isolated VectorStore pre-populated with corpus embeddings."""
        store = VectorStore(path=str(tmp_path), collection_name="test_rerank_collection")
        indexer = CorpusIndexer(vector_store=store, dimension=768)
        indexer.run_indexing(reset_collection=True)
        return store

    def test_score_candidate_exact_id_boost(self):
        """Task 2: Re-ranker identifies exact runbook code and gives significant score boost."""
        query = "How to triage DB-RB-402 database replica latency?"
        matching_text = "# Database Replica Latency Triage Runbook (DB-RB-402)\nEmergency mitigation steps."
        unrelated_text = "Cluster infrastructure health report export."

        score_match, reason_match = ReRanker.score_candidate(query, matching_text)
        score_unrelated, reason_unrelated = ReRanker.score_candidate(query, unrelated_text)

        assert score_match > 0.5
        assert score_unrelated < 0.2
        assert "Exact ID match" in reason_match
        assert ReRanker.score_candidate("", matching_text)[0] == 0.0

    def test_two_stage_candidate_retrieval_and_reranking(self, indexed_store):
        """Task 1 & 2: Pipeline retrieves candidate pool and trims to top-k after re-ranking."""
        pipeline = TwoStageRetriever(
            vector_store=indexed_store,
            candidate_count=4,
            final_top_k=2
        )
        query = "DB-RB-402 database replica latency"
        candidates, reranked = pipeline.retrieve_and_rerank(query, candidate_count=4, top_k=2)

        assert len(candidates) == 4  # Stage 1 candidate pool
        assert len(reranked) == 2    # Stage 2 trimmed top-k

        # Assert top chunk after re-ranking is the target runbook
        assert isinstance(reranked[0], ReRankedChunk)
        assert reranked[0].rerank_rank == 1
        assert "runbook_database_lag" in reranked[0].id
        assert reranked[0].rerank_score >= reranked[1].rerank_score

    def test_compare_before_and_after_tracking(self, indexed_store):
        """Task 3 & 4: Before and after comparison tracks rank deltas and score progressions."""
        pipeline = TwoStageRetriever(
            vector_store=indexed_store,
            candidate_count=4,
            final_top_k=3
        )
        query = "What is the critical_response_time_seconds threshold in the metrics schema?"
        comp = pipeline.compare_before_and_after(query, candidate_count=4, top_k=3)

        assert comp["candidate_count"] == 4
        assert comp["final_top_k"] == 3
        assert len(comp["initial_candidates"]) == 4
        assert len(comp["reranked_results"]) == 3

        # Re-ranked top result should be metrics_schema.json
        top_result = comp["reranked_results"][0]
        assert "metrics_schema" in top_result.id
        assert top_result.rerank_rank == 1

    def test_rerank_demo_execution(self, tmp_path):
        """Task 5: run_reranking_demo executes cleanly and produces formatted report."""
        report = run_reranking_demo(
            persist_path=str(tmp_path),
            collection_name="demo_rerank_coll"
        )
        assert "Alert_IQ - Two-Stage Candidate Retrieval & Re-Ranking Demonstration" in report
        assert "[STAGE 1] INITIAL CANDIDATES RETRIEVED" in report
        assert "[STAGE 2] RE-RANKED RESULTS" in report
        assert "BEFORE-AND-AFTER IMPACT ANALYSIS" in report
        assert "FINAL RE-RANKING PIPELINE VALIDATION" in report
