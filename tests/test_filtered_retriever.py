"""
Unit tests for Alert_IQ Filtered & Hybrid Semantic Retriever.
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
from src.filtered_retriever import (
    FilteredRetriever,
    HybridRetrievedChunk,
    calculate_keyword_score,
    run_filtered_retrieval_demo,
)


class TestFilteredRetriever:
    """Test suite covering metadata filtering, hybrid keyword scoring, and precision comparison."""

    @pytest.fixture
    def indexed_store(self, tmp_path):
        """Sets up an isolated vector store populated with the standard corpus."""
        store = VectorStore(path=str(tmp_path), collection_name="test_filtered_collection")
        indexer = CorpusIndexer(vector_store=store, dimension=768)
        indexer.run_indexing(reset_collection=True)
        return store

    def test_calculate_keyword_score(self):
        """Task 3: Keyword scorer gives high score for exact matches and 0 for unrelated text."""
        query = "DB-RB-402 database replica latency"
        matching_text = "# Database Replica Latency Triage Runbook (DB-RB-402)\nEmergency mitigation steps."
        unrelated_text = "The cafeteria menu has hot soup today."

        score_match = calculate_keyword_score(query, matching_text)
        score_unrelated = calculate_keyword_score(query, unrelated_text)

        assert score_match > 0.5
        assert score_unrelated == 0.0
        assert calculate_keyword_score("", matching_text) == 0.0
        assert calculate_keyword_score(query, "") == 0.0

    def test_metadata_filtering_restricts_corpus_subset(self, indexed_store):
        """Task 1: Metadata filter restricts retrieval strictly to the filtered document."""
        retriever = FilteredRetriever(vector_store=indexed_store, dimension=768)
        query = "What is the policy for incident escalation?"
        
        # Scoped filter
        filtered_results = retriever.retrieve_filtered(
            query=query,
            top_k=4,
            metadata_filter={"source_document": "incident_policy.txt"}
        )

        assert len(filtered_results) == 1
        assert filtered_results[0].metadata["source_document"] == "incident_policy.txt"
        assert "ALERT_IQ ON-CALL ESCALATION POLICY" in filtered_results[0].text

    def test_compare_filtered_and_unfiltered_results(self, indexed_store):
        """Task 2: Comparing filtered vs unfiltered shows eliminated out-of-scope chunks."""
        retriever = FilteredRetriever(vector_store=indexed_store, dimension=768)
        query = "Escalation procedures and SLA guidelines"
        filter_dict = {"source_document": "incident_policy.txt"}

        comparison = retriever.compare_filtered_vs_unfiltered(
            query=query,
            metadata_filter=filter_dict,
            top_k=4
        )

        assert comparison["unfiltered_count"] == 4
        assert comparison["filtered_count"] == 1
        assert comparison["out_of_scope_eliminated_count"] == 3
        assert len(comparison["out_of_scope_ids"]) == 3

    def test_hybrid_search_boosts_exact_terms(self, indexed_store):
        """Task 3 & 4: Hybrid search elevates exact runbook ID to rank #1."""
        retriever = FilteredRetriever(vector_store=indexed_store, dimension=768)
        query = "DB-RB-402 triage mitigation"

        hybrid_results = retriever.retrieve_hybrid(query=query, top_k=3, alpha=0.4)

        assert len(hybrid_results) >= 1
        top_chunk = hybrid_results[0]
        assert isinstance(top_chunk, HybridRetrievedChunk)
        assert top_chunk.rank == 1
        assert top_chunk.keyword_score > 0.0
        assert "runbook_database_lag" in top_chunk.id
        assert "DB-RB-402" in top_chunk.text

    def test_hybrid_invalid_alpha(self, indexed_store):
        """Edge case: alpha outside [0.0, 1.0] raises ValueError."""
        retriever = FilteredRetriever(vector_store=indexed_store, dimension=768)
        with pytest.raises(ValueError, match="alpha must be between 0.0 and 1.0"):
            retriever.retrieve_hybrid("test", alpha=1.5)
        with pytest.raises(ValueError, match="alpha must be between 0.0 and 1.0"):
            retriever.retrieve_hybrid("test", alpha=-0.1)

    def test_filtered_retrieval_demo_execution(self, tmp_path):
        """Task 5: run_filtered_retrieval_demo generates full formatted comparison report."""
        report = run_filtered_retrieval_demo(
            persist_path=str(tmp_path),
            collection_name="demo_filtered_coll"
        )
        assert "Alert_IQ - Filtered & Hybrid Semantic Retrieval Demonstration" in report
        assert "[DEMO 1] METADATA FILTERING VS. UNFILTERED SEARCH" in report
        assert "[DEMO 2] HYBRID (VECTOR + KEYWORD) RETRIEVAL" in report
        assert "[DEMO 3] COMBINED METADATA FILTERING + HYBRID EXACT KEY RETRIEVAL" in report
        assert "FINAL EVALUATION & AUDIT OUTCOME" in report
