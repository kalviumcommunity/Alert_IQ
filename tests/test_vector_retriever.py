"""
Unit tests for Alert_IQ Vector Retriever, top-k similarity search, metadata preservation, and k variations.
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
from src.vector_retriever import VectorRetriever, RetrievedChunk, run_retrieval_demo


class TestVectorRetriever:
    """Test suite for semantic query retrieval against vector store."""

    @pytest.fixture
    def indexed_store(self, tmp_path):
        """Sets up an isolated vector store populated with the standard corpus."""
        store = VectorStore(path=str(tmp_path), collection_name="test_retrieval_collection")
        indexer = CorpusIndexer(vector_store=store, dimension=768)
        indexer.run_indexing(reset_collection=True)
        return store

    def test_query_embedding_dimension(self, indexed_store):
        """Task 1: Query embedding matches expected dimension."""
        retriever = VectorRetriever(vector_store=indexed_store, dimension=768)
        vec = retriever.embed_query("Database latency spike")
        assert len(vec) == 768
        assert all(isinstance(x, float) for x in vec)

        with pytest.raises(ValueError, match="cannot be empty"):
            retriever.embed_query("   ")

    def test_top_k_retrieval_with_scores_and_metadata(self, indexed_store):
        """Task 2 & 3: Top-k search returns ordered chunks with scores and metadata."""
        retriever = VectorRetriever(vector_store=indexed_store, dimension=768)
        query = "How do we handle database replication lag?"
        results = retriever.retrieve(query=query, top_k=2)

        assert len(results) == 2
        for idx, chunk in enumerate(results, 1):
            assert isinstance(chunk, RetrievedChunk)
            assert chunk.rank == idx
            assert 0.0 <= chunk.score <= 1.0
            assert len(chunk.text) > 0
            assert "source_document" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert chunk.id is not None

        # Verify scores are sorted in descending order
        assert results[0].score >= results[1].score

    def test_demonstrate_changing_k(self, indexed_store):
        """Task 4: Changing k adjusts the number of returned chunks dynamically."""
        retriever = VectorRetriever(vector_store=indexed_store, dimension=768)
        query = "Escalation SLA thresholds for on-call engineers"

        k_results = retriever.compare_k_values(query=query, k_values=[1, 2, 3, 4])

        assert len(k_results[1]) == 1
        assert len(k_results[2]) == 2
        assert len(k_results[3]) == 3
        assert len(k_results[4]) == 4

        # Verify that the top result in k=1 is identical to the top result in k=4
        assert k_results[1][0].id == k_results[4][0].id
        assert k_results[1][0].score == k_results[4][0].score

    def test_invalid_k_and_empty_collection(self, tmp_path):
        """Edge case: Invalid k values raise errors; empty store returns empty list."""
        empty_store = VectorStore(path=str(tmp_path), collection_name="empty_test_collection")
        retriever = VectorRetriever(vector_store=empty_store, dimension=768)

        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.retrieve("sample query", top_k=0)

        # Empty store returns empty list safely
        assert retriever.retrieve("sample query", top_k=3) == []

    def test_retrieval_demo_execution(self, tmp_path):
        """Task 5: run_retrieval_demo executes and returns formatted summary."""
        output = run_retrieval_demo(persist_path=str(tmp_path), collection_name="demo_retrieval_coll")
        assert "Alert_IQ - Semantic Retrieval & Top-K Query Demonstration Report" in output
        assert "Query 1: Database Replica Latency & Triage" in output
        assert "Results with k = 1" in output
        assert "Results with k = 2" in output
        assert "K-VARIATION ANALYSIS" in output
        assert "RETRIEVAL VALIDATION SUMMARY" in output
