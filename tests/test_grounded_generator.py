"""
Unit tests for Alert_IQ Grounded Answer Generation & Source Accuracy Verification Engine.
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
from src.rag_pipeline import RAGContextChunk
from src.grounded_generator import (
    GroundedGenerator,
    SourceAccuracyCheck,
    GroundedAnswerResult,
    GroundingComparisonResult,
    FALLBACK_MESSAGE,
    run_grounded_generation_demo,
)


class TestGroundedGenerator:
    """Test suite covering grounded answer generation, source accuracy audits, and fallback behavior."""

    @pytest.fixture
    def indexed_store(self, tmp_path):
        """Sets up an isolated VectorStore with indexed corpus documents."""
        store = VectorStore(path=str(tmp_path), collection_name="test_grounded_gen_collection")
        indexer = CorpusIndexer(vector_store=store, dimension=768)
        indexer.run_indexing(reset_collection=True)
        return store

    def test_validate_source_accuracy_checks(self):
        """Task 2: Source accuracy validator verifies grounded claims and flags unsupported tokens."""
        generator = GroundedGenerator()
        chunks = [
            RAGContextChunk(
                id="c1",
                rank=1,
                score=0.9,
                source_document="runbook_database_lag.md",
                chunk_index=0,
                text="# DB-RB-402 Database Replica Latency Triage\nExecute pg_terminate_backend(pid) for queries running over 30 minutes."
            )
        ]

        # Factual answer containing terms present in chunk
        factual_answer = "Execute pg_terminate_backend for queries over 30 minutes referencing DB-RB-402 [1]."
        audit_pass = generator.validate_source_accuracy(factual_answer, chunks)

        assert audit_pass.is_accurate is True
        assert audit_pass.accuracy_score == 1.0
        assert len(audit_pass.grounded_facts) > 0
        assert len(audit_pass.unsupported_claims) == 0

        # Answer with ungrounded/hallucinated command
        hallucinated_answer = "Execute pg_terminate_backend for queries over 30 minutes and run pg_kill_everything now."
        audit_fail = generator.validate_source_accuracy(hallucinated_answer, chunks)

        assert audit_fail.is_accurate is False
        assert "pg_kill_everything" in audit_fail.unsupported_claims

    def test_grounded_generation_with_citations(self, indexed_store):
        """Task 1 & 2: Grounded generator synthesizes answer from context with citations."""
        generator = GroundedGenerator(vector_store=indexed_store, dimension=768)
        query = "How to triage DB-RB-402 database replica latency?"

        result = generator.generate(query=query, top_k=2)

        assert isinstance(result, GroundedAnswerResult)
        assert result.is_fallback is False
        assert len(result.retrieved_chunks) > 0
        assert "DB-RB-402" in result.answer or "pg_terminate_backend" in result.answer
        assert len(result.citations_used) > 0
        assert result.source_accuracy.is_accurate is True

    def test_missing_context_fallback(self, indexed_store):
        """Task 3: Out-of-domain query with no supporting context triggers safety fallback."""
        generator = GroundedGenerator(vector_store=indexed_store, dimension=768)
        out_of_domain_query = "What is the Stripe payment webhook secret HMAC key rotation schedule?"

        # High threshold triggers fallback
        result = generator.generate(query=out_of_domain_query, min_relevance_threshold=0.85)

        assert isinstance(result, GroundedAnswerResult)
        assert result.is_fallback is True
        assert result.answer == FALLBACK_MESSAGE
        assert result.source_accuracy.is_accurate is True

    def test_compare_with_and_without_retrieval(self, indexed_store):
        """Task 4: compare_with_and_without_retrieval produces structured side-by-side comparison."""
        generator = GroundedGenerator(vector_store=indexed_store, dimension=768)
        query = "What are the mitigation steps for DB-RB-402 replica latency?"

        comp = generator.compare_with_and_without_retrieval(query=query)

        assert isinstance(comp, GroundingComparisonResult)
        assert len(comp.with_retrieval_answer) > 20
        assert len(comp.without_retrieval_answer) > 20
        assert len(comp.grounded_sources) > 0
        assert len(comp.key_differences) > 0

    def test_demo_execution(self, tmp_path):
        """Task 5: run_grounded_generation_demo executes cleanly and produces formatted report."""
        report = run_grounded_generation_demo(
            persist_path=str(tmp_path),
            collection_name="demo_grounded_coll"
        )
        assert "Alert_IQ - Grounded Answer Generation & Source Accuracy Audit" in report
        assert "[DEMO 1] GROUNDED ANSWER GENERATION & SOURCE ACCURACY AUDIT" in report
        assert "[DEMO 2] MISSING-CONTEXT QUERY & SAFETY FALLBACK" in report
        assert "[DEMO 3] COMPARATIVE AUDIT: WITH RETRIEVAL VS. WITHOUT RETRIEVAL" in report
        assert "FINAL GROUNDED GENERATION AUDIT OUTCOME" in report
