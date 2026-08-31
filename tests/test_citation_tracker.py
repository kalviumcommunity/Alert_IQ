"""
Unit tests for Alert_IQ Citation Tracking & Source Verifiability Engine.
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
from src.citation_tracker import (
    CitationTracker,
    CitationReference,
    VerifiableRAGResponse,
    run_citation_demo,
)


class TestCitationTracker:
    """Test suite covering citation generation, metadata mapping, store verification, and anti-fabrication."""

    @pytest.fixture
    def indexed_store(self, tmp_path):
        """Sets up an isolated VectorStore with indexed corpus documents."""
        store = VectorStore(path=str(tmp_path), collection_name="test_citation_collection")
        indexer = CorpusIndexer(vector_store=store, dimension=768)
        indexer.run_indexing(reset_collection=True)
        return store

    def test_build_citations_metadata_mapping(self):
        """Task 1 & 2: Citations map chunks to source document, chunk ID, index, section, and quote."""
        tracker = CitationTracker()
        chunks = [
            RAGContextChunk(
                id="runbook_database_lag_md::chunk_000",
                rank=1,
                score=0.88,
                source_document="runbook_database_lag.md",
                chunk_index=0,
                text="# Database Replica Latency Triage Runbook (DB-RB-402)\nEmergency mitigation steps."
            )
        ]

        citations = tracker.build_citations(chunks)

        assert len(citations) == 1
        c = citations[0]
        assert isinstance(c, CitationReference)
        assert c.marker == "[1]"
        assert c.source_document == "runbook_database_lag.md"
        assert c.chunk_id == "runbook_database_lag_md::chunk_000"
        assert c.chunk_index == 0
        assert "Database Replica" in c.section
        assert len(c.snippet_quote) > 10

    def test_verify_citation_against_vector_store(self, indexed_store):
        """Task 3: User can verify a cited source against the actual stored chunk in vector store."""
        tracker = CitationTracker(vector_store=indexed_store, dimension=768)
        citation = CitationReference(
            marker="[1]",
            source_document="runbook_database_lag.md",
            chunk_id="runbook_database_lag_md::chunk_000",
            chunk_index=0,
            section="Database Replica Latency Triage Runbook",
            snippet_quote="# Database Replica Latency Triage Runbook (DB-RB-402)",
            similarity_score=0.9
        )

        audit = tracker.verify_citation_against_store(citation)

        assert audit["verified"] is True
        assert audit["source_matches"] is True
        assert audit["quote_matches"] is True
        assert audit["chunk_id"] == "runbook_database_lag_md::chunk_000"

    def test_detect_fabricated_citations(self):
        """Task 4: Anti-fabrication check detects ungrounded / hallucinated citation markers."""
        tracker = CitationTracker()
        valid_citations = [
            CitationReference(
                marker="[1]",
                source_document="incident_policy.txt",
                chunk_id="c1",
                chunk_index=0,
                section="Policy",
                snippet_quote="Level 1 escalation rules.",
                similarity_score=0.9
            )
        ]

        # Valid answer referencing [1]
        valid_answer = "Escalate incident within 5 minutes [1]."
        has_fab, fab_list = tracker.detect_fabricated_citations(valid_answer, valid_citations)
        assert has_fab is False
        assert len(fab_list) == 0

        # Fabricated answer referencing [2] and [99]
        fabricated_answer = "Escalate incident within 5 minutes [1] and notify CTO [2] using secret tool [99]."
        has_fab, fab_list = tracker.detect_fabricated_citations(fabricated_answer, valid_citations)
        assert has_fab is True
        assert "[2]" in fab_list
        assert "[99]" in fab_list

    def test_generate_verifiable_answer_and_fallback(self, indexed_store):
        """Task 4 & 5: Valid query generates verified response; out-of-domain query returns fallback without citations."""
        tracker = CitationTracker(vector_store=indexed_store, dimension=768)

        # 1. Valid Query
        valid_resp = tracker.generate_verifiable_answer(
            query="What are the mitigation steps for DB-RB-402 replica latency?",
            top_k=2
        )
        assert isinstance(valid_resp, VerifiableRAGResponse)
        assert valid_resp.is_fallback is False
        assert valid_resp.verification_status == "VERIFIED_100%"
        assert len(valid_resp.citations) > 0
        assert "VERIFIED SOURCE CITATIONS" in valid_resp.footnotes_text

        # 2. Out-of-Domain Query (Triggers Fallback)
        fallback_resp = tracker.generate_verifiable_answer(
            query="What is the Stripe payment gateway webhook HMAC secret key rotation schedule?",
            top_k=2,
            min_relevance_threshold=0.85
        )
        assert fallback_resp.is_fallback is True
        assert fallback_resp.verification_status == "NO_SOURCE_FALLBACK"
        assert len(fallback_resp.citations) == 0

    def test_citation_demo_execution(self, tmp_path):
        """Task 5: run_citation_demo executes cleanly and produces formatted report."""
        report = run_citation_demo(
            persist_path=str(tmp_path),
            collection_name="demo_citation_coll"
        )
        assert "Alert_IQ - Citation Tracking & Source Verifiability Demonstration" in report
        assert "[DEMO 1] VERIFIABLE ANSWER WITH CITATION-TO-METADATA MAPPING" in report
        assert "[DEMO 2] NO-SOURCE FALLBACK (ZERO FABRICATED CITATIONS)" in report
        assert "FINAL CITATION & VERIFIABILITY AUDIT OUTCOME" in report
