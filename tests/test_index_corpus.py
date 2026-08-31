"""
Unit tests for Alert_IQ corpus vector indexing pipeline, count reconciliation, and spot-check integrity.
"""
import sys
import pytest
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.vector_store import VectorStore
from src.index_corpus import (
    CorpusIndexer,
    generate_deterministic_embedding,
    generate_indexing_summary_report,
)


class TestCorpusIndexing:
    """Test suite covering corpus embedding generation, indexing, count validation, and spot checks."""

    @pytest.fixture
    def isolated_store(self, tmp_path):
        """Creates an isolated VectorStore in a temporary directory."""
        return VectorStore(path=str(tmp_path), collection_name="test_indexing_collection")

    def test_deterministic_embedding_generation(self):
        """Task 1 & 2: Embedding generator produces consistent normalized vectors of exact dimension."""
        text1 = "Alert triage escalation procedure"
        v1 = generate_deterministic_embedding(text1, dimension=768)
        v2 = generate_deterministic_embedding(text1, dimension=768)

        assert len(v1) == 768
        assert v1 == v2  # Deterministic repeatability

        # Different text produces distinct vector
        v3 = generate_deterministic_embedding("Completely unrelated cafeteria menu", dimension=768)
        assert len(v3) == 768
        assert v1 != v3

    def test_corpus_indexing_workflow_and_count_match(self, isolated_store):
        """Task 1, 2, 3: Full corpus indexing inserts all chunks, preserves metadata, and confirms counts."""
        indexer = CorpusIndexer(vector_store=isolated_store, dimension=768)
        report, ingestion_res, records = indexer.run_indexing()

        assert report.is_successful is True
        assert report.files_discovered == 5
        assert report.documents_ingested == 4
        assert report.failures_count == 1
        assert report.chunks_created == 4
        assert report.records_indexed == 4
        assert report.count_matches is True

        # Check that corrupted file was tracked as failure
        assert any("corrupt_unsupported.bin" in f["path"] for f in report.failures)

    def test_spot_check_integrity_validation(self, isolated_store):
        """Task 4: Spot checks confirm ID, text, metadata, and vector length match on readback."""
        indexer = CorpusIndexer(vector_store=isolated_store, dimension=768)
        report, ingestion_res, records = indexer.run_indexing()

        assert len(report.spot_checks) == 4
        for sc in report.spot_checks:
            assert sc.is_valid is True
            assert sc.matches_id is True
            assert sc.matches_text is True
            assert sc.matches_metadata is True
            assert sc.vector_length == 768

    def test_summary_report_generation(self, tmp_path):
        """Task 5: generate_indexing_summary_report outputs comprehensive formatted log."""
        output = generate_indexing_summary_report(
            persist_path=str(tmp_path),
            collection_name="summary_test_coll"
        )
        assert "Alert_IQ - Corpus Vector Indexing & Integrity Validation Report" in output
        assert "CORPUS INGESTION & VECTOR STORAGE" in output
        assert "INDEXED COUNT RECONCILIATION" in output
        assert "SPOT-CHECK READBACK INTEGRITY AUDIT" in output
        assert "FINAL INDEXING & VALIDATION OUTCOME" in output
        assert "READY FOR RETRIEVAL" in output
