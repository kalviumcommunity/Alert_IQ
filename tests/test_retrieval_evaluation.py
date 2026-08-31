"""
Unit tests for Alert_IQ Retrieval Quality Evaluation & Failure Inspection Suite.
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
from src.retrieval_evaluation import (
    RetrievalEvaluator,
    LabelledQuery,
    LABELLED_DATASET,
    generate_evaluation_report,
)


class TestRetrievalEvaluation:
    """Test suite covering labelled query dataset, recall/precision metrics, and failure diagnostics."""

    @pytest.fixture
    def indexed_store(self, tmp_path):
        """Sets up an isolated VectorStore with corpus documents."""
        store = VectorStore(path=str(tmp_path), collection_name="test_eval_collection")
        indexer = CorpusIndexer(vector_store=store, dimension=768)
        indexer.run_indexing(reset_collection=True)
        return store

    def test_labelled_dataset_schema(self):
        """Task 1: Labelled queries have valid chunk IDs, categories, and descriptions."""
        assert len(LABELLED_DATASET) >= 6
        for q in LABELLED_DATASET:
            assert isinstance(q, LabelledQuery)
            assert len(q.relevant_chunk_ids) > 0
            assert q.expected_source.endswith((".txt", ".md", ".json", ".html"))
            assert q.difficulty in {"easy", "medium", "hard"}
            assert len(q.intent_description) > 0

    def test_evaluation_metric_calculations(self, indexed_store):
        """Task 2 & 3: Evaluator computes valid Recall@k, Precision@k, F1, and MRR."""
        evaluator = RetrievalEvaluator(vector_store=indexed_store, dimension=768)
        summary = evaluator.evaluate_dataset(dataset=LABELLED_DATASET)

        assert summary.total_queries == len(LABELLED_DATASET)
        assert 0.0 <= summary.mean_recall_at_1 <= 1.0
        assert 0.0 <= summary.mean_recall_at_2 <= 1.0
        assert summary.mean_recall_at_3 >= 0.85  # High coverage across top-3
        assert 0.0 <= summary.mean_precision_at_1 <= 1.0
        assert 0.0 <= summary.mean_precision_at_2 <= 1.0
        assert 0.0 <= summary.mrr <= 1.0
        assert summary.mean_f1_at_2 > 0.0

    def test_failure_inspection_diagnostics(self, indexed_store):
        """Task 4: Evaluator detects and annotates failure cases for under-specified queries."""
        evaluator = RetrievalEvaluator(vector_store=indexed_store, dimension=768)
        summary = evaluator.evaluate_dataset(dataset=LABELLED_DATASET)

        # Ensure every result has valid status
        for qr in summary.query_results:
            assert qr.first_hit_rank is not None or not qr.is_success
            if qr.failure_reason:
                assert ":" in qr.failure_reason  # Structured category: explanation

    def test_evaluation_report_generation(self, tmp_path):
        """Task 5: generate_evaluation_report produces complete formatted report."""
        report = generate_evaluation_report(
            persist_path=str(tmp_path),
            collection_name="demo_eval_coll"
        )
        assert "Alert_IQ - Systematic Retrieval Quality Evaluation" in report
        assert "LABELLED GROUND-TRUTH QUERY BENCHMARK" in report
        assert "RECALL, PRECISION, AND RANKING QUALITY METRICS" in report
        assert "FAILURE & LOW-SCORING CASE ROOT CAUSE ANALYSIS" in report
        assert "ERROR TAXONOMY & REMEDIATION MATRIX" in report
