"""
Unit tests for Alert_IQ Retrieval Tuning & Benchmark Evaluation Suite.
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
from src.retrieval_tuning import (
    RetrievalTuner,
    RetrievalConfig,
    BENCHMARK_QUERIES,
    generate_tuning_report,
)


class TestRetrievalTuning:
    """Test suite for retrieval parameter tuning, relevance metrics, and config optimization."""

    @pytest.fixture
    def indexed_store(self, tmp_path):
        """Creates an isolated, pre-indexed VectorStore instance."""
        store = VectorStore(path=str(tmp_path), collection_name="test_tuning_collection")
        indexer = CorpusIndexer(vector_store=store, dimension=768)
        indexer.run_indexing(reset_collection=True)
        return store

    def test_benchmark_queries_structure(self):
        """Task 1: Benchmark queries have expected fields and ground truth sources."""
        assert len(BENCHMARK_QUERIES) >= 4
        for q in BENCHMARK_QUERIES:
            assert len(q.query_text) > 10
            assert q.expected_source.endswith((".txt", ".md", ".json", ".html"))
            assert len(q.expected_keywords) > 0
            assert q.category is not None

    def test_evaluate_single_configuration(self, indexed_store):
        """Task 2 & 3: Evaluate configuration and compute valid relevance metrics."""
        tuner = RetrievalTuner(vector_store=indexed_store, dimension=768)
        config = RetrievalConfig(
            config_id="TEST-CFG",
            name="Test Hybrid Config",
            top_k=2,
            alpha=0.5,
            score_threshold=0.0,
            use_metadata_filter=False,
            description="Test config"
        )
        summary = tuner.evaluate_configuration(config, benchmark=BENCHMARK_QUERIES)

        assert summary.total_queries == len(BENCHMARK_QUERIES)
        assert 0.0 <= summary.top1_hit_rate <= 1.0
        assert 0.0 <= summary.recall_at_k <= 1.0
        assert 0.0 <= summary.precision_at_k <= 1.0
        assert 0.0 <= summary.mrr <= 1.0
        assert len(summary.query_results) == len(BENCHMARK_QUERIES)

    def test_run_tuning_suite_selects_optimal_config(self, indexed_store):
        """Task 4: Tuning suite compares multiple settings and recommends the best setup."""
        tuner = RetrievalTuner(vector_store=indexed_store, dimension=768)
        summaries, best = tuner.run_tuning_suite()

        assert len(summaries) >= 4
        assert best is not None
        assert best.top1_hit_rate >= 0.75  # Optimal setup achieves high Top-1 hit rate
        assert best.mrr >= 0.8
        assert best.config.config_id in {"CONFIG-D", "CONFIG-E"}

    def test_tuning_report_generation(self, tmp_path):
        """Task 5: generate_tuning_report creates complete formatted audit log."""
        report = generate_tuning_report(
            persist_path=str(tmp_path),
            collection_name="demo_tuning_coll"
        )
        assert "Alert_IQ - Retrieval Parameter Tuning & Relevance Benchmark Report" in report
        assert "TEST QUERY BENCHMARK SPECIFICATION" in report
        assert "RETRIEVAL CONFIGURATION COMPARISON & RELEVANCE METRICS" in report
        assert "CHOSEN PRODUCTION SETTINGS & EMPIRICAL JUSTIFICATION" in report
        assert "Optimal Parameters" in report
