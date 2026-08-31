"""
Unit tests for Alert_IQ Modular End-to-End RAG Pipeline.
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
from src.rag_pipeline import (
    embed_stage,
    retrieve_stage,
    assemble_stage,
    generate_stage,
    RAGPipeline,
    RAGContextChunk,
    RAGResponse,
    run_pipeline_demo,
)


class TestRAGPipeline:
    """Test suite covering each isolated pipeline stage and end-to-end execution."""

    @pytest.fixture
    def indexed_store(self, tmp_path):
        """Sets up an isolated VectorStore with indexed corpus documents."""
        store = VectorStore(path=str(tmp_path), collection_name="test_rag_pipeline_collection")
        indexer = CorpusIndexer(vector_store=store, dimension=768)
        indexer.run_indexing(reset_collection=True)
        return store

    def test_embed_stage(self):
        """Task 2 & 4: Embed stage generates correct 768-d unit-normalized embedding vector."""
        query = "How to escalate P1 alerts?"
        vec = embed_stage(query, dimension=768)

        assert isinstance(vec, list)
        assert len(vec) == 768
        assert all(isinstance(x, float) for x in vec)

        with pytest.raises(ValueError, match="cannot be empty"):
            embed_stage("")

    def test_retrieve_stage(self, indexed_store):
        """Task 2 & 4: Retrieve stage returns structured RAGContextChunks with ranking and metadata."""
        query = "DB-RB-402 database replica latency"
        chunks = retrieve_stage(
            query=query,
            vector_store=indexed_store,
            top_k=2,
            use_reranker=True,
            dimension=768
        )

        assert len(chunks) == 2
        assert isinstance(chunks[0], RAGContextChunk)
        assert chunks[0].rank == 1
        assert "runbook_database_lag" in chunks[0].id
        assert chunks[0].source_document == "runbook_database_lag.md"

    def test_assemble_stage(self):
        """Task 2 & 4: Assemble stage constructs annotated context blocks with source headers."""
        chunks = [
            RAGContextChunk(
                id="doc1::chunk_000",
                rank=1,
                score=0.85,
                source_document="incident_policy.txt",
                chunk_index=0,
                text="Level 1 on-call notification within 5 minutes."
            ),
            RAGContextChunk(
                id="doc2::chunk_000",
                rank=2,
                score=0.75,
                source_document="metrics_schema.json",
                chunk_index=0,
                text="critical_response_time_seconds threshold is 60."
            )
        ]

        context = assemble_stage(chunks)

        assert "--- [DOCUMENT CHUNK 1 | Source: incident_policy.txt" in context
        assert "Level 1 on-call notification within 5 minutes." in context
        assert "--- [DOCUMENT CHUNK 2 | Source: metrics_schema.json" in context

    def test_generate_stage(self):
        """Task 2 & 4: Generate stage produces grounded answer adhering to context."""
        query = "What is the Level 1 notification SLA?"
        context = "--- [DOCUMENT CHUNK 1] ---\n1. Level 1 Notification (0 - 5 Minutes): Automated alert dispatch."

        answer = generate_stage(query=query, context=context)

        assert len(answer) > 20
        assert "Alert_IQ" in answer or "Level 1" in answer
        assert "Notification" in answer

    def test_end_to_end_pipeline_execution(self, indexed_store):
        """Task 3: Full pipeline runs end-to-end returning answer and retrieved sources."""
        pipeline = RAGPipeline(vector_store=indexed_store, dimension=768)
        query = "What are the mitigation steps for DB-RB-402 database replica latency?"

        response = pipeline.run(query=query, top_k=2)

        assert isinstance(response, RAGResponse)
        assert response.query == query
        assert len(response.answer) > 20
        assert len(response.retrieved_sources) == 2
        assert response.retrieved_sources[0]["source_document"] == "runbook_database_lag.md"
        assert "DOCUMENT CHUNK" in response.context_assembled
        assert "total_latency_ms" in response.stage_metrics
        assert response.stage_metrics["total_latency_ms"] > 0.0

    def test_pipeline_demo_report_generation(self, tmp_path):
        """Task 5: run_pipeline_demo produces complete formatted execution report."""
        report = run_pipeline_demo(
            persist_path=str(tmp_path),
            collection_name="demo_pipeline_coll"
        )
        assert "Alert_IQ - End-to-End RAG Pipeline Execution Demonstration" in report
        assert "[STAGE 1 & 2: RETRIEVED SOURCES]" in report
        assert "[STAGE 3: ASSEMBLED GROUNDING CONTEXT]" in report
        assert "[STAGE 4: GENERATED GROUNDED ANSWER]" in report
        assert "PIPELINE STAGE METRICS" in report
        assert "FINAL END-TO-END RAG PIPELINE VALIDATION" in report
