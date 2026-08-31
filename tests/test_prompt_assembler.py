"""
Unit tests for Alert_IQ Grounded Prompt Assembler & Token Budgeting Engine.
"""
import sys
import pytest
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag_pipeline import RAGContextChunk
from src.prompt_assembler import (
    PromptAssembler,
    TokenBudget,
    AugmentedPromptResult,
    run_assembly_demo,
)


class TestPromptAssembler:
    """Test suite covering context chunk injection, token budgeting, source markers, and grounding."""

    def test_assemble_context_source_markers(self):
        """Task 1 & 3: Context assembly injects bracketed source markers ([1], [2]) and maps sources."""
        assembler = PromptAssembler(total_token_limit=2048)
        chunks = [
            RAGContextChunk(
                id="doc_a::chunk_000",
                rank=1,
                score=0.9,
                source_document="incident_policy.txt",
                chunk_index=0,
                text="Level 1 escalation rules."
            ),
            RAGContextChunk(
                id="doc_b::chunk_000",
                rank=2,
                score=0.8,
                source_document="runbook_database_lag.md",
                chunk_index=0,
                text="DB-RB-402 mitigation steps."
            ),
        ]

        context_text, included, source_map, allocated, omitted = assembler.assemble_context(
            chunks=chunks,
            available_context_budget=1000
        )

        assert len(included) == 2
        assert omitted == 0
        assert "[SOURCE REF: [1]" in context_text
        assert "[SOURCE REF: [2]" in context_text
        assert "[1]" in source_map
        assert "[2]" in source_map
        assert "incident_policy.txt" in source_map["[1]"]

    def test_token_budget_boundary_omission(self):
        """Task 2: Chunks exceeding the available token budget are gracefully omitted."""
        assembler = PromptAssembler(total_token_limit=2048)
        # Large chunk that consumes substantial tokens
        large_text = "Detailed diagnostic command execution " * 80
        chunks = [
            RAGContextChunk(
                id="c1",
                rank=1,
                score=0.9,
                source_document="doc1.txt",
                chunk_index=0,
                text="Short high-priority triage step."
            ),
            RAGContextChunk(
                id="c2",
                rank=2,
                score=0.5,
                source_document="doc2.txt",
                chunk_index=0,
                text=large_text
            )
        ]

        # Allocate tight budget sufficient only for chunk 1
        context_text, included, source_map, allocated, omitted = assembler.assemble_context(
            chunks=chunks,
            available_context_budget=50
        )

        assert len(included) == 1
        assert omitted == 1
        assert included[0].id == "c1"

    def test_build_prompt_full_package(self):
        """Task 4: build_prompt generates system prompt, user prompt, and budget accounting."""
        assembler = PromptAssembler(total_token_limit=1024, response_token_reserve=200)
        chunks = [
            RAGContextChunk(
                id="c1",
                rank=1,
                score=0.9,
                source_document="metrics_schema.json",
                chunk_index=0,
                text="critical_response_time_seconds: 300"
            )
        ]

        result = assembler.build_prompt(
            query="What is the critical response threshold?",
            retrieved_chunks=chunks,
            alert_metadata={"alert_id": "ALT-123", "severity": "CRITICAL"}
        )

        assert isinstance(result, AugmentedPromptResult)
        assert "STRICT GROUNDING RULES" in result.system_prompt
        assert "Insufficient alert context" in result.system_prompt
        assert "[ALERT METADATA]" in result.user_prompt
        assert "ALT-123" in result.user_prompt
        assert "[RETRIEVED RUNBOOK & POLICY CONTEXT]" in result.user_prompt

        b = result.token_budget
        assert b.total_limit == 1024
        assert b.response_reserve == 200
        assert b.total_prompt_tokens + b.remaining_headroom_tokens + b.response_reserve <= b.total_limit + 50

    def test_assembly_demo_execution(self, tmp_path):
        """Task 5: run_assembly_demo executes cleanly and produces formatted output."""
        report = run_assembly_demo(
            persist_path=str(tmp_path),
            collection_name="demo_prompt_coll"
        )
        assert "Alert_IQ - Grounded Augmented Prompt Assembly" in report
        assert "TOKEN BUDGET BREAKDOWN" in report
        assert "SOURCE REFERENCE MAP" in report
        assert "RENDERED AUGMENTED PROMPT PREVIEW" in report
        assert "FINAL PROMPT ASSEMBLY VALIDATION" in report
