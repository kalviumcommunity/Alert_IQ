"""Tests for the end-to-end corpus ingestion and validation pipeline."""
import tempfile
import unittest
from pathlib import Path

from src.ingestion_pipeline import IngestionPipeline, validate_ingestion_result


class TestIngestionPipeline(unittest.TestCase):

    def test_pipeline_reconciles_success_and_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "valid.txt").write_text(
                "Alert triage requires checking latency and recent deployments.",
                encoding="utf-8",
            )
            (root / "unsupported.bin").write_bytes(b"not a supported document")

            result = IngestionPipeline(chunk_size=20, overlap=4).ingest(root)

            self.assertEqual(result.file_count, 2)
            self.assertEqual(result.document_count, 1)
            self.assertEqual(result.failure_count, 1)
            self.assertEqual(result.document_count + result.failure_count, result.file_count)
            self.assertTrue(result.is_complete)

    def test_pipeline_creates_traceable_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "runbook.md").write_text(
                "# Triage\n\nCheck latency before escalating the alert. " * 10,
                encoding="utf-8",
            )

            result = IngestionPipeline(chunk_size=30, overlap=5).ingest(root)

            self.assertEqual(result.document_count, 1)
            self.assertGreater(result.chunk_count, 0)
            chunk = result.chunks[0]
            self.assertEqual(chunk.metadata["filename"], "runbook.md")
            self.assertEqual(chunk.metadata["chunk_strategy"], "token")
            self.assertIn("chunk_index", chunk.metadata)
            self.assertIn("token_count", chunk.metadata)

    def test_validation_summary_reports_all_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "one.txt").write_text("A useful incident runbook.", encoding="utf-8")

            result = IngestionPipeline(chunk_size=20, overlap=2).ingest(root)
            summary = validate_ingestion_result(result)

            self.assertEqual(summary["files_discovered"], 1)
            self.assertEqual(summary["documents_ingested"], 1)
            self.assertEqual(summary["failures"], 0)
            self.assertTrue(summary["accounting_reconciles"])
            self.assertGreater(summary["chunks_created"], 0)

    def test_missing_folder_is_accounted_for(self):
        missing = Path(tempfile.gettempdir()) / "alert_iq_folder_that_does_not_exist"
        result = IngestionPipeline().ingest(missing)
        self.assertEqual(result.file_count, 0)
        self.assertEqual(result.document_count, 0)
        self.assertEqual(result.failure_count, 0)
        self.assertTrue(result.is_complete)


if __name__ == "__main__":
    unittest.main()
