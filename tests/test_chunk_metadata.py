"""Unit tests for chunk metadata and source tracking."""
import unittest

from src.chunking import Chunker
from src.document_loader import Document


class TestChunkMetadata(unittest.TestCase):

    def setUp(self):
        self.document = Document(
            content=(
                "# Triage\n\n"
                "Inspect active connections before taking action.\n\n"
                "# Escalation\n\n"
                "Escalate when replication lag exceeds the threshold."
            ),
            source="data/corpus/runbook_database_lag.md",
            metadata={
                "filename": "runbook_database_lag.md",
                "file_type": "md",
                "file_size_bytes": 123,
                "character_count": 138,
                "token_count": 24,
            },
        )

    def test_metadata_contains_consistent_source_and_position_fields(self):
        chunks = Chunker.chunk_document(self.document, strategy="paragraph")
        self.assertEqual(len(chunks), 4)

        for index, chunk in enumerate(chunks):
            metadata = chunk.metadata
            self.assertEqual(metadata["source"], self.document.source)
            self.assertEqual(metadata["filename"], "runbook_database_lag.md")
            self.assertEqual(metadata["chunk_index"], index)
            self.assertEqual(metadata["chunk_strategy"], "paragraph")
            self.assertIn("char_start", metadata)
            self.assertIn("char_end", metadata)
            self.assertLess(metadata["char_start"], metadata["char_end"])
            self.assertEqual(chunk.content, self.document.content[metadata["char_start"]:metadata["char_end"]])

    def test_section_metadata_tracks_markdown_heading(self):
        chunks = Chunker.chunk_document(self.document, strategy="paragraph")
        self.assertEqual(chunks[1].metadata["section"], "Triage")
        self.assertEqual(chunks[3].metadata["section"], "Escalation")

    def test_parent_metadata_is_preserved(self):
        chunk = Chunker.chunk_document(self.document, strategy="paragraph")[0]
        self.assertEqual(chunk.metadata["file_type"], "md")
        self.assertEqual(chunk.metadata["file_size_bytes"], 123)
        self.assertEqual(chunk.metadata["character_count"], 138)
        self.assertEqual(chunk.metadata["token_count"], 24)

    def test_citation_is_traceable(self):
        chunk = Chunker.chunk_document(self.document, strategy="paragraph")[1]
        self.assertEqual(
            chunk.citation(),
            "runbook_database_lag.md (chunk 1, section Triage)",
        )

    def test_fixed_chunks_also_track_positions(self):
        chunks = Chunker.chunk_document(self.document, strategy="fixed", size=30, overlap=5)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["char_start"], 0)
        self.assertEqual(chunks[0].metadata["char_end"], len(chunks[0].content))
        self.assertEqual(chunks[1].metadata["char_start"] < chunks[0].metadata["char_end"], True)


if __name__ == "__main__":
    unittest.main()
