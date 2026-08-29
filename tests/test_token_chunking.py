"""Unit tests for token-aware chunk sizing and overlap."""
import unittest

from src.document_loader import Document
from src.token_chunking import TokenChunker


class TestTokenChunking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.chunker = TokenChunker()

    def test_chunks_are_sized_by_tokens(self):
        text = "Alert triage requires checking latency, errors, saturation, and recent deployments. " * 20
        chunks = self.chunker.token_chunks(text, size=40, overlap=0)
        counts = [len(self.chunker.encoding.encode(chunk)) for chunk in chunks]
        self.assertTrue(chunks)
        self.assertTrue(all(count <= 40 for count in counts))
        self.assertGreater(len(chunks), 1)

    def test_overlap_preserves_boundary_tokens(self):
        text = " ".join(f"token{i}" for i in range(120))
        chunks = self.chunker.token_chunks(text, size=30, overlap=6)
        first_tokens = self.chunker.encoding.encode(chunks[0])
        second_tokens = self.chunker.encoding.encode(chunks[1])
        self.assertEqual(second_tokens[:6], first_tokens[-6:])

    def test_overlap_increases_chunk_count_or_total_tokens(self):
        text = "Alert context is important for reliable retrieval. " * 40
        no_overlap = self.chunker.compare_overlap(text, size=50, overlaps=(0,))
        with_overlap = self.chunker.compare_overlap(text, size=50, overlaps=(10,))
        self.assertGreaterEqual(with_overlap[10]["chunk_count"], no_overlap[0]["chunk_count"])
        self.assertGreater(with_overlap[10]["total_tokens"], no_overlap[0]["total_tokens"])

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            self.chunker.token_chunks("some text", size=20, overlap=20)

    def test_document_keeps_source_and_metadata(self):
        document = Document(
            content="Alert triage and escalation guidance. " * 30,
            source="runbook.md",
            metadata={"filename": "runbook.md", "file_type": "md", "effective_date": "2026-01-01"},
        )
        chunks = self.chunker.chunk_document(document, size=40, overlap=6)
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].source, "runbook.md")
        self.assertEqual(chunks[0].metadata["effective_date"], "2026-01-01")
        self.assertEqual(chunks[0].metadata["chunk_strategy"], "token")
        self.assertEqual(chunks[0].metadata["token_overlap"], 6)
        self.assertEqual(chunks[0].metadata["overlap_tokens"], 0)
        self.assertEqual(chunks[1].metadata["overlap_tokens"], 6)

    def test_recommended_choice_is_400_tokens_and_15_percent_overlap(self):
        choice = self.chunker.explain_choice()
        self.assertEqual(choice["chunk_size_tokens"], 400)
        self.assertEqual(choice["overlap_tokens"], 60)
        self.assertEqual(choice["overlap_percent"], 15.0)


if __name__ == "__main__":
    unittest.main()
