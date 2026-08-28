"""Unit tests for Alert_IQ document chunking strategies."""
import unittest

from src.chunking import Chunker, choose_corpus_strategy
from src.document_loader import Document


class TestChunking(unittest.TestCase):

    def setUp(self):
        self.document = Document(
            content=(
                "First paragraph explains alert triage. It contains enough context.\n\n"
                "Second paragraph explains escalation. It should remain a separate unit."
            ),
            source="incident_policy.txt",
            metadata={"filename": "incident_policy.txt", "file_type": "txt"},
        )

    def test_fixed_chunks_respect_size_and_overlap(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = Chunker.fixed_chunks(text, size=10, overlap=2)
        self.assertEqual(chunks[0], "abcdefghij")
        self.assertEqual(chunks[1], "ijklmnopqr")
        self.assertEqual(chunks[1][:2], chunks[0][-2:])
        self.assertEqual("".join(chunks), "abcdefghijijklmnopqrqrstuvwxyz")

    def test_paragraph_chunks_preserve_boundaries(self):
        chunks = Chunker.paragraph_chunks(self.document.content)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith("First paragraph"))
        self.assertTrue(chunks[1].startswith("Second paragraph"))

    def test_chunk_document_preserves_source_and_index(self):
        chunks = Chunker.chunk_document(self.document, strategy="paragraph")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source, "incident_policy.txt")
        self.assertEqual(chunks[0].metadata["filename"], "incident_policy.txt")
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)
        self.assertEqual(chunks[1].metadata["chunk_index"], 1)

    def test_invalid_fixed_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            Chunker.fixed_chunks("some text", size=10, overlap=10)

    def test_compare_reports_both_strategies(self):
        comparison = Chunker.compare_strategies(self.document, fixed_size=40, fixed_overlap=5)
        self.assertIn("fixed", comparison)
        self.assertIn("paragraph", comparison)
        self.assertGreater(comparison["fixed"]["chunk_count"], 0)
        self.assertEqual(comparison["paragraph"]["chunk_count"], 2)

    def test_corpus_strategy_choice(self):
        strategy, decision = choose_corpus_strategy([self.document])
        self.assertEqual(strategy, "paragraph")
        self.assertEqual(decision["documents_evaluated"], 1)


if __name__ == "__main__":
    unittest.main()
