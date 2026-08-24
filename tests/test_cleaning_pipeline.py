"""
Unit tests for TextCleaner document cleaning and normalization pipeline.
"""
import unittest
from src.cleaning_pipeline import TextCleaner
from src.document_loader import Document


class TestCleaningPipeline(unittest.TestCase):

    def test_unicode_and_encoding_normalization(self):
        raw = "â€œImportantâ€\x9d notice: latency\u00a0is\u200b â€” 850ms."
        cleaned = TextCleaner.clean_text(raw)
        self.assertNotIn("â€œ", cleaned)
        self.assertNotIn("\u00a0", cleaned)
        self.assertNotIn("\u200b", cleaned)
        self.assertIn('"Important" notice: latency is - 850ms.', cleaned)

    def test_boilerplate_removal(self):
        raw = (
            "CONFIDENTIAL - INTERNAL ONLY\n"
            "Home > Docs > Triage > Alert_IQ\n"
            "Actual content line 1.\n"
            "Page 3 of 15\n"
            "Actual content line 2.\n"
            "Copyright (c) 2026 Alert_IQ Inc. All rights reserved.\n"
        )
        cleaned = TextCleaner.remove_boilerplate(raw)
        self.assertNotIn("CONFIDENTIAL", cleaned)
        self.assertNotIn("Home > Docs", cleaned)
        self.assertNotIn("Page 3 of 15", cleaned)
        self.assertNotIn("Copyright", cleaned)
        self.assertIn("Actual content line 1.", cleaned)
        self.assertIn("Actual content line 2.", cleaned)

    def test_broken_line_wrap_reconnection(self):
        raw = "We must optimize the data-\nbase connec-\ntion pool."
        cleaned = TextCleaner.fix_broken_line_wraps(raw)
        self.assertIn("database", cleaned)
        self.assertIn("connection", cleaned)
        self.assertNotIn("data-\nbase", cleaned)

    def test_whitespace_collapsing(self):
        raw = "Line 1    with    spaces.\n\n\n\n\nLine 2 after huge gap."
        cleaned = TextCleaner.collapse_whitespace(raw)
        self.assertNotIn("    ", cleaned)
        self.assertNotIn("\n\n\n", cleaned)
        self.assertIn("Line 1 with spaces.\n\nLine 2 after huge gap.", cleaned)

    def test_clean_document_and_corpus(self):
        doc = Document(
            content="Page 1 of 2\n\nHigh memory connec-\ntion leak.\n\n\n\nPage 2 of 2",
            source="test.md",
            metadata={"filename": "test.md"}
        )
        cleaned_doc, stats = TextCleaner.clean_document(doc)
        self.assertIn("connection", cleaned_doc.content)
        self.assertNotIn("Page 1 of 2", cleaned_doc.content)
        self.assertTrue(cleaned_doc.metadata["is_cleaned"])
        self.assertGreater(stats["char_reduction"], 0)

        # Test corpus clean
        docs, all_stats = TextCleaner.clean_corpus([doc])
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(all_stats), 1)


if __name__ == "__main__":
    unittest.main()
