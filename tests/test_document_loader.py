"""
Unit tests for DocumentLoader multi-format ingestion and error handling.
"""
import unittest
from pathlib import Path
from src.document_loader import DocumentLoader, Document, SimpleHTMLTextExtractor


class TestDocumentLoader(unittest.TestCase):

    def setUp(self):
        self.corpus_dir = Path(__file__).resolve().parent.parent / "data" / "corpus"

    def test_load_markdown_document(self):
        file_path = self.corpus_dir / "runbook_database_lag.md"
        doc, err = DocumentLoader.load_file(file_path)
        self.assertIsNotNone(doc)
        self.assertIsNone(err)
        self.assertEqual(doc.metadata["file_type"], "md")
        self.assertIn("Database Replica Latency", doc.content)
        self.assertGreater(doc.token_count, 0)
        self.assertGreater(doc.character_count, 0)

    def test_load_html_document(self):
        file_path = self.corpus_dir / "service_health_export.html"
        doc, err = DocumentLoader.load_file(file_path)
        self.assertIsNotNone(doc)
        self.assertIsNone(err)
        self.assertEqual(doc.metadata["file_type"], "html")
        self.assertIn("Authentication Service", doc.content)
        # Verify HTML tags were stripped
        self.assertNotIn("<html>", doc.content)
        self.assertNotIn("<body>", doc.content)

    def test_load_text_document(self):
        file_path = self.corpus_dir / "incident_policy.txt"
        doc, err = DocumentLoader.load_file(file_path)
        self.assertIsNotNone(doc)
        self.assertIsNone(err)
        self.assertEqual(doc.metadata["file_type"], "txt")
        self.assertIn("ON-CALL ESCALATION POLICY", doc.content)

    def test_load_json_document(self):
        file_path = self.corpus_dir / "metrics_schema.json"
        doc, err = DocumentLoader.load_file(file_path)
        self.assertIsNotNone(doc)
        self.assertIsNone(err)
        self.assertEqual(doc.metadata["file_type"], "json")
        self.assertIn("critical_response_time_seconds", doc.content)

    def test_missing_file_handled_gracefully(self):
        missing_path = self.corpus_dir / "does_not_exist_file.md"
        doc, err = DocumentLoader.load_file(missing_path)
        self.assertIsNone(doc)
        self.assertIsNotNone(err)
        self.assertIn("File Not Found", err)

    def test_unsupported_binary_file_handled_gracefully(self):
        bin_path = self.corpus_dir / "corrupt_unsupported.bin"
        doc, err = DocumentLoader.load_file(bin_path)
        self.assertIsNone(doc)
        self.assertIsNotNone(err)
        self.assertIn("Unsupported Format", err)

    def test_load_directory_integration(self):
        loaded, skipped = DocumentLoader.load_directory(self.corpus_dir)
        self.assertGreaterEqual(len(loaded), 4)  # .md, .html, .txt, .json
        self.assertGreaterEqual(len(skipped), 1)  # .bin file
        # Check source identity preservation
        for d in loaded:
            self.assertTrue(bool(d.source))
            self.assertIn("filename", d.metadata)


if __name__ == "__main__":
    unittest.main()
