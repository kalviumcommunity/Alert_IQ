from pathlib import Path

from src.document_upload import DocumentUploadService


def test_rejects_unsupported_extension():
    service = DocumentUploadService.__new__(DocumentUploadService)
    try:
        service.validate_filename("malware.exe")
    except ValueError as exc:
        assert "Unsupported file type" in str(exc)
    else:
        raise AssertionError("unsupported extension was accepted")


def test_rejects_empty_upload():
    service = DocumentUploadService.__new__(DocumentUploadService)
    service.upload_dir = Path("/tmp/alert-iq-test-uploads")
    service.max_bytes = 100
    try:
        service.store_bytes("empty.md", b"")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("empty upload was accepted")


def test_rejects_oversized_upload():
    service = DocumentUploadService.__new__(DocumentUploadService)
    service.upload_dir = Path("/tmp/alert-iq-test-uploads")
    service.max_bytes = 4
    try:
        service.store_bytes("large.md", b"12345")
    except ValueError as exc:
        assert "limit" in str(exc).lower()
    else:
        raise AssertionError("oversized upload was accepted")


def test_safe_filename_and_storage(tmp_path):
    service = DocumentUploadService.__new__(DocumentUploadService)
    service.upload_dir = tmp_path
    service.max_bytes = 100
    path = service.store_bytes("../../safe.md", b"hello runtime indexing")
    assert path.parent == tmp_path
    assert path.name == "safe.md"
    assert path.read_bytes() == b"hello runtime indexing"


def test_process_indexes_without_reset(monkeypatch, tmp_path):
    service = DocumentUploadService.__new__(DocumentUploadService)
    service.upload_dir = tmp_path
    service.max_bytes = 1000
    service.dimension = 768

    class FakeStore:
        collection_name = "test"
        def ensure_dimension(self, dimension): pass
        def upsert_records(self, records, dimension):
            self.records = records

    class FakeChunker:
        chunk_size = 400
        overlap = 60
        def chunk_document(self, document, size, overlap):
            return [type("Chunk", (), {"content": document.content, "source": document.source, "metadata": document.metadata})()]

    class FakeIngestion:
        chunker = FakeChunker()

    class FakeIndexer:
        def prepare_records(self, chunks):
            return chunks

    monkeypatch.setattr("src.document_upload.DocumentLoader.load_file", lambda path: (type("Doc", (), {"content": "runtime searchable policy", "source": str(path), "metadata": {"filename": path.name}})(), None))
    monkeypatch.setattr("src.document_upload.TextCleaner.clean_document", lambda doc: (doc, {}))
    service.ingestion = FakeIngestion()
    service.indexer = FakeIndexer()
    service.vector_store = FakeStore()

    path = service.store_bytes("runtime.md", b"runtime searchable policy")
    summary = service.process(path)
    assert summary["chunks"] == 1
    assert summary["indexed"] == 1
    assert len(service.vector_store.records) == 1
