from src.api import app


def test_documents_requires_file():
    client = app.test_client()
    response = client.post("/documents")
    assert response.status_code == 400
    assert response.get_json()["status"] == "error"


def test_documents_rejects_unsupported_file():
    client = app.test_client()
    response = client.post(
        "/documents",
        data={"file": ("notes.exe", b"not supported")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 415
    assert "Unsupported file type" in response.get_json()["error"]


def test_documents_rejects_empty_file():
    client = app.test_client()
    response = client.post(
        "/documents",
        data={"file": ("empty.md", b"")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 415
    assert "empty" in response.get_json()["error"].lower()


def test_documents_indexes_and_returns_summary(monkeypatch, tmp_path):
    class FakeService:
        def __init__(self):
            self.upload_dir = tmp_path

        def store_bytes(self, filename, content):
            path = tmp_path / filename
            path.write_bytes(content)
            return path

        def process(self, path):
            return {"document": path.as_posix(), "chunks": 2, "indexed": 2, "collection": "test"}

    monkeypatch.setattr("src.api.DocumentUploadService", FakeService)
    client = app.test_client()
    response = client.post(
        "/documents",
        data={"file": ("runtime.md", b"new searchable policy")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    body = response.get_json()
    assert body["status"] == "indexed"
    assert body["filename"] == "runtime.md"
    assert body["summary"]["chunks"] == 2
    assert body["summary"]["indexed"] == 2


def test_documents_maps_processing_failure_to_500(monkeypatch):
    class FailingService:
        def __init__(self): pass
        def store_bytes(self, filename, content): return __import__("pathlib").Path("runtime.md")
        def process(self, path): raise RuntimeError("vector database unavailable")

    monkeypatch.setattr("src.api.DocumentUploadService", FailingService)
    client = app.test_client()
    response = client.post(
        "/documents",
        data={"file": ("runtime.md", b"content")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 500
    assert response.get_json()["status"] == "error"
