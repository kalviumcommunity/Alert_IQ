from src.api import app


def test_query_requires_json():
    client = app.test_client()
    response = client.post("/query", data="not json")
    assert response.status_code == 415
    assert response.get_json()["status"] == "error"


def test_query_requires_question():
    client = app.test_client()
    response = client.post("/query", json={})
    assert response.status_code == 400
    assert "question" in response.get_json()["error"]


def test_query_rejects_blank_question():
    client = app.test_client()
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 400


def test_query_returns_structured_rag_response(monkeypatch):
    class FakeResponse:
        query = "How do I triage replica lag?"
        answer = "Check replication slots and long-running transactions."
        retrieved_sources = [
            {"id": "chunk-1", "rank": 1, "score": 0.91, "source_document": "runbook.md", "chunk_index": 2}
        ]
        stage_metrics = {"total_latency_ms": 12.3, "retrieved_chunk_count": 1}

    fake_pipeline = type("FakePipeline", (), {"run": lambda self, query: FakeResponse()})()
    monkeypatch.setattr("src.api.create_pipeline", lambda: fake_pipeline)

    client = app.test_client()
    response = client.post("/query", json={"question": "How do I triage replica lag?"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"
    assert body["answer"] == FakeResponse.answer
    assert body["sources"][0]["source_document"] == "runbook.md"
    assert body["metadata"]["query"] == FakeResponse.query
    assert body["metadata"]["stage_metrics"]["retrieved_chunk_count"] == 1


def test_query_maps_pipeline_failure_to_500(monkeypatch):
    def fail():
        raise RuntimeError("pipeline unavailable")

    monkeypatch.setattr("src.api.create_pipeline", fail)
    client = app.test_client()
    response = client.post("/query", json={"question": "test"})

    assert response.status_code == 500
    assert response.get_json()["status"] == "error"
