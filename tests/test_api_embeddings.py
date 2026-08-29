"""Tests for OpenAI-compatible API embedding integration."""
from types import SimpleNamespace

import pytest

from src.api_embeddings import APIEmbeddingClient


class FakeEmbeddingsAPI:
    def create(self, model, input):
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.4, 0.5, 0.6]),
                SimpleNamespace(index=0, embedding=[0.1, 0.2, 0.3]),
            ]
        )


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddingsAPI()


def test_api_embeddings_preserve_input_order_and_model():
    client = APIEmbeddingClient(api_key="test-key", model="test-model", client=FakeClient())
    vectors = client.embed_texts(["first", "second"])

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert client.model == "test-model"


def test_chunk_embeddings_keep_text_and_metadata():
    chunks = [
        SimpleNamespace(
            content="Password reset instructions.",
            source="account-guide.md",
            metadata={"chunk_index": 0, "section": "Recovery"},
        ),
        SimpleNamespace(
            content="Use your registered email.",
            source="account-guide.md",
            metadata={"chunk_index": 1, "section": "Recovery"},
        ),
    ]
    client = APIEmbeddingClient(api_key="test-key", client=FakeClient())

    records = client.embed_chunks(chunks)

    assert records[0]["text"] == chunks[0].content
    assert records[0]["metadata"] == chunks[0].metadata
    assert records[0]["source"] == chunks[0].source
    assert records[0]["embedding"] == [0.1, 0.2, 0.3]
    assert records[0]["embedding_model"] == "text-embedding-3-small"


def test_report_exposes_vector_length_and_sample_values():
    client = APIEmbeddingClient(api_key="test-key", client=FakeClient())
    records = client.embed_chunks(
        [SimpleNamespace(content="hello", source="a.md", metadata={"chunk_index": 0})]
    )

    report = client.describe_records(records)

    assert report["records"] == 1
    assert report["vector_length"] == 3
    assert report["sample_values"] == [0.1, 0.2, 0.3]
    assert report["source_preserved"] is True
    assert report["metadata_preserved"] is True


def test_missing_api_key_is_rejected_when_real_client_is_needed(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = APIEmbeddingClient()

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        _ = client.client


def test_empty_text_batch_returns_empty():
    client = APIEmbeddingClient(api_key="test-key", client=FakeClient())
    assert client.embed_texts([]) == []


def test_blank_text_is_rejected():
    client = APIEmbeddingClient(api_key="test-key", client=FakeClient())
    with pytest.raises(ValueError, match="non-empty strings"):
        client.embed_texts([" "])
