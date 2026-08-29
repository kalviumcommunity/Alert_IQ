from types import SimpleNamespace

import pytest

from src.batch_embeddings import BatchEmbeddingPipeline, BatchEmbeddingSummary, format_summary


class FakeEmbeddingClient:
    model = "test-model"

    def __init__(self, failures=0):
        self.failures = failures
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        if self.failures:
            self.failures -= 1
            raise RuntimeError("temporary failure")
        return [[float(i), 1.0] for i, _ in enumerate(texts)]


def chunk(source, index, text):
    return SimpleNamespace(content=text, source=source, metadata={"chunk_index": index})


def test_batches_split_items():
    batches = list(BatchEmbeddingPipeline.batches([1, 2, 3, 4, 5], 2))
    assert batches == [[1, 2], [3, 4], [5]]


def test_existing_embeddings_are_skipped_and_remaining_are_batched():
    client = FakeEmbeddingClient()
    pipeline = BatchEmbeddingPipeline(client, batch_size=2, token_counter=lambda text: len(text))
    chunks = [chunk("a.md", i, f"text-{i}") for i in range(4)]
    saved = []

    summary = pipeline.embed(chunks, {"a.md::1"}, saved.extend)

    assert summary.total_chunks == 4
    assert summary.skipped_existing == 1
    assert summary.embedded == 3
    assert summary.failed == 0
    assert [len(call) for call in client.calls] == [2, 1]
    assert len(saved) == 3
    assert summary.input_tokens == sum(len(c.content) for c in chunks if c.metadata["chunk_index"] != 1)


def test_retry_uses_exponential_backoff():
    waits = []
    client = FakeEmbeddingClient(failures=2)
    pipeline = BatchEmbeddingPipeline(client, max_attempts=3, base_wait_seconds=2, sleep_fn=waits.append)

    vectors = pipeline.embed_with_retry(["hello"])

    assert vectors == [[0.0, 1.0]]
    assert waits == [2, 4]
    assert len(client.calls) == 3


def test_permanent_failure_is_counted_without_hiding_it():
    client = FakeEmbeddingClient(failures=5)
    pipeline = BatchEmbeddingPipeline(client, max_attempts=2, base_wait_seconds=0, sleep_fn=lambda _: None)
    chunks = [chunk("a.md", 0, "hello")]

    summary = pipeline.embed(chunks)

    assert summary.embedded == 0
    assert summary.failed == 1
    assert "temporary failure" in summary.failures[0]["error"]


def test_cost_estimate_and_summary_format():
    summary = BatchEmbeddingSummary(total_chunks=10, skipped_existing=2, embedded=8, input_tokens=2000,
                                    estimated_cost_usd=0.00004)
    text = format_summary(summary)
    assert "embedded=8" in text
    assert "skipped_existing=2" in text
    assert "estimated_cost_usd=0.000040" in text


def test_invalid_batch_settings_are_rejected():
    with pytest.raises(ValueError):
        BatchEmbeddingPipeline(FakeEmbeddingClient(), batch_size=0)
