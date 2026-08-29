"""Batch, resumable, retry-aware embedding jobs for Alert_IQ."""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


@dataclass
class BatchEmbeddingSummary:
    total_chunks: int
    skipped_existing: int = 0
    embedded: int = 0
    failed: int = 0
    input_tokens: int = 0
    estimated_cost_usd: float = 0.0
    failures: List[Dict[str, str]] = field(default_factory=list)


class BatchEmbeddingPipeline:
    """Embed pending chunks in batches and resume safely from existing IDs."""

    def __init__(self, embedding_client: Any, batch_size: int = 64, max_attempts: int = 5,
                 base_wait_seconds: float = 2.0, price_per_1k_tokens: float = 0.00002,
                 token_counter: Any | None = None, sleep_fn: Any = time.sleep) -> None:
        if batch_size <= 0 or max_attempts <= 0:
            raise ValueError("batch_size and max_attempts must be positive")
        self.embedding_client = embedding_client
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self.base_wait_seconds = base_wait_seconds
        self.price_per_1k_tokens = price_per_1k_tokens
        self.token_counter = token_counter
        self.sleep_fn = sleep_fn

    @staticmethod
    def batches(items: Sequence[Any], size: int):
        if size <= 0:
            raise ValueError("size must be positive")
        for start in range(0, len(items), size):
            yield list(items[start:start + size])

    @staticmethod
    def chunk_id(chunk: Any) -> str:
        if isinstance(chunk, dict):
            if chunk.get("id") is not None:
                return str(chunk["id"])
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", chunk.get("source", ""))
            index = metadata.get("chunk_index", 0)
        else:
            metadata = getattr(chunk, "metadata", {})
            source = getattr(chunk, "source", "")
            index = metadata.get("chunk_index", 0)
        return f"{source}::{index}"

    def pending_chunks(self, chunks: Sequence[Any], existing_embedding_ids: set[str]):
        pending = [c for c in chunks if self.chunk_id(c) not in existing_embedding_ids]
        return pending, len(chunks) - len(pending)

    def embed_with_retry(self, texts: Sequence[str]):
        for attempt in range(self.max_attempts):
            try:
                return self.embedding_client.embed_texts(texts)
            except Exception:
                if attempt == self.max_attempts - 1:
                    raise
                self.sleep_fn(self.base_wait_seconds * (2 ** attempt))

    def estimate_tokens(self, texts: Sequence[str]) -> int:
        if self.token_counter is not None:
            return sum(self.token_counter(text) for text in texts)
        return sum(max(1, len(text.split())) for text in texts)

    def embed(self, chunks: Sequence[Any], existing_embedding_ids: set[str] | None = None,
              save_embeddings: Any | None = None) -> BatchEmbeddingSummary:
        existing_embedding_ids = existing_embedding_ids or set()
        pending, skipped = self.pending_chunks(chunks, existing_embedding_ids)
        summary = BatchEmbeddingSummary(total_chunks=len(chunks), skipped_existing=skipped)

        for batch in self.batches(pending, self.batch_size):
            texts = [c.content if hasattr(c, "content") else c["text"] for c in batch]
            summary.input_tokens += self.estimate_tokens(texts)
            try:
                vectors = self.embed_with_retry(texts)
                if len(vectors) != len(batch):
                    raise RuntimeError("Embedding API returned a different number of vectors than inputs")
                records = []
                for chunk, vector in zip(batch, vectors):
                    if hasattr(chunk, "content"):
                        text, metadata, source = chunk.content, dict(chunk.metadata), chunk.source
                    else:
                        text = chunk["text"]
                        metadata = dict(chunk.get("metadata", {}))
                        source = metadata.get("source", chunk.get("source"))
                    records.append({"id": self.chunk_id(chunk), "text": text, "metadata": metadata,
                                    "source": source, "embedding": vector,
                                    "embedding_model": self.embedding_client.model})
                if save_embeddings is not None:
                    save_embeddings(records)
                summary.embedded += len(records)
            except Exception as error:
                summary.failed += len(batch)
                summary.failures.append({"chunk_start_id": self.chunk_id(batch[0]), "error": str(error)})

        summary.estimated_cost_usd = round(summary.input_tokens / 1000 * self.price_per_1k_tokens, 6)
        return summary


def format_summary(summary: BatchEmbeddingSummary) -> str:
    return (f"total_chunks={summary.total_chunks} skipped_existing={summary.skipped_existing} "
            f"embedded={summary.embedded} failed={summary.failed} input_tokens={summary.input_tokens} "
            f"estimated_cost_usd={summary.estimated_cost_usd:.6f}")
