"""OpenAI-compatible API embeddings with chunk metadata preserved."""
import os
from typing import Any, Dict, List, Sequence


class APIEmbeddingClient:
    """Generate embeddings through an OpenAI-compatible embeddings API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._client = client

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not configured. Set it in the environment before making API calls."
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "openai is required for API embeddings. Install dependencies from requirements.txt."
                ) from exc

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of texts and return vectors in API response order."""
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain non-empty strings")

        response = self.client.embeddings.create(model=self.model, input=list(texts))
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]

    def embed_chunks(self, chunks: Sequence[Any]) -> List[Dict[str, Any]]:
        """Attach each API vector to its original chunk text and metadata."""
        if not chunks:
            return []

        texts = [chunk.content if hasattr(chunk, "content") else chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts)
        if len(embeddings) != len(chunks):
            raise RuntimeError("Embedding API returned a different number of vectors than inputs")

        records: List[Dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings):
            if hasattr(chunk, "content"):
                text = chunk.content
                metadata = dict(chunk.metadata)
                source = chunk.source
            else:
                text = chunk["text"]
                metadata = dict(chunk.get("metadata", {}))
                source = metadata.get("source")

            records.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "embedding": embedding,
                    "embedding_model": self.model,
                    "source": source,
                }
            )
        return records

    def describe_records(self, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Return evidence for inspecting a successful embedding run."""
        first = records[0] if records else None
        return {
            "model": self.model,
            "records": len(records),
            "vector_length": len(first["embedding"]) if first else 0,
            "sample_values": first["embedding"][:5] if first else [],
            "source_preserved": bool(first and first.get("source")),
            "metadata_preserved": bool(first and first.get("metadata")),
        }
