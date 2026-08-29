"""Local Chroma vector-store adapter for Alert_IQ chunks."""
from typing import Any, Dict, List, Sequence


class VectorStore:
    """Persist embeddings with their text and metadata using Chroma."""

    def __init__(self, path: str = "data/vector_store", collection_name: str = "rag_chunks") -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is required for the vector store. Install dependencies from requirements.txt."
            ) from exc
        self.client = chromadb.PersistentClient(path=path)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _validate_dimension(vector: Sequence[float], expected_dimension: int) -> None:
        if len(vector) != expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dimension}, got {len(vector)}"
            )

    def ensure_dimension(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        existing = self.collection.count()
        if existing:
            sample = self.collection.get(limit=1, include=["embeddings"])
            vectors = sample.get("embeddings") or []
            if vectors and len(vectors[0]) != dimension:
                raise ValueError(
                    f"Collection contains {len(vectors[0])}-dimensional vectors; expected {dimension}"
                )
        self.dimension = dimension

    def upsert_records(self, records: Sequence[Dict[str, Any]], dimension: int | None = None) -> None:
        if not records:
            return
        inferred_dimension = dimension or len(records[0]["embedding"])
        self.ensure_dimension(inferred_dimension)
        for record in records:
            self._validate_dimension(record["embedding"], inferred_dimension)

        self.collection.upsert(
            ids=[str(record["id"]) for record in records],
            embeddings=[record["embedding"] for record in records],
            documents=[record["text"] for record in records],
            metadatas=[dict(record.get("metadata", {})) for record in records],
        )

    def get_record(self, record_id: str) -> Dict[str, Any] | None:
        result = self.collection.get(ids=[str(record_id)], include=["embeddings", "documents", "metadatas"])
        if not result.get("ids"):
            return None
        return {
            "id": result["ids"][0],
            "vector": result["embeddings"][0],
            "text": result["documents"][0],
            "metadata": result["metadatas"][0],
        }

    def count(self) -> int:
        return self.collection.count()

    def collection_info(self) -> Dict[str, Any]:
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "dimension": getattr(self, "dimension", None),
            "metric": "cosine",
        }


def verify_record(store: VectorStore, record_id: str, expected_dimension: int) -> Dict[str, Any]:
    """Read a test record back and verify its vector/text/metadata shape."""
    record = store.get_record(record_id)
    if record is None:
        raise AssertionError(f"Record not found: {record_id}")
    if len(record["vector"]) != expected_dimension:
        raise AssertionError("Read-back vector has the wrong dimension")
    if not record["text"]:
        raise AssertionError("Read-back text is empty")
    if not record["metadata"].get("source"):
        raise AssertionError("Read-back metadata has no source")
    return {
        "id": record["id"],
        "vector_length": len(record["vector"]),
        "text": record["text"],
        "metadata": record["metadata"],
    }
