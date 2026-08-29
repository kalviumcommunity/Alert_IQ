"""Embedding fundamentals and vector-similarity helpers for Alert_IQ."""
from typing import Any, Dict, List, Sequence

import numpy as np


class EmbeddingModel:
    """Lazy wrapper around a local sentence-transformers embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for embeddings. "
                    "Install dependencies from requirements.txt."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate one embedding vector for each input text."""
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain non-empty strings")
        vectors = self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False)
        return vectors.tolist()

    def dimension(self) -> int:
        """Return the number of coordinates in each embedding vector."""
        return int(self.model.get_sentence_embedding_dimension())


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine similarity for non-zero vectors."""
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    if va.ndim != 1 or vb.ndim != 1 or va.shape != vb.shape:
        raise ValueError("vectors must be one-dimensional and have the same dimension")
    denominator = np.linalg.norm(va) * np.linalg.norm(vb)
    if denominator == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.dot(va, vb) / denominator)


def compare_text_pairs(
    embeddings: Sequence[Sequence[float]],
    similar_pair: tuple[int, int] = (0, 1),
    dissimilar_pair: tuple[int, int] = (0, 2),
) -> Dict[str, float]:
    """Compare a semantically similar pair against an unrelated pair."""
    similar = cosine_similarity(embeddings[similar_pair[0]], embeddings[similar_pair[1]])
    dissimilar = cosine_similarity(embeddings[dissimilar_pair[0]], embeddings[dissimilar_pair[1]])
    return {
        "similarity_score": similar,
        "dissimilarity_score": dissimilar,
        "similar_pair_scores_higher": similar > dissimilar,
    }


def embedding_report(texts: Sequence[str], model: EmbeddingModel | None = None) -> Dict[str, Any]:
    """Generate the evidence required by W.E 3.25 for sample texts."""
    model = model or EmbeddingModel()
    vectors = model.embed(texts)
    dimension = len(vectors[0]) if vectors else 0
    scores = compare_text_pairs(vectors) if len(vectors) >= 3 else None
    return {
        "model": model.model_name,
        "text_count": len(texts),
        "dimension": dimension,
        "first_8_values": vectors[0][:8] if vectors else [],
        "similarity": scores,
        "interpretation": (
            "An embedding is a numeric representation of text where the complete vector pattern "
            "captures semantic information. Similar meanings should occupy nearby directions in vector space."
        ),
    }


SAMPLE_TEXTS = [
    "How do I reset my account password?",
    "Steps to recover access to my login",
    "The cafeteria menu has pasta today",
]


if __name__ == "__main__":
    report = embedding_report(SAMPLE_TEXTS)
    print(f"model: {report['model']}")
    print(f"dimension: {report['dimension']}")
    print(f"first 8 values: {report['first_8_values']}")
    print(f"password vs login recovery: {report['similarity']['similarity_score']:.4f}")
    print(f"password vs cafeteria menu: {report['similarity']['dissimilarity_score']:.4f}")
