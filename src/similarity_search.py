"""Similarity scoring and ranking for embedding-based retrieval."""
from typing import Any, Dict, List, Sequence

import numpy as np


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    if va.ndim != 1 or vb.ndim != 1 or va.shape != vb.shape:
        raise ValueError("vectors must be one-dimensional and have the same dimension")
    denominator = np.linalg.norm(va) * np.linalg.norm(vb)
    if denominator == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.dot(va, vb) / denominator)


def rank_chunks(
    query_embedding: Sequence[float],
    chunk_records: Sequence[Dict[str, Any]],
    top_k: int | None = None,
) -> List[Dict[str, Any]]:
    """Score every chunk against a query and rank highest similarity first."""
    ranked: List[Dict[str, Any]] = []
    for record in chunk_records:
        if "embedding" not in record:
            raise ValueError("each chunk record must contain an embedding")
        score = cosine_similarity(query_embedding, record["embedding"])
        ranked.append({**record, "score": score})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        ranked = ranked[:top_k]
    return ranked


def similarity_report(
    query: str,
    query_embedding: Sequence[float],
    chunk_records: Sequence[Dict[str, Any]],
    top_k: int = 3,
) -> Dict[str, Any]:
    """Return retrieval-ranking evidence for a query."""
    ranked = rank_chunks(query_embedding, chunk_records, top_k=top_k)
    return {
        "query": query,
        "candidate_count": len(chunk_records),
        "top_k": len(ranked),
        "ranked": ranked,
        "most_similar": ranked[0] if ranked else None,
        "least_similar": ranked[-1] if ranked else None,
    }
