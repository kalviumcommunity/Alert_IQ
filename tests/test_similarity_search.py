"""Tests for embedding similarity and chunk ranking."""
import pytest

from src.similarity_search import cosine_similarity, rank_chunks, similarity_report


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_rank_chunks_orders_highest_similarity_first():
    query = [1.0, 0.0]
    records = [
        {"text": "unrelated", "metadata": {"source": "campus.md"}, "embedding": [0.0, 1.0]},
        {"text": "strong match", "metadata": {"source": "account.md"}, "embedding": [1.0, 0.0]},
        {"text": "partial match", "metadata": {"source": "guide.md"}, "embedding": [0.8, 0.6]},
    ]

    ranked = rank_chunks(query, records)

    assert [item["text"] for item in ranked] == ["strong match", "partial match", "unrelated"]
    assert ranked[0]["score"] == pytest.approx(1.0)
    assert ranked[-1]["score"] == pytest.approx(0.0)


def test_top_k_limits_ranked_results():
    records = [
        {"text": "a", "embedding": [1.0, 0.0]},
        {"text": "b", "embedding": [0.8, 0.6]},
        {"text": "c", "embedding": [0.0, 1.0]},
    ]
    assert len(rank_chunks([1.0, 0.0], records, top_k=2)) == 2


def test_scores_keep_source_metadata():
    record = {"text": "password reset", "metadata": {"source": "account-guide.md", "chunk_index": 0}, "embedding": [1.0, 0.0]}
    ranked = rank_chunks([1.0, 0.0], [record])
    assert ranked[0]["metadata"] == record["metadata"]
    assert ranked[0]["score"] == pytest.approx(1.0)


def test_similarity_report_identifies_most_and_least_similar():
    records = [
        {"text": "password reset", "metadata": {"source": "account.md"}, "embedding": [1.0, 0.0]},
        {"text": "cafeteria menu", "metadata": {"source": "campus.md"}, "embedding": [0.0, 1.0]},
    ]
    report = similarity_report("reset password", [1.0, 0.0], records, top_k=2)
    assert report["most_similar"]["text"] == "password reset"
    assert report["least_similar"]["text"] == "cafeteria menu"


def test_zero_vector_is_rejected():
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


def test_mismatched_dimensions_are_rejected():
    with pytest.raises(ValueError, match="same dimension"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
