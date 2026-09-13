from types import SimpleNamespace

from src.embedding_sanity import EmbeddingSanityChecker, explain_failures, format_sanity_report


class FakeEmbeddingClient:
    def embed_texts(self, texts):
        vectors = {
            "How can a learner reset their password?": [1.0, 0.0],
            "When does the cafeteria menu change?": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]


def records():
    return [
        {"text": "Reset password", "metadata": {"source": "account-guide.md"}, "embedding": [1.0, 0.0]},
        {"text": "Cafeteria menu", "metadata": {"source": "campus-guide.md"}, "embedding": [0.0, 1.0]},
        {"text": "Unrelated", "metadata": {"source": "other.md"}, "embedding": [-1.0, 0.0]},
    ]


def test_known_related_sources_rank_first():
    checker = EmbeddingSanityChecker(FakeEmbeddingClient())
    report = checker.run(
        [
            {"query": "How can a learner reset their password?", "expected_source": "account-guide.md"},
            {"query": "When does the cafeteria menu change?", "expected_source": "campus-guide.md"},
        ],
        records(),
        top_k=3,
    )
    assert report["tests"] == 2
    assert report["passed"] == 2
    assert report["failed"] == 0
    assert report["all_passed"] is True


def test_failed_case_is_visible_and_explained():
    checker = EmbeddingSanityChecker(FakeEmbeddingClient())
    report = checker.run(
        [{"query": "How can a learner reset their password?", "expected_source": "missing.md"}],
        records(),
        top_k=3,
    )
    assert report["failed"] == 1
    assert report["results"][0]["passed"] is False
    assert explain_failures(report)
    assert "model consistency" in explain_failures(report)[0]


def test_report_format_contains_pass_fail_counts():
    checker = EmbeddingSanityChecker(FakeEmbeddingClient())
    report = checker.run(
        [{"query": "How can a learner reset their password?", "expected_source": "account-guide.md"}],
        records(),
    )
    text = format_sanity_report(report)
    assert "tests: 1 passed: 1 failed: 0" in text
    assert "account-guide.md" in text
