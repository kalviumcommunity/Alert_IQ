"""Smoke tests for embedding and retrieval relevance."""
from typing import Any, Dict, List, Sequence

from src.embedding_similarity import rank_chunks


class EmbeddingSanityChecker:
    """Run known query/source cases against stored chunk embeddings."""

    def __init__(self, embedding_client: Any) -> None:
        self.embedding_client = embedding_client

    def run_case(self, query: str, expected_source: str, chunk_records: Sequence[Dict[str, Any]], top_k: int = 3) -> Dict[str, Any]:
        query_embedding = self.embedding_client.embed_texts([query])[0]
        ranked = rank_chunks(query_embedding, chunk_records, top_k=top_k)
        top = ranked[0] if ranked else None
        expected_rank = next(
            (index + 1 for index, item in enumerate(ranked) if item.get("metadata", {}).get("source") == expected_source),
            None,
        )
        return {
            "query": query,
            "expected_source": expected_source,
            "top_source": top.get("metadata", {}).get("source") if top else None,
            "top_score": round(top["score"], 4) if top else None,
            "expected_rank": expected_rank,
            "passed": expected_rank == 1,
            "top_k": top_k,
        }

    def run(self, test_cases: Sequence[Dict[str, str]], chunk_records: Sequence[Dict[str, Any]], top_k: int = 3) -> Dict[str, Any]:
        results = [
            self.run_case(case["query"], case["expected_source"], chunk_records, top_k)
            for case in test_cases
        ]
        passed = sum(result["passed"] for result in results)
        failed = len(results) - passed
        return {
            "tests": len(results),
            "passed": passed,
            "failed": failed,
            "results": results,
            "all_passed": failed == 0,
        }


def explain_failures(report: Dict[str, Any]) -> List[str]:
    """Return concrete debugging risks for failed sanity cases."""
    explanations = []
    for result in report["results"]:
        if result["passed"]:
            continue
        explanations.append(
            f"Query '{result['query']}' expected '{result['expected_source']}' at rank 1 "
            f"but got '{result['top_source']}'. Check model consistency, vector/chunk alignment, "
            "metadata, text cleaning, and the similarity metric."
        )
    return explanations


def format_sanity_report(report: Dict[str, Any]) -> str:
    """Format a simple human-readable sanity report."""
    lines = [
        "embedding sanity report",
        f"tests: {report['tests']} passed: {report['passed']} failed: {report['failed']}",
    ]
    for row in report["results"]:
        lines.append(
            f"- {row['query']} | expected={row['expected_source']} | "
            f"top={row['top_source']} | score={row['top_score']} | "
            f"rank={row['expected_rank']} | passed={row['passed']}"
        )
    failures = explain_failures(report)
    if failures:
        lines.append("risks:")
        lines.extend(f"- {item}" for item in failures)
    return "\n".join(lines)


DEFAULT_TEST_CASES = [
    {
        "query": "How can a learner reset their password?",
        "expected_source": "account-guide.md",
    },
    {
        "query": "When does the cafeteria menu change?",
        "expected_source": "campus-guide.md",
    },
]
