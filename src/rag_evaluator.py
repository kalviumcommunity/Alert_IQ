"""Offline evaluator for correctness, grounding, and citation quality."""
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass
class EvaluationResult:
    id: str
    question: str
    answer: str
    correctness: float
    grounding: float
    citation_quality: float
    cited_sources: List[str]
    expected_sources: List[str]
    failure: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_-]+", text.lower()))


def score_correctness(answer: str, expected_answer: str) -> float:
    """Score expected-answer coverage using normalized token overlap."""
    expected = _tokens(expected_answer)
    actual = _tokens(answer)
    if not expected:
        return 1.0 if not actual else 0.0
    return round(len(expected & actual) / len(expected), 2)


def score_grounding(answer: str, context: Sequence[Any]) -> float:
    """Score how much answer vocabulary is supported by retrieved context."""
    context_text = " ".join(
        c if isinstance(c, str) else getattr(c, "text", str(c)) for c in context
    )
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 1.0
    supported = _tokens(context_text)
    return round(len(answer_tokens & supported) / len(answer_tokens), 2)


def score_citations(cited_sources: Sequence[str], expected_sources: Sequence[str]) -> float:
    """Score citation accuracy: cited sources must match supporting expected sources."""
    cited = set(cited_sources)
    expected = set(expected_sources)
    if not expected:
        return 1.0 if not cited else 0.0
    if not cited:
        return 0.0
    return round(len(cited & expected) / len(expected), 2)


def evaluate_case(case: Dict[str, Any], answer: str, context: Sequence[Any], cited_sources: Sequence[str]) -> EvaluationResult:
    correctness = score_correctness(answer, case["expected_answer"])
    grounding = score_grounding(answer, context)
    citation_quality = score_citations(cited_sources, case["expected_sources"])
    failures = []
    if correctness < 0.8:
        failures.append("low correctness")
    if grounding < 0.8:
        failures.append("weak grounding")
    if citation_quality < 1.0:
        failures.append("citation mismatch")
    return EvaluationResult(
        id=case["id"], question=case["question"], answer=answer,
        correctness=correctness, grounding=grounding,
        citation_quality=citation_quality, cited_sources=list(cited_sources),
        expected_sources=list(case["expected_sources"]), failure="; ".join(failures)
    )


def summarize(results: Sequence[EvaluationResult]) -> Dict[str, Any]:
    if not results:
        return {"cases": 0, "correctness": 0, "grounding": 0, "citation_quality": 0, "overall_quality": 0, "failures": []}
    correctness = round(sum(r.correctness for r in results) / len(results), 2)
    grounding = round(sum(r.grounding for r in results) / len(results), 2)
    citations = round(sum(r.citation_quality for r in results) / len(results), 2)
    failures = [{"id": r.id, "failure": r.failure} for r in results if r.failure]
    return {
        "cases": len(results), "correctness": correctness,
        "grounding": grounding, "citation_quality": citations,
        "overall_quality": round((correctness + grounding + citations) / 3, 2),
        "failures": failures,
        "likely_causes": [
            "Lexical scoring is conservative when wording differs from the expected answer.",
            "Citation failures indicate source attribution did not match the expected supporting document.",
        ] if failures else []
    }


def run_evaluation(cases: Sequence[Dict[str, Any]], runner) -> Dict[str, Any]:
    """Run cases through a supplied RAG runner and return scored results."""
    results = []
    for case in cases:
        output = runner(case["question"])
        results.append(evaluate_case(
            case, output["answer"], output.get("context", []), output.get("cited_sources", [])
        ))
    return {"results": [r.to_dict() for r in results], "summary": summarize(results)}


def load_test_set(path: str | Path) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    print("Use run_evaluation(cases, runner) to evaluate a RAG implementation.")
