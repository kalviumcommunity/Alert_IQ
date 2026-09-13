# RAG Evaluation & Answer Quality Scoring

## Dimensions

Alert_IQ evaluates three dimensions separately:

1. **Correctness** — does the answer cover the expected answer?
2. **Grounding** — are the answer's terms supported by retrieved context?
3. **Citation quality** — do cited source documents match the sources expected to support the answer?

Scores are normalized from 0 to 1. A case is flagged when correctness or grounding is below 0.80, or when citation quality is below 1.00.

## Test set

`data/rag_evaluation_test_set.json` contains four cases: database replica triage, connection-pool follow-up guidance, telemetry SLA threshold, and an out-of-domain Stripe question that should be refused rather than hallucinated.

## Results

`logs/rag_evaluation_results.json` records the answer, retrieved-support expectation, citation check, and aggregate summary. The committed benchmark produced 1.00 correctness, 1.00 grounding, and 1.00 citation quality across four cases.

## Failure analysis

The evaluator reports failures rather than hiding them. The current benchmark has no failures. In production, lexical overlap can under-score a semantically correct answer when wording differs; this is a reason to complement the deterministic checks with human review or an LLM-as-judge rubric. Citation mismatches should trigger inspection of source selection and citation mapping.

## Running an evaluation

```python
from src.rag_evaluator import load_test_set, run_evaluation

cases = load_test_set("data/rag_evaluation_test_set.json")
report = run_evaluation(cases, runner=my_rag_runner)
print(report["summary"])
```

The runner should return `answer`, `context`, and `cited_sources`. This keeps evaluation independent from a particular model provider.

## Grounding long conversations

For long conversational RAG sessions, retain recent turns under a token budget, rewrite follow-ups into standalone queries, retrieve fresh evidence for each turn, and require citations to map to that retrieved evidence. Older conversational history should be summarized or trimmed rather than allowing unbounded text into the generation prompt.
