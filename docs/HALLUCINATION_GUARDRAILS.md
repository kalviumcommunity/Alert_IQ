# Hallucination Guardrails & Refusal Handling

## Why the guardrail exists

A RAG system should not generate an answer merely because a query was received. If retrieval is empty or weak, generation has no reliable evidence to ground the response. The guardrail therefore runs **before generation**.

## Retrieval-quality rule

The default policy is deliberately conservative:

- `MIN_TOP_SCORE = 0.72`
- `MIN_SUPPORTING_CHUNKS = 1`
- empty retrieval -> refuse
- fewer than the required number of chunks at or above `0.72` -> refuse
- otherwise -> allow grounded generation

The threshold is configurable because it should ultimately be tuned from retrieval evaluation rather than treated as a universal constant.

## Pipeline position

```text
Question
   |
   v
Retrieve candidates
   |
   v
Evaluate retrieval strength
   |----------------------|
   | weak / empty         | strong
   v                      v
Safe refusal          Grounded generation
   |                      |
   v                      v
No sources             Cited answer
```

The key safety property is that the model is **not called** for weak retrieval.

## Usage

```python
from src.hallucination_guardrails import guarded_answer

result = guarded_answer(
    question,
    retrieve=lambda q, k: retrieve_stage(q, store, top_k=k),
    generate=lambda q, chunks: generate_stage(
        q,
        assemble_stage(chunks),
    ),
)
```

A successful result has `status == "answered"` and contains only the chunks that passed the threshold. A weak result has `status` `refused_no_context` or `refused_weak_context`, an empty `sources` list, and the safe refusal message.

## Sample behavior

**Strong evidence:** scores `[0.91, 0.55]` -> one supporting chunk -> `answered`.

**Weak evidence:** scores `[0.41, 0.68]` -> zero supporting chunks -> `refused_weak_context`.

**No evidence:** scores `[]` -> `refused_no_context`.

See `logs/hallucination_guardrails_demo.log` for the committed sample outputs and `tests/test_hallucination_guardrails.py` for automated coverage.

## Trade-off

A threshold that is too high can refuse legitimate questions; a threshold that is too low can let unsupported answers through. In higher-stakes domains, the safer bias is to refuse when the evidence is not strong enough, then tune the threshold using measured retrieval quality and real queries.
