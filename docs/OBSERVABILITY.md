# Caching, Logging & Usage Monitoring

## Query caching

`QueryCache` uses a SHA-256 key built from the normalized question, filters, and relevant settings. Entries expire after a configurable TTL (`RAG_CACHE_TTL_SECONDS`, default 15 minutes). Model settings are included so changing the model does not reuse an incompatible cached response.

A cache hit returns the stored RAG response without running embedding, retrieval, or generation again.

## Structured request logs

Each `/query` request records a UTC timestamp, request ID, question, answer preview, retrieved sources, cache hit/miss, model, latency, token estimates, cost estimate, and any error. The log helper emits JSON so records can be searched or processed by monitoring tools.

Do not log API keys, authentication headers, or unnecessary sensitive user data.

## Usage and cost

The monitor records model token usage when supplied by the integration and otherwise uses a documented offline approximation of roughly four characters per token. Estimated cost uses:

- input: `$0.00015 / 1K tokens`
- output: `$0.00060 / 1K tokens`

These rates are configuration/documentation examples for monitoring and are not presented as live provider billing rates.

## Usage report

`summarize_usage()` aggregates total requests, cache hits/rate, errors, input/output tokens, estimated cost, and average latency. The committed `logs/usage_summary.json` demonstrates the expected report shape from `logs/usage_sample.jsonl`.

## Debugging a bad answer

A request ID lets an operator correlate the question, retrieved source IDs, cache state, latency, usage, and errors. This makes it possible to distinguish retrieval problems, stale/reused results, generation failures, and expensive slow requests.
