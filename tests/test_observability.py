import json

from src.query_cache import QueryCache
from src.usage_monitor import build_usage, estimate_cost, summarize_usage


def test_identical_queries_share_cache_entry():
    cache = QueryCache(ttl_seconds=60)
    response = {"answer": "grounded"}
    cache.set("  What is P1? ", response, settings={"model_name": "test"})
    assert cache.get("what is p1?", settings={"model_name": "test"}) == response
    assert cache.get("what is p1?", settings={"model_name": "other"}) is None


def test_cache_expires(monkeypatch):
    cache = QueryCache(ttl_seconds=10)
    cache.set("hello", {"answer": "world"})
    import src.query_cache as module
    now = module.time.time()
    monkeypatch.setattr(module.time, "time", lambda: now + 11)
    assert cache.get("hello") is None


def test_usage_cost_and_summary():
    usage = build_usage("abcd", "abcdefgh", False, "test-model")
    assert usage["input_tokens"] == 1
    assert usage["output_tokens"] == 2
    assert usage["estimated_cost"] == estimate_cost(1, 2)

    records = [
        {"cache_hit": False, "estimated_cost": 0.001, "latency_ms": 100, "input_tokens": 10, "output_tokens": 20},
        {"cache_hit": True, "estimated_cost": 0.0001, "latency_ms": 5, "input_tokens": 10, "output_tokens": 5},
    ]
    summary = summarize_usage(records)
    assert summary["total_requests"] == 2
    assert summary["cache_hits"] == 1
    assert summary["cache_hit_rate"] == 0.5
    assert summary["total_estimated_cost"] == 0.0011
    assert summary["average_latency_ms"] == 52.5
