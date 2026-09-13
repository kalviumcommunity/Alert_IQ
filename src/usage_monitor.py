"""Structured RAG request logging, token/cost tracking, and usage summaries."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

MODEL_INPUT_COST_PER_1K = 0.00015
MODEL_OUTPUT_COST_PER_1K = 0.00060

logger = logging.getLogger("rag_app")


def estimate_tokens(text: str) -> int:
    """Conservative offline approximation: roughly four characters per token."""
    return max(1, (len(text) + 3) // 4) if text else 0


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return round(
        (input_tokens / 1000) * MODEL_INPUT_COST_PER_1K
        + (output_tokens / 1000) * MODEL_OUTPUT_COST_PER_1K,
        6,
    )


def build_usage(question: str, answer: str, cache_hit: bool, model_name: str = "gemini-2.5-flash") -> Dict[str, Any]:
    input_tokens = estimate_tokens(question)
    output_tokens = estimate_tokens(answer)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": estimate_cost(input_tokens, output_tokens),
        "cache_hit": cache_hit,
        "model_name": model_name,
    }


def log_rag_request(
    request_id: str,
    question: str,
    answer: str,
    sources: Any,
    cache_hit: bool,
    latency_ms: float,
    error: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    usage = usage or build_usage(question, answer, cache_hit)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "question": question,
        "answer_preview": answer[:180],
        "sources": sources,
        "cache_hit": cache_hit,
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "estimated_cost": usage["estimated_cost"],
        "model_name": usage.get("model_name", "unknown"),
        "latency_ms": round(latency_ms, 2),
        "error": error,
    }
    logger.info(json.dumps(record, ensure_ascii=False))
    return record


def summarize_usage(log_records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(log_records)
    total_requests = len(records)
    cache_hits = sum(1 for item in records if item.get("cache_hit"))
    errors = sum(1 for item in records if item.get("error"))
    total_cost = sum(float(item.get("estimated_cost", 0)) for item in records)
    total_input = sum(int(item.get("input_tokens", 0)) for item in records)
    total_output = sum(int(item.get("output_tokens", 0)) for item in records)
    average_latency = sum(float(item.get("latency_ms", 0)) for item in records) / max(total_requests, 1)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_requests": total_requests,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / max(total_requests, 1), 2),
        "errors": errors,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_estimated_cost": round(total_cost, 6),
        "average_latency_ms": round(average_latency, 2),
    }


def read_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    records = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records
