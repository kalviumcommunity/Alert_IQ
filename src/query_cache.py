"""Small in-memory TTL cache for deterministic RAG query responses."""
import hashlib
import time
from typing import Any, Dict, Optional


class QueryCache:
    """Cache responses using normalized questions plus relevant settings."""

    def __init__(self, ttl_seconds: int = 15 * 60) -> None:
        self.ttl_seconds = ttl_seconds
        self._entries: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def cache_key(question: str, filters: Optional[Dict[str, Any]] = None, settings: Optional[Dict[str, Any]] = None) -> str:
        raw = {
            "question": question.strip().lower(),
            "filters": filters or {},
            "settings": settings or {},
        }
        return hashlib.sha256(repr(raw).encode("utf-8")).hexdigest()

    def get(self, question: str, filters: Optional[Dict[str, Any]] = None, settings: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        key = self.cache_key(question, filters, settings)
        entry = self._entries.get(key)
        if entry is None:
            return None
        if time.time() - entry["created_at"] > self.ttl_seconds:
            self._entries.pop(key, None)
            return None
        return entry["response"]

    def set(self, question: str, response: Any, filters: Optional[Dict[str, Any]] = None, settings: Optional[Dict[str, Any]] = None) -> None:
        key = self.cache_key(question, filters, settings)
        self._entries[key] = {"created_at": time.time(), "response": response}

    def clear(self) -> None:
        self._entries.clear()

    def size(self) -> int:
        return len(self._entries)
