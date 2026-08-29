import sys
from types import SimpleNamespace

import pytest

from src.vector_store import VectorStore, verify_record


class FakeCollection:
    def __init__(self):
        self.records = {}

    def count(self):
        return len(self.records)

    def get(self, ids=None, limit=None, include=None):
        selected = list(self.records.values()) if ids is None else [self.records[i] for i in ids if i in self.records]
        if limit:
            selected = selected[:limit]
        return {"ids": [r["id"] for r in selected], "embeddings": [r["vector"] for r in selected], "documents": [r["text"] for r in selected], "metadatas": [r["metadata"] for r in selected]}

    def upsert(self, ids, embeddings, documents, metadatas):
        for values in zip(ids, embeddings, documents, metadatas):
            i, v, t, m = values
            self.records[i] = {"id": i, "vector": v, "text": t, "metadata": m}


def test_vector_store_round_trip(monkeypatch, tmp_path):
    class FakeClient:
        def __init__(self, path): self.collection = FakeCollection()
        def get_or_create_collection(self, name, metadata): return self.collection
    fake_module = SimpleNamespace(PersistentClient=FakeClient)
    monkeypatch.setitem(sys.modules, "chromadb", fake_module)

    store = VectorStore(str(tmp_path))
    record = {"id": "account-guide.md:0", "embedding": [0.1, 0.2, 0.3], "text": "Reset password.", "metadata": {"source": "account-guide.md", "chunk_index": 0, "section": "Account access"}}
    store.upsert_records([record], dimension=3)
    stored = verify_record(store, record["id"], 3)

    assert stored["id"] == record["id"]
    assert stored["vector_length"] == 3
    assert stored["text"] == record["text"]
    assert stored["metadata"]["source"] == "account-guide.md"


def test_wrong_dimension_fails(monkeypatch, tmp_path):
    class FakeClient:
        def __init__(self, path): self.collection = FakeCollection()
        def get_or_create_collection(self, name, metadata): return self.collection
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=FakeClient))
    store = VectorStore(str(tmp_path))
    with pytest.raises(ValueError, match="dimension"):
        store.upsert_records([{"id": "x", "embedding": [1.0, 2.0], "text": "x", "metadata": {"source": "x"}}], dimension=3)
