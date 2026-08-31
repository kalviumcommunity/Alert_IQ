"""
Unit tests for Alert_IQ Vector Database storage, collection sizing, schema design, and read-back verification.
"""
import sys
from types import SimpleNamespace
from pathlib import Path
import pytest

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.vector_store import VectorStore, VectorRecord, verify_record, run_readback_demo


class FakeCollection:
    def __init__(self):
        self.records = {}

    def count(self):
        return len(self.records)

    def get(self, ids=None, limit=None, include=None):
        selected = list(self.records.values()) if ids is None else [self.records[i] for i in ids if i in self.records]
        if limit:
            selected = selected[:limit]
        return {
            "ids": [r["id"] for r in selected],
            "embeddings": [r["vector"] for r in selected],
            "documents": [r["text"] for r in selected],
            "metadatas": [r["metadata"] for r in selected]
        }

    def upsert(self, ids, embeddings, documents, metadatas):
        for values in zip(ids, embeddings, documents, metadatas):
            i, v, t, m = values
            self.records[i] = {"id": i, "vector": v, "text": t, "metadata": m}


def test_vector_store_round_trip(monkeypatch, tmp_path):
    """Test round-trip insertion and read-back with mock persistent client."""
    class FakeClient:
        def __init__(self, path): self.collection = FakeCollection()
        def get_or_create_collection(self, name, metadata): return self.collection
        def heartbeat(self): return 123456
    fake_module = SimpleNamespace(PersistentClient=FakeClient, EphemeralClient=FakeClient)
    monkeypatch.setitem(sys.modules, "chromadb", fake_module)

    store = VectorStore(str(tmp_path))
    record = {
        "id": "account-guide.md:0",
        "embedding": [0.1, 0.2, 0.3],
        "text": "Reset password.",
        "metadata": {"source": "account-guide.md", "chunk_index": 0, "section": "Account access"}
    }
    store.upsert_records([record], dimension=3)
    stored = verify_record(store, record["id"], 3)

    assert stored["id"] == record["id"]
    assert stored["vector_length"] == 3
    assert stored["text"] == record["text"]
    assert stored["metadata"]["source"] == "account-guide.md"


class TestVectorStore:
    """Test suite covering vector DB reachability, collection sizing, schema, and readback."""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Provides an isolated vector store instance backed by a temporary directory."""
        return VectorStore(path=str(tmp_path), collection_name="test_collection")

    @pytest.fixture
    def ephemeral_store(self):
        """Provides an in-memory vector store instance."""
        return VectorStore(in_memory=True, collection_name="ephemeral_test")

    def test_reachability_and_diagnostics(self, temp_store):
        """Task 1: Confirm vector database is reachable and provides diagnostics."""
        assert temp_store.is_reachable() is True
        diag = temp_store.verify_reachability()
        assert diag["status"] == "online"
        assert diag["collection_name"] == "test_collection"
        assert diag["distance_metric"] == "cosine"
        assert diag["item_count"] == 0

    def test_ephemeral_reachability(self, ephemeral_store):
        """Task 1: Confirm in-memory vector database is reachable."""
        assert ephemeral_store.is_reachable() is True
        diag = ephemeral_store.verify_reachability()
        assert diag["storage_type"] == "in-memory"

    def test_collection_dimension_locking(self, temp_store):
        """Task 2: Confirm collection properly locks and checks vector dimensions."""
        temp_store.ensure_dimension(768)
        assert temp_store.dimension == 768

        with pytest.raises(ValueError, match="dimension must be positive"):
            temp_store.ensure_dimension(0)

    def test_wrong_dimension_fails(self, temp_store):
        """Task 2: Confirm dimension mismatches are rejected on insert."""
        record = {
            "id": "bad_dim_record",
            "embedding": [0.1, 0.2],
            "text": "sample text",
            "metadata": {"source": "test.txt"}
        }
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            temp_store.upsert_records([record], dimension=4)

    def test_record_schema_and_dict_conversion(self):
        """Task 3: Confirm VectorRecord schema structure and dictionary round-trip."""
        record = VectorRecord(
            id="rec_001",
            vector=[0.1, 0.2, 0.3],
            text="Database connection pool exhausted.",
            metadata={
                "source_document": "runbook_db.md",
                "source": "runbook_db.md",
                "chunk_index": 2,
                "section": "Troubleshooting",
                "page": 1
            }
        )
        d = record.to_dict()
        assert d["id"] == "rec_001"
        assert d["embedding"] == [0.1, 0.2, 0.3]
        assert d["text"] == "Database connection pool exhausted."
        assert d["metadata"]["chunk_index"] == 2

        restored = VectorRecord.from_dict(d)
        assert restored.id == record.id
        assert restored.vector == record.vector
        assert restored.text == record.text
        assert restored.metadata == record.metadata

    def test_insert_and_readback_record(self, temp_store):
        """Task 4: Insert a test record and read it back successfully, asserting all fields."""
        record = VectorRecord(
            id="incident_policy_001",
            vector=[0.05, -0.12, 0.88],
            text="P1 incidents require response within 15 minutes.",
            metadata={
                "source_document": "incident_policy.txt",
                "source": "incident_policy.txt",
                "chunk_index": 0,
                "section": "SLA Matrix",
                "page": 1
            }
        )
        temp_store.insert_record(record, dimension=3)
        assert temp_store.count() == 1

        stored = verify_record(temp_store, "incident_policy_001", expected_dimension=3)
        assert stored["id"] == "incident_policy_001"
        assert stored["vector_length"] == 3
        assert stored["text"] == "P1 incidents require response within 15 minutes."
        assert stored["metadata"]["source_document"] == "incident_policy.txt"
        assert stored["metadata"]["section"] == "SLA Matrix"
        assert stored["metadata"]["page"] == 1

    def test_get_nonexistent_record(self, temp_store):
        """Task 4: Reading back a non-existent record returns None."""
        assert temp_store.get_record("missing_id") is None

        with pytest.raises(AssertionError, match="Record not found"):
            verify_record(temp_store, "missing_id", expected_dimension=3)

    def test_readback_demo_execution(self, tmp_path):
        """Task 4 & 5: Confirm the readback demo runs and returns formatted report."""
        report = run_readback_demo(persist_path=str(tmp_path), collection_name="demo_coll")
        assert "Alert_IQ - Vector Database Setup & Read-Back Verification Report" in report
        assert "ONLINE" in report
        assert "doc_incident_policy_chunk_001" in report
        assert "768 dimensions" in report
