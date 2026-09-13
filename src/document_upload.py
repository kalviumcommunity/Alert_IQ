"""Runtime document upload processing for the Alert_IQ RAG service."""
import os
from pathlib import Path
from typing import Any, Dict

from src.cleaning_pipeline import TextCleaner
from src.document_loader import DocumentLoader
from src.index_corpus import CorpusIndexer
from src.ingestion_pipeline import IngestionPipeline
from src.vector_store import VectorStore


class DocumentUploadService:
    """Store, ingest, embed, and index one uploaded document without resetting the corpus."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

    def __init__(self) -> None:
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", "uploads"))
        self.max_bytes = int(os.getenv("UPLOAD_MAX_BYTES", str(5 * 1024 * 1024)))
        self.dimension = int(os.getenv("EMBEDDING_DIMENSION", "768"))
        self.vector_store = VectorStore(
            path=os.getenv("VECTOR_DB_PERSIST_DIR", "data/vector_store"),
            collection_name=os.getenv("VECTOR_COLLECTION_NAME", "alert_iq_knowledge_base"),
        )
        self.vector_store.ensure_dimension(self.dimension)
        self.ingestion = IngestionPipeline(
            chunk_size=int(os.getenv("INGESTION_CHUNK_SIZE", "400")),
            overlap=int(os.getenv("INGESTION_CHUNK_OVERLAP", "60")),
        )
        self.indexer = CorpusIndexer(vector_store=self.vector_store, dimension=self.dimension)

    def validate_filename(self, filename: str) -> str:
        """Validate extension and return the normalized suffix."""
        if not filename or not filename.strip():
            raise ValueError("A filename is required.")
        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{suffix or '[none]}'. Supported: .txt, .md, .pdf")
        return suffix

    def store_bytes(self, filename: str, content: bytes) -> Path:
        """Validate, size-check, and safely store upload bytes."""
        self.validate_filename(filename)
        if not content:
            raise ValueError("Uploaded file is empty.")
        if len(content) > self.max_bytes:
            raise ValueError(f"Uploaded file exceeds the {self.max_bytes} byte limit.")

        safe_name = Path(filename).name
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        path = self.upload_dir / safe_name
        path.write_bytes(content)
        return path

    def process(self, path: Path) -> Dict[str, Any]:
        """Run load -> clean -> chunk -> embed -> upsert for one document."""
        document, load_error = DocumentLoader.load_file(path)
        if document is None:
            raise ValueError(load_error or "Document could not be loaded.")

        cleaned, _ = TextCleaner.clean_document(document)
        chunks = self.ingestion.chunker.chunk_document(
            cleaned,
            size=self.ingestion.chunk_size,
            overlap=self.ingestion.overlap,
        )
        if not chunks:
            raise ValueError("No chunks were produced from the uploaded document.")

        records = self.indexer.prepare_records(chunks)
        self.vector_store.upsert_records(records, dimension=self.dimension)

        return {
            "document": path.as_posix(),
            "chunks": len(chunks),
            "indexed": len(records),
            "collection": self.vector_store.collection_name,
        }
