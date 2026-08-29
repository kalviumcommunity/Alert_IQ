"""End-to-end corpus ingestion and validation for the Alert_IQ RAG pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from src.cleaning_pipeline import TextCleaner
from src.document_loader import Document, DocumentLoader
from src.token_chunking import TokenChunker


@dataclass
class IngestionFailure:
    """A file that could not complete ingestion, with an actionable reason."""

    path: str
    error: str


@dataclass
class IngestionResult:
    """Complete, auditable result of one corpus ingestion run."""

    files: List[str] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    chunks: list = field(default_factory=list)
    failures: List[IngestionFailure] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def is_complete(self) -> bool:
        """True only when every discovered file has a success or failure outcome."""
        return self.document_count + self.failure_count == self.file_count

    @property
    def total_tokens(self) -> int:
        return sum(chunk.metadata.get("token_count", chunk.token_count) for chunk in self.chunks)

    def validate(self) -> None:
        """Fail loudly if a file disappeared from the ingestion accounting."""
        if not self.is_complete:
            raise AssertionError(
                "A document was silently dropped: "
                f"files={self.file_count}, documents={self.document_count}, "
                f"failures={self.failure_count}"
            )


class IngestionPipeline:
    """Run load -> clean -> token-chunk -> metadata as one auditable operation."""

    def __init__(self, chunk_size: int = 400, overlap: int = 60) -> None:
        self.chunker = TokenChunker()
        self.chunk_size = chunk_size
        self.overlap = overlap

    @staticmethod
    def discover_files(folder: str | Path) -> List[Path]:
        """Discover every file recursively, including unsupported files."""
        root = Path(folder)
        if not root.exists():
            return []
        return sorted(path for path in root.rglob("*") if path.is_file())

    def ingest_file(self, path: Path) -> tuple[Document | None, list, str | None]:
        """Load, clean, and chunk one file while returning its failure separately."""
        document, error = DocumentLoader.load_file(path)
        if document is None:
            return None, [], error or "Unknown ingestion error"

        try:
            cleaned, _ = TextCleaner.clean_document(document)
            chunks = self.chunker.chunk_document(
                cleaned,
                size=self.chunk_size,
                overlap=self.overlap,
            )
            if not chunks:
                return None, [], "No chunks produced after cleaning."
            return cleaned, chunks, None
        except Exception as exc:
            return None, [], f"Pipeline error: {exc}"

    def ingest(self, folder: str | Path) -> IngestionResult:
        """Run the complete pipeline and account for every discovered file."""
        files = self.discover_files(folder)
        result = IngestionResult(files=[str(path.as_posix()) for path in files])

        for path in files:
            document, chunks, error = self.ingest_file(path)
            if error:
                result.failures.append(
                    IngestionFailure(path=path.as_posix(), error=error)
                )
                continue

            result.documents.append(document)
            result.chunks.extend(chunks)

        result.validate()
        return result


def validate_ingestion_result(result: IngestionResult) -> Dict[str, Any]:
    """Return machine-readable validation evidence for an ingestion run."""
    result.validate()
    return {
        "files_discovered": result.file_count,
        "documents_ingested": result.document_count,
        "chunks_created": result.chunk_count,
        "failures": result.failure_count,
        "accounting_reconciles": result.is_complete,
        "total_chunk_tokens": result.total_tokens,
    }


def generate_ingestion_report(
    corpus_dir: str | Path | None = None,
    sample_count: int = 3,
) -> str:
    """Run the corpus pipeline and print validation plus sample chunk evidence."""
    if corpus_dir is None:
        corpus_dir = Path(__file__).resolve().parent.parent / "data" / "corpus"

    pipeline = IngestionPipeline()
    result = pipeline.ingest(corpus_dir)
    summary = validate_ingestion_result(result)

    lines = [
        "=" * 80,
        "🚀 Alert_IQ - End-to-End Corpus Ingestion Validation",
        "=" * 80,
        f"📂 Corpus Directory : {corpus_dir}",
        f"📄 Files Discovered : {summary['files_discovered']}",
        f"✅ Documents Ingested: {summary['documents_ingested']}",
        f"🧩 Chunks Created   : {summary['chunks_created']}",
        f"⚠️ Failures         : {summary['failures']}",
        f"🔎 Accounting Check : {summary['files_discovered']} = "
        f"{summary['documents_ingested']} + {summary['failures']} "
        f"→ {summary['accounting_reconciles']}",
        f"🔢 Chunk Tokens     : {summary['total_chunk_tokens']:,}",
        "",
    ]

    if result.failures:
        lines.append("❌ FAILED FILES")
        lines.append("-" * 80)
        for failure in result.failures:
            lines.append(f"• {failure.path}: {failure.error}")
        lines.append("")
    else:
        lines.append("✅ No ingestion failures.")
        lines.append("")

    lines.extend(["🔬 SAMPLE RETRIEVABLE CHUNKS", "-" * 80])
    for index, chunk in enumerate(result.chunks[:sample_count], 1):
        lines.extend(
            [
                f"[{index}] {chunk.source}",
                f"    text     : {chunk.content[:120]!r}",
                f"    metadata : {chunk.metadata}",
                "",
            ]
        )

    lines.append("=" * 80)
    lines.append("🏁 VALIDATION PASSED: every discovered file has a recorded outcome.")
    lines.append("=" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_ingestion_report())
