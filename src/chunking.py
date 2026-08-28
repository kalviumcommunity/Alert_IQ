"""
Document Chunking Strategies for Alert_IQ RAG Pipeline
Compares fixed-size and paragraph-based chunking and reports chunk statistics.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.cleaning_pipeline import TextCleaner
from src.document_loader import Document, DocumentLoader
from src.token_counter import count_tokens


@dataclass
class Chunk:
    """A retrievable piece of a source document."""
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def character_count(self) -> int:
        return len(self.content)

    @property
    def token_count(self) -> int:
        return count_tokens(self.content)


class Chunker:
    """Provides comparable document chunking strategies."""

    @staticmethod
    def fixed_chunks(text: str, size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into fixed character windows with boundary overlap."""
        if not text or size <= 0:
            return []
        if overlap < 0 or overlap >= size:
            raise ValueError("overlap must be >= 0 and smaller than size")

        chunks: List[str] = []
        step = size - overlap
        for start in range(0, len(text), step):
            chunk = text[start:start + size].strip()
            if chunk:
                chunks.append(chunk)
            if start + size >= len(text):
                break
        return chunks

    @staticmethod
    def paragraph_chunks(text: str) -> List[str]:
        """Split text at paragraph boundaries while discarding empty paragraphs."""
        if not text:
            return []
        return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]

    @classmethod
    def chunk_document(
        cls,
        document: Document,
        strategy: str = "paragraph",
        size: int = 500,
        overlap: int = 50,
    ) -> List[Chunk]:
        """Chunk one document using the requested strategy."""
        strategy = strategy.lower().strip()
        if strategy == "fixed":
            texts = cls.fixed_chunks(document.content, size=size, overlap=overlap)
        elif strategy == "paragraph":
            texts = cls.paragraph_chunks(document.content)
        else:
            raise ValueError("strategy must be 'fixed' or 'paragraph'")

        return [
            Chunk(
                content=text,
                source=document.source,
                metadata={
                    "filename": document.metadata.get("filename", Path(document.source).name),
                    "chunk_index": index,
                    "chunk_strategy": strategy,
                },
            )
            for index, text in enumerate(texts)
        ]

    @classmethod
    def compare_strategies(
        cls,
        document: Document,
        fixed_size: int = 500,
        fixed_overlap: int = 50,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare fixed-size and paragraph strategies on the same document."""
        results: Dict[str, Dict[str, Any]] = {}
        for strategy in ("fixed", "paragraph"):
            chunks = cls.chunk_document(
                document,
                strategy=strategy,
                size=fixed_size,
                overlap=fixed_overlap,
            )
            sizes = [chunk.character_count for chunk in chunks]
            results[strategy] = {
                "chunks": chunks,
                "chunk_count": len(chunks),
                "average_chars": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
                "min_chars": min(sizes) if sizes else 0,
                "max_chars": max(sizes) if sizes else 0,
                "sample": chunks[0].content[:100] if chunks else "",
            }
        return results


def choose_corpus_strategy(documents: List[Document]) -> Tuple[str, Dict[str, Any]]:
    """
    Compare both strategies over the corpus and choose paragraph boundaries.

    The sample corpus contains policies, runbooks, and structured service-health
    sections, so preserving paragraph/section boundaries is preferred for retrieval.
    """
    if not documents:
        return "paragraph", {"reason": "No documents available for comparison."}

    comparisons = [Chunker.compare_strategies(document) for document in documents]
    fixed_count = sum(item["fixed"]["chunk_count"] for item in comparisons)
    paragraph_count = sum(item["paragraph"]["chunk_count"] for item in comparisons)

    return "paragraph", {
        "reason": (
            "The Alert_IQ corpus is structured as policies, runbooks, and report "
            "sections. Paragraph chunking preserves those semantic boundaries, "
            "whereas fixed-size chunking can cut an instruction or sentence in half."
        ),
        "documents_evaluated": len(documents),
        "fixed_chunk_count": fixed_count,
        "paragraph_chunk_count": paragraph_count,
    }


def generate_chunking_report(corpus_dir: str | Path | None = None) -> str:
    """Run the chunking comparison over the sample corpus and print evidence."""
    if corpus_dir is None:
        corpus_dir = Path(__file__).resolve().parent.parent / "data" / "corpus"

    raw_docs, skipped = DocumentLoader.load_directory(corpus_dir)
    documents, _ = TextCleaner.clean_corpus(raw_docs)

    lines = [
        "=" * 80,
        "📚 Alert_IQ - Document Chunking Strategy Comparison",
        "=" * 80,
        f"📂 Corpus Directory : {corpus_dir}",
        f"📄 Documents Loaded : {len(documents)}",
        f"⚠️ Files Skipped    : {len(skipped)}",
        "",
    ]

    total_fixed = 0
    total_paragraph = 0

    for document in documents:
        comparison = Chunker.compare_strategies(document)
        fixed = comparison["fixed"]
        paragraph = comparison["paragraph"]
        total_fixed += fixed["chunk_count"]
        total_paragraph += paragraph["chunk_count"]

        filename = document.metadata.get("filename", Path(document.source).name)
        lines.extend([
            f"📄 {filename}",
            f"  fixed     : {fixed['chunk_count']} chunks, avg {fixed['average_chars']} chars",
            f"  paragraph : {paragraph['chunk_count']} chunks, avg {paragraph['average_chars']} chars",
            f"  fixed sample     : {fixed['sample']!r}",
            f"  paragraph sample : {paragraph['sample']!r}",
            "",
        ])

    strategy, decision = choose_corpus_strategy(documents)
    lines.extend([
        "=" * 80,
        "🏁 CORPUS STRATEGY DECISION",
        "=" * 80,
        f"• Fixed-size chunks      : {total_fixed}",
        f"• Paragraph chunks       : {total_paragraph}",
        f"• Selected strategy      : {strategy}",
        f"• Justification          : {decision['reason']}",
        "=" * 80,
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    print(generate_chunking_report())
