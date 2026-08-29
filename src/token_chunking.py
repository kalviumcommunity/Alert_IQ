"""Token-aware chunking with controlled overlap for the Alert_IQ RAG pipeline."""
from typing import Any, Dict, List

from src.chunking import Chunk
from src.document_loader import Document

try:
    import tiktoken
except ImportError as exc:
    tiktoken = None
    _TIKTOKEN_IMPORT_ERROR = exc
else:
    _TIKTOKEN_IMPORT_ERROR = None


class TokenChunker:
    """Split documents by tokenizer tokens instead of character count."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        if tiktoken is None:
            raise RuntimeError(
                "tiktoken is required for token-aware chunking. "
                "Install dependencies from requirements.txt."
            ) from _TIKTOKEN_IMPORT_ERROR
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)

    def token_chunks(self, text: str, size: int = 400, overlap: int = 60) -> List[str]:
        """Return token-sized text chunks with overlap between adjacent chunks."""
        if not text or size <= 0:
            return []
        if overlap < 0 or overlap >= size:
            raise ValueError("overlap must be >= 0 and smaller than size")

        tokens = self.encoding.encode(text)
        chunks: List[str] = []
        step = size - overlap
        for start in range(0, len(tokens), step):
            chunk_tokens = tokens[start:start + size]
            if not chunk_tokens:
                break
            chunk = self.encoding.decode(chunk_tokens).strip()
            if chunk:
                chunks.append(chunk)
            if start + size >= len(tokens):
                break
        return chunks

    def chunk_document(self, document: Document, size: int = 400, overlap: int = 60) -> List[Chunk]:
        """Create traceable token-sized chunks while retaining source metadata."""
        if not document.content:
            return []
        if size <= 0 or overlap < 0 or overlap >= size:
            raise ValueError("size must be positive and overlap must be >= 0 and smaller than size")

        tokens = self.encoding.encode(document.content)
        chunks: List[Chunk] = []
        step = size - overlap

        for index, start in enumerate(range(0, len(tokens), step)):
            chunk_tokens = tokens[start:start + size]
            if not chunk_tokens:
                break
            content = self.encoding.decode(chunk_tokens).strip()
            if not content:
                continue

            char_start = len(self.encoding.decode(tokens[:start]))
            char_end = len(self.encoding.decode(tokens[:start + len(chunk_tokens)]))
            metadata: Dict[str, Any] = dict(document.metadata)
            metadata.update({
                "filename": document.metadata.get("filename", document.source),
                "chunk_index": index,
                "chunk_strategy": "token",
                "token_size": size,
                "token_overlap": overlap,
                "token_start": start,
                "token_end": start + len(chunk_tokens),
                "token_count": len(chunk_tokens),
                "char_start": char_start,
                "char_end": char_end,
                "overlap_tokens": overlap if index > 0 else 0,
                "encoding": self.encoding_name,
            })
            chunks.append(Chunk(content=content, source=document.source, metadata=metadata))
            if start + size >= len(tokens):
                break

        return chunks

    def compare_overlap(self, text: str, size: int = 400, overlaps: tuple[int, ...] = (0, 60)) -> Dict[int, Dict[str, Any]]:
        """Show how changing overlap affects chunk count and token volume."""
        result: Dict[int, Dict[str, Any]] = {}
        for overlap in overlaps:
            chunks = self.token_chunks(text, size=size, overlap=overlap)
            token_counts = [len(self.encoding.encode(chunk)) for chunk in chunks]
            result[overlap] = {
                "chunk_count": len(chunks),
                "total_tokens": sum(token_counts),
                "average_tokens": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0.0,
            }
        return result

    def explain_choice(self, size: int = 400, overlap: int = 60) -> Dict[str, Any]:
        """Document the starting token-size/overlap choice for the current pipeline."""
        return {
            "encoding": self.encoding_name,
            "chunk_size_tokens": size,
            "overlap_tokens": overlap,
            "overlap_percent": round((overlap / size) * 100, 2),
            "justification": (
                "400 tokens is within the recommended 300–500 token starting range, "
                "while 60 tokens provides 15% overlap to preserve context across boundaries "
                "without duplicating excessive text. Exact values should be tuned with retrieval tests."
            ),
        }
