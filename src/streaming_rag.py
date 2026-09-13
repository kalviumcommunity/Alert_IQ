"""Server-sent event streaming adapter for the Alert_IQ RAG pipeline."""
import json
import time
from typing import Any, Dict, Iterator

from src.rag_pipeline import RAGPipeline, assemble_stage, retrieve_stage, generate_stage


def sse_event(payload: Dict[str, Any]) -> str:
    """Serialize one event using the Server-Sent Events data format."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class StreamingRAGService:
    """Run retrieval once, emit citations, then progressively emit the answer."""

    def __init__(self, pipeline: RAGPipeline | None = None) -> None:
        self.pipeline = pipeline or RAGPipeline()

    def events(self, question: str, chunk_words: int = 4, delay_seconds: float = 0.01) -> Iterator[str]:
        if not question or not question.strip():
            yield sse_event({"type": "error", "message": "Question cannot be empty."})
            return

        try:
            chunks = retrieve_stage(
                query=question.strip(),
                vector_store=self.pipeline.store,
                top_k=2,
                dimension=self.pipeline.dimension,
            )
            context = assemble_stage(chunks)

            sources = [
                {
                    "id": chunk.id,
                    "label": f"[{index}]",
                    "document": chunk.source_document,
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "score": chunk.score,
                    "chunk_index": chunk.chunk_index,
                }
                for index, chunk in enumerate(chunks, 1)
            ]
            yield sse_event({"type": "citations", "sources": sources})

            answer = generate_stage(
                query=question.strip(),
                context=context,
                chat_client=self.pipeline.client,
            )

            words = answer.split()
            size = max(1, chunk_words)
            for start in range(0, len(words), size):
                text = " ".join(words[start:start + size])
                if start + size < len(words):
                    text += " "
                yield sse_event({"type": "token", "text": text})
                if delay_seconds > 0:
                    time.sleep(delay_seconds)

            yield sse_event({"type": "done"})
        except Exception:
            yield sse_event({"type": "error", "message": "The answer stopped streaming. Please retry."})
