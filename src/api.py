"""HTTP API for the Alert_IQ grounded RAG service."""
import os
from typing import Any, Dict

from flask import Flask, jsonify, request

from src.document_upload import DocumentUploadService
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore

app = Flask(__name__)


def load_config() -> Dict[str, Any]:
    """Load service configuration from environment variables."""
    return {
        "model_name": os.getenv("OPENAI_MODEL", "gemini-2.5-flash"),
        "vector_db_path": os.getenv("VECTOR_DB_PERSIST_DIR", "data/vector_store"),
        "collection_name": os.getenv("VECTOR_COLLECTION_NAME", "alert_iq_knowledge_base"),
        "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
        "upload_dir": os.getenv("UPLOAD_DIR", "uploads"),
        "upload_max_bytes": int(os.getenv("UPLOAD_MAX_BYTES", str(5 * 1024 * 1024))),
    }


def create_pipeline() -> RAGPipeline:
    """Build the existing RAG pipeline using environment-based vector settings."""
    config = load_config()
    store = VectorStore(path=config["vector_db_path"], collection_name=config["collection_name"])
    return RAGPipeline(vector_store=store)


def response_from_rag(question: str) -> Dict[str, Any]:
    """Run the RAG pipeline and serialize its grounded response."""
    result = create_pipeline().run(query=question)
    return {
        "answer": result.answer,
        "sources": result.retrieved_sources,
        "status": "success",
        "metadata": {
            "query": result.query,
            "model_name": load_config()["model_name"],
            "stage_metrics": result.stage_metrics,
        },
    }


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "Alert_IQ RAG API"}), 200


@app.post("/query")
def query():
    """Accept {\"question\": \"...\"} and return a grounded RAG response."""
    if not request.is_json:
        return jsonify({"status": "error", "error": "Request body must be JSON."}), 415

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"status": "error", "error": "Request body must be a JSON object."}), 400

    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"status": "error", "error": "Missing required field: question."}), 400

    try:
        return jsonify(response_from_rag(question.strip())), 200
    except Exception as exc:
        app.logger.exception("RAG query failed")
        return jsonify({"status": "error", "error": "RAG service failed.", "detail": str(exc)}), 500


@app.post("/documents")
def upload_document():
    """Store, ingest, embed, and index a multipart document at runtime."""
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"status": "error", "error": "Missing required multipart field: file."}), 400
    if not uploaded.filename:
        return jsonify({"status": "error", "error": "Uploaded file must have a filename."}), 400

    config = load_config()
    if request.content_length and request.content_length > config["upload_max_bytes"] + 1024 * 1024:
        return jsonify({"status": "error", "error": "Upload request exceeds the configured size limit."}), 413

    try:
        content = uploaded.read(config["upload_max_bytes"] + 1)
        if len(content) > config["upload_max_bytes"]:
            return jsonify({"status": "error", "error": "Uploaded file exceeds the configured size limit."}), 413

        service = DocumentUploadService()
        path = service.store_bytes(uploaded.filename, content)
        summary = service.process(path)
        return jsonify({
            "status": "indexed",
            "filename": uploaded.filename,
            "summary": summary,
        }), 201
    except ValueError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 415
    except Exception as exc:
        app.logger.exception("Document indexing failed")
        return jsonify({"status": "error", "error": "Document indexing failed.", "detail": str(exc)}), 500


if __name__ == "__main__":
    app.run(host=os.getenv("RAG_API_HOST", "127.0.0.1"), port=int(os.getenv("RAG_API_PORT", "5000")))
