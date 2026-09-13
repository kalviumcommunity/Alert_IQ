"""
Multi-Format Document Ingestion & Normalization Loader for Alert_IQ RAG Pipeline
Loads Markdown, HTML, Plain Text, JSON, and PDF documents into unified plain text.
"""
import os
import sys
import json
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.token_counter import count_tokens


@dataclass
class Document:
    """Standardized in-memory plain text representation of an ingested document."""
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def character_count(self) -> int:
        return len(self.content)

    @property
    def token_count(self) -> int:
        return count_tokens(self.content)

    def sample_preview(self, max_chars: int = 120) -> str:
        flattened = " ".join(self.content.split())
        return flattened[:max_chars] + "..." if len(flattened) > max_chars else flattened


class SimpleHTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {"script", "style", "noscript"}
        self.in_skip = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.skip_tags:
            self.in_skip = True
        elif tag.lower() in {"p", "h1", "h2", "h3", "h4", "li", "div", "br", "tr", "section"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag.lower() in self.skip_tags:
            self.in_skip = False
        elif tag.lower() in {"p", "h1", "h2", "h3", "h4", "li", "div", "tr", "section"}:
            self.text_parts.append("\n")

    def handle_data(self, data):
        if not self.in_skip:
            cleaned = data.strip()
            if cleaned:
                self.text_parts.append(cleaned + " ")

    def get_text(self) -> str:
        raw_text = "".join(self.text_parts)
        return "\n".join(line.strip() for line in raw_text.splitlines() if line.strip())


class DocumentLoader:
    """Multi-format loader supporting Markdown, HTML, text, JSON, and PDF."""

    SUPPORTED_EXTENSIONS = {".md", ".html", ".htm", ".txt", ".json", ".pdf"}

    @classmethod
    def load_file(cls, file_path: str | Path) -> Tuple[Optional[Document], Optional[str]]:
        path = Path(file_path)
        if not path.exists():
            return None, f"File Not Found: '{path}' does not exist."
        if not path.is_file():
            return None, f"Invalid Target: '{path}' is a directory, not a file."

        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            return None, f"Unsupported Format: '{ext}' is not a supported document type."

        try:
            if ext == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                raw_content = "\n".join((page.extract_text() or "") for page in reader.pages)
            else:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        raw_content = f.read()
                except UnicodeDecodeError:
                    with open(path, "r", encoding="latin-1") as f:
                        raw_content = f.read()

            source_name = str(path.as_posix())
            file_size_bytes = path.stat().st_size
            normalized_text = cls._normalize_content(raw_content, ext)

            if not normalized_text.strip():
                return None, f"Empty Document: '{path.name}' contains no readable textual content."

            metadata = {
                "filename": path.name,
                "file_type": ext.lstrip("."),
                "file_size_bytes": file_size_bytes,
                "character_count": len(normalized_text),
                "token_count": count_tokens(normalized_text),
            }
            return Document(content=normalized_text, source=source_name, metadata=metadata), None
        except PermissionError:
            return None, f"Permission Denied: Unable to read '{path.name}'."
        except Exception as exc:
            return None, f"Unreadable File: Error parsing '{path.name}': {exc}"

    @classmethod
    def _normalize_content(cls, raw: str, ext: str) -> str:
        if ext in {".html", ".htm"}:
            parser = SimpleHTMLTextExtractor()
            parser.feed(raw)
            return parser.get_text()
        if ext == ".json":
            try:
                return json.dumps(json.loads(raw), indent=2)
            except Exception:
                return raw.strip()
        return raw.strip()

    @classmethod
    def load_directory(cls, directory_path: str | Path, recursive: bool = False) -> Tuple[List[Document], List[Dict[str, str]]]:
        dir_path = Path(directory_path)
        if not dir_path.exists():
            return [], [{"path": str(dir_path), "reason": "Directory does not exist."}]
        loaded_docs: List[Document] = []
        skipped: List[Dict[str, str]] = []
        iterator = dir_path.rglob("*") if recursive else dir_path.iterdir()
        for item in sorted(iterator):
            if item.is_file():
                doc, err = cls.load_file(item)
                if doc:
                    loaded_docs.append(doc)
                else:
                    skipped.append({"path": item.name, "reason": err or "Unknown error"})
        return loaded_docs, skipped
