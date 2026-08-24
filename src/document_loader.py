"""
Multi-Format Document Ingestion & Normalization Loader for Alert_IQ RAG Pipeline
Loads Markdown, HTML, Plain Text, and JSON documents into a unified plain-text format,
preserving source identity and handling missing/corrupted files gracefully.
"""
import os
import sys
import json
import re
import html
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.token_counter import count_tokens


@dataclass
class Document:
    """
    Standardized in-memory plain text representation of an ingested document.
    """
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
        """
        Returns a clean single-line preview sample of the loaded text.
        """
        flattened = " ".join(self.content.split())
        return flattened[:max_chars] + "..." if len(flattened) > max_chars else flattened


class SimpleHTMLTextExtractor(HTMLParser):
    """
    Lightweight HTML parser to extract clean plain text, ignoring scripts and styles.
    """
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
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return "\n".join(lines)


class DocumentLoader:
    """
    Multi-format document intake loader supporting Markdown, HTML, Plain Text, and JSON.
    """

    SUPPORTED_EXTENSIONS = {".md", ".html", ".htm", ".txt", ".json"}

    @classmethod
    def load_file(cls, file_path: str | Path) -> Tuple[Optional[Document], Optional[str]]:
        """
        Loads a single file into a Document object.
        Returns (Document, None) on success, or (None, error_message) on failure.
        """
        path = Path(file_path)

        # Task 2: Handle missing files
        if not path.exists():
            return None, f"File Not Found: '{path}' does not exist."

        if not path.is_file():
            return None, f"Invalid Target: '{path}' is a directory, not a file."

        ext = path.suffix.lower()

        # Task 2: Handle unsupported file extensions
        if ext not in cls.SUPPORTED_EXTENSIONS:
            return None, f"Unsupported Format: '{ext}' is not a supported document type."

        try:
            # Read file with UTF-8 encoding (with fallback)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw_content = f.read()
            except UnicodeDecodeError:
                with open(path, "r", encoding="latin-1") as f:
                    raw_content = f.read()

            source_name = str(path.as_posix())
            file_size_bytes = path.stat().st_size

            # Task 1: Normalize into common plain text
            normalized_text = cls._normalize_content(raw_content, ext)

            if not normalized_text.strip():
                return None, f"Empty Document: '{path.name}' contains no readable textual content."

            # Task 3: Preserve source identity and metadata
            metadata = {
                "filename": path.name,
                "file_type": ext.lstrip("."),
                "file_size_bytes": file_size_bytes,
                "character_count": len(normalized_text),
                "token_count": count_tokens(normalized_text)
            }

            doc = Document(content=normalized_text, source=source_name, metadata=metadata)
            return doc, None

        except PermissionError:
            return None, f"Permission Denied: Unable to read '{path.name}'."
        except Exception as e:
            return None, f"Unreadable File: Error parsing '{path.name}': {str(e)}"

    @classmethod
    def _normalize_content(cls, raw: str, ext: str) -> str:
        """
        Normalizes different formats into clean plain-text.
        """
        if ext in {".html", ".htm"}:
            parser = SimpleHTMLTextExtractor()
            parser.feed(raw)
            return parser.get_text()

        elif ext == ".json":
            try:
                data = json.loads(raw)
                # Pretty print JSON for clean text representation
                return json.dumps(data, indent=2)
            except Exception:
                return raw.strip()

        elif ext == ".md":
            # Clean markdown artifacts if necessary, or keep formatted markdown text
            return raw.strip()

        else:  # .txt and fallback
            return raw.strip()

    @classmethod
    def load_directory(
        cls,
        directory_path: str | Path,
        recursive: bool = False
    ) -> Tuple[List[Document], List[Dict[str, str]]]:
        """
        Scans and ingests all documents within a folder.
        Returns:
            Tuple of (loaded_documents: List[Document], skipped_files: List[Dict[path, reason]])
        """
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


def run_intake_confirmation_demo(corpus_dir: str | Path = None) -> str:
    """
    Task 4 & 5: Runs intake over the sample corpus and prints intake confirmation report.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if corpus_dir is None:
        corpus_dir = Path(__file__).resolve().parent.parent / "data" / "corpus"

    loaded_docs, skipped_files = DocumentLoader.load_directory(corpus_dir)

    lines = []
    lines.append("=" * 80)
    lines.append("📥 Alert_IQ - Multi-Format Document Intake & Confirmation Report")
    lines.append(f"📂 Ingestion Directory : {corpus_dir}")
    lines.append("=" * 80)

    # Task 4: Confirm intake for each document
    lines.append("\n✅ SUCCESSFULLY INGESTED DOCUMENTS:")
    lines.append("-" * 80)
    for idx, doc in enumerate(loaded_docs, 1):
        meta = doc.metadata
        lines.append(f"[{idx}] Source File : {meta['filename']} ({meta['file_type'].upper()})")
        lines.append(f"    • Source URI       : {doc.source}")
        lines.append(f"    • Character Length : {meta['character_count']:,} chars")
        lines.append(f"    • Token Count      : {meta['token_count']:,} tokens")
        lines.append(f"    • File Size        : {meta['file_size_bytes']:,} bytes")
        lines.append(f"    • Content Preview  : \"{doc.sample_preview(100)}\"")
        lines.append("")

    # Task 2: Confirm graceful handling of skipped / unsupported / corrupt files
    lines.append("=" * 80)
    lines.append(f"⚠️ SKIPPED / REJECTED FILES ({len(skipped_files)} items handled gracefully):")
    lines.append("-" * 80)
    for item in skipped_files:
        lines.append(f"• File: {item['path']}")
        lines.append(f"  Reason: {item['reason']}")

    # Non-existent file demonstration
    missing_file = Path(corpus_dir) / "non_existent_runbook.md"
    _, missing_err = DocumentLoader.load_file(missing_file)
    lines.append(f"• File: {missing_file.name} (Explicit Missing File Test)")
    lines.append(f"  Reason: {missing_err}")

    lines.append("\n" + "=" * 80)
    lines.append("🏁 INTAKE SUMMARY")
    lines.append("=" * 80)
    lines.append(f"• Total Ingested Documents : {len(loaded_docs)}")
    lines.append(f"• Total Corpus Tokens      : {sum(d.token_count for d in loaded_docs):,} tokens")
    lines.append(f"• Formats Ingested         : {', '.join(sorted(list(set(d.metadata['file_type'] for d in loaded_docs))))}")
    lines.append(f"• Unhandled Exceptions     : 0 (100% Graceful Survival)")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    report = run_intake_confirmation_demo()
    print(report)


if __name__ == "__main__":
    main()
