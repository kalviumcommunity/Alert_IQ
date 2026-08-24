"""
Document Cleaning & Text Normalization Pipeline for Alert_IQ RAG
Transforms raw extracted document text into clean, standardized, retrieval-ready content.
Handles: Unicode NFKC normalization, encoding artifacts, hyphenated line-wraps, boilerplate removal, and whitespace collapse.
"""
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.document_loader import Document, DocumentLoader
from src.token_counter import count_tokens


class TextCleaner:
    """
    Composable, deterministic text cleaning and normalization pipeline.
    """

    # Common boilerplate regex patterns (Task 1)
    BOILERPLATE_PATTERNS = [
        # Page numbering: "Page 1 of 12", "Page 3/10", "[Page 4]"
        r"(?i)\bpage\s+\d+\s*(?:of|/)\s*\d+\b",
        r"(?i)\[\s*page\s+\d+\s*\]",
        # Navigation breadcrumbs: "Home > Docs > Alert_IQ > Triage"
        r"(?i)^.*?Home\s*>\s*Docs\s*>.*$",
        # Confidentiality & copyright headers/footers
        r"(?i)^\s*(?:CONFIDENTIAL|INTERNAL ONLY|PROPRIETARY)\b.*$",
        r"(?i)^\s*Copyright\s*(?:©|\(c\))\s*\d{4}\s+Alert_IQ.*?(?:All rights reserved\.?)?\s*$",
        # URL nav link artifacts
        r"(?i)^\s*\[?(?:Back to top|Table of Contents|Previous Page|Next Page)\]?\s*$",
    ]

    @classmethod
    def normalize_unicode(cls, text: str) -> str:
        """
        Applies Unicode NFKC normalization to standardize characters, ligatures, and symbols.
        """
        if not text:
            return ""
        return unicodedata.normalize("NFKC", text)

    @classmethod
    def fix_encoding_artifacts(cls, text: str) -> str:
        """
        Replaces smart quotes, non-breaking spaces, and common mojibake characters.
        """
        if not text:
            return ""
        result = (
            # 1. Multi-character mojibake sequences (must run before single-char replacements)
            text.replace("\u00e2\u20ac\u2014", " - ")
            .replace("\u00e2\u20ac\u2013", "-")
            .replace("\u00e2\u20ac\u0153", '"')
            .replace("\u00e2\u20ac\x9d", '"')
            .replace("\u00e2\u20ac\x9c", '"')
            .replace("\u00e2\u20ac\u2122", "'")
            .replace("\u00e2\u2014", " - ")
            .replace("\u00e2\u2013", "-")
            .replace("\u00e2\u201c", '"')
            .replace("\u00e2\u201d", '"')
            .replace("â€”", " - ")
            .replace("â€“", "-")
            .replace("â€œ", '"')
            .replace("â€\x9d", '"')
            .replace("â€\x9c", '"')
            .replace("â€™", "'")
            .replace("â€", "")
            # 2. Standard Unicode punctuation & spacing normalization
            .replace("\u00a0", " ")
            .replace("\u200b", "")
            .replace("\u200e", "")
            .replace("\u200f", "")
            .replace("\u2018", "'")
            .replace("\u2019", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2013", "-")
            .replace("\u2014", " - ")
            .replace("\u2026", "...")
            .replace("Â ", " ")
            .replace("Â", "")
        )
        return result

    @classmethod
    def remove_boilerplate(cls, text: str) -> str:
        """
        Strips repeated page numbers, navigation text, and legal boilerplate.
        """
        if not text:
            return ""

        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            stripped_line = line.strip()
            # Check if line matches any boilerplate pattern
            is_boilerplate = any(
                re.search(pattern, stripped_line, re.IGNORECASE)
                for pattern in cls.BOILERPLATE_PATTERNS
            )
            if not is_boilerplate:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    @classmethod
    def fix_broken_line_wraps(cls, text: str) -> str:
        """
        Fixes words broken by hyphenated line wraps (e.g. 'data-\nbase' -> 'database')
        and merges orphan single-line breaks within paragraphs.
        """
        if not text:
            return ""

        # Reconnect hyphenated words split across lines: "data-\nbase" -> "database"
        reconnected = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", text)

        # Merge soft line breaks that do not represent bullet points or header boundaries
        # Look for a lowercase or word char ending a line, followed by a lowercase line start
        def merge_soft_breaks(match):
            return match.group(1) + " " + match.group(2)

        # Merge lines where the previous line doesn't end in punctuation or markdown bullet
        merged = re.sub(r"([a-zA-Z0-9,;:])\n([a-zA-Z0-9])", merge_soft_breaks, reconnected)
        return merged

    @classmethod
    def collapse_whitespace(cls, text: str) -> str:
        """
        Collapses runaway spaces, tabs, and excess blank lines down to clean spacing.
        """
        if not text:
            return ""

        # Normalize line-by-line internal spaces
        lines = []
        for line in text.splitlines():
            # Replace tabs and multiple spaces within a line with single space
            collapsed_line = re.sub(r"[ \t]+", " ", line).strip()
            lines.append(collapsed_line)

        # Join lines and collapse 3+ consecutive newlines to maximum 2 newlines (1 blank line)
        joined = "\n".join(lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", joined)
        return cleaned.strip()

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Executes the complete sequential cleaning pipeline on a raw text string.
        """
        if not text:
            return ""

        step1 = cls.fix_encoding_artifacts(text)
        step2 = cls.normalize_unicode(step1)
        step3 = cls.remove_boilerplate(step2)
        step4 = cls.fix_broken_line_wraps(step3)
        step5 = cls.collapse_whitespace(step4)
        return step5

    @classmethod
    def clean_document(cls, doc: Document) -> Tuple[Document, Dict[str, Any]]:
        """
        Cleans a single Document object and attaches transformation audit metrics.
        """
        original_content = doc.content
        original_chars = len(original_content)
        original_tokens = doc.token_count

        cleaned_content = cls.clean_text(original_content)
        cleaned_chars = len(cleaned_content)
        cleaned_tokens = count_tokens(cleaned_content)

        char_reduction = original_chars - cleaned_chars
        reduction_pct = round((char_reduction / original_chars * 100), 2) if original_chars > 0 else 0.0

        stats = {
            "source": doc.source,
            "filename": doc.metadata.get("filename", "unknown"),
            "original_chars": original_chars,
            "cleaned_chars": cleaned_chars,
            "char_reduction": char_reduction,
            "reduction_pct": reduction_pct,
            "original_tokens": original_tokens,
            "cleaned_tokens": cleaned_tokens,
            "token_reduction": original_tokens - cleaned_tokens
        }

        cleaned_metadata = dict(doc.metadata)
        cleaned_metadata.update({
            "is_cleaned": True,
            "character_count": cleaned_chars,
            "token_count": cleaned_tokens,
            "cleaning_reduction_pct": reduction_pct
        })

        cleaned_doc = Document(
            content=cleaned_content,
            source=doc.source,
            metadata=cleaned_metadata
        )

        return cleaned_doc, stats

    @classmethod
    def clean_corpus(cls, documents: List[Document]) -> Tuple[List[Document], List[Dict[str, Any]]]:
        """
        Task 3: Applies uniform cleaning consistently across an entire corpus of documents.
        """
        cleaned_docs: List[Document] = []
        all_stats: List[Dict[str, Any]] = []

        for doc in documents:
            cleaned_doc, stat = cls.clean_document(doc)
            cleaned_docs.append(cleaned_doc)
            all_stats.append(stat)

        return cleaned_docs, all_stats


def generate_cleaning_demo_report(corpus_dir: Optional[Path] = None) -> str:
    """
    Task 4 & 5: Executes cleaning across sample corpus and generates detailed before/after comparison report.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if corpus_dir is None:
        corpus_dir = Path(__file__).resolve().parent.parent / "data" / "corpus"

    # Ingest raw corpus documents
    raw_docs, _ = DocumentLoader.load_directory(corpus_dir)

    # Add an explicitly synthetic noisy sample to demonstrate all cleaning edge cases
    noisy_synthetic_raw = """
CONFIDENTIAL - INTERNAL ONLY
Home > Docs > Runbooks > Incident-Triage

# Database Fail-\nover and Repli-\ncation Recovery

Page 1 of 5

Alert_IQ systems occasionally encounter database connec-\ntion pool saturation during peak flash-sale events.
The primary data-\nbase server utilizes automated replication pools.

Â “Important Notice”: When replication lag exceeds 850ms, the on-call engineer must execute `pg_terminate_backend()` immediatelyâ€”do not hesitate.   


Page 2 of 5
Copyright (c) 2026 Alert_IQ Inc. All rights reserved.
"""
    noisy_synthetic_doc = Document(
        content=noisy_synthetic_raw,
        source="data/corpus/synthetic_noisy_sample.txt",
        metadata={"filename": "synthetic_noisy_sample.txt", "file_type": "txt"}
    )

    all_test_docs = raw_docs + [noisy_synthetic_doc]
    cleaned_docs, cleaning_stats = TextCleaner.clean_corpus(all_test_docs)

    lines = []
    lines.append("=" * 80)
    lines.append("🧹 Alert_IQ - Document Cleaning & Normalization Pipeline Report")
    lines.append("=" * 80)

    # Task 4: Detailed Before & After Evidence
    lines.append("\n🔬 TASK 4: BEFORE vs. AFTER CLEANING DEMONSTRATIONS")
    lines.append("-" * 80)

    # 1. Synthetic Noisy Sample Diff
    syn_raw = noisy_synthetic_doc.content
    syn_clean, syn_stat = TextCleaner.clean_document(noisy_synthetic_doc)

    lines.append("📄 [SAMPLE CASE: Synthetic Noisy Document with Full Artifact Suite]")
    lines.append("--- 🔴 BEFORE CLEANING (Raw Text with Boilerplate & Broken Wraps) ---")
    lines.append(syn_raw.strip())
    lines.append("\n--- 🟢 AFTER CLEANING (Normalized, De-hyphenated, Cleaned) ---")
    lines.append(syn_clean.content)
    lines.append("\n📊 Transformation Metrics:")
    lines.append(f"   • Characters : {syn_stat['original_chars']} -> {syn_stat['cleaned_chars']} (Saved {syn_stat['char_reduction']} chars / -{syn_stat['reduction_pct']}%)")
    lines.append(f"   • Tokens     : {syn_stat['original_tokens']} -> {syn_stat['cleaned_tokens']} (Saved {syn_stat['token_reduction']} tokens)")
    lines.append("=" * 80)

    # 2. Corpus Overview (Task 3)
    lines.append("\n📌 TASK 3: UNIFORM CORPUS-WIDE CLEANING METRICS")
    lines.append("-" * 80)
    lines.append(f"{'Filename':<30} | {'Raw Chars':<10} | {'Clean Chars':<11} | {'Reduction':<10} | {'Raw Tokens':<10} | {'Clean Tokens':<12}")
    lines.append("-" * 80)

    total_orig_tokens = 0
    total_clean_tokens = 0

    for stat in cleaning_stats:
        total_orig_tokens += stat["original_tokens"]
        total_clean_tokens += stat["cleaned_tokens"]
        lines.append(
            f"{stat['filename']:<30} | {stat['original_chars']:<10} | {stat['cleaned_chars']:<11} | "
            f"-{stat['reduction_pct']}%{'':<4} | {stat['original_tokens']:<10} | {stat['cleaned_tokens']:<12}"
        )

    lines.append("-" * 80)
    lines.append("\n🏁 CORPUS CLEANING SUMMARY:")
    lines.append(f"• Total Documents Processed : {len(cleaned_docs)}")
    lines.append(f"• Total Raw Corpus Tokens   : {total_orig_tokens:,} tokens")
    lines.append(f"• Total Clean Corpus Tokens : {total_clean_tokens:,} tokens")
    lines.append(f"• Total Token Savings       : {total_orig_tokens - total_clean_tokens:,} tokens saved (Reduces embedding & context costs)")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    report = generate_cleaning_demo_report()
    print(report)


if __name__ == "__main__":
    main()
