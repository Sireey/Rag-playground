# rag/parsers/pymupdf.py
#
# Concrete implementation of BaseParser using PyMuPDF (fitz).
#
# WHY PyMuPDF over pypdf/pdfplumber?
#   - pypdf: clean text but includes figure caption text inline with body text
#   - pdfplumber: merges words without spaces, interleaves two-column layouts badly
#   - PyMuPDF: best column detection, reliable spacing, most accurate for academic PDFs
#
# The cleaning logic here is ported from src/02_clean.py.
# It lives in the parser because it's specific to how PyMuPDF emits text —
# another parser might not need the same fixes.

import os
import re
from typing import List

import fitz  # PyMuPDF

from rag.base import BaseParser
from rag.document import Document


def _clean_text(text: str) -> str:
    """Remove PyMuPDF-specific artifacts from extracted text.

    Fixes applied (in order):
      1. Rejoin hyphenated line breaks  ("connec-\ntion" → "connection")
      2. Remove standalone page-number lines  ("14\n" at start of line)
      3. Collapse multiple blank lines to a single blank line
      4. Strip leading/trailing whitespace
    """
    # 1. Hyphenated line-break: word ends with '-' then newline → rejoin
    text = re.sub(r"-\n", "", text)

    # 2. Standalone page numbers — a line containing only digits
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

    # 3. Collapse runs of blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Trim
    return text.strip()


class PyMuPDFParser(BaseParser):
    """Parse PDF files page-by-page using PyMuPDF.

    Each page becomes one Document. The text is lightly cleaned before
    being returned. Empty pages (e.g. blank pages, pure-image pages) are
    skipped rather than returned as empty Documents.

    Args:
        clean: If True (default), apply _clean_text() to each page's text.
    """

    def __init__(self, clean: bool = True) -> None:
        self.clean = clean

    def parse(self, file_path: str) -> List[Document]:
        """Parse a PDF file and return one Document per non-empty page.

        Args:
            file_path: Absolute or relative path to a PDF file.

        Returns:
            List of Documents with metadata:
                source — basename of the file  (e.g. "bert.pdf")
                page   — 1-indexed page number
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")

        source = os.path.basename(file_path)
        documents: List[Document] = []

        with fitz.open(file_path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text()

                if self.clean:
                    text = _clean_text(text)

                # Skip pages with no meaningful text (cover images, blank pages)
                if not text.strip():
                    continue

                documents.append(Document(
                    content=text,
                    metadata={"source": source, "page": page_num},
                ))

        return documents
