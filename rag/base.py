# rag/base.py
#
# Abstract Base Classes for the three pluggable stages of the RAG pipeline.
#
# WHY ABCs?
#   They enforce a contract. Any class that claims to be a BaseParser MUST
#   implement `parse()`. If it doesn't, Python raises TypeError at instantiation,
#   not silently at runtime. This makes swapping implementations safe.
#
# The stages that vary between RAG architectures:
#
#   BaseParser    — "How do we turn raw files into text?"
#                   Swap: PyMuPDF ↔ pdfplumber ↔ Docx ↔ HTML scraper
#
#   BaseChunker   — "How do we split text into indexable pieces?"
#                   Swap: token-aware ↔ semantic ↔ sentence ↔ fixed-char
#                   Used INSIDE retrievers, not as a pipeline stage.
#
#   BaseRetriever — "How do we index and find relevant chunks?"
#                   Owns its own chunking. Swap the retriever → chunking changes too.
#                   Swap: vector-only ↔ rerank ↔ parent-document ↔ hybrid BM25+vector

from abc import ABC, abstractmethod
from typing import List

from rag.document import Document


class BaseParser(ABC):
    """Converts a raw file into a list of Documents (one per logical unit, e.g. page)."""

    @abstractmethod
    def parse(self, file_path: str) -> List[Document]:
        """Parse a file at `file_path` and return a flat list of Documents.

        Each Document should carry at minimum:
            metadata["source"] = os.path.basename(file_path)
            metadata["page"]   = page_number (1-indexed)
        """
        ...


class BaseChunker(ABC):
    """Splits a list of Documents into smaller, overlapping chunk Documents."""

    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split `documents` into chunks and return the flat list.

        Source metadata from the input Documents must be preserved in the output
        chunks. Implementations should add:
            metadata["chunk"] = chunk_index_within_source
        """
        ...


class BaseRetriever(ABC):
    """Indexes and retrieves Documents. Owns its own chunking strategy.

    The pipeline passes raw page-level Documents from the parser. The retriever
    decides how to split, store, and retrieve them. This lets each retriever type
    use a different chunking strategy without any changes to the pipeline:
      - ChromaRetriever:         single-level chunks
      - ReRankRetriever:         single-level chunks + reranking
      - ParentDocumentRetriever: child chunks for retrieval, parent chunks returned
    """

    @abstractmethod
    def index(self, pages: List[Document], reset: bool = False) -> int:
        """Chunk and index `pages` into the retriever's backing store.

        Args:
            pages:  Page-level Documents from a BaseParser — one per page.
                    The retriever is responsible for splitting these into chunks.
            reset:  If True, wipe the existing index before indexing.
                    If False (default), add to the existing index.

        Returns:
            Number of chunks added to the index.
        """
        ...

    @abstractmethod
    def retrieve(self, query: str) -> List[Document]:
        """Return the most relevant Documents for `query`.

        The number of Documents returned is configured at construction time (k).
        Implementations may fetch more candidates internally before filtering.
        """
        ...
