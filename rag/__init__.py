"""rag — modular RAG pipeline package.

Public API:
    RAGPipeline   — orchestrates parse → index → retrieve → answer
    RAGFactory    — builds a RAGPipeline from a YAML config file
    Document      — the data contract shared by all components

Concrete implementations:
    parsers.PyMuPDFParser
    chunkers.RecursiveTokenChunker          (used inside retrievers)
    retrievers.ChromaRetriever              — vector similarity
    retrievers.ReRankRetriever              — cross-encoder + multi-query
    retrievers.ParentDocumentRetriever      — child retrieval, parent context
"""

from rag.document import Document
from rag.factory import RAGFactory
from rag.pipeline import RAGPipeline

__all__ = ["Document", "RAGFactory", "RAGPipeline"]
