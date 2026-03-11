# rag/document.py
#
# The core data contract of the pipeline.
#
# Every component — parser, chunker, retriever — speaks in Documents.
# Using our own type (rather than LangChain's) decouples the business logic
# from any specific framework. We convert to LangChain's type only at the
# ChromaDB boundary, inside the retriever.

from dataclasses import dataclass, field


@dataclass
class Document:
    """A piece of text with provenance metadata.

    Attributes:
        content:  The raw text of this document or chunk.
        metadata: Arbitrary key-value pairs. Conventionally includes:
                    "source"  — original filename (e.g. "bert.pdf")
                    "page"    — page number within the source file
                    "chunk"   — chunk index within the source document
    """
    content: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        src = self.metadata.get("source", "?")
        preview = self.content[:60].replace("\n", " ")
        return f"Document(src={src!r}, content={preview!r}...)"
