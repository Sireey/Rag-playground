# rag/chunkers/recursive_token.py
#
# Concrete implementation of BaseChunker using LangChain's
# RecursiveCharacterTextSplitter with tiktoken token counting.
#
# WHY token-aware splitting over character-based?
#   OpenAI embedding models have a token limit (8191 for text-embedding-3-small).
#   Splitting by character count gives inconsistent token counts — a 512-char chunk
#   of dense technical text may be 200 tokens; the same chars of prose may be 100.
#   tiktoken counts the actual subword tokens the model sees, so chunk_size=512
#   means "at most 512 tokens," not "at most 512 characters."
#
# WHY RecursiveCharacterTextSplitter over a naive split?
#   It tries separators in order: ["\n\n", "\n", " ", ""].
#   It only falls back to a coarser separator if the finer one would produce
#   chunks that are too large. This preserves paragraph and sentence boundaries
#   wherever possible, unlike a naive char split that cuts mid-word.
#
# Default settings (mirror src/03_chunk.py's chosen strategy):
#   chunk_size    = 512 tokens
#   chunk_overlap = 50 tokens   ← each chunk re-shares 50 tokens with the next,
#                                  preventing answers from falling on chunk edges

from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

from rag.base import BaseChunker
from rag.document import Document


class RecursiveTokenChunker(BaseChunker):
    """Split Documents into token-bounded chunks with overlap.

    Args:
        chunk_size:    Maximum number of tokens per chunk (default 512).
        chunk_overlap: Number of tokens shared between consecutive chunks (default 50).
        encoding_name: tiktoken encoding — must match the embedding model's tokenizer.
                       "cl100k_base" is used by all text-embedding-3-* models.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split each Document into token-bounded chunks.

        Source metadata (source, page) is propagated to every chunk.
        Each chunk also receives a "chunk" key with its index within that source.

        Args:
            documents: Output of a BaseParser — typically one Document per page.

        Returns:
            Flat list of chunk Documents, ordered by source then chunk index.
        """
        all_chunks: List[Document] = []

        for doc in documents:
            # Convert to LangChain's Document so the splitter preserves metadata
            lc_doc = LangchainDocument(
                page_content=doc.content,
                metadata=doc.metadata,
            )
            lc_chunks = self._splitter.split_documents([lc_doc])

            # Convert back to our Document type, tagging each with its chunk index
            for idx, lc_chunk in enumerate(lc_chunks):
                chunk_meta = {**lc_chunk.metadata, "chunk": idx}
                all_chunks.append(Document(
                    content=lc_chunk.page_content,
                    metadata=chunk_meta,
                ))

        return all_chunks
