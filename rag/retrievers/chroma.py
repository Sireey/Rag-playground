# rag/retrievers/chroma.py
#
# Baseline retriever — pure vector similarity search backed by ChromaDB.
#
# The retriever owns its chunking strategy. The pipeline passes raw page-level
# Documents; this class splits them into chunks before indexing.
#
# Strengths:  Fast, simple, works when query vocabulary matches document vocabulary.
# Weaknesses: Ranks by vocabulary overlap, not semantic relevance.
#             → Solved in ReRankRetriever (which inherits index() unchanged).

from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LangchainDocument
from langchain_openai import OpenAIEmbeddings

from rag.base import BaseChunker, BaseRetriever
from rag.document import Document


class ChromaRetriever(BaseRetriever):
    """Vector similarity retriever backed by a local ChromaDB collection.

    Args:
        embedder:        LangChain OpenAIEmbeddings instance.
        chunker:         BaseChunker used to split pages before indexing.
        chroma_dir:      Path to the on-disk ChromaDB directory.
        collection_name: Name of the ChromaDB collection.
        k:               Number of chunks to return from retrieve().
    """

    def __init__(
        self,
        embedder: OpenAIEmbeddings,
        chunker: BaseChunker,
        chroma_dir: str = "./chroma_db",
        collection_name: str = "rag_papers",
        k: int = 3,
    ) -> None:
        self._embedder = embedder
        self._chunker = chunker
        self._chroma_dir = chroma_dir
        self._collection_name = collection_name
        self._k = k
        self._vectorstore: Chroma | None = None
        self._load_vectorstore()

    def _load_vectorstore(self) -> None:
        """Attach to an existing ChromaDB collection if one exists on disk."""
        try:
            self._vectorstore = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embedder,
                persist_directory=self._chroma_dir,
            )
        except Exception:
            self._vectorstore = None

    # ── BaseRetriever interface ────────────────────────────────────────────

    def index(self, pages: List[Document], reset: bool = False) -> int:
        """Chunk pages and add them to the ChromaDB collection.

        Args:
            pages:  Page-level Documents from the parser.
            reset:  If True, wipe the existing collection before indexing.

        Returns:
            Number of chunks indexed.
        """
        if reset and self._vectorstore is not None:
            self._vectorstore.delete_collection()
            self._vectorstore = None

        chunks = self._chunker.chunk(pages)
        lc_docs = [
            LangchainDocument(page_content=doc.content, metadata=doc.metadata)
            for doc in chunks
        ]

        if self._vectorstore is None:
            self._vectorstore = Chroma.from_documents(
                documents=lc_docs,
                embedding=self._embedder,
                collection_name=self._collection_name,
                persist_directory=self._chroma_dir,
            )
        else:
            self._vectorstore.add_documents(lc_docs)

        return len(chunks)

    def retrieve(self, query: str) -> List[Document]:
        """Return the top-k most similar chunks for `query`."""
        if self._vectorstore is None:
            raise RuntimeError(
                "No index loaded. Call index() first or point chroma_dir at an "
                "existing collection."
            )
        results = self._vectorstore.similarity_search_with_score(query, k=self._k)
        return [
            Document(content=doc.page_content, metadata=doc.metadata)
            for doc, _score in results
        ]
