# rag/retrievers/parent_document.py
#
# ParentDocumentRetriever — solves the chunk size dilemma.
#
# The problem:
#   Small chunks → precise retrieval (the query embedding matches a tight piece of text)
#                  but the LLM gets too little context to reason with.
#   Large chunks → rich context for the LLM
#                  but noisy retrieval (the chunk contains irrelevant sentences too).
#   You cannot optimise both with a single chunk size.
#
# The fix — two-level chunking:
#   1. Split pages into PARENT chunks (large, e.g. 512 tokens) — stored in a docstore.
#   2. Split each parent into CHILD chunks (small, e.g. 128 tokens) — indexed in ChromaDB.
#   3. Retrieval: search child chunks (precision) → look up their parent (rich context).
#   4. The LLM receives parent chunks — rich context, no precision loss.
#
# Docstore:
#   Parents are stored as a JSON file on disk so they survive process restarts.
#   Location: {chroma_dir}/parents_{collection_name}.json
#   Format:   { parent_id: { "content": "...", "metadata": {...} } }
#
# Parent ID scheme:
#   "{source}_parent_{i}"  where i is the sequential index across all parents.
#   This is stable as long as you don't change the parent chunker settings
#   or the order of files in ingest().

import json
from pathlib import Path
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LangchainDocument
from langchain_openai import OpenAIEmbeddings

from rag.base import BaseChunker, BaseRetriever
from rag.document import Document


class ParentDocumentRetriever(BaseRetriever):
    """Retrieve child chunks, return their parent Documents to the LLM.

    Args:
        embedder:        LangChain OpenAIEmbeddings instance.
        child_chunker:   BaseChunker for small chunks — indexed for retrieval.
                         Recommended: chunk_size=128, chunk_overlap=20
        parent_chunker:  BaseChunker for large chunks — returned to the LLM.
                         Recommended: chunk_size=512, chunk_overlap=0
        chroma_dir:      Path to ChromaDB directory.
        collection_name: ChromaDB collection name for child chunks.
        k:               Number of parent Documents to return.
    """

    def __init__(
        self,
        embedder: OpenAIEmbeddings,
        child_chunker: BaseChunker,
        parent_chunker: BaseChunker,
        chroma_dir: str = "./chroma_db",
        collection_name: str = "rag_papers_pdr",
        k: int = 3,
    ) -> None:
        self._embedder = embedder
        self._child_chunker = child_chunker
        self._parent_chunker = parent_chunker
        self._chroma_dir = chroma_dir
        self._collection_name = collection_name
        self._k = k

        # Docstore: parent_id → Document, persisted to disk
        self._docstore: dict[str, Document] = {}
        self._docstore_path = Path(chroma_dir) / f"parents_{collection_name}.json"

        self._vectorstore: Chroma | None = None

        self._load_docstore()
        self._load_vectorstore()

    # ── Persistence helpers ────────────────────────────────────────────────

    def _load_docstore(self) -> None:
        """Load the parent docstore from disk if it exists."""
        if self._docstore_path.exists():
            with open(self._docstore_path) as f:
                data = json.load(f)
            self._docstore = {
                pid: Document(content=d["content"], metadata=d["metadata"])
                for pid, d in data.items()
            }

    def _save_docstore(self) -> None:
        """Persist the parent docstore to disk."""
        self._docstore_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._docstore_path, "w") as f:
            json.dump(
                {
                    pid: {"content": doc.content, "metadata": doc.metadata}
                    for pid, doc in self._docstore.items()
                },
                f,
                indent=2,
            )

    def _load_vectorstore(self) -> None:
        """Attach to an existing child chunk collection if one exists."""
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
        """Build the two-level index from raw pages.

        Steps:
          1. parent_chunker splits pages → parent chunks
          2. Each parent gets a stable parent_id and is stored in the docstore
          3. child_chunker splits each parent → child chunks, tagged with parent_id
          4. Child chunks are indexed in ChromaDB

        Args:
            pages:  Page-level Documents from the parser.
            reset:  If True, wipe both the docstore and ChromaDB before indexing.

        Returns:
            Total number of child chunks indexed.
        """
        if reset:
            self._docstore = {}
            if self._docstore_path.exists():
                self._docstore_path.unlink()
            if self._vectorstore is not None:
                self._vectorstore.delete_collection()
                self._vectorstore = None

        # Step 1 + 2: Create parent chunks and register them in the docstore
        parents = self._parent_chunker.chunk(pages)
        parent_offset = len(self._docstore)  # stable IDs across incremental ingests

        child_lc_docs: List[LangchainDocument] = []

        for i, parent in enumerate(parents):
            parent_id = f"{parent.metadata.get('source', 'doc')}_parent_{parent_offset + i}"
            parent.metadata["parent_id"] = parent_id
            self._docstore[parent_id] = parent

            # Step 3: Chunk this parent into children, tag each with parent_id
            children = self._child_chunker.chunk([parent])
            for child in children:
                child.metadata["parent_id"] = parent_id
                child_lc_docs.append(
                    LangchainDocument(page_content=child.content, metadata=child.metadata)
                )

        self._save_docstore()

        # Step 4: Index child chunks in ChromaDB
        if not child_lc_docs:
            return 0

        if self._vectorstore is None:
            self._vectorstore = Chroma.from_documents(
                documents=child_lc_docs,
                embedding=self._embedder,
                collection_name=self._collection_name,
                persist_directory=self._chroma_dir,
            )
        else:
            self._vectorstore.add_documents(child_lc_docs)

        return len(child_lc_docs)

    def retrieve(self, query: str) -> List[Document]:
        """Search child chunks, return their parent Documents (deduplicated).

        Searches k*3 child chunks to ensure enough candidates after deduplication
        (multiple children may share the same parent, reducing the unique count).

        Returns:
            Up to k unique parent Documents, ordered by first child match.
        """
        if self._vectorstore is None:
            raise RuntimeError("No index loaded. Call index() first.")

        # Fetch a wide set of child candidates to survive dedup
        child_results = self._vectorstore.similarity_search(query, k=self._k * 3)

        seen_parents: dict[str, Document] = {}
        for child_doc in child_results:
            parent_id = child_doc.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parents:
                parent = self._docstore.get(parent_id)
                if parent:
                    seen_parents[parent_id] = parent
            if len(seen_parents) >= self._k:
                break

        return list(seen_parents.values())
