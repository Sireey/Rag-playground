# rag/retrievers/rerank.py
#
# ReRankRetriever — extends ChromaRetriever with two upgrades:
#
#   1. Cross-Encoder Re-Ranking
#      Retrieve a large candidate set (k_retrieve=10), then score every
#      (query, chunk) pair jointly with a cross-encoder. The cross-encoder
#      reads both texts together, so it can tell whether a chunk actually
#      ANSWERS the question, not just shares vocabulary with it.
#      Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, accurate, free)
#
#   2. Optional Multi-Query Expansion
#      A single query embeds to one point in vector space. If the phrasing
#      misses the document's vocabulary, the right chunks never appear.
#      Fix: LLM generates 3 rephrased variants, all 4 run against ChromaDB,
#      results deduplicated, then re-ranked with the cross-encoder.
#
# index() is inherited unchanged from ChromaRetriever — chunking strategy
# is the same (single level). Only retrieve() is overridden.

from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sentence_transformers import CrossEncoder

from rag.base import BaseChunker
from rag.document import Document
from rag.retrievers.chroma import ChromaRetriever


_EXPANSION_PROMPT = """\
You are an AI assistant helping improve document retrieval.
Given a user question, generate {n} alternative phrasings of the same question.
Each rephrasing should approach the topic from a different angle — different vocabulary,
different level of specificity, different framing — while preserving the original intent.

Output ONLY the {n} alternative questions, one per line. No numbering. No explanation.

Original question: {question}
"""


class ReRankRetriever(ChromaRetriever):
    """Vector retrieval + cross-encoder re-ranking + optional multi-query expansion.

    Inherits index() from ChromaRetriever — single-level chunking, same ChromaDB
    storage. Only retrieve() is overridden to add reranking and multi-query.

    Args:
        embedder:            LangChain OpenAIEmbeddings instance.
        chunker:             BaseChunker for splitting pages (passed to ChromaRetriever).
        chroma_dir:          Path to the ChromaDB directory.
        collection_name:     ChromaDB collection name.
        k_retrieve:          Candidate pool size for the cross-encoder.
        k_final:             Chunks returned to the LLM after reranking.
        cross_encoder_model: HuggingFace model id for the cross-encoder.
        multi_query:         If True, generate query variants before retrieval.
        multi_query_variants: Number of LLM-generated query variants.
        llm:                 ChatOpenAI instance, required when multi_query=True.
    """

    def __init__(
        self,
        embedder: OpenAIEmbeddings,
        chunker: BaseChunker,
        chroma_dir: str = "./chroma_db",
        collection_name: str = "rag_papers",
        k_retrieve: int = 10,
        k_final: int = 3,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        multi_query: bool = False,
        multi_query_variants: int = 3,
        llm: ChatOpenAI | None = None,
    ) -> None:
        super().__init__(
            embedder=embedder,
            chunker=chunker,
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            k=k_retrieve,  # ChromaRetriever.retrieve() uses this as candidate pool
        )
        self._k_retrieve = k_retrieve
        self._k_final = k_final
        self._cross_encoder = CrossEncoder(cross_encoder_model)
        self._multi_query = multi_query
        self._multi_query_variants = multi_query_variants
        self._llm = llm

        if multi_query and llm is None:
            raise ValueError("multi_query=True requires an llm to be provided.")

    # ── Private helpers ────────────────────────────────────────────────────

    def _generate_variants(self, question: str) -> List[str]:
        prompt = _EXPANSION_PROMPT.format(n=self._multi_query_variants, question=question)
        response = self._llm.invoke(prompt)
        lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
        return lines[:self._multi_query_variants]

    def _fetch_candidates(self, question: str) -> List[Document]:
        """Retrieve and deduplicate candidates across all query variants."""
        if not self._multi_query:
            # Single query — delegate to parent's retrieve() for the wide pool
            results = self._vectorstore.similarity_search_with_score(question, k=self._k_retrieve)
            return [Document(content=d.page_content, metadata=d.metadata) for d, _ in results]

        queries = [question] + self._generate_variants(question)
        seen: dict[str, Document] = {}
        for q in queries:
            results = self._vectorstore.similarity_search_with_score(q, k=self._k_retrieve)
            for lc_doc, _score in results:
                key = lc_doc.page_content
                if key not in seen:
                    seen[key] = Document(content=lc_doc.page_content, metadata=lc_doc.metadata)
        return list(seen.values())

    # ── BaseRetriever interface ────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve candidates, re-rank with cross-encoder, return top k_final."""
        if self._vectorstore is None:
            raise RuntimeError("No index loaded. Call index() first.")

        candidates = self._fetch_candidates(query)
        if not candidates:
            return []

        pairs = [[query, doc.content] for doc in candidates]
        ce_scores = self._cross_encoder.predict(pairs)

        reranked = sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _score, doc in reranked[:self._k_final]]
