# rag/pipeline.py
#
# RAGPipeline — orchestrates three stages:
#
#   Ingest  : parse raw files → pass pages to retriever (retriever owns chunking)
#   Retrieve: find the most relevant Documents for a query
#   Answer  : build a prompt from the Documents and call the LLM
#
# The pipeline is intentionally thin. It does not know about chunking —
# that is the retriever's responsibility. Swapping a retriever in the YAML
# config changes the entire indexing + retrieval strategy in one line.
#
# Output format
# ─────────────
# query() returns:
#   {"query": str, "result": str, "source_documents": List[Document]}
#
# query_for_ragas() reformats for RAGAS:
#   {"question": str, "answer": str, "contexts": List[str], "ground_truth": str}

from typing import List

from langchain_openai import ChatOpenAI

from rag.base import BaseParser, BaseRetriever
from rag.document import Document


_DEFAULT_PROMPT = """\
You are a research assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say \
"The retrieved context doesn't cover this."

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """End-to-end RAG pipeline.

    Args:
        parser:          Converts raw files to page-level Documents.
        retriever:       Owns chunking, indexing, and retrieval.
        llm:             ChatOpenAI instance for answer generation.
        prompt_template: f-string with {context} and {question} placeholders.
    """

    def __init__(
        self,
        parser: BaseParser,
        retriever: BaseRetriever,
        llm: ChatOpenAI,
        prompt_template: str = _DEFAULT_PROMPT,
    ) -> None:
        self._parser = parser
        self._retriever = retriever
        self._llm = llm
        self._prompt_template = prompt_template

    # ── Public API ─────────────────────────────────────────────────────────

    def ingest(self, file_paths: List[str], reset: bool = False) -> int:
        """Parse files and hand pages to the retriever for chunking + indexing.

        Args:
            file_paths: Paths to PDF (or other supported) files.
            reset:      If True, wipe the existing index before indexing.

        Returns:
            Total number of chunks indexed (reported by the retriever).
        """
        all_pages: List[Document] = []
        for path in file_paths:
            all_pages.extend(self._parser.parse(path))
        return self._retriever.index(all_pages, reset=reset)

    def query(self, question: str) -> dict:
        """Retrieve relevant Documents and generate an answer.

        Returns:
            {
                "query":            str,
                "result":           str,
                "source_documents": List[Document],
            }
        """
        source_docs = self._retriever.retrieve(question)
        context = "\n\n---\n\n".join(doc.content for doc in source_docs)
        prompt = self._prompt_template.format(context=context, question=question)
        response = self._llm.invoke(prompt)
        return {
            "query": question,
            "result": response.content,
            "source_documents": source_docs,
        }

    def query_for_ragas(self, question: str, ground_truth: str = "") -> dict:
        """Run query() and reformat for direct use with RAGAS.

        Returns:
            {"question": str, "answer": str, "contexts": List[str], "ground_truth": str}
        """
        result = self.query(question)
        return {
            "question": result["query"],
            "answer": result["result"],
            "contexts": [doc.content for doc in result["source_documents"]],
            "ground_truth": ground_truth,
        }
