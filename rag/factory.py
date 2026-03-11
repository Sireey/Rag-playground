# rag/factory.py
#
# RAGFactory — reads a YAML config file and returns an initialized RAGPipeline.
#
# WHY a factory + config file?
#   Swapping the retriever strategy (chroma → rerank → parent_document) is
#   a one-line YAML change, not a code change. Adding a new retriever type
#   means registering it here — nothing else in the codebase needs to change.
#
# YAML structure:
#
#   ssl_patch: true
#
#   llm:
#     model: gpt-4o-mini
#     temperature: 0
#
#   embedder:
#     model: text-embedding-3-small
#
#   parser:
#     type: pymupdf
#     clean: true
#
#   retriever:
#     type: chroma | rerank | parent_document
#     chroma_dir: ./chroma_db
#     collection_name: rag_papers
#     k_final: 3                    # chroma / rerank
#     k_retrieve: 10                # rerank only
#     multi_query: true             # rerank only
#     multi_query_variants: 3       # rerank only
#     chunker:                      # chroma / rerank
#       type: recursive_token
#       chunk_size: 512
#       chunk_overlap: 50
#     child_chunker:                # parent_document only
#       type: recursive_token
#       chunk_size: 128
#       chunk_overlap: 20
#     parent_chunker:               # parent_document only
#       type: recursive_token
#       chunk_size: 512
#       chunk_overlap: 0

import os
from typing import Any

import yaml
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag.chunkers.recursive_token import RecursiveTokenChunker
from rag.parsers.pymupdf import PyMuPDFParser
from rag.pipeline import RAGPipeline
from rag.retrievers.chroma import ChromaRetriever
from rag.retrievers.parent_document import ParentDocumentRetriever
from rag.retrievers.rerank import ReRankRetriever


# ── Registries ─────────────────────────────────────────────────────────────
# To add a new parser/chunker: implement the ABC, import it, add to the dict.

_PARSER_REGISTRY = {
    "pymupdf": PyMuPDFParser,
}

_CHUNKER_REGISTRY = {
    "recursive_token": RecursiveTokenChunker,
}


def _build_chunker(cfg: dict):
    """Build a chunker from a config dict with a 'type' key."""
    chunker_type = cfg.get("type", "recursive_token")
    if chunker_type not in _CHUNKER_REGISTRY:
        raise ValueError(
            f"Unknown chunker type {chunker_type!r}. Available: {list(_CHUNKER_REGISTRY)}"
        )
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    return _CHUNKER_REGISTRY[chunker_type](**kwargs)


class RAGFactory:
    """Builds a RAGPipeline from a YAML file or a plain Python dict.

    Usage:
        pipeline = RAGFactory.from_yaml("configs/default.yaml")
        pipeline = RAGFactory.from_dict({"retriever": {"type": "rerank", ...}, ...})
        pipeline.ingest(["data/raw/bert.pdf"])
        result = pipeline.query("What are BERT's pre-training tasks?")
    """

    @classmethod
    def from_yaml(cls, config_path: str) -> RAGPipeline:
        """Load config from a YAML file and return a RAGPipeline."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        with open(config_path, "r") as f:
            cfg: dict[str, Any] = yaml.safe_load(f)
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> RAGPipeline:
        """Build a RAGPipeline from a plain Python config dict.

        Same structure as the YAML config. Useful for programmatic
        pipeline construction (e.g. from a UI) without writing temp files.
        """
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        # ── 1. Corporate SSL fix ───────────────────────────────────────────
        if cfg.get("ssl_patch", False):
            from rag import ssl_patch
            ssl_patch.apply()

        # ── 2. Shared LLM + Embedder ───────────────────────────────────────
        llm_cfg = cfg.get("llm", {})
        llm = ChatOpenAI(
            api_key=os.getenv("LLMFOUNDRY_TOKEN"),
            base_url=os.getenv("LLMFOUNDRY_BASE_URL"),
            model=llm_cfg.get("model", "gpt-4o-mini"),
            temperature=llm_cfg.get("temperature", 0),
        )

        emb_cfg = cfg.get("embedder", {})
        embedder = OpenAIEmbeddings(
            api_key=os.getenv("LLMFOUNDRY_TOKEN"),
            base_url=os.getenv("LLMFOUNDRY_BASE_URL"),
            model=emb_cfg.get("model", "text-embedding-3-small"),
        )

        # ── 3. Parser ──────────────────────────────────────────────────────
        parser_cfg = cfg.get("parser", {})
        parser_type = parser_cfg.get("type", "pymupdf")
        if parser_type not in _PARSER_REGISTRY:
            raise ValueError(
                f"Unknown parser type {parser_type!r}. Available: {list(_PARSER_REGISTRY)}"
            )
        parser_kwargs = {k: v for k, v in parser_cfg.items() if k != "type"}
        parser = _PARSER_REGISTRY[parser_type](**parser_kwargs)

        # ── 4. Retriever (owns its own chunker(s)) ─────────────────────────
        ret_cfg = cfg.get("retriever", {})
        ret_type = ret_cfg.get("type", "chroma")
        chroma_dir = ret_cfg.get("chroma_dir", "./chroma_db")
        collection_name = ret_cfg.get("collection_name", "rag_papers")

        if ret_type == "chroma":
            chunker = _build_chunker(ret_cfg.get("chunker", {}))
            retriever = ChromaRetriever(
                embedder=embedder,
                chunker=chunker,
                chroma_dir=chroma_dir,
                collection_name=collection_name,
                k=ret_cfg.get("k_final", 3),
            )

        elif ret_type == "rerank":
            chunker = _build_chunker(ret_cfg.get("chunker", {}))
            retriever = ReRankRetriever(
                embedder=embedder,
                chunker=chunker,
                chroma_dir=chroma_dir,
                collection_name=collection_name,
                k_retrieve=ret_cfg.get("k_retrieve", 10),
                k_final=ret_cfg.get("k_final", 3),
                cross_encoder_model=ret_cfg.get(
                    "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                ),
                multi_query=ret_cfg.get("multi_query", False),
                multi_query_variants=ret_cfg.get("multi_query_variants", 3),
                llm=llm if ret_cfg.get("multi_query", False) else None,
            )

        elif ret_type == "parent_document":
            child_chunker = _build_chunker(ret_cfg.get("child_chunker", {}))
            parent_chunker = _build_chunker(ret_cfg.get("parent_chunker", {}))
            retriever = ParentDocumentRetriever(
                embedder=embedder,
                child_chunker=child_chunker,
                parent_chunker=parent_chunker,
                chroma_dir=chroma_dir,
                collection_name=collection_name,
                k=ret_cfg.get("k", 3),
            )

        else:
            raise ValueError(
                f"Unknown retriever type {ret_type!r}. "
                f"Available: 'chroma', 'rerank', 'parent_document'"
            )

        # ── 5. Assemble pipeline ───────────────────────────────────────────
        kwargs: dict[str, Any] = {"parser": parser, "retriever": retriever, "llm": llm}
        prompt_template = cfg.get("prompt_template")
        if prompt_template:
            kwargs["prompt_template"] = prompt_template

        return RAGPipeline(**kwargs)
