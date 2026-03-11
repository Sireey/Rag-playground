# RAG Pipeline Playground

A modular, plug-and-play Retrieval-Augmented Generation (RAG) pipeline built from scratch for learning and experimentation. Swap parsers, chunkers, and retrieval strategies via a YAML config file or an interactive Streamlit UI — no code changes required.

## What's inside

Three retrieval strategies, each addressing a different failure mode:

| Retriever | What it fixes |
|-----------|--------------|
| `chroma` | Baseline — fast vector similarity search |
| `rerank` | Cross-encoder re-ranking + optional multi-query expansion |
| `parent_document` | Small chunks for precise retrieval, large parent chunks returned to the LLM |

Evaluation with [RAGAS](https://github.com/explodinggradients/ragas) — four metrics that decompose retrieval quality from generation quality:
- **Faithfulness** — did the LLM stay grounded in the context?
- **Answer Relevancy** — does the answer address the question?
- **Context Precision** — were the retrieved chunks relevant?
- **Context Recall** — was anything important missed?

---

## Project structure

```
practice-rag/
├── rag/                            ← pipeline package
│   ├── document.py                 ← Document dataclass (shared data contract)
│   ├── base.py                     ← ABCs: BaseParser, BaseChunker, BaseRetriever
│   ├── pipeline.py                 ← RAGPipeline: ingest() + query() + query_for_ragas()
│   ├── factory.py                  ← RAGFactory: YAML/dict → RAGPipeline
│   ├── ssl_patch.py                ← corporate SSL fix (optional)
│   ├── parsers/
│   │   └── pymupdf.py              ← PyMuPDFParser
│   ├── chunkers/
│   │   └── recursive_token.py      ← RecursiveTokenChunker (token-aware)
│   └── retrievers/
│       ├── chroma.py               ← ChromaRetriever
│       ├── rerank.py               ← ReRankRetriever
│       └── parent_document.py      ← ParentDocumentRetriever
├── configs/
│   ├── baseline.yaml               ← vector similarity only
│   ├── default.yaml                ← rerank + multi-query (recommended)
│   └── pdr.yaml                    ← parent-document retriever
├── data/
│   └── raw/                        ← put your PDFs here
├── app.py                          ← Streamlit UI
├── run.py                          ← CLI entry point
└── requirements.txt
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd practice-rag

python -m venv .venv
source .venv/Scripts/activate   # Windows
# source .venv/bin/activate     # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
LLMFOUNDRY_TOKEN=your_api_key_here
LLMFOUNDRY_BASE_URL=https://llmfoundry.straive.com/openai/v1
```

> The pipeline uses an OpenAI-compatible API. If you have a direct OpenAI key, set `LLMFOUNDRY_TOKEN=sk-...` and `LLMFOUNDRY_BASE_URL=https://api.openai.com/v1`.

### 4. Add PDFs

Place PDF files in `data/raw/`:

```bash
data/raw/attention.pdf
data/raw/bert.pdf
data/raw/rag_paper.pdf
```

---

## Running

### Streamlit UI (recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Configure the pipeline in the sidebar, click **Build Pipeline**, then **Ingest** your PDFs, then ask questions.

### CLI

```bash
# Query using the existing index
python run.py

# Choose a config
python run.py --config configs/baseline.yaml
python run.py --config configs/default.yaml
python run.py --config configs/pdr.yaml

# Index PDFs first (required on first run or after adding new files)
python run.py --ingest

# Wipe the index and rebuild from scratch
python run.py --reset
```

---

## Configs

Each config file defines the full pipeline. The key section is `retriever` — it owns the chunking strategy too.

### `configs/baseline.yaml` — vector similarity only
```yaml
retriever:
  type: chroma
  chunker:
    type: recursive_token
    chunk_size: 512
    chunk_overlap: 50
```

### `configs/default.yaml` — rerank + multi-query
```yaml
retriever:
  type: rerank
  k_retrieve: 10        # candidate pool for the cross-encoder
  k_final: 3            # chunks returned to the LLM
  multi_query: true     # LLM generates 3 query variants to widen recall
  chunker:
    type: recursive_token
    chunk_size: 512
    chunk_overlap: 50
```

### `configs/pdr.yaml` — parent-document retriever
```yaml
retriever:
  type: parent_document
  child_chunker:         # small — indexed for retrieval precision
    chunk_size: 128
  parent_chunker:        # large — returned to the LLM for rich context
    chunk_size: 512
```

> When changing retriever type or chunker settings, always run with `--reset` to rebuild the index cleanly.

---

## Using the package directly

```python
from rag import RAGFactory

# Build from a YAML config
pipeline = RAGFactory.from_yaml("configs/default.yaml")

# Or build from a dict (useful for programmatic use)
pipeline = RAGFactory.from_dict({
    "ssl_patch": True,
    "llm": {"model": "gpt-4o-mini", "temperature": 0},
    "embedder": {"model": "text-embedding-3-small"},
    "parser": {"type": "pymupdf"},
    "retriever": {
        "type": "rerank",
        "chroma_dir": "./chroma_db",
        "collection_name": "my_docs",
        "k_retrieve": 10,
        "k_final": 3,
        "multi_query": True,
        "chunker": {"type": "recursive_token", "chunk_size": 512, "chunk_overlap": 50},
    },
})

# Index documents
pipeline.ingest(["data/raw/bert.pdf"], reset=True)

# Query
result = pipeline.query("What are BERT's two pre-training tasks?")
print(result["result"])
print(result["source_documents"])

# RAGAS-ready output
row = pipeline.query_for_ragas(
    "What are BERT's two pre-training tasks?",
    ground_truth="Masked Language Modeling and Next Sentence Prediction"
)
# row = {"question": ..., "answer": ..., "contexts": [...], "ground_truth": ...}
```

### Evaluating with RAGAS

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

golden = [
    ("What is RAG-Sequence?", "RAG-Sequence uses the same retrieved document..."),
    ("What are BERT's pre-training tasks?", "MLM and NSP"),
]

rows = [pipeline.query_for_ragas(q, gt) for q, gt in golden]
ds = Dataset.from_list(rows)

result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(result)
```

---

## Adding a new retriever

1. Create `rag/retrievers/your_retriever.py` implementing `BaseRetriever`
2. Implement `index(pages, reset) -> int` and `retrieve(query) -> List[Document]`
3. Register it in `rag/factory.py` under `_build_retriever`
4. Add a `configs/your_retriever.yaml`

No other files need to change.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `pymupdf` | PDF parsing |
| `langchain`, `langchain-openai` | LLM + embeddings client |
| `langchain-community` | ChromaDB integration |
| `chromadb` | Local vector store |
| `sentence-transformers` | Cross-encoder re-ranking |
| `tiktoken` | Token-aware chunking |
| `ragas` | RAG evaluation metrics |
| `streamlit` | Interactive UI |
| `python-dotenv` | `.env` file loading |
