# RAG Learning Project — Plan

## Objective
Learn RAG (Retrieval-Augmented Generation) in depth — not just "how" but "why."
Build a modular RAG pipeline piece-by-piece, treating it like a data engineering project.
Understand failure modes at each stage before adding upgrades.

---

## Environment

| Item | Value |
|------|-------|
| OS | Windows 11, bash shell |
| Python | 3.11, pip + venv (.venv) |
| Project dir | `c:\Users\e430503\Desktop\practice-rag` |
| LLM backend | LLMFoundry (company OpenAI proxy) |
| LLM model | `gpt-4o-mini` |
| Embedding model | `text-embedding-3-small` |
| Vector store | ChromaDB (local, on disk at `./chroma_db/`) |

### .env file variables
```
LLMFOUNDRY_TOKEN=<token>
LLMFOUNDRY_PROJECT=<project_id>
LLMFOUNDRY_BASE_URL=https://llmfoundry.straive.com/openai/v1
```

### Corporate SSL Fix
Handled automatically via `rag/ssl_patch.py`. Set `ssl_patch: true` in any YAML config.

---

## Documents in Use
All stored in `data/raw/` and indexed in ChromaDB:

| File | Paper | Chunks |
|------|-------|--------|
| attention.pdf | "Attention Is All You Need" (Transformer) | 28 |
| bert.pdf | "BERT: Pre-training of Deep Bidirectional Transformers" | 49 |
| rag_paper.pdf | "Retrieval-Augmented Generation" (Lewis et al.) | 49 |

Total: **126 chunks** in ChromaDB collection `"rag_papers"`

---

## Project Structure
```
practice-rag/
├── rag/                        ← modular pipeline package
│   ├── document.py             ← Document dataclass (shared data contract)
│   ├── base.py                 ← ABCs: BaseParser, BaseChunker, BaseRetriever
│   ├── ssl_patch.py            ← corporate SSL fix (applied once at startup)
│   ├── pipeline.py             ← RAGPipeline: ingest() + query() + query_for_ragas()
│   ├── factory.py              ← RAGFactory: reads YAML → returns RAGPipeline
│   ├── parsers/pymupdf.py      ← PyMuPDFParser
│   ├── chunkers/recursive_token.py ← RecursiveTokenChunker
│   ├── retrievers/chroma.py    ← ChromaRetriever (vector similarity)
│   └── retrievers/rerank.py    ← ReRankRetriever (cross-encoder + multi-query)
├── configs/
│   ├── default.yaml            ← full pipeline (rerank + multi-query)
│   └── baseline.yaml           ← vector-only (for RAGAS A/B comparison)
├── data/raw/                   ← PDFs
├── chroma_db/                  ← vector index (on disk, no server needed)
├── run.py                      ← entry point
├── .env                        ← API keys (not committed)
└── requirements.txt
```

### Running the application
```bash
source .venv/Scripts/activate

python run.py                            # query using existing index
python run.py --ingest                   # index PDFs then query
python run.py --reset                    # wipe + rebuild index then query
python run.py --config configs/baseline.yaml  # use a different config
```

### Using the package directly
```python
from rag import RAGFactory

pipeline = RAGFactory.from_yaml("configs/default.yaml")
pipeline.ingest(["data/raw/bert.pdf"])          # parse → chunk → index

result = pipeline.query("What is BERT?")
# {"query": ..., "result": ..., "source_documents": [...]}

row = pipeline.query_for_ragas("What is BERT?", ground_truth="...")
# {"question": ..., "answer": ..., "contexts": [...], "ground_truth": ...}
```

---

## Progress Log

### Parsing ✅
Compared pypdf, pdfplumber, PyMuPDF. Chose PyMuPDF — best column handling, reliable spacing.
Cleaning logic (hyphen rejoining, page number removal) baked into `PyMuPDFParser`.

### Chunking ✅
Tested 3 strategies. Chose token-aware recursive split: 512 tokens, 50 overlap.
Token-aware because character count ≠ token count for OpenAI models.

### Embedding + Indexing ✅
`text-embedding-3-small` (1536 dims) via LLMFoundry proxy. Stored in local ChromaDB.

### Baseline RAG — Failure Modes Established ✅
Three failure modes identified:

| Query | Failure | Root Cause |
|-------|---------|------------|
| RAG-Sequence vs RAG-Token | Figure text ranked 2nd | Keyword match on "BART" |
| Self-attention + BERT bidirectionality | "context doesn't cover this" | Right doc found, wrong chunk ranked |
| RAG state-of-the-art tasks | Figure text ranked 2nd | Same keyword contamination |

### Cross-Encoder Re-Ranking ✅
Added `cross-encoder/ms-marco-MiniLM-L-6-v2`. Retrieve k=10, rerank, return top 3.
- Eliminated figure text from all results
- Fixed the cross-paper question (promoted correct chunk from rank 4 to rank 1)
- **Key insight:** Bi-encoder matches vocabulary. Cross-encoder matches meaning.

### Multi-Query Expansion ✅
LLM generates 3 query variants. All 4 run against ChromaDB. Results deduplicated, then reranked.
- **Key insight:** Multi-query maximises recall. Cross-encoder maximises precision. They are complementary.

### RAGAS Evaluation ✅
4 metrics, 5 golden questions, baseline vs multi-query comparison.

| Metric | What it measures | Points to |
|--------|-----------------|-----------|
| Faithfulness | LLM stayed grounded, no hallucination | Generation stage |
| Answer Relevancy | Answer addresses the question | Generation + prompt |
| Context Precision | Retrieved chunks were relevant | Retriever (ranking) |
| Context Recall | Retrieved chunks covered the answer | Retriever (coverage) |

**Key insight:** Each metric points to a different component to fix.

### Modular OOP Architecture ✅
Refactored from scripts into a proper Python package.
- ABCs enforce the contract — swapping `PyMuPDFParser` for a Docx parser requires zero changes elsewhere
- `RAGFactory` reads YAML — changing the retrieval strategy is a config change, not a code change
- `ReRankRetriever` extends `ChromaRetriever` — indexing is inherited, only `retrieve()` is overridden
- `query_for_ragas()` outputs RAGAS-ready dicts directly

---

## Remaining Roadmap

### Next: Parent-Document Retriever
**Problem:** Small chunks = precise retrieval but the LLM sees too little context. Large chunks = rich context but noisy retrieval. You can't optimise both with a single chunk size.

**Fix:** Index small chunks (256 tokens) for retrieval precision. Store their full parent paragraph separately. When a small chunk is retrieved, return its parent to the LLM instead.

**Implementation:** New `ParentDocumentRetriever` class extending `BaseRetriever`. Two ChromaDB collections: one for small chunks (indexed), one for parents (fetched by ID).

---

### Retrieval Improvements

**Hybrid Search (BM25 + Dense Vectors)**
Vector search misses exact keyword matches (model names, version numbers, proper nouns).
BM25 is the classic keyword ranker. Merge both result sets with Reciprocal Rank Fusion (RRF).

**HyDE — Hypothetical Document Embeddings**
A query and its answer embed differently. HyDE asks the LLM to hallucinate a plausible answer,
then embeds that instead of the query — landing closer to where real answer chunks live.

**Contextual Compression**
Retrieved chunks contain irrelevant sentences around the relevant ones.
Compression extracts only the relevant sentences before passing them to the LLM.

---

### Indexing Improvements

**Semantic Chunking**
Split where meaning changes (embedding similarity between sentences) rather than by token count.
Chunk boundaries align with concept boundaries, not arbitrary token limits.

**Hierarchical Indexing**
Index both chunk-level and document-level summaries.
Route broad questions to summaries, specific questions to chunks.

---

### Generation Improvements

**Citations / Grounding**
Force the LLM to cite which chunk each claim came from.
Makes hallucinations detectable and cross-checkable programmatically.

**Corrective RAG (CRAG)**
After retrieval, a grader checks whether the chunks are actually relevant.
If not, reformulate the query or fall back to a web search before generating.

---

### Architecture Shifts

**Conversational RAG**
Stateful pipeline — maintains chat history and rephrases follow-up questions
("what about BERT?" → full question with context) before retrieval.

**Agentic RAG**
The LLM decides when to retrieve, what to retrieve, and whether to retrieve again.
Turns RAG from a fixed pipeline into a reasoning loop.

---

## Key Conceptual Lessons

1. **Parser choice is not trivial.** Figure text contamination is a real, measurable problem.

2. **Character count ≠ token count.** Always use token-aware splitting for OpenAI models.

3. **Vector similarity finds vocabulary overlap, not relevant answers.** Figure text scored highly because it shared the word "BART."

4. **Bi-encoder recalls. Cross-encoder ranks.** They solve different problems and work best together.

5. **Multi-query expands recall before the cross-encoder improves precision.** The cross-encoder can only rerank what was retrieved.

6. **Retrieval failure ≠ LLM failure.** When the LLM says "I don't know," check the retrieved chunks first.

7. **Each RAGAS metric points to a different component.** Faithfulness = LLM. Context Recall = retriever coverage. Context Precision = retriever ranking. Answer Relevancy = prompt + LLM.

8. **ABCs + factory pattern = swappable components.** Changing retrieval strategy is a YAML edit, not a code change.
