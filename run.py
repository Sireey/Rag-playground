# run.py  — entry point for the RAG pipeline
#
# Usage:
#   python run.py                          # query mode (uses existing index)
#   python run.py --ingest                 # (re)index all PDFs in data/raw/, then query
#   python run.py --config configs/baseline.yaml   # use a different config
#   python run.py --reset                  # wipe and rebuild the index

import argparse
import glob
import os

from rag import RAGFactory

QUESTIONS = [
    "What is the difference between RAG-Sequence and RAG-Token?",
    "How does the self-attention mechanism in the Transformer relate to BERT's bidirectional pre-training?",
    "What tasks did the RAG model achieve state-of-the-art results on?",
]

def main():
    parser = argparse.ArgumentParser(description="Run the RAG pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--ingest", action="store_true", help="Index PDFs before querying")
    parser.add_argument("--reset", action="store_true", help="Wipe and rebuild the index")
    args = parser.parse_args()

    pipeline = RAGFactory.from_yaml(args.config)

    if args.ingest or args.reset:
        pdfs = sorted(glob.glob("data/raw/*.pdf"))
        if not pdfs:
            print("No PDFs found in data/raw/")
            return
        print(f"Indexing {len(pdfs)} file(s): {[os.path.basename(p) for p in pdfs]}")
        n = pipeline.ingest(pdfs, reset=args.reset)
        print(f"Indexed {n} chunks.\n")

    for q in QUESTIONS:
        print(f"\n{'─'*70}")
        print(f"Q: {q}")
        result = pipeline.query(q)
        print(f"A: {result['result']}")
        print("Sources:", [doc.metadata.get('source') for doc in result['source_documents']])

if __name__ == "__main__":
    main()
