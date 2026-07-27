from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunking import FixedSizeChunker, SemanticChunker, StructuralChunker
from chunking.models import Chunker
from chunking.utils import load_document_jsonl
from retrieval import DenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk one document, index chunks in local Qdrant, and test retrieval."
    )
    parser.add_argument("jsonl_path", help="Path to ingestion JSONL output")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "structural", "semantic"],
        default="structural",
    )
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--collection", default="rag_eval_smoke")
    parser.add_argument("--qdrant-path", default=":memory:")
    args = parser.parse_args()

    document = load_document_jsonl(args.jsonl_path)
    chunker = _chunker(args.strategy, document.text)
    chunks = chunker.chunk(document)

    retriever = DenseRetriever(
        collection_name=args.collection,
        qdrant_path=args.qdrant_path,
    )
    retriever.index(chunks)

    queries = args.query or _prompt_queries()
    print(f"Indexed {len(chunks)} {chunker.chunk_strategy} chunks into {args.collection}")
    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, k=args.k)
        for rank, (chunk, score) in enumerate(results, start=1):
            preview = " ".join(chunk.text.split())[:180]
            print(
                f"{rank}. score={score:.4f} pages={chunk.pages} "
                f"chunk_id={chunk.chunk_id}\n   {preview}"
            )


def _chunker(strategy: str, sample_text: str) -> Chunker:
    if strategy == "fixed":
        return FixedSizeChunker()
    if strategy == "semantic":
        return SemanticChunker()
    return StructuralChunker.from_sample(sample_text)


def _prompt_queries() -> list[str]:
    queries = []
    for index in range(1, 4):
        queries.append(input(f"Manual test query {index}: ").strip())
    return [query for query in queries if query]


if __name__ == "__main__":
    main()
