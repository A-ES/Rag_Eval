from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunking import FixedSizeChunker, SemanticChunker, StructuralChunker
from chunking.models import Chunk, Chunker, Document
from chunking.utils import load_document_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all chunkers on one ingestion JSONL document."
    )
    parser.add_argument("jsonl_path", help="Path to ingestion JSONL output")
    parser.add_argument("--fixed-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--semantic-threshold", type=float, default=0.35)
    args = parser.parse_args()

    document = load_document_jsonl(args.jsonl_path)
    chunkers: list[Chunker] = [
        FixedSizeChunker(
            chunk_size_tokens=args.fixed_size,
            overlap_tokens=args.overlap,
        ),
        StructuralChunker.from_sample(document.text),
        SemanticChunker(distance_threshold=args.semantic_threshold),
    ]

    for chunker in chunkers:
        chunks = chunker.chunk(document)
        print(_summary(document, chunker, chunks))


def _summary(document: Document, chunker: Chunker, chunks: list[Chunk]) -> str:
    lengths = [len(chunk.text.split()) for chunk in chunks]
    average_length = mean(lengths) if lengths else 0
    return (
        f"{document.doc_id} | {chunker.chunk_strategy}: "
        f"{len(chunks)} chunks, avg {average_length:.1f} words/chunk"
    )


if __name__ == "__main__":
    main()
