from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chunking import FixedSizeChunker, SemanticChunker, StructuralChunker
from chunking.models import Chunk, Chunker, Document
from chunking.utils import load_document_jsonl
from eval.golden_set import load_golden_set
from eval.run_eval import EvalConfig, RetrievalResult, run_evaluation
from generation.generator import ContextGenerator
from retrieval import (
    CrossEncoderReranker,
    DenseRetriever,
    SparseRetriever,
)


VALID_CHUNK_STRATEGIES = {"fixed", "structural", "semantic"}


@dataclass(frozen=True)
class AblationVariant:
    name: str
    chunk_strategy: str
    chunk_size: int | None
    embedding_model: str
    reranker_on: bool


@dataclass(frozen=True)
class AblationConfig:
    golden_set_path: str
    source_jsonl_path: str
    output_path: str | None
    retrieve_k: int
    final_k: int
    variants: list[AblationVariant]


class PassthroughReranker:
    def __init__(self, top_n: int) -> None:
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        return candidates[: self.top_n]


def run_ablation(config: AblationConfig) -> dict[str, Any]:
    golden_items = load_golden_set(config.golden_set_path)
    document = load_document_jsonl(config.source_jsonl_path)
    variant_results = [
        _run_variant(
            variant=variant,
            document=document,
            golden_items=golden_items,
            retrieve_k=config.retrieve_k,
            final_k=config.final_k,
        )
        for variant in config.variants
    ]
    sorted_results = sorted(
        variant_results,
        key=lambda result: result["aggregate"].get("recall_at_k", 0.0),
        reverse=True,
    )
    return {
        "variants": sorted_results,
        "comparison_table": format_comparison_table(sorted_results),
    }


def load_ablation_config(path: Path | str) -> AblationConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    variants = [_variant_from_record(record) for record in payload["variants"]]
    return AblationConfig(
        golden_set_path=str(payload["golden_set_path"]),
        source_jsonl_path=str(payload["source_jsonl_path"]),
        output_path=(
            str(payload["output_path"]) if payload.get("output_path") else None
        ),
        retrieve_k=int(payload.get("retrieve_k", 20)),
        final_k=int(payload.get("final_k", 5)),
        variants=variants,
    )


def format_comparison_table(variant_results: list[dict[str, Any]]) -> str:
    metric_names = sorted(
        {
            metric
            for result in variant_results
            for metric in result.get("aggregate", {})
        }
    )
    columns = [
        "variant",
        "chunk_strategy",
        "chunk_size",
        "embedding_model",
        "reranker_on",
        *metric_names,
    ]
    rows = [
        [
            result["variant"]["name"],
            result["variant"]["chunk_strategy"],
            str(result["variant"].get("chunk_size") or "-"),
            result["variant"]["embedding_model"],
            str(result["variant"]["reranker_on"]).lower(),
            *[
                _format_metric(result.get("aggregate", {}).get(metric))
                for metric in metric_names
            ],
        ]
        for result in variant_results
    ]
    widths = [
        max(len(column), *(len(row[index]) for row in rows)) if rows else len(column)
        for index, column in enumerate(columns)
    ]
    lines = [
        "  ".join(column.ljust(widths[index]) for index, column in enumerate(columns)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run config-driven RAG ablations and print a comparison table."
    )
    parser.add_argument("config_path")
    args = parser.parse_args(argv)

    config = load_ablation_config(args.config_path)
    results = run_ablation(config)
    print(results["comparison_table"])

    if config.output_path:
        Path(config.output_path).write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote ablation results to {config.output_path}")

    return 0


def _run_variant(
    *,
    variant: AblationVariant,
    document: Document,
    golden_items: Any,
    retrieve_k: int,
    final_k: int,
) -> dict[str, Any]:
    chunks = _chunk_document(variant, document)
    dense_retriever = DenseRetriever(
        collection_name=f"ablate_{_safe_name(variant.name)}",
        model_name=variant.embedding_model,
        qdrant_path=":memory:",
    )
    dense_retriever.index(chunks)
    sparse_retriever = SparseRetriever(chunks)
    reranker = (
        CrossEncoderReranker(candidate_limit=retrieve_k, top_n=final_k)
        if variant.reranker_on
        else PassthroughReranker(top_n=final_k)
    )
    generator = ContextGenerator(max_context_chunks=final_k)
    results = run_evaluation(
        golden_items=golden_items,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        reranker=reranker,
        generator=generator,
        config=EvalConfig(retrieve_k=retrieve_k, final_k=final_k),
    )
    return {
        "variant": asdict(variant),
        "chunk_count": len(chunks),
        "aggregate": results["aggregate"],
        "questions": results["questions"],
    }


def _chunk_document(variant: AblationVariant, document: Document) -> list[Chunk]:
    chunker = _chunker_for_variant(variant, document)
    return chunker.chunk(document)


def _chunker_for_variant(
    variant: AblationVariant,
    document: Document,
) -> Chunker:
    if variant.chunk_strategy == "fixed":
        return FixedSizeChunker(chunk_size_tokens=variant.chunk_size or 512)
    if variant.chunk_strategy == "structural":
        return StructuralChunker.from_sample(document.text)
    if variant.chunk_strategy == "semantic":
        return SemanticChunker(model_name=variant.embedding_model)
    raise ValueError(f"Unsupported chunk_strategy: {variant.chunk_strategy}")


def _variant_from_record(record: dict[str, Any]) -> AblationVariant:
    chunk_strategy = str(record["chunk_strategy"])
    if chunk_strategy not in VALID_CHUNK_STRATEGIES:
        raise ValueError(
            f"chunk_strategy must be one of {sorted(VALID_CHUNK_STRATEGIES)}"
        )
    chunk_size = record.get("chunk_size")
    return AblationVariant(
        name=str(record.get("name") or _default_variant_name(record)),
        chunk_strategy=chunk_strategy,
        chunk_size=int(chunk_size) if chunk_size is not None else None,
        embedding_model=str(
            record.get("embedding_model")
            or "sentence-transformers/all-MiniLM-L6-v2"
        ),
        reranker_on=bool(record.get("reranker_on", True)),
    )


def _default_variant_name(record: dict[str, Any]) -> str:
    reranker = "rerank" if record.get("reranker_on", True) else "no-rerank"
    return f"{record['chunk_strategy']}-{record.get('chunk_size', 'na')}-{reranker}"


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
