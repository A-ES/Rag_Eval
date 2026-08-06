from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Protocol, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chunking.models import Chunk
from eval.golden_set import GoldenSetItem, load_golden_set
from eval.metrics import faithfulness, mrr, precision_at_k, recall_at_k, relevance
from generation.generator import ContextGenerator, GenerationResponse
from retrieval import (
    CrossEncoderReranker,
    DenseRetriever,
    RRFConfig,
    SparseRetriever,
    reciprocal_rank_fusion,
)


RetrievalResult = tuple[Chunk, float]


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Return ranked chunk results."""


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Rerank candidate chunks."""


class Generator(Protocol):
    def generate(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> GenerationResponse:
        """Generate an answer from reranked candidates."""


@dataclass(frozen=True)
class EvalConfig:
    retrieve_k: int = 20
    final_k: int = 5
    rrf_rank_constant: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5


def run_evaluation(
    *,
    golden_items: list[GoldenSetItem],
    dense_retriever: Retriever,
    sparse_retriever: Retriever,
    reranker: Reranker,
    generator: Generator,
    config: EvalConfig | None = None,
    judge: Any | None = None,
) -> dict[str, Any]:
    eval_config = config or EvalConfig()
    question_results = [
        _evaluate_item(
            item=item,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            reranker=reranker,
            generator=generator,
            config=eval_config,
            judge=judge,
        )
        for item in golden_items
    ]
    return {
        "config": asdict(eval_config),
        "aggregate": _aggregate(question_results),
        "questions": question_results,
    }


def load_chunks_jsonl(path: Path | str) -> list[Chunk]:
    chunks = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        try:
            chunks.append(
                Chunk(
                    text=str(record["text"]),
                    doc_id=str(record["doc_id"]),
                    source=str(record["source"]),
                    pages=[int(page) for page in record["pages"]],
                    chunk_strategy=str(record["chunk_strategy"]),
                    chunk_id=str(record["chunk_id"]),
                )
            )
        except KeyError as exc:
            raise ValueError(
                f"Chunk index line {line_number} is missing field {exc}"
            ) from exc
    return chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run full RAG evaluation over a golden set."
    )
    parser.add_argument("golden_set_path")
    parser.add_argument("chunk_index_path")
    parser.add_argument("output_path")
    parser.add_argument("--retrieve-k", type=int, default=20)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--sparse-weight", type=float, default=0.5)
    parser.add_argument("--rrf-rank-constant", type=int, default=60)
    parser.add_argument("--qdrant-path", default=":memory:")
    parser.add_argument("--collection", default="rag_eval_chunks")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N golden-set items, useful for fast CI subsets.",
    )
    args = parser.parse_args(argv)

    golden_items = load_golden_set(args.golden_set_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        golden_items = golden_items[: args.limit]
    chunks = load_chunks_jsonl(args.chunk_index_path)

    dense_retriever = DenseRetriever(
        collection_name=args.collection,
        qdrant_path=args.qdrant_path,
    )
    dense_retriever.index(chunks)
    sparse_retriever = SparseRetriever(chunks)
    reranker = CrossEncoderReranker(
        candidate_limit=args.retrieve_k,
        top_n=args.final_k,
    )
    generator = ContextGenerator(max_context_chunks=args.final_k)

    results = run_evaluation(
        golden_items=golden_items,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        reranker=reranker,
        generator=generator,
        config=EvalConfig(
            retrieve_k=args.retrieve_k,
            final_k=args.final_k,
            rrf_rank_constant=args.rrf_rank_constant,
            dense_weight=args.dense_weight,
            sparse_weight=args.sparse_weight,
        ),
    )
    Path(args.output_path).write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote evaluation results to {args.output_path}")
    return 0


def _evaluate_item(
    *,
    item: GoldenSetItem,
    dense_retriever: Retriever,
    sparse_retriever: Retriever,
    reranker: Reranker,
    generator: Generator,
    config: EvalConfig,
    judge: Any | None,
) -> dict[str, Any]:
    dense_results = dense_retriever.retrieve(item.question, k=config.retrieve_k)
    sparse_results = sparse_retriever.retrieve(item.question, k=config.retrieve_k)
    fused_results = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        config=RRFConfig(
            rank_constant=config.rrf_rank_constant,
            dense_weight=config.dense_weight,
            sparse_weight=config.sparse_weight,
        ),
        limit=config.retrieve_k,
    )
    reranked_results = reranker.rerank(item.question, fused_results)
    response = generator.generate(item.question, reranked_results)

    retrieved_chunk_ids = [chunk.chunk_id for chunk, _score in reranked_results]
    retrieved_chunks = [chunk for chunk, _score in reranked_results]
    faithfulness_result = faithfulness(response.answer, retrieved_chunks, judge=judge)
    relevance_result = relevance(item.question, response.answer, judge=judge)
    scores = {
        "recall_at_k": recall_at_k(
            retrieved_chunk_ids,
            item.expected_chunk_ids,
            config.final_k,
        ),
        "precision_at_k": precision_at_k(
            retrieved_chunk_ids,
            item.expected_chunk_ids,
            config.final_k,
        ),
        "mrr": mrr(retrieved_chunk_ids, item.expected_chunk_ids),
        "faithfulness": faithfulness_result.score,
        "relevance": relevance_result.score,
    }

    return {
        "question": item.question,
        "question_type": item.question_type.value,
        "expected_chunk_ids": item.expected_chunk_ids,
        "expected_answer_summary": item.expected_answer_summary,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "dense_chunk_ids": [chunk.chunk_id for chunk, _score in dense_results],
        "sparse_chunk_ids": [chunk.chunk_id for chunk, _score in sparse_results],
        "fused_chunk_ids": [chunk.chunk_id for chunk, _score in fused_results],
        "answer": response.answer,
        "citations": [asdict(citation) for citation in response.citations],
        "unsupported_citations": [
            asdict(citation) for citation in response.unsupported_citations
        ],
        "insufficient_context": response.insufficient_context,
        "scores": scores,
        "faithfulness_claims": [
            asdict(claim) for claim in faithfulness_result.claims
        ],
        "relevance_reason": relevance_result.reason,
    }


def _aggregate(question_results: list[dict[str, Any]]) -> dict[str, float]:
    if not question_results:
        return {}
    metric_names = sorted(question_results[0]["scores"])
    return {
        metric_name: mean(
            float(result["scores"][metric_name]) for result in question_results
        )
        for metric_name in metric_names
    }


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
