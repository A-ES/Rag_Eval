from __future__ import annotations

from dataclasses import dataclass

from chunking.models import Chunk


RetrievalResult = tuple[Chunk, float]


@dataclass(frozen=True)
class RRFConfig:
    rank_constant: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.rank_constant < 0:
            raise ValueError("rank_constant cannot be negative")
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ValueError("retriever weights cannot be negative")


def reciprocal_rank_fusion(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    *,
    config: RRFConfig | None = None,
    limit: int | None = None,
) -> list[RetrievalResult]:
    rrf_config = config or RRFConfig()
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}

    _add_rrf_scores(
        results=dense_results,
        weight=rrf_config.dense_weight,
        rank_constant=rrf_config.rank_constant,
        scores=scores,
        chunks=chunks,
    )
    _add_rrf_scores(
        results=sparse_results,
        weight=rrf_config.sparse_weight,
        rank_constant=rrf_config.rank_constant,
        scores=scores,
        chunks=chunks,
    )

    fused = [
        (chunks[chunk_id], score)
        for chunk_id, score in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    return fused if limit is None else fused[:limit]


def _add_rrf_scores(
    *,
    results: list[RetrievalResult],
    weight: float,
    rank_constant: int,
    scores: dict[str, float],
    chunks: dict[str, Chunk],
) -> None:
    for rank, (chunk, _score) in enumerate(results, start=1):
        chunks.setdefault(chunk.chunk_id, chunk)
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight * (
            1.0 / (rank_constant + rank)
        )
