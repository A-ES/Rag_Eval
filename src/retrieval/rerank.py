from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from chunking.models import Chunk
from retrieval.fusion import RetrievalResult


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CANDIDATE_LIMIT = 20
DEFAULT_TOP_N = 5


class CrossEncoderReranker:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        model: Any | None = None,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
        top_n: int = DEFAULT_TOP_N,
    ) -> None:
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
        if top_n <= 0:
            raise ValueError("top_n must be positive")

        self.model_name = model_name
        self._model = model
        self.candidate_limit = candidate_limit
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        limited_candidates = list(candidates[: self.candidate_limit])
        if not limited_candidates:
            return []

        chunks = [chunk for chunk, _score in limited_candidates]
        scores = self._predict([(query, chunk.text) for chunk in chunks])
        ranked = sorted(
            zip(chunks, scores),
            key=lambda item: item[1],
            reverse=True,
        )
        return [(chunk, float(score)) for chunk, score in ranked[: self.top_n]]

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

        scores = self._model.predict(pairs)
        return _as_float_list(scores)


def rerank(
    query: str,
    candidates: Sequence[RetrievalResult],
) -> list[RetrievalResult]:
    return CrossEncoderReranker().rerank(query, candidates)


def _as_float_list(scores: Any) -> list[float]:
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    return [float(score) for score in scores]
