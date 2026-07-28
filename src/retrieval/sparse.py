from __future__ import annotations

import re
from typing import Any

from chunking.models import Chunk


TOKEN_RE = re.compile(r"[a-z0-9]+")


class SparseRetriever:
    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        *,
        bm25: Any | None = None,
        bm25_factory: Any | None = None,
    ) -> None:
        self.chunks: list[Chunk] = []
        self._bm25 = bm25
        self._bm25_factory = bm25_factory
        if chunks:
            self.index(chunks)

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        tokenized_corpus = [tokenize(chunk.text) for chunk in self.chunks]

        if self._bm25 is None:
            bm25_factory = self._bm25_factory or _load_bm25_factory()
            self._bm25 = bm25_factory(tokenized_corpus)

    def retrieve(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        if k <= 0 or not self.chunks:
            return []
        if self._bm25 is None:
            raise ValueError("SparseRetriever has no index. Call index(chunks) first.")

        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            enumerate(scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            (self.chunks[index], float(score))
            for index, score in ranked[:k]
        ]


def tokenize(text: str) -> list[str]:
    """Lowercase and strip punctuation while preserving numeric clause tokens."""
    return TOKEN_RE.findall(text.lower())


def _load_bm25_factory() -> Any:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError(
            "SparseRetriever requires rank-bm25. Install project dependencies "
            "with `uv sync --dev` or `pip install rank-bm25`."
        ) from exc

    return BM25Okapi
