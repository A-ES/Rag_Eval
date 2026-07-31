from __future__ import annotations

import pytest

from chunking.models import Chunk
from retrieval.rerank import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.seen_pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.seen_pairs = pairs
        return [
            3.0 if "best" in text else 2.0 if "good" in text else 1.0
            for _query, text in pairs
        ]


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        text=text,
        doc_id="doc",
        source="source.pdf",
        pages=[1],
        chunk_strategy="structural",
        chunk_id=chunk_id,
    )


def test_cross_encoder_reranker_returns_top_scored_candidates() -> None:
    model = FakeCrossEncoder()
    reranker = CrossEncoderReranker(model=model, top_n=2)
    candidates = [
        (chunk("a", "adequate answer"), 0.9),
        (chunk("b", "best answer"), 0.8),
        (chunk("c", "good answer"), 0.7),
    ]

    results = reranker.rerank("question", candidates)

    assert [result.chunk_id for result, _score in results] == ["b", "c"]
    assert [score for _result, score in results] == [3.0, 2.0]
    assert model.seen_pairs == [
        ("question", "adequate answer"),
        ("question", "best answer"),
        ("question", "good answer"),
    ]


def test_cross_encoder_reranker_only_scores_top_candidate_limit() -> None:
    model = FakeCrossEncoder()
    reranker = CrossEncoderReranker(model=model, candidate_limit=20, top_n=5)
    candidates = [
        (chunk(str(index), f"candidate {index}"), float(index))
        for index in range(25)
    ]

    results = reranker.rerank("query", candidates)

    assert len(model.seen_pairs) == 20
    assert len(results) == 5
    assert all(int(result.chunk_id) < 20 for result, _score in results)


def test_cross_encoder_reranker_returns_empty_for_no_candidates() -> None:
    assert CrossEncoderReranker(model=FakeCrossEncoder()).rerank("query", []) == []


def test_cross_encoder_reranker_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError):
        CrossEncoderReranker(candidate_limit=0)
    with pytest.raises(ValueError):
        CrossEncoderReranker(top_n=0)
