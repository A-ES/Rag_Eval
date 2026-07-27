from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from chunking.models import Chunk
from retrieval.dense import DenseRetriever


@dataclass
class Hit:
    payload: dict[str, Any]
    score: float


@dataclass
class QueryResult:
    points: list[Hit]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.points: list[Any] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def recreate_collection(self, collection_name: str, vectors_config: Any) -> None:
        self.collections.add(collection_name)
        self.points = []

    def upsert(self, collection_name: str, points: list[Any]) -> None:
        self.collections.add(collection_name)
        self.points.extend(points)

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        limit: int,
        with_payload: bool,
    ) -> QueryResult:
        hits = [
            Hit(payload=point.payload, score=_cosine(query, point.vector))
            for point in self.points
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return QueryResult(points=hits[:limit])


class FakeEmbeddingModel:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [_embed(text) for text in texts]


def test_dense_retriever_indexes_chunks_and_returns_ranked_results() -> None:
    chunks = [
        Chunk(
            text="alpha beta retrieval metrics",
            doc_id="doc",
            source="sample.pdf",
            pages=[1],
            chunk_strategy="structural",
            chunk_id="doc:structural:1",
        ),
        Chunk(
            text="gamma delta compliance clauses",
            doc_id="doc",
            source="sample.pdf",
            pages=[2],
            chunk_strategy="structural",
            chunk_id="doc:structural:2",
        ),
    ]
    retriever = DenseRetriever(
        client=FakeQdrantClient(),
        embedding_model=FakeEmbeddingModel(),
    )

    retriever.index(chunks)
    results = retriever.retrieve("retrieval alpha", k=1)

    assert len(results) == 1
    chunk, score = results[0]
    assert chunk.chunk_id == "doc:structural:1"
    assert chunk.pages == [1]
    assert score > 0


def _embed(text: str) -> list[float]:
    lowered = text.lower()
    return [
        float("alpha" in lowered or "retrieval" in lowered),
        float("gamma" in lowered or "compliance" in lowered),
    ]


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)
