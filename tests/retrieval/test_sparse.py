from __future__ import annotations

from chunking.models import Chunk
from retrieval.sparse import SparseRetriever, tokenize


class FakeBM25:
    def __init__(self, corpus: list[list[str]]) -> None:
        self.corpus = corpus

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        query = set(query_tokens)
        return [float(len(query.intersection(document))) for document in self.corpus]


def test_tokenize_keeps_numbers_and_clause_parts() -> None:
    assert tokenize("Section 4.2(a), applies!") == ["section", "4", "2", "a", "applies"]


def test_sparse_retriever_returns_ranked_chunks() -> None:
    chunks = [
        Chunk(
            text="Section 4.2(a) requires emissions monitoring records.",
            doc_id="doc",
            source="reg.pdf",
            pages=[4],
            chunk_strategy="structural",
            chunk_id="doc:structural:1",
        ),
        Chunk(
            text="Section 9 covers unrelated inspection fees.",
            doc_id="doc",
            source="reg.pdf",
            pages=[9],
            chunk_strategy="structural",
            chunk_id="doc:structural:2",
        ),
    ]
    retriever = SparseRetriever(chunks, bm25_factory=FakeBM25)

    results = retriever.retrieve("What does Section 4.2(a) require?", k=1)

    assert len(results) == 1
    chunk, score = results[0]
    assert chunk.chunk_id == "doc:structural:1"
    assert score > 0


def test_sparse_retriever_returns_empty_for_non_positive_k() -> None:
    retriever = SparseRetriever([])

    assert retriever.retrieve("anything", k=0) == []
