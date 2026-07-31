import pytest

from chunking.models import Chunk
from retrieval.fusion import RRFConfig, reciprocal_rank_fusion


def chunk(chunk_id: str) -> Chunk:
    return Chunk(
        text=f"text for {chunk_id}",
        doc_id="doc",
        source="source.pdf",
        pages=[1],
        chunk_strategy="structural",
        chunk_id=chunk_id,
    )


def test_reciprocal_rank_fusion_combines_dense_and_sparse_rankings() -> None:
    a = chunk("a")
    b = chunk("b")
    c = chunk("c")

    fused = reciprocal_rank_fusion(
        dense_results=[(a, 0.9), (b, 0.8)],
        sparse_results=[(b, 12.0), (c, 9.0)],
        config=RRFConfig(rank_constant=60, dense_weight=0.5, sparse_weight=0.5),
    )

    assert [result.chunk_id for result, _score in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx((0.5 / 62) + (0.5 / 61))


def test_reciprocal_rank_fusion_respects_weighting_config() -> None:
    dense_winner = chunk("dense")
    sparse_winner = chunk("sparse")

    fused = reciprocal_rank_fusion(
        dense_results=[(dense_winner, 0.99), (sparse_winner, 0.1)],
        sparse_results=[(sparse_winner, 12.0), (dense_winner, 0.2)],
        config=RRFConfig(rank_constant=0, dense_weight=0.7, sparse_weight=0.3),
    )

    assert fused[0][0].chunk_id == "dense"


def test_reciprocal_rank_fusion_supports_limit() -> None:
    fused = reciprocal_rank_fusion(
        dense_results=[(chunk("a"), 1.0), (chunk("b"), 0.5)],
        sparse_results=[(chunk("c"), 2.0)],
        limit=2,
    )

    assert len(fused) == 2


def test_rrf_config_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError):
        RRFConfig(dense_weight=-0.1)
