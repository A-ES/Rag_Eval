from __future__ import annotations

import json

import pytest

from chunking.models import Chunk
from eval.metrics import faithfulness, mrr, precision_at_k, recall_at_k, relevance


class FakeJudge:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.system_prompt = ""
        self.user_prompt = ""

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return json.dumps(self.payload)


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        text=text,
        doc_id="doc",
        source="source.pdf",
        pages=[1],
        chunk_strategy="structural",
        chunk_id=chunk_id,
    )


def test_recall_at_k() -> None:
    assert recall_at_k(["a", "b", "c"], ["b", "d"], 2) == 0.5
    assert recall_at_k(["a"], ["b"], 0) == 0.0
    assert recall_at_k(["a"], [], 5) == 1.0


def test_precision_at_k() -> None:
    assert precision_at_k(["a", "b", "c"], ["b", "c"], 2) == 0.5
    assert precision_at_k(["a", "b", "c"], ["b", "c"], 3) == pytest.approx(2 / 3)
    assert precision_at_k([], ["a"], 3) == 0.0


def test_mrr() -> None:
    assert mrr(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
    assert mrr(["a", "b"], ["x"]) == 0.0
    assert mrr(["a"], []) == 0.0


def test_faithfulness_uses_judge_and_returns_per_claim_breakdown() -> None:
    judge = FakeJudge(
        {
            "claims": [
                {
                    "claim": "Monitoring is required.",
                    "supported": True,
                    "reason": "chunk states monitoring is required",
                },
                {
                    "claim": "Audits are monthly.",
                    "supported": False,
                    "reason": "no monthly audit support",
                },
            ]
        }
    )

    result = faithfulness(
        "Monitoring is required. Audits are monthly.",
        [chunk("chunk-1", "Monitoring is required.")],
        judge=judge,
    )

    assert result.score == 0.5
    assert result.claims[0].supported is True
    assert result.claims[1].reason == "no monthly audit support"
    assert "strict faithfulness judge" in judge.system_prompt
    assert "chunk_id=chunk-1" in judge.user_prompt


def test_faithfulness_empty_answer_returns_zero_without_judge() -> None:
    assert faithfulness("", []).score == 0.0


def test_relevance_uses_judge_and_clamps_score() -> None:
    judge = FakeJudge({"score": 1.2, "reason": "directly answers the question"})

    result = relevance("What is required?", "Monitoring is required.", judge=judge)

    assert result.score == 1.0
    assert result.reason == "directly answers the question"
    assert "strict answer relevance judge" in judge.system_prompt


def test_relevance_blank_input_returns_zero_without_judge() -> None:
    assert relevance("", "answer").score == 0.0
