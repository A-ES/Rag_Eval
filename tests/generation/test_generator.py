from __future__ import annotations

import json

import pytest

from chunking.models import Chunk
from generation import ContextGenerator, generate_answer


class FakeLLMClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.system_prompt = ""
        self.user_prompt = ""

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return json.dumps(self.payload)


class FakeCitationJudge:
    def __init__(self, unsupported_markers: set[str] | None = None) -> None:
        self.unsupported_markers = unsupported_markers or set()
        self.checked: list[tuple[str, str]] = []

    def supports(self, *, claim: str, chunk: Chunk) -> tuple[bool, str]:
        self.checked.append((claim, chunk.chunk_id))
        unsupported = any(marker in claim for marker in self.unsupported_markers)
        if unsupported:
            return False, "chunk does not state this claim"
        return True, "supported"


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        text=text,
        doc_id="doc",
        source="source.pdf",
        pages=[1],
        chunk_strategy="structural",
        chunk_id=chunk_id,
    )


def test_generator_maps_bracketed_citations_to_chunk_ids() -> None:
    client = FakeLLMClient(
        {
            "answer": "The rule requires monitoring records [1] and retention [2].",
            "insufficient_context": False,
        }
    )
    candidates = [
        (chunk("chunk-a", "Monitoring records are required."), 3.0),
        (chunk("chunk-b", "Records must be retained."), 2.0),
    ]

    judge = FakeCitationJudge()

    response = ContextGenerator(llm_client=client, citation_judge=judge).generate(
        "What is required?",
        candidates,
    )

    assert response.to_dict() == {
        "answer": "The rule requires monitoring records [1] and retention [2].",
        "citations": [
            {"marker": "[1]", "chunk_id": "chunk-a"},
            {"marker": "[2]", "chunk_id": "chunk-b"},
        ],
        "insufficient_context": False,
        "unsupported_citations": [],
    }
    assert judge.checked == [
        ("The rule requires monitoring records [1] and retention [2].", "chunk-a"),
        ("The rule requires monitoring records [1] and retention [2].", "chunk-b"),
    ]
    assert "answer questions only from the provided context" in client.system_prompt
    assert "[1] chunk_id=chunk-a" in client.user_prompt


def test_generator_returns_insufficient_context_without_citations() -> None:
    client = FakeLLMClient(
        {"answer": "insufficient context", "insufficient_context": True}
    )

    response = generate_answer(
        "Unsupported question?",
        [(chunk("chunk-a", "Unrelated context."), 1.0)],
        llm_client=client,
        citation_judge=FakeCitationJudge(),
    )

    assert response.answer == "insufficient context"
    assert response.citations == []
    assert response.insufficient_context is True
    assert response.unsupported_citations == []


def test_generator_limits_context_chunks() -> None:
    client = FakeLLMClient({"answer": "Only first chunk [1].", "insufficient_context": False})
    candidates = [
        (chunk("chunk-a", "First."), 3.0),
        (chunk("chunk-b", "Second."), 2.0),
    ]

    response = ContextGenerator(
        llm_client=client,
        citation_judge=FakeCitationJudge(),
        max_context_chunks=1,
    ).generate(
        "Question",
        candidates,
    )

    assert response.citations[0].chunk_id == "chunk-a"
    assert "chunk-a" in client.user_prompt
    assert "chunk-b" not in client.user_prompt


def test_generator_flags_unsupported_citations() -> None:
    client = FakeLLMClient(
        {
            "answer": "The rule requires monitoring [1]. It also requires audits [2].",
            "insufficient_context": False,
        }
    )
    judge = FakeCitationJudge(unsupported_markers={"[2]"})
    candidates = [
        (chunk("chunk-a", "Monitoring is required."), 3.0),
        (chunk("chunk-b", "No audit language appears here."), 2.0),
    ]

    response = ContextGenerator(llm_client=client, citation_judge=judge).generate(
        "What is required?",
        candidates,
    )

    assert response.unsupported_citations
    unsupported = response.unsupported_citations[0]
    assert unsupported.marker == "[2]"
    assert unsupported.chunk_id == "chunk-b"
    assert unsupported.claim == "It also requires audits [2]."
    assert unsupported.reason == "chunk does not state this claim"


def test_generator_rejects_invalid_json_response() -> None:
    class BadLLMClient:
        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            return "not json"

    with pytest.raises(ValueError):
        ContextGenerator(
            llm_client=BadLLMClient(),
            citation_judge=FakeCitationJudge(),
        ).generate("query", [])
