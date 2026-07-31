from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from chunking.models import Chunk

SENTENCE_RE = re.compile(r"[^.!?\n]*(?:\[[0-9]+\])[^.!?\n]*[.!?]?")


class LLMClient(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return a model response as text."""


class CitationLike(Protocol):
    marker: str
    chunk_id: str


class CitationSupportJudge(Protocol):
    def supports(self, *, claim: str, chunk: Chunk) -> tuple[bool, str]:
        """Return whether a chunk supports a claim and the reason."""


@dataclass(frozen=True)
class UnsupportedCitation:
    marker: str
    chunk_id: str
    claim: str
    reason: str


class LLMCitationSupportJudge:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def supports(self, *, claim: str, chunk: Chunk) -> tuple[bool, str]:
        raw_response = self.llm_client.complete(
            system_prompt=_judge_system_prompt(),
            user_prompt=_judge_user_prompt(claim, chunk),
        )
        payload = _parse_judge_json(raw_response)
        supported = bool(payload.get("supported", False))
        reason = str(payload.get("reason", "")).strip()
        return supported, reason


def verify_citations(
    *,
    answer: str,
    citations: list[CitationLike],
    marker_to_chunk: dict[str, Chunk],
    judge: CitationSupportJudge,
) -> list[UnsupportedCitation]:
    unsupported = []
    for citation in citations:
        chunk = marker_to_chunk.get(citation.marker)
        if chunk is None:
            unsupported.append(
                UnsupportedCitation(
                    marker=citation.marker,
                    chunk_id=citation.chunk_id,
                    claim="",
                    reason="citation marker is not mapped to a provided chunk",
                )
            )
            continue

        claim = _claim_for_marker(answer, citation.marker)
        supported, reason = judge.supports(claim=claim, chunk=chunk)
        if not supported:
            unsupported.append(
                UnsupportedCitation(
                    marker=citation.marker,
                    chunk_id=citation.chunk_id,
                    claim=claim,
                    reason=reason or "cited chunk does not support the claim",
                )
            )

    return unsupported


def _claim_for_marker(answer: str, marker: str) -> str:
    for match in SENTENCE_RE.finditer(answer):
        sentence = match.group(0).strip()
        if marker in sentence:
            return sentence
    return answer.strip()


def _judge_system_prompt() -> str:
    return (
        "You verify one citation at a time. Decide whether the provided chunk "
        "directly supports the cited claim. Answer only JSON with keys "
        '"supported" (boolean) and "reason" (short string).'
    )


def _judge_user_prompt(claim: str, chunk: Chunk) -> str:
    return (
        f"Claim:\n{claim}\n\n"
        f"Chunk id: {chunk.chunk_id}\n"
        f"Chunk text:\n{chunk.text}\n\n"
        "Does this chunk support the claim?"
    )


def _parse_judge_json(raw_response: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("citation judge response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("citation judge response JSON must be an object")
    return payload
