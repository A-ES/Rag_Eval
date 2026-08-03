from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

from chunking.models import Chunk
from generation.generator import GroqLLMClient


class LLMJudge(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return judge output as JSON text."""


@dataclass(frozen=True)
class ClaimFaithfulness:
    claim: str
    supported: bool
    reason: str


@dataclass(frozen=True)
class FaithfulnessResult:
    score: float
    claims: list[ClaimFaithfulness]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "claims": [asdict(claim) for claim in self.claims],
        }


@dataclass(frozen=True)
class RelevanceResult:
    score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def recall_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    expected_ids = set(expected)
    if not expected_ids:
        return 1.0
    retrieved_at_k = set(retrieved[:k])
    return len(expected_ids.intersection(retrieved_at_k)) / len(expected_ids)


def precision_at_k(retrieved: Sequence[str], expected: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved_at_k = list(retrieved[:k])
    if not retrieved_at_k:
        return 0.0
    expected_ids = set(expected)
    return sum(1 for chunk_id in retrieved_at_k if chunk_id in expected_ids) / len(
        retrieved_at_k
    )


def mrr(retrieved: Sequence[str], expected: Sequence[str]) -> float:
    expected_ids = set(expected)
    if not expected_ids:
        return 0.0
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected_ids:
            return 1.0 / rank
    return 0.0


def faithfulness(
    answer: str,
    retrieved_chunks: Sequence[Chunk],
    *,
    judge: LLMJudge | None = None,
) -> FaithfulnessResult:
    if not answer.strip():
        return FaithfulnessResult(score=0.0, claims=[])

    llm_judge = judge or GroqLLMClient()
    payload = _judge_json(
        llm_judge,
        system_prompt=_faithfulness_system_prompt(),
        user_prompt=_faithfulness_user_prompt(answer, retrieved_chunks),
    )
    claims = [
        ClaimFaithfulness(
            claim=str(item.get("claim", "")).strip(),
            supported=bool(item.get("supported", False)),
            reason=str(item.get("reason", "")).strip(),
        )
        for item in payload.get("claims", [])
        if isinstance(item, dict)
    ]
    if not claims:
        return FaithfulnessResult(score=0.0, claims=[])

    supported_count = sum(1 for claim in claims if claim.supported)
    return FaithfulnessResult(
        score=supported_count / len(claims),
        claims=claims,
    )


def relevance(
    question: str,
    answer: str,
    *,
    judge: LLMJudge | None = None,
) -> RelevanceResult:
    if not question.strip() or not answer.strip():
        return RelevanceResult(score=0.0, reason="question or answer is blank")

    llm_judge = judge or GroqLLMClient()
    payload = _judge_json(
        llm_judge,
        system_prompt=_relevance_system_prompt(),
        user_prompt=_relevance_user_prompt(question, answer),
    )
    return RelevanceResult(
        score=_clamp_score(payload.get("score", 0.0)),
        reason=str(payload.get("reason", "")).strip(),
    )


def _judge_json(
    judge: LLMJudge,
    *,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    raw_response = judge.complete(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM judge response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM judge response JSON must be an object")
    return payload


def _faithfulness_system_prompt() -> str:
    return (
        "You are a strict faithfulness judge. Split the answer into factual "
        "claims and decide whether each claim is directly supported by the "
        "provided chunks. Return only JSON: "
        '{"claims":[{"claim":"...","supported":true|false,"reason":"..."}]}.'
    )


def _faithfulness_user_prompt(answer: str, retrieved_chunks: Sequence[Chunk]) -> str:
    context = "\n\n".join(
        f"chunk_id={chunk.chunk_id}; source={chunk.source}; pages={chunk.pages}\n{chunk.text}"
        for chunk in retrieved_chunks
    )
    return (
        f"Answer:\n{answer}\n\n"
        f"Retrieved chunks:\n{context or 'No chunks provided.'}\n\n"
        "Does every factual claim in the answer trace to the provided chunks?"
    )


def _relevance_system_prompt() -> str:
    return (
        "You are a strict answer relevance judge. Decide whether the answer "
        "addresses the question asked. Return only JSON with keys "
        '"score" (number from 0 to 1) and "reason" (short string).'
    )


def _relevance_user_prompt(question: str, answer: str) -> str:
    return f"Question:\n{question}\n\nAnswer:\n{answer}\n\nDoes the answer address the question?"


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, score))
