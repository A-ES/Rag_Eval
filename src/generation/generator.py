from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from chunking.models import Chunk
from config import settings


RerankedCandidate = tuple[Chunk, float]
MARKER_RE = re.compile(r"\[(\d+)\]")


class LLMClient(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return a model response as text."""


@dataclass(frozen=True)
class Citation:
    marker: str
    chunk_id: str


@dataclass(frozen=True)
class GenerationResponse:
    answer: str
    citations: list[Citation]
    insufficient_context: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [asdict(citation) for citation in self.citations],
            "insufficient_context": self.insufficient_context,
        }


class GroqLLMClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.generation_model
        self._client: Any | None = None

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for Groq generation")
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""


class ContextGenerator:
    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        max_context_chunks: int = 5,
    ) -> None:
        if max_context_chunks <= 0:
            raise ValueError("max_context_chunks must be positive")
        self.llm_client = llm_client or GroqLLMClient()
        self.max_context_chunks = max_context_chunks

    def generate(
        self,
        query: str,
        candidates: list[RerankedCandidate],
    ) -> GenerationResponse:
        context_chunks = [chunk for chunk, _score in candidates[: self.max_context_chunks]]
        marker_to_chunk_id = {
            f"[{index}]": chunk.chunk_id
            for index, chunk in enumerate(context_chunks, start=1)
        }
        raw_response = self.llm_client.complete(
            system_prompt=_system_prompt(),
            user_prompt=_user_prompt(query, context_chunks),
        )
        payload = _parse_model_json(raw_response)
        answer = str(payload.get("answer", "")).strip()
        insufficient_context = _is_insufficient(answer, payload)
        citations = _citations_from_answer(answer, marker_to_chunk_id)

        if insufficient_context:
            citations = []

        return GenerationResponse(
            answer=answer or "insufficient context",
            citations=citations,
            insufficient_context=insufficient_context,
        )


def generate_answer(
    query: str,
    candidates: list[RerankedCandidate],
    *,
    llm_client: LLMClient | None = None,
    max_context_chunks: int = 5,
) -> GenerationResponse:
    return ContextGenerator(
        llm_client=llm_client,
        max_context_chunks=max_context_chunks,
    ).generate(query, candidates)


def _system_prompt() -> str:
    return (
        "You answer questions only from the provided context. "
        "If the context does not support the answer, say exactly "
        '"insufficient context". Include bracketed citations like [1] and [2] '
        "for every supported factual claim. Return only JSON with keys "
        '"answer" and "insufficient_context".'
    )


def _user_prompt(query: str, chunks: list[Chunk]) -> str:
    context = "\n\n".join(
        (
            f"[{index}] chunk_id={chunk.chunk_id}; "
            f"source={chunk.source}; pages={chunk.pages}\n{chunk.text}"
        )
        for index, chunk in enumerate(chunks, start=1)
    )
    return (
        f"Query:\n{query}\n\n"
        f"Context:\n{context or 'No context provided.'}\n\n"
        "Answer using only the context. Cite supporting chunks with bracketed "
        "markers that match the context numbers."
    )


def _parse_model_json(raw_response: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object")
    return payload


def _is_insufficient(answer: str, payload: dict[str, Any]) -> bool:
    explicit = payload.get("insufficient_context")
    if isinstance(explicit, bool):
        return explicit
    return answer.strip().lower() == "insufficient context"


def _citations_from_answer(
    answer: str,
    marker_to_chunk_id: dict[str, str],
) -> list[Citation]:
    citations = []
    seen = set()
    for match in MARKER_RE.finditer(answer):
        marker = match.group(0)
        chunk_id = marker_to_chunk_id.get(marker)
        if chunk_id and marker not in seen:
            citations.append(Citation(marker=marker, chunk_id=chunk_id))
            seen.add(marker)
    return citations
