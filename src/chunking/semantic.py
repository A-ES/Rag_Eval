from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

from chunking.models import Chunk, Document
from chunking.utils import make_chunk


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker:
    chunk_strategy = "semantic"

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        distance_threshold: float = 0.35,
        model: Any | None = None,
    ) -> None:
        if not 0 <= distance_threshold <= 2:
            raise ValueError("distance_threshold must be between 0 and 2")
        self.model_name = model_name
        self.distance_threshold = distance_threshold
        self._model = model

    def chunk(self, document: Document) -> list[Chunk]:
        sentences_with_pages = _sentences_with_pages(document)
        if not sentences_with_pages:
            return []

        sentences = [sentence for sentence, _ in sentences_with_pages]
        embeddings = self._encode(sentences)

        chunks: list[Chunk] = []
        current_sentences = [sentences_with_pages[0][0]]
        current_pages = [sentences_with_pages[0][1]]

        for index in range(1, len(sentences_with_pages)):
            distance = _cosine_distance(embeddings[index - 1], embeddings[index])
            if distance >= self.distance_threshold:
                chunks.append(
                    make_chunk(
                        text=" ".join(current_sentences),
                        document=document,
                        pages=current_pages,
                        strategy=self.chunk_strategy,
                        index=len(chunks) + 1,
                    )
                )
                current_sentences = []
                current_pages = []

            current_sentences.append(sentences_with_pages[index][0])
            current_pages.append(sentences_with_pages[index][1])

        if current_sentences:
            chunks.append(
                make_chunk(
                    text=" ".join(current_sentences),
                    document=document,
                    pages=current_pages,
                    strategy=self.chunk_strategy,
                    index=len(chunks) + 1,
                )
            )

        return chunks

    def _encode(self, sentences: list[str]) -> Sequence[Sequence[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(sentences)


def _sentences_with_pages(document: Document) -> list[tuple[str, int]]:
    sentences = []
    for page in document.pages:
        for sentence in SENTENCE_RE.split(page.text):
            cleaned = sentence.strip()
            if cleaned:
                sentences.append((cleaned, page.page))
    return sentences


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 1.0
    return 1 - dot / (left_norm * right_norm)
