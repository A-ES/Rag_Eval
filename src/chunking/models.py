from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DocumentPage:
    text: str
    page: int


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    pages: list[DocumentPage]

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)


@dataclass(frozen=True)
class Chunk:
    text: str
    doc_id: str
    source: str
    pages: list[int]
    chunk_strategy: str
    chunk_id: str


class Chunker(Protocol):
    chunk_strategy: str

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks."""
