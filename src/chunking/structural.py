from __future__ import annotations

import re

from chunking.models import Chunk, Document
from chunking.utils import make_chunk


DEFAULT_HEADER_PATTERNS = [
    r"^\s*(?:section|chapter|part|article)\s+\d+[A-Za-z0-9.\-()]*\b",
    r"^\s*\d+(?:\.\d+)*[.)]?\s+[A-Z][A-Za-z0-9,;:()\- ]{3,}$",
    r"^\s*\([a-zA-Z0-9]+\)\s+[A-Z][A-Za-z0-9,;:()\- ]{3,}$",
]

SNIFFED_PATTERN_CANDIDATES = [
    r"^\s*\d+\.\s+.+$",
    r"^\s*\d+(?:\.\d+)+\s+.+$",
    r"^\s*\d+\)\s+.+$",
    r"^\s*\([a-zA-Z0-9]+\)\s+.+$",
    r"^\s*(?:section|chapter|part|article)\s+\d+[A-Za-z0-9.\-()]*\b.*$",
]


class StructuralChunker:
    chunk_strategy = "structural"

    def __init__(
        self,
        header_patterns: list[str] | None = None,
        *,
        sample_text: str | None = None,
    ) -> None:
        if header_patterns is None and sample_text:
            header_patterns = sniff_header_patterns(sample_text)
        self.header_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in (header_patterns or DEFAULT_HEADER_PATTERNS)
        ]

    @classmethod
    def from_sample(cls, sample_text: str) -> "StructuralChunker":
        return cls(sample_text=sample_text)

    def chunk(self, document: Document) -> list[Chunk]:
        sections: list[tuple[list[str], list[int]]] = []
        current_lines: list[str] = []
        current_pages: list[int] = []

        for page in document.pages:
            for line in page.text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if self._is_header(stripped) and current_lines:
                    sections.append((current_lines, current_pages))
                    current_lines = []
                    current_pages = []
                current_lines.append(stripped)
                current_pages.append(page.page)

        if current_lines:
            sections.append((current_lines, current_pages))

        if not sections and document.text.strip():
            sections.append(([document.text.strip()], [page.page for page in document.pages]))

        return [
            make_chunk(
                text="\n".join(lines),
                document=document,
                pages=pages,
                strategy=self.chunk_strategy,
                index=index,
            )
            for index, (lines, pages) in enumerate(sections, start=1)
            if "\n".join(lines).strip()
        ]

    def _is_header(self, line: str) -> bool:
        return any(pattern.match(line) for pattern in self.header_patterns)


def sniff_header_patterns(sample_text: str) -> list[str]:
    """Pick likely section-header patterns from representative document text."""
    lines = [line.strip() for line in sample_text.splitlines() if line.strip()]
    matched_patterns = []
    for pattern in SNIFFED_PATTERN_CANDIDATES:
        compiled = re.compile(pattern, re.IGNORECASE)
        if sum(1 for line in lines if compiled.match(line)) >= 2:
            matched_patterns.append(pattern)

    return matched_patterns or DEFAULT_HEADER_PATTERNS
