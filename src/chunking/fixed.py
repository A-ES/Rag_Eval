from __future__ import annotations

from typing import Protocol

from chunking.models import Chunk, Document
from chunking.utils import make_chunk


class TokenEncoding(Protocol):
    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""

    def decode(self, tokens: list[int]) -> str:
        """Decode token ids into text."""


class FixedSizeChunker:
    chunk_strategy = "fixed_size"

    def __init__(
        self,
        *,
        chunk_size_tokens: int = 512,
        overlap_tokens: int = 64,
        encoding_name: str = "cl100k_base",
        encoding: TokenEncoding | None = None,
    ) -> None:
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")
        if overlap_tokens >= chunk_size_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")

        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = encoding or _load_tiktoken_encoding(encoding_name)

    def chunk(self, document: Document) -> list[Chunk]:
        page_spans = []
        tokens: list[int] = []
        for page in document.pages:
            page_tokens = self.encoding.encode(page.text)
            start = len(tokens)
            tokens.extend(page_tokens)
            page_spans.append((page.page, start, len(tokens)))

        chunks: list[Chunk] = []
        step = self.chunk_size_tokens - self.overlap_tokens
        for index, start in enumerate(range(0, len(tokens), step), start=1):
            end = min(start + self.chunk_size_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                continue
            text = self.encoding.decode(chunk_tokens)
            pages = [
                page_number
                for page_number, page_start, page_end in page_spans
                if page_start < end and page_end > start
            ]
            chunks.append(
                make_chunk(
                    text=text,
                    document=document,
                    pages=pages,
                    strategy=self.chunk_strategy,
                    index=index,
                )
            )
            if end == len(tokens):
                break

        return chunks


def _load_tiktoken_encoding(encoding_name: str) -> TokenEncoding:
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError(
            "FixedSizeChunker requires tiktoken. Install project dependencies "
            "with `uv sync --dev` or `pip install tiktoken`."
        ) from exc

    return tiktoken.get_encoding(encoding_name)
