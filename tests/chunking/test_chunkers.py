from chunking import FixedSizeChunker, SemanticChunker, StructuralChunker
from chunking.structural import sniff_header_patterns
from chunking.models import Document, DocumentPage


def sample_document() -> Document:
    return Document(
        doc_id="doc-1",
        source="sample.pdf",
        pages=[
            DocumentPage(
                page=1,
                text=(
                    "1. Introduction\n"
                    "This document describes retrieval evaluation. "
                    "It explains why measurement matters."
                ),
            ),
            DocumentPage(
                page=2,
                text=(
                    "2. Requirements\n"
                    "The system shall keep retrieval separate from evaluation. "
                    "The system shall report repeatable metrics."
                ),
            ),
        ],
    )


def test_fixed_size_chunker_creates_overlapping_token_chunks() -> None:
    chunker = FixedSizeChunker(
        chunk_size_tokens=8,
        overlap_tokens=2,
        encoding=WhitespaceEncoding(),
    )

    chunks = chunker.chunk(sample_document())

    assert len(chunks) > 1
    assert all(chunk.chunk_strategy == "fixed_size" for chunk in chunks)
    assert chunks[0].doc_id == "doc-1"
    assert chunks[0].source == "sample.pdf"
    assert chunks[0].pages
    assert chunks[0].chunk_id.startswith("doc-1:fixed_size:")


class WhitespaceEncoding:
    def __init__(self) -> None:
        self.token_to_word: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        tokens = []
        for word in text.split():
            token = len(self.token_to_word) + 1
            self.token_to_word[token] = word
            tokens.append(token)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return " ".join(self.token_to_word[token] for token in tokens)


def test_structural_chunker_splits_numbered_sections() -> None:
    chunker = StructuralChunker()

    chunks = chunker.chunk(sample_document())

    assert len(chunks) == 2
    assert chunks[0].text.startswith("1. Introduction")
    assert chunks[0].pages == [1]
    assert chunks[1].text.startswith("2. Requirements")
    assert chunks[1].pages == [2]
    assert all(chunk.chunk_strategy == "structural" for chunk in chunks)


def test_structural_chunker_can_sniff_numbering_from_sample_text() -> None:
    sample = "1) Scope\nBody\n2) Definitions\nBody\n3) Requirements\nBody"

    patterns = sniff_header_patterns(sample)
    chunks = StructuralChunker.from_sample(sample).chunk(
        Document(
            doc_id="doc-3",
            source="sample.txt",
            pages=[DocumentPage(page=1, text=sample)],
        )
    )

    assert patterns
    assert len(chunks) == 3
    assert chunks[0].text.startswith("1) Scope")


class FakeEmbeddingModel:
    def encode(self, sentences: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.02, 0.98],
        ][: len(sentences)]


def test_semantic_chunker_splits_on_embedding_distance_threshold() -> None:
    document = Document(
        doc_id="doc-2",
        source="semantic.pdf",
        pages=[
            DocumentPage(
                page=1,
                text=(
                    "Cats sleep on mats. Cats chase toys. "
                    "Revenue increased this quarter. Profit margins expanded."
                ),
            )
        ],
    )
    chunker = SemanticChunker(model=FakeEmbeddingModel(), distance_threshold=0.4)

    chunks = chunker.chunk(document)

    assert len(chunks) == 2
    assert chunks[0].text == "Cats sleep on mats. Cats chase toys."
    assert chunks[1].text == "Revenue increased this quarter. Profit margins expanded."
    assert all(chunk.pages == [1] for chunk in chunks)
    assert all(chunk.chunk_strategy == "semantic" for chunk in chunks)
