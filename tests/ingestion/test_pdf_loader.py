import json
import logging
from pathlib import Path

import pytest

import ingestion.pdf_loader as pdf_loader
from ingestion.pdf_loader import load_pdf_pages, save_pdf_pages_jsonl


FIXTURE_PDF = Path(__file__).resolve().parents[1] / "fixtures" / "sample.pdf"


class FakeImage:
    original = object()


class FakePage:
    def __init__(
        self,
        text: str = "",
        tables: list[list[list[str | None]]] | None = None,
    ) -> None:
        self.text = text
        self.tables = tables or []

    def extract_text(self) -> str:
        return self.text

    def extract_tables(self) -> list[list[list[str | None]]]:
        return self.tables

    def to_image(self, resolution: int = 300) -> FakeImage:
        return FakeImage()


class FakePdf:
    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages

    def __enter__(self) -> "FakePdf":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_load_pdf_pages_classifies_text_table_ocr_and_failed_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.touch()
    pages = [
        FakePage("A plain text page with enough extracted characters."),
        FakePage("Name Amount", [[["Name", "Amount"], ["Ada", "10"]]]),
        FakePage(""),
        FakePage(""),
    ]

    monkeypatch.setattr(pdf_loader.pdfplumber, "open", lambda _: FakePdf(pages))
    ocr_results = iter(["OCR recovered text", ""])
    monkeypatch.setattr(pdf_loader, "_ocr_page", lambda _: next(ocr_results))

    caplog.set_level(logging.INFO, logger="ingestion.pdf_loader")
    records = load_pdf_pages(tmp_path)

    assert [record["content_type"] for record in records] == [
        "text",
        "table",
        "ocr",
        "failed",
    ]
    assert records[0]["text"] == "A plain text page with enough extracted characters."
    assert records[1]["tables"] == [{"rows": [["Name", "Amount"], ["Ada", "10"]]}]
    assert records[2]["text"] == "OCR recovered text"
    assert records[3]["extraction_failed"] is True
    assert "1 pages text, 1 pages table, 1 pages OCR" in caplog.text
    assert "Pages failed both extraction and OCR: ['sample.pdf page 4']" in caplog.text


def test_load_pdf_pages_cleans_repeated_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.touch()
    pages = [
        FakePage("ACME Header\nFirst page has enough content."),
        FakePage("ACME Header\nSecond page has enough content."),
        FakePage("ACME Header\nThird page has enough content."),
        FakePage("ACME Header\nFourth page has enough content."),
        FakePage("ACME Header\nFifth page has enough content."),
    ]

    monkeypatch.setattr(pdf_loader.pdfplumber, "open", lambda _: FakePdf(pages))

    records = load_pdf_pages(tmp_path)

    assert "ACME Header" not in records[0]["text"]
    assert records[0]["text"] == "First page has enough content."


@pytest.mark.skipif(
    not FIXTURE_PDF.exists(),
    reason="Add a small sample PDF at tests/fixtures/sample.pdf to run this test.",
)
def test_load_pdf_pages_extracts_page_records() -> None:
    records = load_pdf_pages(FIXTURE_PDF.parent)

    sample_records = [
        record for record in records if record["source"] == FIXTURE_PDF.name
    ]
    assert sample_records
    assert sample_records[0]["page"] == 1
    assert sample_records[0]["content_type"] in {"text", "table", "ocr", "failed"}
    assert sample_records[0]["doc_id"]


@pytest.mark.skipif(
    not FIXTURE_PDF.exists(),
    reason="Add a small sample PDF at tests/fixtures/sample.pdf to run this test.",
)
def test_save_pdf_pages_jsonl_writes_one_record_per_page(tmp_path: Path) -> None:
    output_path = tmp_path / "pages.jsonl"

    save_pdf_pages_jsonl(FIXTURE_PDF.parent, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    sample_records = [
        record for record in records if record["source"] == FIXTURE_PDF.name
    ]

    assert sample_records
    assert {"source", "page", "doc_id", "content_type"}.issubset(sample_records[0])
    assert sample_records[0]["page"] == 1
