import json
from pathlib import Path

import pytest

from ingestion.pdf_loader import load_pdf_pages, save_pdf_pages_jsonl


FIXTURE_PDF = Path(__file__).resolve().parents[1] / "fixtures" / "sample.pdf"


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
    assert sample_records[0]["text"].strip()
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
    assert set(sample_records[0]) == {"text", "source", "page", "doc_id"}
    assert sample_records[0]["page"] == 1
