from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pdfplumber


PageRecord = dict[str, Any]


def load_pdf_pages(pdf_dir: Path | str) -> list[PageRecord]:
    """Extract page-level text and metadata from every PDF in a folder."""
    folder = Path(pdf_dir)
    if not folder.exists():
        raise FileNotFoundError(f"PDF folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a folder of PDFs: {folder}")

    records: list[PageRecord] = []
    for pdf_path in sorted(folder.glob("*.pdf")):
        records.extend(_load_single_pdf(pdf_path))

    return records


def save_pdf_pages_jsonl(pdf_dir: Path | str, output_path: Path | str) -> Path:
    """Extract page-level PDF records and write them as JSONL."""
    records = load_pdf_pages(pdf_dir)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return destination


def _load_single_pdf(pdf_path: Path) -> list[PageRecord]:
    with pdfplumber.open(pdf_path) as pdf:
        page_texts = [(page.extract_text() or "").strip() for page in pdf.pages]

    doc_id = _document_title(pdf_path, page_texts)

    return [
        {
            "text": text,
            "source": pdf_path.name,
            "page": page_number,
            "doc_id": doc_id,
        }
        for page_number, text in enumerate(page_texts, start=1)
    ]


def _document_title(pdf_path: Path, page_texts: list[str]) -> str:
    first_line = next(
        (
            line.strip()
            for page_text in page_texts
            for line in page_text.splitlines()
            if line.strip()
        ),
        "",
    )
    return first_line or pdf_path.stem
