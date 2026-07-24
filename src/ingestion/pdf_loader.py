from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pdfplumber

from ingestion.clean import clean_text, detect_repeated_lines


PageRecord = dict[str, Any]
logger = logging.getLogger(__name__)

MIN_EXTRACTED_TEXT_CHARS = 25


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

    _log_summary(records)
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
        extracted_pages = [
            {
                "page": page,
                "text": (page.extract_text() or "").strip(),
            }
            for page in pdf.pages
        ]
        page_texts = [page_data["text"] for page_data in extracted_pages]
        repeated_lines = detect_repeated_lines(page_texts)
        cleaned_texts = [
            clean_text(page_text, repeated_lines=repeated_lines)
            for page_text in page_texts
        ]

        doc_id = _document_title(pdf_path, page_texts)

        records = [
            _page_record(
                pdf_path=pdf_path,
                page=page_data["page"],
                page_number=page_number,
                text=cleaned_texts[page_number - 1],
                doc_id=doc_id,
            )
            for page_number, page_data in enumerate(extracted_pages, start=1)
        ]

    return records


def _page_record(
    pdf_path: Path,
    page: Any,
    page_number: int,
    text: str,
    doc_id: str,
) -> PageRecord:
    base_record: PageRecord = {
        "source": pdf_path.name,
        "page": page_number,
        "doc_id": doc_id,
    }

    tables = _extract_tables(page)
    if tables:
        return {
            **base_record,
            "content_type": "table",
            "tables": tables,
        }

    if _has_extractable_text(text):
        return {
            **base_record,
            "content_type": "text",
            "text": text,
        }

    ocr_text = _ocr_page(page)
    if ocr_text:
        return {
            **base_record,
            "content_type": "ocr",
            "text": ocr_text,
        }

    return {
        **base_record,
        "content_type": "failed",
        "text": "",
        "extraction_failed": True,
    }


def _extract_tables(page: Any) -> list[dict[str, list[list[str | None]]]]:
    raw_tables = page.extract_tables() or []
    tables = []
    for rows in raw_tables:
        cleaned_rows = [
            [cell.strip() if isinstance(cell, str) else cell for cell in row]
            for row in rows
            if any(cell for cell in row)
        ]
        if cleaned_rows:
            tables.append({"rows": cleaned_rows})

    return tables


def _has_extractable_text(text: str) -> bool:
    return len(text.strip()) >= MIN_EXTRACTED_TEXT_CHARS


def _ocr_page(page: Any) -> str:
    try:
        import pytesseract
    except ImportError:
        logger.warning("pytesseract is not installed; skipping OCR fallback")
        return ""

    try:
        image = page.to_image(resolution=300).original
        return clean_text(pytesseract.image_to_string(image))
    except Exception as exc:
        logger.warning("OCR failed for page: %s", exc)
        return ""


def _log_summary(records: list[PageRecord]) -> None:
    text_count = _count_pages(records, "text")
    table_count = _count_pages(records, "table")
    ocr_count = _count_pages(records, "ocr")
    failed_pages = [
        f"{record['source']} page {record['page']}"
        for record in records
        if record["content_type"] == "failed"
    ]

    logger.info(
        "Ingestion summary: %s pages text, %s pages table, %s pages OCR",
        text_count,
        table_count,
        ocr_count,
    )
    if failed_pages:
        logger.warning(
            "Pages failed both extraction and OCR: %s",
            failed_pages,
        )


def _count_pages(records: list[PageRecord], content_type: str) -> int:
    return len(
        {
            (record["source"], record["page"])
            for record in records
            if record["content_type"] == content_type
        }
    )


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
