from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from chunking.models import Chunk, Document, DocumentPage


def document_from_ingestion_records(records: list[dict[str, Any]]) -> Document:
    text_records = [
        record for record in records if record.get("content_type") in {None, "text", "ocr"}
    ]
    if not text_records:
        raise ValueError("No text or OCR records found to chunk")

    first = text_records[0]
    return Document(
        doc_id=str(first["doc_id"]),
        source=str(first["source"]),
        pages=[
            DocumentPage(text=str(record.get("text", "")), page=int(record["page"]))
            for record in text_records
        ],
    )


def load_document_jsonl(path: Path | str) -> Document:
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return document_from_ingestion_records(records)


def make_chunk(
    *,
    text: str,
    document: Document,
    pages: list[int],
    strategy: str,
    index: int,
) -> Chunk:
    stable_hash = hashlib.sha1(
        f"{document.doc_id}:{strategy}:{index}:{text}".encode("utf-8")
    ).hexdigest()[:12]
    return Chunk(
        text=text.strip(),
        doc_id=document.doc_id,
        source=document.source,
        pages=sorted(set(pages)),
        chunk_strategy=strategy,
        chunk_id=f"{document.doc_id}:{strategy}:{index}:{stable_hash}",
    )
