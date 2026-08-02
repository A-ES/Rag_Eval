from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QuestionType(StrEnum):
    LOOKUP = "lookup"
    MULTI_HOP = "multi-hop"
    NO_ANSWER = "no-answer"
    AMBIGUOUS = "ambiguous"


class GoldenSetItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    expected_chunk_ids: list[str]
    expected_answer_summary: str = Field(min_length=1)
    question_type: QuestionType

    @field_validator("question", "expected_answer_summary")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field cannot be blank")
        return stripped

    @field_validator("expected_chunk_ids")
    @classmethod
    def strip_chunk_ids(cls, value: list[str]) -> list[str]:
        stripped = [chunk_id.strip() for chunk_id in value]
        if any(not chunk_id for chunk_id in stripped):
            raise ValueError("expected_chunk_ids cannot contain blank values")
        return stripped


@dataclass(frozen=True)
class GoldenSetValidationResult:
    item_count: int
    missing_chunk_ids: dict[int, list[str]]

    @property
    def is_valid(self) -> bool:
        return not self.missing_chunk_ids


def load_golden_set(path: Path | str) -> list[GoldenSetItem]:
    items = []
    for line_number, record in _load_jsonl(path):
        try:
            items.append(GoldenSetItem.model_validate(record))
        except Exception as exc:
            raise ValueError(f"Invalid golden set item on line {line_number}: {exc}") from exc
    return items


def validate_golden_set(
    *,
    golden_set_path: Path | str,
    chunk_index_path: Path | str,
    min_items: int = 80,
    max_items: int = 150,
) -> GoldenSetValidationResult:
    items = load_golden_set(golden_set_path)
    if not min_items <= len(items) <= max_items:
        raise ValueError(
            f"Golden set must contain {min_items}-{max_items} items; found {len(items)}"
        )

    chunk_ids = load_chunk_ids(chunk_index_path)
    missing_chunk_ids = {
        line_number: missing
        for line_number, item in enumerate(items, start=1)
        if (missing := sorted(set(item.expected_chunk_ids) - chunk_ids))
    }
    return GoldenSetValidationResult(
        item_count=len(items),
        missing_chunk_ids=missing_chunk_ids,
    )


def load_chunk_ids(path: Path | str) -> set[str]:
    chunk_ids = set()
    for line_number, record in _load_jsonl(path):
        chunk_id = record.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"Chunk index line {line_number} is missing chunk_id")
        chunk_ids.add(chunk_id.strip())
    return chunk_ids


def golden_set_json_schema() -> dict[str, Any]:
    return GoldenSetItem.model_json_schema()


def _load_jsonl(path: Path | str) -> list[tuple[int, dict[str, Any]]]:
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"JSONL line {line_number} must be an object")
        records.append((line_number, record))
    return records
