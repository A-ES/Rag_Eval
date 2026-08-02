from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.golden_set import QuestionType, golden_set_json_schema, validate_golden_set


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_validate_golden_set_accepts_existing_chunk_ids(tmp_path: Path) -> None:
    golden_path = tmp_path / "golden_set.jsonl"
    chunk_index_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        golden_path,
        [
            {
                "question": "What does Section 4.2 require?",
                "expected_chunk_ids": ["chunk-1"],
                "expected_answer_summary": "Section 4.2 requires monitoring.",
                "question_type": "lookup",
            }
        ],
    )
    write_jsonl(chunk_index_path, [{"chunk_id": "chunk-1", "text": "Monitoring."}])

    result = validate_golden_set(
        golden_set_path=golden_path,
        chunk_index_path=chunk_index_path,
        min_items=1,
        max_items=1,
    )

    assert result.is_valid
    assert result.item_count == 1


def test_validate_golden_set_flags_missing_chunk_ids(tmp_path: Path) -> None:
    golden_path = tmp_path / "golden_set.jsonl"
    chunk_index_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        golden_path,
        [
            {
                "question": "What does Section 4.2 require?",
                "expected_chunk_ids": ["missing-chunk"],
                "expected_answer_summary": "Section 4.2 requires monitoring.",
                "question_type": "lookup",
            }
        ],
    )
    write_jsonl(chunk_index_path, [{"chunk_id": "chunk-1"}])

    result = validate_golden_set(
        golden_set_path=golden_path,
        chunk_index_path=chunk_index_path,
        min_items=1,
        max_items=1,
    )

    assert not result.is_valid
    assert result.missing_chunk_ids == {1: ["missing-chunk"]}


def test_validate_golden_set_enforces_item_count(tmp_path: Path) -> None:
    golden_path = tmp_path / "golden_set.jsonl"
    chunk_index_path = tmp_path / "chunks.jsonl"
    write_jsonl(golden_path, [])
    write_jsonl(chunk_index_path, [{"chunk_id": "chunk-1"}])

    with pytest.raises(ValueError, match="80-150"):
        validate_golden_set(
            golden_set_path=golden_path,
            chunk_index_path=chunk_index_path,
        )


def test_schema_exposes_question_type_enum() -> None:
    schema = golden_set_json_schema()

    assert QuestionType.LOOKUP == "lookup"
    assert "question_type" in schema["properties"]
