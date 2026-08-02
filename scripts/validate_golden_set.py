from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.golden_set import golden_set_json_schema, validate_golden_set


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate golden_set.jsonl against a chunk-index JSONL."
    )
    parser.add_argument("golden_set_path", help="Path to golden_set.jsonl")
    parser.add_argument("chunk_index_path", help="Path to chunk index JSONL")
    parser.add_argument("--min-items", type=int, default=80)
    parser.add_argument("--max-items", type=int, default=150)
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print the JSON schema for one golden_set.jsonl line and exit.",
    )
    args = parser.parse_args()

    if args.print_schema:
        print(json.dumps(golden_set_json_schema(), indent=2))
        return 0

    try:
        result = validate_golden_set(
            golden_set_path=args.golden_set_path,
            chunk_index_path=args.chunk_index_path,
            min_items=args.min_items,
            max_items=args.max_items,
        )
    except ValueError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1

    if not result.is_valid:
        print("Validation failed: missing expected_chunk_ids", file=sys.stderr)
        for line_number, chunk_ids in result.missing_chunk_ids.items():
            print(
                f"  line {line_number}: {', '.join(chunk_ids)}",
                file=sys.stderr,
            )
        return 1

    print(f"Golden set valid: {result.item_count} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
