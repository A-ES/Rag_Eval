from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare aggregate metrics from two evaluation result JSON files."
    )
    parser.add_argument("before_path")
    parser.add_argument("after_path")
    args = parser.parse_args()

    before = _load_aggregate(args.before_path)
    after = _load_aggregate(args.after_path)
    metrics = sorted(set(before) | set(after))

    print(f"{'metric':<20} {'before':>12} {'after':>12} {'delta':>12}")
    print("-" * 59)
    for metric in metrics:
        before_value = before.get(metric)
        after_value = after.get(metric)
        delta = (
            None
            if before_value is None or after_value is None
            else after_value - before_value
        )
        print(
            f"{metric:<20} "
            f"{_format_value(before_value):>12} "
            f"{_format_value(after_value):>12} "
            f"{_format_value(delta, signed=True):>12}"
        )
    return 0


def _load_aggregate(path: str) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate", {})
    if not isinstance(aggregate, dict):
        raise ValueError(f"{path} does not contain an aggregate object")
    return {
        key: float(value)
        for key, value in aggregate.items()
        if isinstance(value, int | float)
    }


def _format_value(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
