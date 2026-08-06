from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail if selected aggregate metrics regress beyond a threshold."
    )
    parser.add_argument("baseline_path")
    parser.add_argument("current_path")
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        default=[],
        help="Aggregate metric to gate. Can be repeated.",
    )
    parser.add_argument(
        "--max-drop",
        type=float,
        default=0.03,
        help="Maximum allowed absolute drop, e.g. 0.03 for 3 points.",
    )
    args = parser.parse_args()

    metrics = args.metrics or ["recall_at_k", "faithfulness"]
    baseline = _load_aggregate(args.baseline_path)
    current = _load_aggregate(args.current_path)

    failures = []
    for metric in metrics:
        if metric not in baseline:
            failures.append(f"{metric}: missing from baseline")
            continue
        if metric not in current:
            failures.append(f"{metric}: missing from current run")
            continue

        drop = baseline[metric] - current[metric]
        if drop > args.max_drop:
            failures.append(
                f"{metric}: baseline={baseline[metric]:.4f}, "
                f"current={current[metric]:.4f}, drop={drop:.4f} "
                f"> max_drop={args.max_drop:.4f}"
            )

    if failures:
        print("Metric regression gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(
        "Metric regression gate passed for "
        f"{', '.join(metrics)} with max_drop={args.max_drop:.4f}"
    )
    return 0


def _load_aggregate(path: str) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(f"{path} does not contain an aggregate object")
    return {
        metric: float(value)
        for metric, value in aggregate.items()
        if isinstance(value, int | float)
    }


if __name__ == "__main__":
    raise SystemExit(main())
