from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_compare_runs_prints_metric_delta(tmp_path) -> None:
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(
        json.dumps({"aggregate": {"recall_at_k": 0.5, "mrr": 0.25}}),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps({"aggregate": {"recall_at_k": 0.75, "mrr": 0.25}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_runs.py",
            str(before_path),
            str(after_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "recall_at_k" in result.stdout
    assert "+0.2500" in result.stdout
