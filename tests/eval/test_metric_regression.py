from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_check_metric_regression_passes_within_threshold(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    baseline_path.write_text(
        json.dumps({"aggregate": {"recall_at_k": 0.90, "faithfulness": 0.80}}),
        encoding="utf-8",
    )
    current_path.write_text(
        json.dumps({"aggregate": {"recall_at_k": 0.88, "faithfulness": 0.80}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_metric_regression.py",
            str(baseline_path),
            str(current_path),
            "--metric",
            "recall_at_k",
            "--metric",
            "faithfulness",
            "--max-drop",
            "0.03",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "passed" in result.stdout


def test_check_metric_regression_fails_beyond_threshold(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    baseline_path.write_text(
        json.dumps({"aggregate": {"recall_at_k": 0.90}}),
        encoding="utf-8",
    )
    current_path.write_text(
        json.dumps({"aggregate": {"recall_at_k": 0.85}}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_metric_regression.py",
            str(baseline_path),
            str(current_path),
            "--metric",
            "recall_at_k",
            "--max-drop",
            "0.03",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Metric regression gate failed" in result.stderr
