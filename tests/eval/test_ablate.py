from __future__ import annotations

import json

from eval.ablate import (
    AblationVariant,
    format_comparison_table,
    load_ablation_config,
)


def test_load_ablation_config_parses_variants(tmp_path) -> None:
    config_path = tmp_path / "ablate.json"
    config_path.write_text(
        json.dumps(
            {
                "golden_set_path": "data/golden_set.jsonl",
                "source_jsonl_path": "data/processed/pages.jsonl",
                "output_path": "artifacts/ablate.json",
                "retrieve_k": 10,
                "final_k": 5,
                "variants": [
                    {
                        "name": "fixed-small",
                        "chunk_strategy": "fixed",
                        "chunk_size": 256,
                        "embedding_model": "mini",
                        "reranker_on": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = load_ablation_config(config_path)

    assert config.retrieve_k == 10
    assert config.variants == [
        AblationVariant(
            name="fixed-small",
            chunk_strategy="fixed",
            chunk_size=256,
            embedding_model="mini",
            reranker_on=False,
        )
    ]


def test_format_comparison_table_sorts_input_order_preserved_by_caller() -> None:
    table = format_comparison_table(
        [
            {
                "variant": {
                    "name": "better",
                    "chunk_strategy": "fixed",
                    "chunk_size": 256,
                    "embedding_model": "mini",
                    "reranker_on": True,
                },
                "aggregate": {"recall_at_k": 0.8, "faithfulness": 0.7},
            },
            {
                "variant": {
                    "name": "worse",
                    "chunk_strategy": "structural",
                    "chunk_size": None,
                    "embedding_model": "mini",
                    "reranker_on": False,
                },
                "aggregate": {"recall_at_k": 0.4, "faithfulness": 0.9},
            },
        ]
    )

    lines = table.splitlines()
    assert "recall_at_k" in lines[0]
    assert lines[2].startswith("better")
    assert "0.8000" in lines[2]
