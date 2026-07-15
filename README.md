# rag-eval-harness

A Python project skeleton for building a retrieval-augmented generation evaluation harness.

## Package Manager

This project uses `uv`.

I chose `uv` because it is fast, uses standard `pyproject.toml` project metadata, and keeps dependency management simple for a small harness that will grow over time. It also supports development dependency groups without requiring extra project-specific tooling.

## Folder Structure

```text
.
├── data/
│   ├── raw/             # Source PDFs and untouched input files
│   └── processed/       # Cleaned, normalized, or extracted artifacts
├── src/
│   ├── api/             # API layer and service entry points
│   ├── chunking/        # Document splitting and chunk metadata logic
│   ├── eval/            # Measurement, scoring, and experiment evaluation
│   ├── ingestion/       # Source loading and raw-to-clean document preparation
│   │   └── pdf_loader.py # Page-by-page PDF extraction to JSONL
│   ├── retrieval/       # Indexing, search, ranking, and retrieval adapters
│   └── config.py        # Environment-driven project settings
├── tests/
│   ├── api/
│   ├── chunking/
│   ├── eval/
│   ├── ingestion/
│   └── retrieval/
└── pyproject.toml
```

## Configuration

Runtime configuration lives in `src/config.py` and is powered by `pydantic-settings`.

Settings can be loaded from environment variables or a local `.env` file. API key fields default to `None` so the project can be imported and tested before any provider credentials are configured.

## Checkpoint: Why Separate Chunking, Retrieval, and Eval?

Chunking, retrieval, and evaluation should live in separate packages because each one is a different source of system behavior and failure.

Chunking controls what information is available to retrieve. Retrieval controls which chunks are selected for a query. Evaluation measures whether those choices actually helped. If these concerns are collapsed into one large `pipeline.py`, it becomes too easy to change multiple variables at once and then mistake a lucky output for real progress.

Keeping retrieval and measurement separated is especially important before Week 4 because a RAG system only improves when we can tell which part improved. If retrieval quality changes, evaluation should catch that independently. If evaluation changes, retrieval should not silently change with it. This separation makes experiments repeatable, failures easier to diagnose, and progress measurable instead of anecdotal.

In short: the pipeline can orchestrate later, but the harness needs clean boundaries first.

## Development

Install dependencies:

```bash
uv sync --dev
```

Run tests:

```bash
uv run pytest
```
# Rag_Eval
# Rag_Eval
