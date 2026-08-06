from __future__ import annotations

from dataclasses import dataclass

from chunking.models import Chunk
from eval.golden_set import GoldenSetItem
from eval.run_eval import EvalConfig, load_chunks_jsonl, run_evaluation
from generation.generator import Citation, GenerationResponse


class FakeRetriever:
    def __init__(self, results: list[tuple[Chunk, float]]) -> None:
        self.results = results

    def retrieve(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        return self.results[:k]


class FakeReranker:
    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        return candidates[:2]


class FakeGenerator:
    def generate(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
    ) -> GenerationResponse:
        return GenerationResponse(
            answer="Monitoring is required [1].",
            citations=[Citation(marker="[1]", chunk_id=candidates[0][0].chunk_id)],
            insufficient_context=False,
            unsupported_citations=[],
        )


@dataclass
class FakeJudge:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        if "faithfulness" in system_prompt:
            return (
                '{"claims":[{"claim":"Monitoring is required.",'
                '"supported":true,"reason":"supported"}]}'
            )
        return '{"score":1.0,"reason":"answers question"}'


def chunk(chunk_id: str) -> Chunk:
    return Chunk(
        text="Monitoring is required.",
        doc_id="doc",
        source="source.pdf",
        pages=[1],
        chunk_strategy="structural",
        chunk_id=chunk_id,
    )


def test_run_evaluation_computes_per_question_and_aggregate_scores() -> None:
    chunk_a = chunk("chunk-a")
    chunk_b = chunk("chunk-b")
    golden_items = [
        GoldenSetItem(
            question="What is required?",
            expected_chunk_ids=["chunk-a"],
            expected_answer_summary="Monitoring is required.",
            question_type="lookup",
        )
    ]

    results = run_evaluation(
        golden_items=golden_items,
        dense_retriever=FakeRetriever([(chunk_a, 0.9), (chunk_b, 0.8)]),
        sparse_retriever=FakeRetriever([(chunk_b, 2.0), (chunk_a, 1.0)]),
        reranker=FakeReranker(),
        generator=FakeGenerator(),
        config=EvalConfig(retrieve_k=2, final_k=2),
        judge=FakeJudge(),
    )

    question_result = results["questions"][0]
    assert question_result["scores"]["recall_at_k"] == 1.0
    assert question_result["scores"]["precision_at_k"] == 0.5
    assert question_result["scores"]["mrr"] == 1.0
    assert question_result["scores"]["faithfulness"] == 1.0
    assert question_result["scores"]["relevance"] == 1.0
    assert results["aggregate"]["recall_at_k"] == 1.0


def test_load_chunks_jsonl(tmp_path) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    chunk_path.write_text(
        (
            '{"text":"Text","doc_id":"doc","source":"source.pdf","pages":[1],'
            '"chunk_strategy":"structural","chunk_id":"chunk-a"}\n'
        ),
        encoding="utf-8",
    )

    chunks = load_chunks_jsonl(chunk_path)

    assert chunks[0].chunk_id == "chunk-a"
    assert chunks[0].pages == [1]
