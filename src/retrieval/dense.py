from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from chunking.models import Chunk


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever:
    def __init__(
        self,
        *,
        collection_name: str = "rag_eval_chunks",
        model_name: str = DEFAULT_MODEL_NAME,
        qdrant_path: str = ":memory:",
        embedding_model: Any | None = None,
        client: Any | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.model_name = model_name
        self._embedding_model = embedding_model
        self.client = client or _local_qdrant_client(qdrant_path)
        self._vector_size: int | None = None

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        vectors = self._embed([chunk.text for chunk in chunks])
        self._vector_size = len(vectors[0])
        self._ensure_collection(self._vector_size)

        points = [
            _point_struct(
                point_id=_point_id(chunk),
                vector=vector,
                payload=_chunk_to_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def retrieve(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        if k <= 0:
            return []

        query_vector = self._embed([query])[0]
        hits = self._search(query_vector, k)
        return [
            (_chunk_from_payload(hit.payload), float(hit.score))
            for hit in hits
            if getattr(hit, "payload", None)
        ]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.model_name)

        embeddings = self._embedding_model.encode(texts)
        return [_as_float_list(vector) for vector in embeddings]

    def _ensure_collection(self, vector_size: int) -> None:
        if hasattr(self.client, "collection_exists") and self.client.collection_exists(
            self.collection_name
        ):
            return

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=_vector_config(vector_size),
        )

    def _search(self, query_vector: list[float], k: int) -> Sequence[Any]:
        if hasattr(self.client, "query_points"):
            result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=k,
                with_payload=True,
            )
            return result.points

        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
        )


def _local_qdrant_client(qdrant_path: str) -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError(
            "DenseRetriever requires qdrant-client. Install project dependencies "
            "with `uv sync --dev` or `pip install qdrant-client`."
        ) from exc

    return QdrantClient(path=qdrant_path)


def _point_struct(*, point_id: str, vector: list[float], payload: dict[str, Any]) -> Any:
    try:
        from qdrant_client.models import PointStruct
    except ImportError:
        return _FallbackPoint(id=point_id, vector=vector, payload=payload)

    return PointStruct(id=point_id, vector=vector, payload=payload)


def _vector_config(vector_size: int) -> Any:
    try:
        from qdrant_client.models import Distance, VectorParams
    except ImportError:
        return {"size": vector_size, "distance": "Cosine"}

    return VectorParams(size=vector_size, distance=Distance.COSINE)


@dataclass(frozen=True)
class _FallbackPoint:
    id: str
    vector: list[float]
    payload: dict[str, Any]


def _point_id(chunk: Chunk) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))


def _chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "text": chunk.text,
        "doc_id": chunk.doc_id,
        "source": chunk.source,
        "pages": chunk.pages,
        "chunk_strategy": chunk.chunk_strategy,
        "chunk_id": chunk.chunk_id,
    }


def _chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    return Chunk(
        text=str(payload["text"]),
        doc_id=str(payload["doc_id"]),
        source=str(payload["source"]),
        pages=[int(page) for page in payload["pages"]],
        chunk_strategy=str(payload["chunk_strategy"]),
        chunk_id=str(payload["chunk_id"]),
    )


def _as_float_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]
