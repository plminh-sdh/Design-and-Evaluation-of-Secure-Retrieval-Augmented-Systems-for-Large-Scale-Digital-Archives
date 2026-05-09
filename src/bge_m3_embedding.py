"""BGE-M3 embedding utilities for dense and sparse Qdrant indexing."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from qdrant_client.http import models

from src.archive_schema import ArchiveChunk, archive_chunk_to_qdrant_payload


BGE_M3_MODEL_NAME = "BAAI/bge-m3"
BGE_M3_DENSE_VECTOR_SIZE = 1024


@dataclass
class EmbeddedText:
    dense: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]

    def to_qdrant_vectors(self) -> Dict[str, Any]:
        return {
            "dense": self.dense,
            "sparse": models.SparseVector(
                indices=self.sparse_indices,
                values=self.sparse_values,
            ),
        }


class BGEM3Embedder:
    """Wrapper around FlagEmbedding's BGE-M3 dense+sparse encoder."""

    def __init__(
        self,
        model_name: str = BGE_M3_MODEL_NAME,
        use_fp16: bool = True,
        device: Optional[str] = None,
    ) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is required for BGE-M3 sparse+dense embeddings. "
                "Install it with: pip install -U FlagEmbedding"
            ) from exc

        model_kwargs: Dict[str, Any] = {"use_fp16": use_fp16}
        if device:
            model_kwargs["device"] = device

        self.model_name = model_name
        self.model = BGEM3FlagModel(model_name, **model_kwargs)

    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
        max_length: int = 8192,
    ) -> List[EmbeddedText]:
        if not texts:
            return []

        output = self.model.encode(
            list(texts),
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vectors = output["dense_vecs"]
        lexical_weights = output["lexical_weights"]

        return [
            EmbeddedText(
                dense=self._dense_to_list(dense_vector),
                sparse_indices=sparse["indices"],
                sparse_values=sparse["values"],
            )
            for dense_vector, sparse in zip(
                dense_vectors,
                (self._sparse_to_qdrant(weight_map) for weight_map in lexical_weights),
            )
        ]

    def encode_chunks(
        self,
        chunks: Sequence[ArchiveChunk],
        batch_size: int = 8,
        max_length: int = 8192,
    ) -> List[EmbeddedText]:
        return self.encode_texts(
            [chunk.embedding_text for chunk in chunks],
            batch_size=batch_size,
            max_length=max_length,
        )

    @staticmethod
    def _dense_to_list(vector: Any) -> List[float]:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return [float(value) for value in vector]

    @staticmethod
    def _sparse_to_qdrant(lexical_weights: Any) -> Dict[str, List[Any]]:
        if hasattr(lexical_weights, "items"):
            items = lexical_weights.items()
        else:
            items = lexical_weights

        sparse_items = []
        for index, value in items:
            index = int(index)
            value = float(value)
            if value == 0.0:
                continue
            sparse_items.append((index, value))

        sparse_items.sort(key=lambda item: item[0])
        return {
            "indices": [index for index, _value in sparse_items],
            "values": [value for _index, value in sparse_items],
        }


def qdrant_point_id(chunk_id: str) -> str:
    """Create a stable Qdrant-compatible UUID point ID from an archive chunk ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"archive-chunk:{chunk_id}"))


def qdrant_point_ids(chunks: Sequence[ArchiveChunk]) -> List[str]:
    """Create stable Qdrant-compatible UUID point IDs for archive chunks."""
    return [qdrant_point_id(chunk.chunk_id) for chunk in chunks]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "item"):
        return _json_safe(value.item())

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    return str(value)


def build_qdrant_points(
    chunks: Sequence[ArchiveChunk],
    embeddings: Sequence[EmbeddedText],
) -> List[models.PointStruct]:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    points: List[models.PointStruct] = []
    for chunk, embedding in zip(chunks, embeddings):
        payload = _json_safe(archive_chunk_to_qdrant_payload(chunk))
        payload["embedding_model"] = BGE_M3_MODEL_NAME
        payload["dense_vector_ready"] = True
        payload["sparse_vector_ready"] = True

        points.append(
            models.PointStruct(
                id=qdrant_point_id(chunk.chunk_id),
                vector=embedding.to_qdrant_vectors(),
                payload=payload,
            )
        )

    return points


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
