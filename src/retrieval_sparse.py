"""Qdrant sparse-vector retrieval for the retrieval component notebook."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from qdrant_client import models

from src.bge_m3_embedding import BGEM3Embedder
from src.retrieval_cache import (
    DEFAULT_SPARSE_RETRIEVAL_CACHE_PATH,
    append_retrieval_cache_result,
    load_retrieval_cache_result,
)
from src.retrieval_drivers import QdrantRetrievalDriver


QDRANT_SPARSE_VECTOR_NAME = "sparse"


def qdrant_points_to_retrieval_results(
    points: list[Any],
    *,
    query: str,
    query_id: str,
    retrieval_method: str,
    latency_ms: float,
    retrieval_stage: str,
) -> pd.DataFrame:
    """Convert Qdrant scored points into the common retrieval result schema."""

    rows = []
    for rank, point in enumerate(points, start=1):
        payload = dict(getattr(point, "payload", None) or {})
        retrieval_text = str(
            payload.get("embedding_text")
            or payload.get("masked_text")
            or payload.get("summary")
            or ""
        )
        rows.append(
            {
                "query_id": query_id,
                "query": query,
                "retrieval_method": retrieval_method,
                "chunk_id": payload.get("chunk_id") or str(getattr(point, "id", "")),
                "rank": rank,
                "score": float(getattr(point, "score", 0.0) or 0.0),
                "document_id": payload.get("document_id", ""),
                "dataset": payload.get("dataset", ""),
                "modality": payload.get("modality", ""),
                "title": payload.get("title", ""),
                "latency_ms": latency_ms,
                "retrieval_stage": retrieval_stage,
                "retrieval_text": retrieval_text,
                "retrieval_text_preview": retrieval_text[:500],
            }
        )
    return pd.DataFrame(rows)


@dataclass
class QdrantSparseRetriever:
    """Sparse retriever using BGE-M3 query sparse vectors and Qdrant search."""

    qdrant_driver: QdrantRetrievalDriver
    embedder: BGEM3Embedder
    vector_name: str = QDRANT_SPARSE_VECTOR_NAME
    method_name: str = "qdrant_sparse"
    embedding_batch_size: int = 1
    embedding_max_length: int = 1024
    retrieval_cache: dict[str, pd.DataFrame] = field(default_factory=dict)
    retrieval_cache_path: str | Path | None = DEFAULT_SPARSE_RETRIEVAL_CACHE_PATH

    def _cache_enabled(
        self,
        *,
        query_filter: Any | None,
        with_payload: bool,
    ) -> bool:
        return query_filter is None and with_payload

    def _cached_results(self, query_id: str, top_k: int) -> pd.DataFrame | None:
        if self.retrieval_cache_path is not None:
            cached_df, _ = load_retrieval_cache_result(
                self.retrieval_cache_path,
                query_id=str(query_id),
                top_k=top_k,
            )
            return cached_df
        cached_df = self.retrieval_cache.get(str(query_id))
        if cached_df is None or len(cached_df) < top_k:
            return None
        return cached_df.head(top_k).copy()

    def _store_cache(self, query_id: str, results_df: pd.DataFrame) -> None:
        if self.retrieval_cache_path is not None:
            append_retrieval_cache_result(
                self.retrieval_cache_path,
                cache_source=self.method_name,
                retriever_class=type(self).__name__,
                query_id=str(query_id),
                results_df=results_df,
            )
            return
        cached_df = self.retrieval_cache.get(str(query_id))
        if cached_df is None or len(results_df) > len(cached_df):
            self.retrieval_cache[str(query_id)] = results_df.copy()

    def encode_query_sparse_vector(self, query: str) -> models.SparseVector:
        embedding = self.embedder.encode_texts(
            [query],
            batch_size=self.embedding_batch_size,
            max_length=self.embedding_max_length,
        )[0]
        return models.SparseVector(
            indices=embedding.sparse_indices,
            values=embedding.sparse_values,
        )

    def retrieve(
        self,
        query: str,
        *,
        query_id: str = "sample_query",
        top_k: int = 10,
        query_filter: Any | None = None,
        with_payload: bool = True,
    ) -> pd.DataFrame:
        if self._cache_enabled(query_filter=query_filter, with_payload=with_payload):
            cached_df = self._cached_results(query_id, top_k)
            if cached_df is not None:
                return cached_df

        started_at = time.perf_counter()
        query_sparse_vector = self.encode_query_sparse_vector(query)
        points = self.qdrant_driver.search_sparse(
            query_sparse_vector,
            vector_name=self.vector_name,
            top_k=top_k,
            query_filter=query_filter,
            with_payload=with_payload,
            with_vectors=False,
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        results_df = qdrant_points_to_retrieval_results(
            points,
            query=query,
            query_id=query_id,
            retrieval_method=self.method_name,
            latency_ms=latency_ms,
            retrieval_stage="sparse",
        )
        if self._cache_enabled(query_filter=query_filter, with_payload=with_payload):
            self._store_cache(query_id, results_df)
        return results_df
