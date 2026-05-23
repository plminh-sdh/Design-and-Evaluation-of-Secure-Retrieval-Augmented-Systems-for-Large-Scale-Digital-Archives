"""Qdrant dense-vector retrieval for the retrieval component notebook."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.bge_m3_embedding import BGEM3Embedder
from src.retrieval_drivers import QdrantRetrievalDriver
from src.retrieval_sparse import qdrant_points_to_retrieval_results


QDRANT_DENSE_VECTOR_NAME = "dense"


@dataclass
class QdrantDenseRetriever:
    """Dense retriever using BGE-M3 query dense vectors and Qdrant search."""

    qdrant_driver: QdrantRetrievalDriver
    embedder: BGEM3Embedder
    vector_name: str = QDRANT_DENSE_VECTOR_NAME
    method_name: str = "qdrant_dense"
    embedding_batch_size: int = 1
    embedding_max_length: int = 1024
    retrieval_cache: dict[str, pd.DataFrame] = field(default_factory=dict)

    def _cache_enabled(
        self,
        *,
        query_filter: Any | None,
        with_payload: bool,
    ) -> bool:
        return query_filter is None and with_payload

    def _cached_results(self, query_id: str, top_k: int) -> pd.DataFrame | None:
        cached_df = self.retrieval_cache.get(str(query_id))
        if cached_df is None or len(cached_df) < top_k:
            return None
        return cached_df.head(top_k).copy()

    def _store_cache(self, query_id: str, results_df: pd.DataFrame) -> None:
        cached_df = self.retrieval_cache.get(str(query_id))
        if cached_df is None or len(results_df) > len(cached_df):
            self.retrieval_cache[str(query_id)] = results_df.copy()

    def encode_query_dense_vector(self, query: str) -> list[float]:
        embedding = self.embedder.encode_texts(
            [query],
            batch_size=self.embedding_batch_size,
            max_length=self.embedding_max_length,
        )[0]
        return embedding.dense

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
        query_dense_vector = self.encode_query_dense_vector(query)
        points = self.qdrant_driver.search_dense(
            query_dense_vector,
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
            retrieval_stage="dense",
        )
        if self._cache_enabled(query_filter=query_filter, with_payload=with_payload):
            self._store_cache(query_id, results_df)
        return results_df
