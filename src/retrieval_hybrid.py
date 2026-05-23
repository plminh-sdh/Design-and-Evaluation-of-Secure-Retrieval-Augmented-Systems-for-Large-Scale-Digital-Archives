"""Qdrant dense+sparse hybrid retrieval for the retrieval component notebook."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from qdrant_client import models

from src.bge_m3_embedding import BGEM3Embedder
from src.retrieval_dense import QDRANT_DENSE_VECTOR_NAME
from src.retrieval_drivers import QdrantRetrievalDriver
from src.retrieval_sparse import (
    QDRANT_SPARSE_VECTOR_NAME,
    qdrant_points_to_retrieval_results,
)


@dataclass
class QdrantHybridRrfRetriever:
    """Hybrid retriever using Qdrant's server-side dense+sparse RRF fusion."""

    qdrant_driver: QdrantRetrievalDriver
    embedder: BGEM3Embedder
    dense_vector_name: str = QDRANT_DENSE_VECTOR_NAME
    sparse_vector_name: str = QDRANT_SPARSE_VECTOR_NAME
    method_name: str = "qdrant_hybrid_rrf"
    embedding_batch_size: int = 1
    embedding_max_length: int = 1024
    prefetch_multiplier: int = 5
    min_prefetch_limit: int = 20
    rrf_k: int | None = None

    def encode_query_vectors(self, query: str) -> tuple[list[float], models.SparseVector]:
        embedding = self.embedder.encode_texts(
            [query],
            batch_size=self.embedding_batch_size,
            max_length=self.embedding_max_length,
        )[0]
        sparse_vector = models.SparseVector(
            indices=embedding.sparse_indices,
            values=embedding.sparse_values,
        )
        return embedding.dense, sparse_vector

    def retrieve(
        self,
        query: str,
        *,
        query_id: str = "sample_query",
        top_k: int = 10,
        query_filter: Any | None = None,
        with_payload: bool = True,
        prefetch_limit: int | None = None,
    ) -> pd.DataFrame:
        started_at = time.perf_counter()
        dense_query_vector, sparse_query_vector = self.encode_query_vectors(query)
        resolved_prefetch_limit = prefetch_limit or max(
            self.min_prefetch_limit,
            top_k * self.prefetch_multiplier,
        )
        points = self.qdrant_driver.search_hybrid_rrf(
            dense_query_vector=dense_query_vector,
            sparse_query_vector=sparse_query_vector,
            dense_vector_name=self.dense_vector_name,
            sparse_vector_name=self.sparse_vector_name,
            top_k=top_k,
            prefetch_limit=resolved_prefetch_limit,
            query_filter=query_filter,
            with_payload=with_payload,
            with_vectors=False,
            rrf_k=self.rrf_k,
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
        return qdrant_points_to_retrieval_results(
            points,
            query=query,
            query_id=query_id,
            retrieval_method=self.method_name,
            latency_ms=latency_ms,
            retrieval_stage="hybrid_rrf",
        )
