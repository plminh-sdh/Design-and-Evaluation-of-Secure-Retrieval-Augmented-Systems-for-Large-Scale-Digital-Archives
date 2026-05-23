"""Qdrant dense-vector retrieval for the retrieval component notebook."""

from __future__ import annotations

import time
from dataclasses import dataclass
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
        return qdrant_points_to_retrieval_results(
            points,
            query=query,
            query_id=query_id,
            retrieval_method=self.method_name,
            latency_ms=latency_ms,
            retrieval_stage="dense",
        )
