"""JSONL-backed cache helpers for retrieval result dataframes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


DEFAULT_RETRIEVAL_RUN_DIR = Path("data") / "evaluation" / "retrieval_runs"
DEFAULT_RETRIEVER_CACHE_DIR = DEFAULT_RETRIEVAL_RUN_DIR / "retriever_caches"

DEFAULT_SPARSE_RETRIEVAL_CACHE_PATH = (
    DEFAULT_RETRIEVER_CACHE_DIR / "qdrant_sparse_retriever_retrieval_cache.jsonl"
)
DEFAULT_DENSE_RETRIEVAL_CACHE_PATH = (
    DEFAULT_RETRIEVER_CACHE_DIR / "qdrant_dense_retriever_retrieval_cache.jsonl"
)
DEFAULT_HYBRID_RETRIEVAL_CACHE_PATH = (
    DEFAULT_RETRIEVER_CACHE_DIR / "qdrant_hybrid_retriever_retrieval_cache.jsonl"
)
DEFAULT_GRAPH_EXPANDED_RETRIEVAL_CACHE_PATH = (
    DEFAULT_RETRIEVER_CACHE_DIR / "graph_expanded_retriever_retrieval_cache.jsonl"
)
DEFAULT_GRAPH_RERANKED_RETRIEVAL_CACHE_PATH = (
    DEFAULT_RETRIEVER_CACHE_DIR / "graph_reranked_retriever_retrieval_cache.jsonl"
)


def default_hybrid_retrieval_cache_path(*, prefetch_multiplier: int) -> Path:
    """Match exported tuning cache names for hybrid retriever variants."""

    return (
        DEFAULT_RETRIEVER_CACHE_DIR
        / f"hybrid_tuning_retrievers__{int(prefetch_multiplier)}___retrieval_cache.jsonl"
    )


def default_graph_expanded_retrieval_cache_path(
    *,
    prefetch_multiplier: int,
    entity_limit: int,
    alias_limit: int,
    neighbor_limit: int,
) -> Path:
    """Match exported tuning cache names for graph-expanded variants."""

    return (
        DEFAULT_RETRIEVER_CACHE_DIR
        / (
            "graph_expanded_tuning_retrievers__"
            f"{int(prefetch_multiplier)}__{int(entity_limit)}__{int(alias_limit)}__"
            f"{int(neighbor_limit)}__retrieval_cache.jsonl"
        )
    )


def default_graph_reranked_retrieval_cache_path(
    *,
    candidate_multiplier: int,
    matched_entity_boost: float,
    neighbor_entity_boost: float,
    typed_relation_boost: float,
) -> Path:
    """Match exported tuning cache names for graph-reranked variants."""

    return (
        DEFAULT_RETRIEVER_CACHE_DIR
        / (
            "graph_reranked_tuning_retrievers__"
            f"{int(candidate_multiplier)}__{matched_entity_boost:g}__"
            f"{neighbor_entity_boost:g}__{typed_relation_boost:g}__"
            "retrieval_cache.jsonl"
        )
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        if pd.isna(value) and not isinstance(value, (list, tuple, dict, pd.DataFrame)):
            return None
    except Exception:
        pass
    return value


def _dataframe_to_records(results_df: pd.DataFrame) -> list[dict[str, Any]]:
    if results_df is None or results_df.empty:
        return []
    clean_df = results_df.copy()
    clean_df = clean_df.where(pd.notna(clean_df), None)
    return clean_df.to_dict(orient="records")


def _records_to_dataframe(records: list[Mapping[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([dict(record) for record in records])


def load_retrieval_cache_result(
    cache_path: str | Path | None,
    *,
    query_id: str,
    top_k: int,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Return a cached retrieval dataframe and debug payload for a query if available.

    The JSONL format is one record per query cache entry:

    ```json
    {"query_id": "...", "result_count": 20, "results": [...], "debug": {...}}
    ```

    If a file contains duplicate query ids from multiple runs, the largest usable
    entry is selected; ties prefer the latest appended entry.
    """

    if cache_path is None:
        return None, {}
    path = Path(cache_path)
    if not path.exists():
        return None, {}

    query_id = str(query_id)
    best_record: dict[str, Any] | None = None
    best_count = -1
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(record.get("query_id", "")) != query_id:
                continue
            results = record.get("results") or []
            result_count = int(record.get("result_count") or len(results))
            if result_count >= int(top_k) and result_count >= best_count:
                best_record = record
                best_count = result_count

    if not best_record:
        return None, {}
    cached_df = _records_to_dataframe(best_record.get("results") or [])
    if cached_df.empty or len(cached_df) < int(top_k):
        return None, {}
    return cached_df.head(int(top_k)).copy(), dict(best_record.get("debug") or {})


def append_retrieval_cache_result(
    cache_path: str | Path | None,
    *,
    cache_source: str,
    retriever_class: str,
    query_id: str,
    results_df: pd.DataFrame,
    debug: Mapping[str, Any] | None = None,
) -> None:
    """Append one query's retrieval results to a JSONL cache file."""

    if cache_path is None:
        return
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "cache_source": str(cache_source),
        "retriever_class": str(retriever_class),
        "query_id": str(query_id),
        "result_count": int(len(results_df)) if results_df is not None else 0,
        "results": _dataframe_to_records(results_df),
    }
    if debug:
        record["debug"] = dict(debug)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=_json_safe) + "\n")
