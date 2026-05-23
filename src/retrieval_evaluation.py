"""Reusable retrieval evaluation helpers for qrel-based notebook tuning."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from tqdm.auto import tqdm


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL records, returning an empty list when the file is absent."""

    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Append mapping records to a JSONL file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def extract_first_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from an LLM response without importing LLM code."""

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("The first JSON value in the LLM response is not an object.")
    return parsed


def load_retrieval_qrels(path: str | Path, *, valid_only: bool = True) -> pd.DataFrame:
    """Load generated retrieval qrels from JSONL."""

    qrels_df = pd.DataFrame(read_jsonl(path))
    if qrels_df.empty:
        return qrels_df
    if valid_only and "is_valid" in qrels_df.columns:
        qrels_df = qrels_df[qrels_df["is_valid"].fillna(True)].copy()
    return qrels_df.reset_index(drop=True)


def retrieve_for_qrels(
    qrels_df: pd.DataFrame,
    retriever: Any,
    *,
    top_k: int,
    query_id_column: str = "query_id",
    query_column: str = "query",
    max_retries: int = 3,
    retry_sleep_seconds: float = 3.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run a retriever for each qrel using only the query fields."""

    frames: list[pd.DataFrame] = []
    iterator = tqdm(
        qrels_df.to_dict(orient="records"),
        desc=f"Retrieving dev queries @ {top_k}",
        unit="query",
        disable=not verbose,
        dynamic_ncols=True,
    )
    for qrel in iterator:
        query_id = str(qrel.get(query_id_column, ""))
        query = str(qrel.get(query_column, ""))
        results_df = pd.DataFrame()
        last_error: Exception | None = None
        for attempt in range(1, max(max_retries, 1) + 1):
            try:
                results_df = retriever.retrieve(query, query_id=query_id, top_k=top_k)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < max(max_retries, 1):
                    if verbose:
                        tqdm.write(
                            "Retrying retrieval after error "
                            f"({attempt}/{max_retries}) for query_id={query_id}: {exc!r}"
                        )
                    time.sleep(retry_sleep_seconds * attempt)
        if last_error is not None:
            if verbose:
                tqdm.write(
                    f"Skipping query_id={query_id} after retrieval retries failed: {last_error!r}"
                )
            continue
        if not results_df.empty:
            frames.append(results_df)
        iterator.set_postfix(retrieved=sum(len(frame) for frame in frames))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _judgment_cache_key(
    *,
    query_id: str,
    chunk_id: str,
    judge_model_id: str,
) -> str:
    """Stable cache key for deterministic query/chunk judgment reuse."""

    raw = "|".join(
        [
            str(query_id),
            str(chunk_id),
            str(judge_model_id),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _legacy_judgment_cache_key(
    *,
    query_id: str,
    chunk_id: str,
    expected_relevant_information: str,
    judge_model_id: str,
) -> str:
    """Previous cache key shape kept so old judgment files remain reusable."""

    raw = "|".join(
        [
            str(query_id),
            str(chunk_id),
            str(expected_relevant_information),
            str(judge_model_id),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_judgment_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load cached judge records keyed by stable judgment key."""

    cache: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        key = str(record.get("judgment_key", ""))
        if key:
            cache.setdefault(key, record)

        query_id = str(record.get("query_id", ""))
        chunk_id = str(record.get("chunk_id", ""))
        judge_model_id = str(record.get("judge_model", ""))
        if query_id and chunk_id and judge_model_id:
            stable_key = _judgment_cache_key(
                query_id=query_id,
                chunk_id=chunk_id,
                judge_model_id=judge_model_id,
            )
            stable_record = dict(record)
            stable_record["judgment_key"] = stable_key
            stable_record["cache_key_version"] = "query_chunk_v1"
            cache.setdefault(stable_key, stable_record)
    return cache


def judge_retrieval_results(
    results_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    judge_llm: Any,
    *,
    output_jsonl: str | Path | None = None,
    resume: bool = True,
    max_new_tokens: int = 256,
    max_retries: int = 5,
    retry_sleep_seconds: float = 2.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Judge retrieved chunks with the relevance-judge LLM.

    The retriever only sees query/query_id. Qrel fields are joined here, after
    retrieval, strictly for evaluation.
    """

    if results_df.empty:
        return pd.DataFrame()
    if qrels_df.empty:
        raise ValueError("qrels_df is empty; cannot judge retrieval results.")

    qrel_lookup = qrels_df.set_index("query_id").to_dict(orient="index")
    judge_model_id = getattr(judge_llm, "model_id", judge_llm.__class__.__name__)
    cache = load_judgment_cache(output_jsonl) if output_jsonl and resume else {}
    judged_records: list[dict[str, Any]] = []
    new_records: list[dict[str, Any]] = []

    iterator = tqdm(
        results_df.sort_values(["query_id", "rank"]).to_dict(orient="records"),
        desc="Judging retrieved chunks",
        unit="result",
        disable=not verbose,
        dynamic_ncols=True,
    )
    for result in iterator:
        query_id = str(result.get("query_id", ""))
        chunk_id = str(result.get("chunk_id", ""))
        qrel = qrel_lookup.get(query_id, {})
        expected = str(qrel.get("expected_relevant_information", ""))
        judgment_key = _judgment_cache_key(
            query_id=query_id,
            chunk_id=chunk_id,
            judge_model_id=judge_model_id,
        )
        legacy_judgment_key = _legacy_judgment_cache_key(
            query_id=query_id,
            chunk_id=chunk_id,
            expected_relevant_information=expected,
            judge_model_id=judge_model_id,
        )

        cached_record = cache.get(judgment_key) or cache.get(legacy_judgment_key)
        if cached_record:
            cached_record = dict(cached_record)
            cached_record["judgment_key"] = judgment_key
            cached_record["cache_key_version"] = "query_chunk_v1"
            judged_records.append(cached_record)
            iterator.set_postfix(cached=len(judged_records), new=len(new_records))
            continue

        raw_response = ""
        last_error: Exception | None = None
        query_text = str(result.get("query", qrel.get("query", "")))
        retrieved_chunk_text = str(result.get("retrieval_text", ""))
        for attempt in range(1, max(max_retries, 1) + 1):
            try:
                raw_response = judge_llm.judge_relevance(
                    query=query_text,
                    expected_relevant_information=expected,
                    retrieved_chunk_text=retrieved_chunk_text,
                    max_new_tokens=max_new_tokens,
                )
                if str(raw_response).strip():
                    break
                last_error = ValueError("Judge returned an empty response.")
            except Exception as exc:
                last_error = exc

            if attempt < max(max_retries, 1):
                if verbose:
                    tqdm.write(
                        "Retrying judge call after empty/error response "
                        f"({attempt}/{max_retries}) for query_id={query_id}, "
                        f"chunk_id={chunk_id}: {last_error!r}"
                    )
                time.sleep(retry_sleep_seconds * attempt)

        try:
            parsed = extract_first_json_object(raw_response)
            relevance_score = int(parsed.get("relevance_score", 0))
        except Exception as exc:
            parsed = {
                "relevance_score": 0,
                "relevance_label": "judge_error",
                "contains_expected_information": False,
                "missing_information": "",
                "supporting_evidence": "",
                "rationale": (
                    "Judge response failed after retries. "
                    f"last_error={last_error!r}; parse_error={exc!r}"
                ),
            }
            relevance_score = 0

        relevance_score = max(0, min(3, relevance_score))
        record = {
            "judgment_key": judgment_key,
            "cache_key_version": "query_chunk_v1",
            "query_id": query_id,
            "query": result.get("query", qrel.get("query", "")),
            "source_chunk_id": qrel.get("source_chunk_id", ""),
            "chunk_id": chunk_id,
            "rank": int(result.get("rank", 0) or 0),
            "retrieval_method": result.get("retrieval_method", ""),
            "retrieval_score": float(result.get("score", 0.0) or 0.0),
            "expected_relevant_information": expected,
            "is_source_chunk": chunk_id == str(qrel.get("source_chunk_id", "")),
            "judge_model": judge_model_id,
            "judge_attempts": attempt,
            "judge_error": repr(last_error) if last_error and not str(raw_response).strip() else "",
            "raw_judge_response": raw_response,
            **parsed,
            "relevance_score": relevance_score,
        }
        judged_records.append(record)
        new_records.append(record)
        if output_jsonl:
            append_jsonl([record], output_jsonl)
        iterator.set_postfix(cached=len(judged_records) - len(new_records), new=len(new_records))

    return pd.DataFrame(judged_records)


def dcg_at_k(relevance_scores: Sequence[int | float], k: int) -> float:
    """Compute graded DCG@k with exponential gain."""

    dcg = 0.0
    for index, score in enumerate(list(relevance_scores)[:k], start=1):
        gain = math.pow(2.0, float(score)) - 1.0
        dcg += gain / math.log2(index + 1)
    return dcg


def ndcg_at_k(relevance_scores: Sequence[int | float], k: int) -> float:
    """Compute judged-list nDCG@k for one ranked list."""

    scores = [float(score) for score in list(relevance_scores)[:k]]
    if not scores:
        return 0.0
    dcg = dcg_at_k(scores, k)
    ideal_dcg = dcg_at_k(sorted(scores, reverse=True), k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def summarize_retrieval_judgments(
    judgments_df: pd.DataFrame,
    *,
    k_values: Sequence[int],
    relevance_threshold: int = 2,
) -> pd.DataFrame:
    """Compute aggregate graded metrics from judged retrieval results."""

    if judgments_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    grouped = judgments_df.sort_values(["query_id", "rank"]).groupby("query_id")
    query_count = grouped.ngroups
    for k in k_values:
        per_query_ndcg: list[float] = []
        per_query_precision: list[float] = []
        per_query_hit: list[float] = []
        per_query_mrr: list[float] = []
        per_query_source_hit: list[float] = []
        judged_results = 0

        for _, group in grouped:
            top_group = group.sort_values("rank").head(k)
            scores = top_group["relevance_score"].fillna(0).astype(float).tolist()
            judged_results += len(top_group)
            per_query_ndcg.append(ndcg_at_k(scores, k))
            relevant_mask = top_group["relevance_score"].fillna(0).astype(float) >= relevance_threshold
            per_query_precision.append(float(relevant_mask.sum()) / max(k, 1))
            per_query_hit.append(float(relevant_mask.any()))
            per_query_source_hit.append(float(top_group["is_source_chunk"].fillna(False).astype(bool).any()))

            first_relevant_rank = None
            for _, row in top_group.iterrows():
                if float(row.get("relevance_score", 0) or 0) >= relevance_threshold:
                    first_relevant_rank = int(row.get("rank", 0) or 0)
                    break
            per_query_mrr.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)

        rows.append(
            {
                "k": int(k),
                "queries": int(query_count),
                "judged_results": int(judged_results),
                "mean_ndcg": sum(per_query_ndcg) / max(len(per_query_ndcg), 1),
                "precision_at_k": sum(per_query_precision) / max(len(per_query_precision), 1),
                "hit_rate_at_k": sum(per_query_hit) / max(len(per_query_hit), 1),
                "mrr_at_k": sum(per_query_mrr) / max(len(per_query_mrr), 1),
                "source_chunk_hit_at_k": sum(per_query_source_hit)
                / max(len(per_query_source_hit), 1),
                "relevance_threshold": relevance_threshold,
            }
        )

    return pd.DataFrame(rows)


def tune_retriever_top_k(
    *,
    strategy_name: str,
    qrels_df: pd.DataFrame,
    retriever: Any,
    judge_llm: Any,
    top_k_candidates: Sequence[int],
    judgments_jsonl: str | Path,
    relevance_threshold: int = 2,
    max_new_tokens: int = 1024,
    retrieval_max_retries: int = 3,
    retrieval_retry_sleep_seconds: float = 3.0,
    max_retries: int = 5,
    retry_sleep_seconds: float = 2.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Retrieve once at max-k, judge once, and summarize candidate top-k values."""

    if not top_k_candidates:
        raise ValueError("top_k_candidates must not be empty.")

    max_k = max(int(k) for k in top_k_candidates)
    retrieval_results_df = retrieve_for_qrels(
        qrels_df,
        retriever,
        top_k=max_k,
        max_retries=retrieval_max_retries,
        retry_sleep_seconds=retrieval_retry_sleep_seconds,
        verbose=verbose,
    )
    if not retrieval_results_df.empty:
        retrieval_results_df = retrieval_results_df.copy()
        retrieval_results_df["evaluation_strategy"] = strategy_name

    judgments_df = judge_retrieval_results(
        retrieval_results_df,
        qrels_df,
        judge_llm,
        output_jsonl=judgments_jsonl,
        resume=True,
        max_new_tokens=max_new_tokens,
        max_retries=max_retries,
        retry_sleep_seconds=retry_sleep_seconds,
        verbose=verbose,
    )
    metrics_df = summarize_retrieval_judgments(
        judgments_df,
        k_values=top_k_candidates,
        relevance_threshold=relevance_threshold,
    )
    if metrics_df.empty:
        raise RuntimeError(f"No tuning metrics were produced for {strategy_name}.")

    metrics_df = metrics_df.copy()
    metrics_df.insert(0, "strategy", strategy_name)
    best_row = metrics_df.sort_values(
        ["mean_ndcg", "hit_rate_at_k", "mrr_at_k"],
        ascending=False,
    ).iloc[0]
    return retrieval_results_df, judgments_df, metrics_df, best_row


def evaluate_retrieval_strategies(
    *,
    strategy_retrievers: Mapping[str, tuple[Any, int]],
    qrels_df: pd.DataFrame,
    judge_llm: Any,
    judgments_dir: str | Path,
    relevance_threshold: int = 2,
    max_new_tokens: int = 1024,
    retrieval_max_retries: int = 3,
    retrieval_retry_sleep_seconds: float = 3.0,
    max_retries: int = 5,
    retry_sleep_seconds: float = 2.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Evaluate named retrievers on a qrel set using their selected top-k values."""

    judgments_dir = Path(judgments_dir)
    metrics_frames: list[pd.DataFrame] = []
    retrieval_results_by_strategy: dict[str, pd.DataFrame] = {}
    judgments_by_strategy: dict[str, pd.DataFrame] = {}

    iterator = tqdm(
        list(strategy_retrievers.items()),
        desc="Evaluating retrieval strategies",
        unit="strategy",
        disable=not verbose,
        dynamic_ncols=True,
    )
    for strategy_name, (retriever, top_k) in iterator:
        iterator.set_postfix(strategy=strategy_name, top_k=top_k)
        retrieval_results_df, judgments_df, metrics_df, _ = tune_retriever_top_k(
            strategy_name=strategy_name,
            qrels_df=qrels_df,
            retriever=retriever,
            judge_llm=judge_llm,
            top_k_candidates=[int(top_k)],
            judgments_jsonl=judgments_dir / f"{strategy_name}_test_judgments.jsonl",
            relevance_threshold=relevance_threshold,
            max_new_tokens=max_new_tokens,
            retrieval_max_retries=retrieval_max_retries,
            retrieval_retry_sleep_seconds=retrieval_retry_sleep_seconds,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
            verbose=verbose,
        )
        retrieval_results_by_strategy[strategy_name] = retrieval_results_df
        judgments_by_strategy[strategy_name] = judgments_df
        metrics_frames.append(metrics_df)

    if not metrics_frames:
        return pd.DataFrame(), retrieval_results_by_strategy, judgments_by_strategy
    return pd.concat(metrics_frames, ignore_index=True), retrieval_results_by_strategy, judgments_by_strategy
