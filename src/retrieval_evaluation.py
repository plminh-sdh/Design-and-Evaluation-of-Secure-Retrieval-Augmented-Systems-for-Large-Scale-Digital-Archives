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

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("No valid JSON object found in LLM response.")


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


def _compact_retrieval_results(results_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Keep only fields required for later relevance judging."""

    if results_df is None or results_df.empty:
        return []

    compact_records: list[dict[str, Any]] = []
    for row in results_df.sort_values("rank").to_dict(orient="records"):
        score = row.get("score")
        compact_records.append(
            {
                "chunk_id": str(row.get("chunk_id", "")),
                "rank": int(row.get("rank", 0) or 0),
                "retrieval_method": str(row.get("retrieval_method", "")),
                "score": float(score) if pd.notna(score) else None,
                "retrieval_text": str(row.get("retrieval_text", "")),
            }
        )
    return compact_records


def load_retrieval_ablation_cache(
    path: str | Path,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Load successful per-query ablation records for resumable retrieval."""

    cache: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in read_jsonl(path):
        if str(record.get("status", "")) != "completed":
            continue
        key = (
            str(record.get("strategy", "")),
            str(record.get("query_id", "")),
            int(record.get("top_k", 0) or 0),
        )
        if key[0] and key[1] and key[2] > 0:
            cache[key] = record
    return cache


def retrieval_ablation_records_to_dataframe(
    records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Flatten per-query ablation records for later judge or audit steps."""

    rows: list[dict[str, Any]] = []
    for record in records:
        if str(record.get("status", "")) != "completed":
            continue
        for result in record.get("results") or []:
            rows.append(
                {
                    "evaluation_strategy": record.get("strategy", ""),
                    "top_k": int(record.get("top_k", 0) or 0),
                    "query_id": record.get("query_id", ""),
                    "query": record.get("query", ""),
                    "source_chunk_id": record.get("source_chunk_id", ""),
                    "expected_relevant_information": record.get(
                        "expected_relevant_information", ""
                    ),
                    "retrieval_elapsed_seconds": record.get("elapsed_seconds"),
                    **dict(result),
                }
            )
    return pd.DataFrame(rows)


def run_retrieval_strategy_ablation(
    *,
    strategy_name: str,
    retriever: Any,
    qrels_df: pd.DataFrame,
    top_k: int,
    output_jsonl: str | Path,
    max_queries: int | None = None,
    resume: bool = True,
    max_retries: int = 3,
    retry_sleep_seconds: float = 3.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run and cache retrieval only for one strategy, without invoking an LLM.

    One JSONL record is appended per query so interrupted runs can resume. The
    artifact contains only the qrel fields and ranked chunk fields needed by a
    later relevance-judging stage, plus retrieval elapsed time.
    """

    if qrels_df.empty:
        raise ValueError("qrels_df is empty; cannot run retrieval ablation.")
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive.")
    if max_queries is not None and int(max_queries) <= 0:
        raise ValueError("max_queries must be positive or None.")

    output_path = Path(output_jsonl)
    selected_qrels_df = (
        qrels_df.copy()
        if max_queries is None
        else qrels_df.head(int(max_queries)).copy()
    )
    cache = load_retrieval_ablation_cache(output_path) if resume else {}
    run_records: list[dict[str, Any]] = []
    cached_queries = 0
    completed_queries = 0
    failed_queries = 0

    iterator = tqdm(
        selected_qrels_df.to_dict(orient="records"),
        desc=f"Retrieval ablation: {strategy_name} @ {int(top_k)}",
        unit="query",
        disable=not verbose,
        dynamic_ncols=True,
    )
    for qrel in iterator:
        query_id = str(qrel.get("query_id", ""))
        query = str(qrel.get("query", ""))
        cache_key = (str(strategy_name), query_id, int(top_k))
        cached_record = cache.get(cache_key)
        if cached_record is not None:
            run_records.append(dict(cached_record))
            cached_queries += 1
            iterator.set_postfix(
                cached=cached_queries,
                completed=completed_queries,
                failed=failed_queries,
            )
            continue

        started_at = time.perf_counter()
        results_df = pd.DataFrame()
        last_error: Exception | None = None
        attempts = 0
        for attempts in range(1, max(int(max_retries), 1) + 1):
            try:
                results_df = retriever.retrieve(
                    query,
                    query_id=query_id,
                    top_k=int(top_k),
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempts < max(int(max_retries), 1):
                    if verbose:
                        tqdm.write(
                            "Retrying retrieval after error "
                            f"({attempts}/{max_retries}) for strategy={strategy_name}, "
                            f"query_id={query_id}: {exc!r}"
                        )
                    time.sleep(float(retry_sleep_seconds) * attempts)

        elapsed_seconds = time.perf_counter() - started_at
        compact_results = _compact_retrieval_results(results_df)
        status = "completed" if last_error is None else "failed"
        record = {
            "artifact_type": "retrieval_ablation_result",
            "artifact_version": "v1",
            "strategy": str(strategy_name),
            "retriever_class": retriever.__class__.__name__,
            "top_k": int(top_k),
            "query_id": query_id,
            "query": query,
            "source_chunk_id": str(qrel.get("source_chunk_id", "")),
            "expected_relevant_information": str(
                qrel.get("expected_relevant_information", "")
            ),
            "status": status,
            "attempts": int(attempts),
            "elapsed_seconds": float(elapsed_seconds),
            "result_count": len(compact_results),
            "error": repr(last_error) if last_error is not None else "",
            "results": compact_results,
        }
        append_jsonl([record], output_path)
        run_records.append(record)
        if status == "completed":
            completed_queries += 1
            cache[cache_key] = record
        else:
            failed_queries += 1
        iterator.set_postfix(
            cached=cached_queries,
            completed=completed_queries,
            failed=failed_queries,
        )

    records_df = pd.DataFrame(run_records)
    completed_df = records_df[records_df.get("status") == "completed"]
    summary_df = pd.DataFrame(
        [
            {
                "strategy": str(strategy_name),
                "top_k": int(top_k),
                "requested_queries": len(selected_qrels_df),
                "completed_queries": int(len(completed_df)),
                "cached_queries": int(cached_queries),
                "new_completed_queries": int(completed_queries),
                "failed_queries": int(failed_queries),
                "retrieved_results": int(
                    completed_df.get("result_count", pd.Series(dtype=int)).sum()
                ),
                "total_elapsed_seconds": float(
                    completed_df.get(
                        "elapsed_seconds", pd.Series(dtype=float)
                    ).sum()
                ),
                "mean_elapsed_seconds": float(
                    completed_df.get(
                        "elapsed_seconds", pd.Series(dtype=float)
                    ).mean()
                )
                if not completed_df.empty
                else 0.0,
                "output_jsonl": str(output_path),
            }
        ]
    )
    return summary_df, records_df


def run_retrieval_ablation_strategies(
    *,
    strategy_retrievers: Mapping[str, tuple[Any, int]],
    qrels_df: pd.DataFrame,
    output_dir: str | Path,
    max_queries: int | None = None,
    selected_strategies: Sequence[str] | None = None,
    resume: bool = True,
    max_retries: int = 3,
    retry_sleep_seconds: float = 3.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run retrieval-only ablation exports for selected or all strategies."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected_names = (
        list(strategy_retrievers)
        if selected_strategies is None
        else [str(name) for name in selected_strategies]
    )
    unknown_names = sorted(set(selected_names) - set(strategy_retrievers))
    if unknown_names:
        raise KeyError(f"Unknown retrieval strategies: {unknown_names}")

    summaries: list[pd.DataFrame] = []
    records_by_strategy: dict[str, pd.DataFrame] = {}
    for strategy_name in tqdm(
        selected_names,
        desc="Exporting retrieval ablation runs",
        unit="strategy",
        disable=not verbose,
        dynamic_ncols=True,
    ):
        retriever, top_k = strategy_retrievers[strategy_name]
        summary_df, records_df = run_retrieval_strategy_ablation(
            strategy_name=strategy_name,
            retriever=retriever,
            qrels_df=qrels_df,
            top_k=int(top_k),
            output_jsonl=output_path / f"{strategy_name}_retrieval_results.jsonl",
            max_queries=max_queries,
            resume=resume,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
            verbose=verbose,
        )
        summaries.append(summary_df)
        records_by_strategy[strategy_name] = records_df

    if not summaries:
        return pd.DataFrame(), records_by_strategy
    return pd.concat(summaries, ignore_index=True), records_by_strategy


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


def load_judgment_cache(
    path: str | Path,
    *,
    include_failed: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load cached judge records keyed by stable judgment key."""

    cache: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        if (
            str(record.get("judgment_status", "")) == "failed"
            and not include_failed
        ):
            continue
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
    progress_description: str | None = None,
    reuse_failed_cache: bool = False,
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
    cache = (
        load_judgment_cache(output_jsonl, include_failed=reuse_failed_cache)
        if output_jsonl and resume
        else {}
    )
    judged_records: list[dict[str, Any]] = []
    new_records: list[dict[str, Any]] = []
    failed_records = 0

    iterator = tqdm(
        results_df.sort_values(["query_id", "rank"]).to_dict(orient="records"),
        desc=progress_description or f"Judging chunks: {judge_model_id}",
        unit="result",
        disable=not verbose,
        dynamic_ncols=True,
        leave=False,
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
            iterator.set_postfix(
                cached=len(judged_records) - len(new_records),
                new=len(new_records),
                failed=failed_records,
            )
            continue

        raw_response = ""
        parsed: dict[str, Any] | None = None
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
                if not str(raw_response).strip():
                    raise ValueError("Judge returned an empty response.")
                parsed = extract_first_json_object(raw_response)
                relevance_score = int(parsed["relevance_score"])
                if not 0 <= relevance_score <= 3:
                    raise ValueError(
                        f"relevance_score must be between 0 and 3, got {relevance_score}."
                    )
                break
            except Exception as exc:
                last_error = exc
                parsed = None

            if attempt < max(max_retries, 1):
                if verbose:
                    tqdm.write(
                        "Retrying judge call after generation/JSON error "
                        f"({attempt}/{max_retries}) for query_id={query_id}, "
                        f"chunk_id={chunk_id}: {last_error!r}"
                    )
                time.sleep(retry_sleep_seconds * attempt)

        if parsed is not None:
            judgment_status = "completed"
        else:
            parsed = {
                "relevance_score": None,
                "relevance_label": "judge_error",
                "contains_expected_information": False,
                "missing_information": "",
                "supporting_evidence": "",
                "rationale": (
                    "Judge response failed after retries. "
                    f"last_error={last_error!r}"
                ),
            }
            relevance_score = None
            judgment_status = "failed"

        if relevance_score is not None:
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
            "evaluation_strategy": result.get("evaluation_strategy", ""),
            "judgment_status": judgment_status,
            "judge_attempts": attempt,
            "judge_error": repr(last_error) if last_error and not str(raw_response).strip() else "",
            "raw_judge_response": raw_response,
            **parsed,
            "relevance_score": relevance_score,
        }
        judged_records.append(record)
        new_records.append(record)
        failed_records += int(judgment_status == "failed")
        if output_jsonl:
            append_jsonl([record], output_jsonl)
        iterator.set_postfix(
            cached=len(judged_records) - len(new_records),
            new=len(new_records),
            failed=failed_records,
        )

    return pd.DataFrame(judged_records)


def _model_file_slug(model_id: str) -> str:
    """Convert a model identifier into a stable filesystem-safe slug."""

    slug = re.sub(r"[^a-zA-Z0-9._-]+", "__", str(model_id)).strip("._-")
    return slug or "judge"


def load_cached_retrieval_ablation(
    retrieval_results_dir: str | Path,
    *,
    selected_strategies: Sequence[str] | None = None,
    max_queries_per_strategy: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load frozen per-strategy retrieval artifacts into flat judge inputs."""

    if max_queries_per_strategy is not None and int(max_queries_per_strategy) <= 0:
        raise ValueError("max_queries_per_strategy must be positive or None.")

    retrieval_results_dir = Path(retrieval_results_dir)
    selected = set(selected_strategies or [])
    results_by_strategy: dict[str, pd.DataFrame] = {}
    for path in sorted(retrieval_results_dir.glob("*_retrieval_results.jsonl")):
        strategy = path.name.removesuffix("_retrieval_results.jsonl")
        if selected and strategy not in selected:
            continue
        flat_df = retrieval_ablation_records_to_dataframe(read_jsonl(path))
        if not flat_df.empty and max_queries_per_strategy is not None:
            selected_query_ids = (
                flat_df["query_id"]
                .drop_duplicates()
                .head(int(max_queries_per_strategy))
                .tolist()
            )
            flat_df = flat_df[flat_df["query_id"].isin(selected_query_ids)].copy()
        if not flat_df.empty:
            results_by_strategy[strategy] = flat_df
    return results_by_strategy


def count_pending_retrieval_judgments(
    *,
    judge_model_id: str,
    retrieval_results_dir: str | Path,
    judgments_root_dir: str | Path,
    selected_strategies: Sequence[str] | None = None,
    max_queries_per_strategy: int | None = None,
    resume: bool = True,
    reuse_failed_cache: bool = False,
) -> tuple[int, int]:
    """Count pending and cached rows without loading the judge model."""

    retrieval_by_strategy = load_cached_retrieval_ablation(
        retrieval_results_dir,
        selected_strategies=selected_strategies,
        max_queries_per_strategy=max_queries_per_strategy,
    )
    judge_dir = Path(judgments_root_dir) / _model_file_slug(judge_model_id)
    pending = 0
    cached = 0
    for strategy, results_df in retrieval_by_strategy.items():
        cache = (
            load_judgment_cache(
                judge_dir / f"{strategy}_judgments.jsonl",
                include_failed=reuse_failed_cache,
            )
            if resume
            else {}
        )
        for result in results_df.to_dict(orient="records"):
            key = _judgment_cache_key(
                query_id=str(result.get("query_id", "")),
                chunk_id=str(result.get("chunk_id", "")),
                judge_model_id=judge_model_id,
            )
            if key in cache:
                cached += 1
            else:
                pending += 1
    return pending, cached


def judge_cached_retrieval_ablation(
    *,
    judge_llm: Any,
    qrels_df: pd.DataFrame,
    retrieval_results_dir: str | Path,
    judgments_root_dir: str | Path,
    selected_strategies: Sequence[str] | None = None,
    max_queries_per_strategy: int | None = None,
    resume: bool = True,
    reuse_failed_cache: bool = False,
    max_new_tokens: int = 256,
    max_retries: int = 3,
    retry_sleep_seconds: float = 2.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Judge frozen retrieval outputs and append each judgment immediately."""

    judge_model_id = str(
        getattr(judge_llm, "model_id", judge_llm.__class__.__name__)
    )
    judge_dir = Path(judgments_root_dir) / _model_file_slug(judge_model_id)
    retrieval_by_strategy = load_cached_retrieval_ablation(
        retrieval_results_dir,
        selected_strategies=selected_strategies,
        max_queries_per_strategy=max_queries_per_strategy,
    )
    if not retrieval_by_strategy:
        raise FileNotFoundError(
            f"No completed retrieval ablation artifacts found in {retrieval_results_dir}"
        )

    summaries: list[dict[str, Any]] = []
    judgments_by_strategy: dict[str, pd.DataFrame] = {}
    strategy_iterator = tqdm(
        list(retrieval_by_strategy.items()),
        desc=f"Strategies: {judge_model_id}",
        unit="strategy",
        disable=not verbose,
        dynamic_ncols=True,
        leave=False,
    )
    for strategy, results_df in strategy_iterator:
        strategy_iterator.set_postfix(
            strategy=strategy,
            queries=int(results_df["query_id"].nunique()),
            results=len(results_df),
        )
        output_jsonl = judge_dir / f"{strategy}_judgments.jsonl"
        judgments_df = judge_retrieval_results(
            results_df,
            qrels_df,
            judge_llm,
            output_jsonl=output_jsonl,
            resume=resume,
            max_new_tokens=max_new_tokens,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
            verbose=verbose,
            progress_description=f"{judge_model_id} | {strategy}",
            reuse_failed_cache=reuse_failed_cache,
        )
        judgments_by_strategy[strategy] = judgments_df
        valid_scores = pd.to_numeric(
            judgments_df.get("relevance_score"), errors="coerce"
        ).between(0, 3)
        if "judgment_status" in judgments_df:
            explicit_completed = judgments_df["judgment_status"].eq("completed")
            legacy_completed = judgments_df["judgment_status"].isna() & valid_scores
            completed = (explicit_completed | legacy_completed).sum()
        else:
            completed = valid_scores.sum()
        strategy_summary = {
            "judge_model": judge_model_id,
            "strategy": strategy,
            "queries": int(results_df["query_id"].nunique()),
            "judgments": int(len(judgments_df)),
            "completed": int(completed),
            "failed": int(len(judgments_df) - completed),
            "cache_file": str(output_jsonl),
        }
        summaries.append(
            {
                **strategy_summary,
            }
        )
        strategy_iterator.set_postfix(
            strategy=strategy,
            completed=int(completed),
            failed=int(len(judgments_df) - completed),
        )
    return pd.DataFrame(summaries), judgments_by_strategy


def run_four_judge_retrieval_ablation(
    *,
    qrels_df: pd.DataFrame,
    retrieval_results_dir: str | Path,
    judgments_root_dir: str | Path,
    project_root: str | Path = ".",
    hf_cache_dir: str | Path = "data/models/huggingface",
    openai_api_key: str | None = None,
    hugging_face_token: str | None = None,
    selected_strategies: Sequence[str] | None = None,
    selected_judges: Sequence[str] | None = None,
    max_queries_per_strategy: int | None = None,
    resume: bool = True,
    reuse_failed_cache: bool = False,
    skip_fully_cached_judges: bool = True,
    max_new_tokens: int = 256,
    max_retries: int = 3,
    retry_sleep_seconds: float = 2.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    """Run configured judges sequentially with dry runs and cached-row skipping."""

    from src.retrieval_llms import (
        LOCAL_RELEVANCE_JUDGE_MODEL_IDS,
        LocalRelevanceJudgeManager,
        OpenAIRelevanceJudgeLLM,
    )

    judge_ids = ["gpt-5-nano", *LOCAL_RELEVANCE_JUDGE_MODEL_IDS.values()]
    if selected_judges is not None:
        requested = set(selected_judges)
        unknown = requested.difference(judge_ids)
        if unknown:
            raise ValueError(f"Unknown selected_judges: {sorted(unknown)}")
        judge_ids = [model_id for model_id in judge_ids if model_id in requested]

    manager = LocalRelevanceJudgeManager(
        project_root=project_root,
        cache_dir=hf_cache_dir,
        token=hugging_face_token,
    )
    summaries: list[pd.DataFrame] = []
    all_judgments: dict[str, dict[str, pd.DataFrame]] = {}
    judge_iterator = tqdm(
        judge_ids,
        desc=f"{len(judge_ids)}-judge ablation",
        unit="judge",
        disable=not verbose,
        dynamic_ncols=True,
    )
    try:
        for model_id in judge_iterator:
            judge_iterator.set_postfix(model=model_id, stage="checking_cache")
            pending, cached = count_pending_retrieval_judgments(
                judge_model_id=model_id,
                retrieval_results_dir=retrieval_results_dir,
                judgments_root_dir=judgments_root_dir,
                selected_strategies=selected_strategies,
                max_queries_per_strategy=max_queries_per_strategy,
                resume=resume,
                reuse_failed_cache=reuse_failed_cache,
            )
            if resume and skip_fully_cached_judges and pending == 0 and cached > 0:
                if verbose:
                    tqdm.write(
                        f"Skipping fully cached judge {model_id}: {cached} judgments cached."
                    )
                cached_judgments = load_retrieval_ablation_judgment_caches(
                    judgments_root_dir,
                    selected_judges=[model_id],
                    selected_strategies=selected_strategies,
                    max_queries_per_strategy=max_queries_per_strategy,
                ).get(model_id, {})
                all_judgments[model_id] = cached_judgments
                summaries.append(
                    pd.DataFrame(
                        [
                            {
                                "judge_model": model_id,
                                "strategy": "all_selected",
                                "queries": max_queries_per_strategy,
                                "judgments": cached,
                                "completed": cached,
                                "failed": 0,
                                "cache_file": str(judgments_root_dir),
                                "run_status": "skipped_fully_cached",
                            }
                        ]
                    )
                )
                judge_iterator.set_postfix(
                    model=model_id,
                    stage="skipped_cached",
                    cached=cached,
                )
                continue

            judge_iterator.set_postfix(
                model=model_id,
                stage="loading",
                pending=pending,
                cached=cached,
            )
            if model_id == "gpt-5-nano":
                judge = OpenAIRelevanceJudgeLLM(
                    model_id=model_id,
                    api_key=openai_api_key,
                )
            else:
                judge = manager.load(model_id)

            judge_iterator.set_postfix(
                model=model_id,
                stage="judging",
                pending=pending,
                cached=cached,
            )
            summary_df, judgments_by_strategy = judge_cached_retrieval_ablation(
                judge_llm=judge,
                qrels_df=qrels_df,
                retrieval_results_dir=retrieval_results_dir,
                judgments_root_dir=judgments_root_dir,
                selected_strategies=selected_strategies,
                max_queries_per_strategy=max_queries_per_strategy,
                resume=resume,
                reuse_failed_cache=reuse_failed_cache,
                max_new_tokens=max_new_tokens,
                max_retries=max_retries,
                retry_sleep_seconds=retry_sleep_seconds,
                verbose=verbose,
            )
            summaries.append(summary_df)
            all_judgments[model_id] = judgments_by_strategy
            if model_id != "gpt-5-nano":
                judge_iterator.set_postfix(model=model_id, stage="unloading")
                manager.unload()
            judge_iterator.set_postfix(model=model_id, stage="completed")
    finally:
        manager.unload()

    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    return summary, all_judgments


def average_retrieval_judgments(
    judgments_by_model: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    expected_judges: int | None = None,
) -> pd.DataFrame:
    """Average valid 0-3 scores after aligning all judges on ranked results."""

    frames: list[pd.DataFrame] = []
    key_columns = ["evaluation_strategy", "query_id", "chunk_id", "rank"]
    metadata_columns = [
        "query",
        "source_chunk_id",
        "retrieval_method",
        "retrieval_score",
        "expected_relevant_information",
        "is_source_chunk",
    ]
    for model_id, strategy_frames in judgments_by_model.items():
        for strategy, frame in strategy_frames.items():
            if frame.empty:
                continue
            current = frame.copy()
            current["evaluation_strategy"] = strategy
            current["judge_model"] = model_id
            current["relevance_score"] = pd.to_numeric(
                current["relevance_score"], errors="coerce"
            )
            current = current[current["relevance_score"].between(0, 3)].copy()
            frames.append(current)

    if not frames:
        return pd.DataFrame()

    judgments_df = pd.concat(frames, ignore_index=True)
    judgments_df = judgments_df.drop_duplicates(
        key_columns + ["judge_model"],
        keep="last",
    )
    aggregation = (
        judgments_df.groupby(key_columns, dropna=False)
        .agg(
            relevance_score=("relevance_score", "mean"),
            relevance_vote_rate=(
                "relevance_score",
                lambda scores: float((scores >= 2).mean()),
            ),
            judge_score_std=("relevance_score", "std"),
            judge_count=("judge_model", "nunique"),
            judge_models=("judge_model", lambda values: sorted(set(values))),
            **{
                column: (column, "first")
                for column in metadata_columns
                if column in judgments_df.columns
            },
        )
        .reset_index()
    )
    aggregation["judge_score_std"] = aggregation["judge_score_std"].fillna(0.0)
    resolved_expected_judges = (
        len(judgments_by_model)
        if expected_judges is None
        else int(expected_judges)
    )
    aggregation["ensemble_complete"] = aggregation["judge_count"].eq(
        resolved_expected_judges
    )
    return aggregation


def summarize_retrieval_judgments_by_model(
    judgments_by_model: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    relevance_threshold: float = 2.0,
) -> pd.DataFrame:
    """Calculate retrieval metrics separately for each LLM judge."""

    strategy_order = [
        "sparse",
        "dense",
        "hybrid_rrf",
        "graph_expanded_hybrid",
        "graph_expanded_hybrid_reranked",
    ]
    rows: list[dict[str, Any]] = []
    for model_id, strategy_frames in judgments_by_model.items():
        for strategy, frame in strategy_frames.items():
            if frame.empty or "relevance_score" not in frame:
                continue
            current = frame.copy()
            current["evaluation_strategy"] = strategy
            current["relevance_score"] = pd.to_numeric(
                current["relevance_score"], errors="coerce"
            )
            current = current[current["relevance_score"].between(0, 3)].copy()
            key_columns = [
                column
                for column in ("query_id", "chunk_id", "rank")
                if column in current.columns
            ]
            if key_columns:
                current = current.drop_duplicates(key_columns, keep="last")
            strategy_k = int(current["rank"].max())
            metrics = summarize_retrieval_judgments(
                current,
                k_values=[strategy_k],
                relevance_threshold=relevance_threshold,
            )
            if metrics.empty:
                continue
            row = metrics.iloc[0].to_dict()
            row["judge_model"] = model_id
            row["strategy"] = strategy
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["_strategy_order"] = pd.Categorical(
        result["strategy"],
        categories=strategy_order,
        ordered=True,
    )
    return (
        result.sort_values(["judge_model", "_strategy_order", "strategy"])
        .drop(columns="_strategy_order")
        .reset_index(drop=True)
    )


def load_retrieval_ablation_judgment_caches(
    judgments_root_dir: str | Path,
    *,
    selected_judges: Sequence[str] | None = None,
    selected_strategies: Sequence[str] | None = None,
    max_queries_per_strategy: int | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Rebuild the nested ensemble input directly from per-judge JSONL caches."""

    root = Path(judgments_root_dir)
    selected_judge_set = set(selected_judges or [])
    selected_strategy_set = set(selected_strategies or [])

    judgments_by_model: dict[str, dict[str, pd.DataFrame]] = {}
    for path in sorted(root.glob("*/*_judgments.jsonl")):
        strategy = path.name.removesuffix("_judgments.jsonl")
        if selected_strategy_set and strategy not in selected_strategy_set:
            continue
        frame = pd.DataFrame(read_jsonl(path))
        if frame.empty or "judge_model" not in frame:
            continue
        model_ids = frame["judge_model"].dropna().astype(str).unique().tolist()
        if len(model_ids) != 1:
            raise ValueError(f"Expected one judge_model in {path}, found {model_ids}")
        model_id = model_ids[0]
        if selected_judge_set and model_id not in selected_judge_set:
            continue
        if max_queries_per_strategy is not None:
            selected_query_ids = (
                frame["query_id"]
                .drop_duplicates()
                .head(int(max_queries_per_strategy))
                .tolist()
            )
            frame = frame[frame["query_id"].isin(selected_query_ids)].copy()
        cache_key_columns = [
            column
            for column in ("query_id", "chunk_id", "rank")
            if column in frame.columns
        ]
        if cache_key_columns:
            frame["_cache_order"] = range(len(frame))
            numeric_scores = pd.to_numeric(
                frame.get("relevance_score"), errors="coerce"
            )
            frame["_valid_score"] = numeric_scores.between(0, 3)
            frame = (
                frame.sort_values(
                    cache_key_columns + ["_valid_score", "_cache_order"]
                )
                .drop_duplicates(cache_key_columns, keep="last")
                .drop(columns=["_cache_order", "_valid_score"])
                .reset_index(drop=True)
            )
        judgments_by_model.setdefault(model_id, {})[strategy] = frame
    return judgments_by_model


def summarize_retrieval_ablation_elapsed_time(
    retrieval_results_dir: str | Path,
    *,
    selected_strategies: Sequence[str] | None = None,
    max_queries_per_strategy: int | None = None,
) -> pd.DataFrame:
    """Summarize per-query retrieval latency from frozen ablation artifacts."""

    strategy_order = [
        "sparse",
        "dense",
        "hybrid_rrf",
        "graph_expanded_hybrid",
        "graph_expanded_hybrid_reranked",
    ]
    selected = set(selected_strategies or [])
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(retrieval_results_dir).glob("*_retrieval_results.jsonl")):
        strategy = path.name.removesuffix("_retrieval_results.jsonl")
        if selected and strategy not in selected:
            continue

        records = [
            record
            for record in read_jsonl(path)
            if str(record.get("status", "")) == "completed"
        ]
        if max_queries_per_strategy is not None:
            records = records[: int(max_queries_per_strategy)]
        elapsed = pd.to_numeric(
            pd.Series(
                [record.get("elapsed_seconds") for record in records],
                dtype="object",
            ),
            errors="coerce",
        ).dropna()
        if elapsed.empty:
            continue
        rows.append(
            {
                "strategy": strategy,
                "queries_timed": int(len(elapsed)),
                "mean_elapsed_seconds": float(elapsed.mean()),
                "median_elapsed_seconds": float(elapsed.median()),
                "p95_elapsed_seconds": float(elapsed.quantile(0.95)),
                "total_elapsed_seconds": float(elapsed.sum()),
            }
        )
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df
    summary_df["_strategy_order"] = pd.Categorical(
        summary_df["strategy"],
        categories=strategy_order,
        ordered=True,
    )
    return (
        summary_df.sort_values(
            ["_strategy_order", "strategy"],
            na_position="last",
        )
        .drop(columns="_strategy_order")
        .reset_index(drop=True)
    )


def plot_retrieval_ablation_elapsed_time(
    elapsed_metrics_df: pd.DataFrame,
    *,
    figsize: tuple[int, int] = (13, 6),
) -> Any:
    """Plot mean retrieval latency per query for each strategy."""

    if elapsed_metrics_df.empty:
        raise ValueError("elapsed_metrics_df is empty.")
    plot_df = elapsed_metrics_df.copy()
    strategy_order = [
        "sparse",
        "dense",
        "hybrid_rrf",
        "graph_expanded_hybrid",
        "graph_expanded_hybrid_reranked",
    ]
    plot_df["_strategy_order"] = pd.Categorical(
        plot_df["strategy"],
        categories=strategy_order,
        ordered=True,
    )
    plot_df = (
        plot_df.sort_values(
            ["_strategy_order", "strategy"],
            na_position="last",
        )
        .drop(columns="_strategy_order")
        .reset_index(drop=True)
    )
    plot_df["strategy_label"] = plot_df["strategy"].str.replace("_", " ")
    axis = plot_df.plot(
        x="strategy_label",
        y="mean_elapsed_seconds",
        kind="bar",
        figsize=figsize,
        width=0.72,
        legend=False,
        title="Mean Retrieval Time per Query",
        color="#4C78A8",
    )
    axis.set_xlabel("strategy")
    axis.set_ylabel("seconds")
    axis.tick_params(axis="x", rotation=5)
    axis.set_xticklabels(axis.get_xticklabels(), ha="center")
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    return axis


def summarize_and_plot_average_judgments(
    averaged_judgments_df: pd.DataFrame,
    *,
    relevance_threshold: float = 2.0,
    require_complete_ensemble: bool = True,
    figsize: tuple[int, int] = (13, 6),
) -> tuple[pd.DataFrame, Any]:
    """Print, return, and plot Section 3-style metrics from averaged scores."""

    strategy_order = [
        "sparse",
        "dense",
        "hybrid_rrf",
        "graph_expanded_hybrid",
        "graph_expanded_hybrid_reranked",
    ]
    if averaged_judgments_df.empty:
        raise ValueError("averaged_judgments_df is empty.")
    evaluated_df = averaged_judgments_df.copy()
    if require_complete_ensemble:
        evaluated_df = evaluated_df[evaluated_df["ensemble_complete"]].copy()
    if evaluated_df.empty:
        raise ValueError("No rows have the required complete judge ensemble.")

    metrics_frames: list[pd.DataFrame] = []
    for strategy, strategy_df in evaluated_df.groupby("evaluation_strategy"):
        strategy_k = int(strategy_df["rank"].max())
        metrics_df = summarize_retrieval_judgments(
            strategy_df,
            k_values=[strategy_k],
            relevance_threshold=relevance_threshold,
        )
        metrics_df.insert(0, "strategy", strategy)
        if "relevance_vote_rate" in strategy_df:
            majority_df = strategy_df.copy()
            majority_df["relevance_score"] = majority_df["relevance_vote_rate"]
            majority_metrics = summarize_retrieval_judgments(
                majority_df,
                k_values=[strategy_k],
                relevance_threshold=0.5,
            ).iloc[0]
            metrics_df["majority_precision_at_k"] = majority_metrics[
                "precision_at_k"
            ]
            metrics_df["majority_hit_rate_at_k"] = majority_metrics[
                "hit_rate_at_k"
            ]
            metrics_df["majority_mrr_at_k"] = majority_metrics["mrr_at_k"]
        metrics_df["mean_judge_std"] = float(strategy_df["judge_score_std"].mean())
        metrics_frames.append(metrics_df)

    ensemble_metrics_df = pd.concat(metrics_frames, ignore_index=True)
    ensemble_metrics_df["_strategy_order"] = pd.Categorical(
        ensemble_metrics_df["strategy"],
        categories=strategy_order,
        ordered=True,
    )
    ensemble_metrics_df = (
        ensemble_metrics_df.sort_values(
            ["_strategy_order", "strategy"],
            na_position="last",
        )
        .drop(columns="_strategy_order")
        .reset_index(drop=True)
    )
    print(ensemble_metrics_df.to_string(index=False))

    plot_df = ensemble_metrics_df.copy()
    plot_df["strategy_label"] = plot_df["strategy"].str.replace("_", " ")
    plot_ax = plot_df.plot(
        x="strategy_label",
        y=[
            "mean_ndcg",
            "hit_rate_at_k",
            "mrr_at_k",
            "source_chunk_hit_at_k",
        ],
        kind="bar",
        figsize=figsize,
        width=0.78,
        title="Retrieval strategy metrics averaged across LLM judges",
    )
    plot_ax.set_xlabel("strategy")
    plot_ax.set_ylabel("score")
    plot_ax.set_ylim(0, 1.05)
    plot_ax.tick_params(axis="x", rotation=5)
    plot_ax.legend(
        ["nDCG", "Hit rate", "MRR", "Source-chunk hit rate"],
        loc="lower right",
    )
    plot_ax.figure.tight_layout()
    return ensemble_metrics_df, plot_ax


def dcg_at_k(relevance_scores: Sequence[int | float], k: int) -> float:
    """Compute graded DCG@k with exponential gain."""

    dcg = 0.0
    for index, score in enumerate(list(relevance_scores)[:k], start=1):
        gain = math.pow(2.0, float(score)) - 1.0
        dcg += gain / math.log2(index + 1)
    return dcg


def ndcg_at_k(relevance_scores: Sequence[int | float], k: int) -> float:
    """Compute nDCG@k using the ideal ordering of the judged relevance labels."""

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
        per_query_mean_relevance: list[float] = []
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
            per_query_mean_relevance.append(sum(scores) / max(k, 1))
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
                "mean_relevance_score": sum(per_query_mean_relevance)
                / max(len(per_query_mean_relevance), 1),
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
