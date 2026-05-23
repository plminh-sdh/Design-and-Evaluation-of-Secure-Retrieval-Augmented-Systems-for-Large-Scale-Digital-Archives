"""Reusable retrieval evaluation helpers for qrel-based notebook tuning."""

from __future__ import annotations

import hashlib
import json
import math
import re
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
        results_df = retriever.retrieve(query, query_id=query_id, top_k=top_k)
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
    expected_relevant_information: str,
    judge_model_id: str,
) -> str:
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
            cache[key] = record
    return cache


def judge_retrieval_results(
    results_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    judge_llm: Any,
    *,
    output_jsonl: str | Path | None = None,
    resume: bool = True,
    max_new_tokens: int = 256,
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
            expected_relevant_information=expected,
            judge_model_id=judge_model_id,
        )

        cached_record = cache.get(judgment_key)
        if cached_record:
            judged_records.append(cached_record)
            iterator.set_postfix(cached=len(judged_records), new=len(new_records))
            continue

        raw_response = judge_llm.judge_relevance(
            query=str(result.get("query", qrel.get("query", ""))),
            expected_relevant_information=expected,
            retrieved_chunk_text=str(result.get("retrieval_text", "")),
            max_new_tokens=max_new_tokens,
        )
        try:
            parsed = extract_first_json_object(raw_response)
            relevance_score = int(parsed.get("relevance_score", 0))
        except Exception as exc:
            parsed = {
                "relevance_score": 0,
                "relevance_label": "parse_error",
                "contains_expected_information": False,
                "missing_information": "",
                "supporting_evidence": "",
                "rationale": f"Judge response parse error: {exc!r}",
            }
            relevance_score = 0

        relevance_score = max(0, min(3, relevance_score))
        record = {
            "judgment_key": judgment_key,
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

