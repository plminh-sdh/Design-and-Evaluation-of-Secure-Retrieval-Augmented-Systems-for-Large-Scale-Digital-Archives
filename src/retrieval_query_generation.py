"""Helpers for generating retrieval-evaluation query/qrel JSONL files."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from tqdm.auto import tqdm

from src.retrieval_llms import extract_first_json_object


CHUNK_ID_COLUMN = "chunk_id"
NEO4J_CHUNK_ID_COLUMN = "chunk_id:ID(Chunk)"
DEFAULT_QREL_FIELDS = [
    "query_id",
    "query",
    "source_chunk_id",
    "document_id",
    "dataset",
    "modality",
    "title",
    "expected_relevant_information",
    "reference_answer",
    "query_type",
    "difficulty",
    "access_level",
    "sensitivity_level",
    "generation_model",
    "generation_prompt_version",
    "grounding_evidence",
    "quality_notes",
]

SOURCE_FRAMING_PATTERN = re.compile(
    r"\b(chunk|passage|document|metadata|excerpt|provided text|source text)\b",
    flags=re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def normalize_chunk_columns(chunks_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Neo4j export column names used by the retrieval notebook."""

    df = chunks_df.copy()
    if NEO4J_CHUNK_ID_COLUMN in df.columns and CHUNK_ID_COLUMN not in df.columns:
        df = df.rename(columns={NEO4J_CHUNK_ID_COLUMN: CHUNK_ID_COLUMN})
    if "chunk_index:int" in df.columns and "chunk_index" not in df.columns:
        df = df.rename(columns={"chunk_index:int": "chunk_index"})
    return df


def load_archive_chunks(
    chunks_csv: str | Path,
    *,
    usecols: list[str] | None = None,
    chunksize: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load exported archive chunks and normalize expected columns."""

    path = Path(chunks_csv)
    if usecols is None:
        usecols = [
            NEO4J_CHUNK_ID_COLUMN,
            "document_id",
            "dataset",
            "modality",
            "chunk_index:int",
            "title",
            "masked_text",
            "summary",
            "sensitivity_level",
            "access_level",
        ]

    if chunksize is None:
        if verbose:
            tqdm.write(f"Loading archive chunks from {path}")
        return normalize_chunk_columns(pd.read_csv(path, usecols=usecols))

    frames = []
    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize)
    for chunk in tqdm(reader, desc="Loading archive chunk batches", disable=not verbose):
        frames.append(chunk)
    if not frames:
        return normalize_chunk_columns(pd.DataFrame(columns=usecols))
    return normalize_chunk_columns(pd.concat(frames, ignore_index=True))


def _text_stats(text: Any) -> dict[str, Any]:
    text = "" if pd.isna(text) else str(text)
    words = WORD_PATTERN.findall(text.lower())
    char_count = len(text)
    alpha_count = sum(char.isalpha() for char in text)
    alnum_count = sum(char.isalnum() for char in text)
    whitespace_count = sum(char.isspace() for char in text)
    nonspace_count = max(char_count - whitespace_count, 1)
    unique_tokens = len(set(words))
    punctuation_noise = max(nonspace_count - alnum_count, 0) / nonspace_count
    alpha_ratio = alpha_count / max(char_count, 1)

    return {
        "text_char_count": char_count,
        "text_token_count": len(words),
        "unique_token_count": unique_tokens,
        "alpha_ratio": alpha_ratio,
        "punctuation_noise_ratio": punctuation_noise,
    }


def add_query_candidate_features(
    chunks_df: pd.DataFrame,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Add lightweight text-quality and sampling features to chunk rows."""

    df = normalize_chunk_columns(chunks_df)
    text_iterable = tqdm(
        df["masked_text"],
        total=len(df),
        desc="Scoring chunk text quality",
        disable=not verbose,
    )
    stats_df = pd.DataFrame([_text_stats(text) for text in text_iterable])
    df = pd.concat([df.reset_index(drop=True), stats_df], axis=1)

    token_count = df["text_token_count"].fillna(0)
    df["length_bucket"] = pd.cut(
        token_count,
        bins=[-1, 80, 180, 400, math.inf],
        labels=["short", "medium", "long", "very_long"],
    ).astype("string")
    df["is_probable_ocr_noise"] = (
        (df["alpha_ratio"] < 0.45)
        | (df["punctuation_noise_ratio"] > 0.35)
        | ((df["unique_token_count"] < 25) & (df["text_token_count"] >= 80))
    )
    df["is_too_short_for_query"] = (
        (df["text_char_count"] < 300)
        | (df["text_token_count"] < 45)
        | (df["unique_token_count"] < 20)
    )
    df["is_metadata_only_like"] = df["masked_text"].fillna("").str.match(
        r"^\s*(dataset|modality|title|summary|metadata)\s*:",
        case=False,
    ) & (df["text_token_count"] < 120)
    df["query_candidate_quality"] = "usable"
    df.loc[df["is_too_short_for_query"], "query_candidate_quality"] = "too_short"
    df.loc[df["is_probable_ocr_noise"], "query_candidate_quality"] = "ocr_noise"
    df.loc[df["is_metadata_only_like"], "query_candidate_quality"] = "metadata_only"
    return df


def filter_public_query_candidates(
    chunks_df: pd.DataFrame,
    *,
    min_tokens: int = 45,
    min_unique_tokens: int = 20,
    min_alpha_ratio: float = 0.45,
    max_punctuation_noise_ratio: float = 0.35,
    keep_feature_columns: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Filter public chunks to candidate rows likely to yield useful qrels."""

    df = add_query_candidate_features(chunks_df, verbose=verbose)
    mask = (
        df["access_level"].fillna("").str.lower().eq("public")
        & df["masked_text"].notna()
        & df["masked_text"].astype(str).str.strip().ne("")
        & (df["text_token_count"] >= min_tokens)
        & (df["unique_token_count"] >= min_unique_tokens)
        & (df["alpha_ratio"] >= min_alpha_ratio)
        & (df["punctuation_noise_ratio"] <= max_punctuation_noise_ratio)
        & (~df["is_metadata_only_like"])
    )
    filtered = df.loc[mask].copy()
    if verbose:
        tqdm.write(
            "Filtered public query candidates: "
            f"{len(filtered):,}/{len(df):,} chunks kept"
        )
    if keep_feature_columns:
        return filtered
    feature_columns = [
        "text_char_count",
        "text_token_count",
        "unique_token_count",
        "alpha_ratio",
        "punctuation_noise_ratio",
        "length_bucket",
        "is_probable_ocr_noise",
        "is_too_short_for_query",
        "is_metadata_only_like",
        "query_candidate_quality",
    ]
    return filtered.drop(columns=[col for col in feature_columns if col in filtered])


def summarize_query_candidates(chunks_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create compact inspection tables for public candidate quality."""

    df = add_query_candidate_features(chunks_df)
    return {
        "access_level": df.groupby("access_level", dropna=False).size().reset_index(name="chunks"),
        "quality": df.groupby(
            ["access_level", "dataset", "modality", "query_candidate_quality"],
            dropna=False,
        )
        .size()
        .reset_index(name="chunks")
        .sort_values(["access_level", "dataset", "modality", "query_candidate_quality"]),
        "length": df.groupby(
            ["access_level", "dataset", "modality", "length_bucket"],
            dropna=False,
        )
        .size()
        .reset_index(name="chunks")
        .sort_values(["access_level", "dataset", "modality", "length_bucket"]),
    }


def stratified_sample_query_candidates(
    candidates_df: pd.DataFrame,
    *,
    target_queries: int,
    reserve_multiplier: int = 4,
    strata_columns: list[str] | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sample a candidate queue with extra rows for rejected/noisy chunks."""

    if strata_columns is None:
        strata_columns = ["dataset", "modality", "length_bucket"]

    df = normalize_chunk_columns(candidates_df).copy()
    df = df.drop_duplicates(subset=[CHUNK_ID_COLUMN])
    sample_size = min(len(df), max(target_queries, target_queries * reserve_multiplier))
    if sample_size <= 0:
        return df.head(0)

    if verbose:
        tqdm.write(
            "Sampling query candidate reserve queue: "
            f"target={target_queries:,}, reserve_multiplier={reserve_multiplier}, "
            f"sample_size={sample_size:,}, available={len(df):,}"
        )

    rng_seed = random_state
    group_columns = [col for col in strata_columns if col in df.columns]
    if not group_columns:
        sampled = df.sample(n=sample_size, random_state=random_state)
    else:
        groups = list(df.groupby(group_columns, dropna=False))
        base_per_group = max(1, sample_size // max(len(groups), 1))
        pieces = []
        remaining = sample_size
        group_iterator = tqdm(
            groups,
            desc="Sampling candidate strata",
            disable=not verbose,
        )
        for index, (_, group) in enumerate(group_iterator):
            groups_left = len(groups) - index
            take = min(len(group), max(1, min(base_per_group, remaining - groups_left + 1)))
            if take <= 0:
                continue
            pieces.append(group.sample(n=take, random_state=rng_seed + index))
            remaining -= take

        sampled = pd.concat(pieces, ignore_index=True) if pieces else df.head(0)
        if len(sampled) < sample_size:
            remaining_df = df.loc[~df[CHUNK_ID_COLUMN].isin(sampled[CHUNK_ID_COLUMN])]
            extra_n = min(sample_size - len(sampled), len(remaining_df))
            if extra_n > 0:
                sampled = pd.concat(
                    [
                        sampled,
                        remaining_df.sample(n=extra_n, random_state=random_state + 999),
                    ],
                    ignore_index=True,
                )

    sampled = sampled.sample(frac=1, random_state=random_state + 2024).reset_index(drop=True)
    sampled["candidate_rank"] = range(1, len(sampled) + 1)
    sampled["target_queries"] = target_queries
    return sampled


def dataframe_to_jsonl(records_df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records_df.to_dict(orient="records"):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def normalize_query_text(query: Any) -> str:
    text = "" if pd.isna(query) else str(query).lower().strip()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def make_query_id(source_chunk_id: str, query: str) -> str:
    digest = hashlib.sha1(f"{source_chunk_id}|{normalize_query_text(query)}".encode()).hexdigest()
    return f"q_{digest[:16]}"


def validate_generated_qrel(record: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Validate a parsed generated qrel record."""

    reasons: list[str] = []
    if bool(record.get("reject")):
        reasons.append("model_rejected")
        return False, reasons

    query = str(record.get("query", "")).strip()
    expected = str(record.get("expected_relevant_information", "")).strip()
    answer = str(record.get("reference_answer", "")).strip()
    query_type = str(record.get("query_type", "")).strip()
    difficulty = str(record.get("difficulty", "")).strip()

    if len(query) < 15:
        reasons.append("query_too_short")
    if SOURCE_FRAMING_PATTERN.search(query):
        reasons.append("query_mentions_source_frame")
    if len(expected) < 30:
        reasons.append("expected_information_too_short")
    if len(expected.split()) < 5:
        reasons.append("expected_information_not_self_contained")
    if not answer:
        reasons.append("missing_reference_answer")
    if not query_type:
        reasons.append("missing_query_type")
    if not difficulty:
        reasons.append("missing_difficulty")

    return not reasons, reasons


def build_qrel_record(
    parsed_record: Mapping[str, Any],
    chunk: Mapping[str, Any],
    *,
    generation_model: str,
    generation_prompt_version: str = "v1",
) -> dict[str, Any]:
    """Merge parsed LLM output with source chunk metadata."""

    chunk_id = str(chunk.get(CHUNK_ID_COLUMN) or chunk.get(NEO4J_CHUNK_ID_COLUMN) or "")
    query = str(parsed_record.get("query", "")).strip()
    qrel = {
        "query_id": make_query_id(chunk_id, query) if query else "",
        "query": query,
        "source_chunk_id": chunk_id,
        "document_id": chunk.get("document_id", ""),
        "dataset": chunk.get("dataset", ""),
        "modality": chunk.get("modality", ""),
        "title": chunk.get("title", ""),
        "expected_relevant_information": str(
            parsed_record.get("expected_relevant_information", "")
        ).strip(),
        "reference_answer": str(parsed_record.get("reference_answer", "")).strip(),
        "query_type": str(parsed_record.get("query_type", "")).strip(),
        "difficulty": str(parsed_record.get("difficulty", "")).strip(),
        "access_level": chunk.get("access_level", ""),
        "sensitivity_level": chunk.get("sensitivity_level", ""),
        "generation_model": generation_model,
        "generation_prompt_version": generation_prompt_version,
        "grounding_evidence": str(parsed_record.get("grounding_evidence", "")).strip(),
        "quality_notes": str(parsed_record.get("quality_notes", "")).strip(),
    }
    is_valid, reasons = validate_generated_qrel(parsed_record)
    qrel["is_valid"] = is_valid
    qrel["validation_reasons"] = reasons
    return qrel


def run_query_generation_llm(
    candidate_chunks_df: pd.DataFrame,
    llm: Any,
    *,
    output_jsonl: str | Path,
    progress_jsonl: str | Path,
    rejects_jsonl: str | Path,
    target_qrels: int,
    prompt_version: str = "v1",
    max_new_tokens: int = 768,
    temperature: float = 0.2,
    top_p: float = 0.9,
    do_sample: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate qrels until target accepted records or candidate queue is exhausted."""

    output_jsonl = Path(output_jsonl)
    progress_jsonl = Path(progress_jsonl)
    rejects_jsonl = Path(rejects_jsonl)
    existing_qrels = read_jsonl(output_jsonl)
    existing_progress = read_jsonl(progress_jsonl)
    completed_chunk_ids = {str(record.get("chunk_id")) for record in existing_progress}
    accepted_query_norms = {
        normalize_query_text(record.get("query", "")) for record in existing_qrels
    }

    accepted_count = len(existing_qrels)
    generated_now = 0
    rejected_now = 0
    error_now = 0
    skipped_now = 0

    df = normalize_chunk_columns(candidate_chunks_df)
    remaining_df = df.loc[~df[CHUNK_ID_COLUMN].astype(str).isin(completed_chunk_ids)].copy()
    if verbose:
        tqdm.write(
            "Starting query generation: "
            f"{accepted_count:,}/{target_qrels:,} accepted already, "
            f"{len(completed_chunk_ids):,} chunks completed, "
            f"{len(remaining_df):,} candidate chunks remaining"
        )

    iterator = tqdm(
        remaining_df.to_dict(orient="records"),
        total=len(remaining_df),
        desc="Generating retrieval qrels",
        disable=not verbose,
        unit="chunk",
        dynamic_ncols=True,
    )
    for chunk in iterator:
        if accepted_count >= target_qrels:
            break

        chunk_id = str(chunk.get(CHUNK_ID_COLUMN, ""))
        if chunk_id in completed_chunk_ids:
            skipped_now += 1
            iterator.set_postfix(
                accepted=accepted_count,
                rejected=rejected_now,
                errors=error_now,
                skipped=skipped_now,
                remaining_target=max(target_qrels - accepted_count, 0),
            )
            continue

        try:
            response = llm.generate_query_record(
                chunk,
                prompt_version=prompt_version,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            parsed_record = extract_first_json_object(response)
            qrel = build_qrel_record(
                parsed_record,
                chunk,
                generation_model=getattr(llm, "model_id", ""),
                generation_prompt_version=prompt_version,
            )
            query_norm = normalize_query_text(qrel.get("query", ""))
            if query_norm in accepted_query_norms:
                qrel["is_valid"] = False
                qrel["validation_reasons"] = list(qrel["validation_reasons"]) + [
                    "duplicate_query"
                ]

            if qrel["is_valid"]:
                append_jsonl([qrel], output_jsonl)
                accepted_query_norms.add(query_norm)
                accepted_count += 1
                generated_now += 1
            else:
                reject_record = {
                    "chunk_id": chunk_id,
                    "response": response,
                    "parsed_record": parsed_record,
                    "validation_reasons": qrel.get("validation_reasons", []),
                }
                append_jsonl([reject_record], rejects_jsonl)
                rejected_now += 1

            append_jsonl(
                [
                    {
                        "chunk_id": chunk_id,
                        "accepted": bool(qrel["is_valid"]),
                        "query_id": qrel.get("query_id", ""),
                    }
                ],
                progress_jsonl,
            )
            completed_chunk_ids.add(chunk_id)
            iterator.set_postfix(
                accepted=accepted_count,
                rejected=rejected_now,
                errors=error_now,
                skipped=skipped_now,
                remaining_target=max(target_qrels - accepted_count, 0),
            )
        except Exception as exc:
            append_jsonl(
                [{"chunk_id": chunk_id, "error": repr(exc)}],
                rejects_jsonl,
            )
            append_jsonl(
                [{"chunk_id": chunk_id, "accepted": False, "error": repr(exc)}],
                progress_jsonl,
            )
            completed_chunk_ids.add(chunk_id)
            error_now += 1
            iterator.set_postfix(
                accepted=accepted_count,
                rejected=rejected_now,
                errors=error_now,
                skipped=skipped_now,
                remaining_target=max(target_qrels - accepted_count, 0),
            )

    return {
        "target_qrels": target_qrels,
        "accepted_total": accepted_count,
        "accepted_now": generated_now,
        "rejected_now": rejected_now,
        "errors_now": error_now,
        "skipped_now": skipped_now,
        "output_jsonl": output_jsonl,
        "progress_jsonl": progress_jsonl,
        "rejects_jsonl": rejects_jsonl,
    }


def load_generated_qrels(path: str | Path) -> pd.DataFrame:
    records = read_jsonl(path)
    return pd.DataFrame(records, columns=DEFAULT_QREL_FIELDS + ["is_valid", "validation_reasons"])


def deduplicate_qrels(qrels_df: pd.DataFrame) -> pd.DataFrame:
    if qrels_df.empty:
        return qrels_df.copy()
    df = qrels_df.copy()
    df["normalized_query"] = df["query"].map(normalize_query_text)
    df = df.sort_values(["source_chunk_id", "query_id"]).drop_duplicates(
        subset=["normalized_query"],
        keep="first",
    )
    return df.drop(columns=["normalized_query"])


def split_qrels_dev_test(
    qrels_df: pd.DataFrame,
    *,
    dev_fraction: float = 0.3,
    random_state: int = 42,
    stratify_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stratify_columns is None:
        stratify_columns = ["dataset", "modality", "query_type"]
    if qrels_df.empty:
        return qrels_df.copy(), qrels_df.copy()

    df = qrels_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    group_columns = [col for col in stratify_columns if col in df.columns]
    if not group_columns:
        dev_n = max(1, int(round(len(df) * dev_fraction)))
        return df.iloc[:dev_n].copy(), df.iloc[dev_n:].copy()

    dev_parts = []
    test_parts = []
    for _, group in df.groupby(group_columns, dropna=False):
        dev_n = int(round(len(group) * dev_fraction))
        if len(group) > 1:
            dev_n = min(max(dev_n, 1), len(group) - 1)
        dev_parts.append(group.iloc[:dev_n])
        test_parts.append(group.iloc[dev_n:])

    dev_df = pd.concat(dev_parts, ignore_index=True) if dev_parts else df.head(0)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else df.head(0)
    return dev_df.sample(frac=1, random_state=random_state).reset_index(drop=True), test_df.sample(
        frac=1, random_state=random_state + 1
    ).reset_index(drop=True)


def qrel_quality_summary(qrels_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if qrels_df.empty:
        empty = pd.DataFrame(columns=["records"])
        return {"dataset_modality": empty, "query_type": empty, "difficulty": empty}
    return {
        "dataset_modality": qrels_df.groupby(["dataset", "modality"], dropna=False)
        .size()
        .reset_index(name="records")
        .sort_values(["dataset", "modality"]),
        "query_type": qrels_df.groupby("query_type", dropna=False)
        .size()
        .reset_index(name="records")
        .sort_values("records", ascending=False),
        "difficulty": qrels_df.groupby("difficulty", dropna=False)
        .size()
        .reset_index(name="records")
        .sort_values("records", ascending=False),
    }
