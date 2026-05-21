"""Typed relation extraction helpers for the graph knowledge-base pipeline.

The helpers in this module recover chunk-local, canonicalized entity mentions
from the existing CSV exports and prepare constrained GLiREL inputs. They are
designed so notebook cells can run a small sample first, while later production
cells can reuse the same loaders and predictors in resumable chunk batches.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm

from src.archive_schema import stable_id
from src.graph_kb_local_coreference import (
    CONTACT_DETAIL_TYPES,
    is_obvious_generic_local_coreference_mention,
)


RELATION_EXPORT_DIR = Path("data") / "graph_kb_exports" / "step_04_relations"
DEFAULT_RELATION_TEXT_COLUMN = "masked_text"
DEFAULT_RELATION_CANDIDATE_TYPES = {
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "EVENT",
    "POLITICAL_GROUP",
    "PROGRAM_INITIATIVE",
    "LAW",
    "CREATIVE_WORK",
    "PRODUCT",
}
LOCAL_RELATION_STATUSES = {
    "EXACT_SURFACE_CLUSTERED",
    "CLUSTERED",
}


@dataclass(frozen=True)
class RelationSchemaEntry:
    """Controlled relation label and type constraints."""

    graph_relation: str
    glirel_label: str
    allowed_head: tuple[str, ...]
    allowed_tail: tuple[str, ...]
    description: str = ""

    def to_glirel_constraint(self) -> Dict[str, list[str]]:
        return {
            "allowed_head": list(self.allowed_head),
            "allowed_tail": list(self.allowed_tail),
        }


RELATION_SCHEMA: tuple[RelationSchemaEntry, ...] = (
    RelationSchemaEntry(
        "AFFILIATED_WITH",
        "affiliated with",
        ("PERSON",),
        ("ORGANIZATION", "POLITICAL_GROUP", "PROGRAM_INITIATIVE"),
        "Person has a stated affiliation with a group, party, or program.",
    ),
    RelationSchemaEntry(
        "WORKS_FOR",
        "works for",
        ("PERSON",),
        ("ORGANIZATION",),
        "Person is employed by or professionally works for an organization.",
    ),
    RelationSchemaEntry(
        "LEADS",
        "leads",
        ("PERSON",),
        ("ORGANIZATION", "POLITICAL_GROUP", "PROGRAM_INITIATIVE"),
        "Person leads, directs, chairs, or heads an entity.",
    ),
    RelationSchemaEntry(
        "MEMBER_OF",
        "member of",
        ("PERSON",),
        ("ORGANIZATION", "POLITICAL_GROUP"),
        "Person is a member of a group, party, team, or organization.",
    ),
    RelationSchemaEntry(
        "FOUNDED_BY",
        "founded by",
        ("ORGANIZATION", "PROGRAM_INITIATIVE", "PRODUCT", "CREATIVE_WORK"),
        ("PERSON", "ORGANIZATION"),
        "Entity, program, product, or work was founded by an actor.",
    ),
    RelationSchemaEntry(
        "LOCATED_IN",
        "located in",
        ("ORGANIZATION", "EVENT", "PROGRAM_INITIATIVE"),
        ("LOCATION",),
        "Entity, event, or program is located in a place.",
    ),
    RelationSchemaEntry(
        "HEADQUARTERED_IN",
        "headquartered in",
        ("ORGANIZATION",),
        ("LOCATION",),
        "Organization has headquarters in a place.",
    ),
    RelationSchemaEntry(
        "PART_OF",
        "part of",
        ("ORGANIZATION", "LOCATION", "PROGRAM_INITIATIVE"),
        ("ORGANIZATION", "LOCATION", "PROGRAM_INITIATIVE"),
        "Entity is a subunit, component, or subdivision of another entity.",
    ),
    RelationSchemaEntry(
        "PARTICIPATED_IN",
        "participated in",
        ("PERSON", "ORGANIZATION", "POLITICAL_GROUP"),
        ("EVENT", "PROGRAM_INITIATIVE"),
        "Actor participated in an event or program.",
    ),
    RelationSchemaEntry(
        "ORGANIZED_BY",
        "organized by",
        ("EVENT", "PROGRAM_INITIATIVE"),
        ("PERSON", "ORGANIZATION", "POLITICAL_GROUP"),
        "Event or program was organized by an actor.",
    ),
    RelationSchemaEntry(
        "OCCURRED_IN",
        "occurred in",
        ("EVENT",),
        ("LOCATION",),
        "Event happened in a location.",
    ),
    RelationSchemaEntry(
        "CREATED_BY",
        "created by",
        ("CREATIVE_WORK", "PRODUCT", "PROGRAM_INITIATIVE"),
        ("PERSON", "ORGANIZATION"),
        "Work, product, or program was created by an actor.",
    ),
    RelationSchemaEntry(
        "PUBLISHED_BY",
        "published by",
        ("CREATIVE_WORK",),
        ("ORGANIZATION", "PERSON"),
        "Creative work was published by an actor.",
    ),
    RelationSchemaEntry(
        "PRODUCED_BY",
        "produced by",
        ("CREATIVE_WORK", "PRODUCT"),
        ("ORGANIZATION", "PERSON"),
        "Creative work or product was produced by an actor.",
    ),
    RelationSchemaEntry(
        "GOVERNS",
        "governs",
        ("LAW",),
        ("ORGANIZATION", "PERSON", "LOCATION", "EVENT"),
        "Law or policy governs an actor, place, or event.",
    ),
    RelationSchemaEntry(
        "ENACTED_BY",
        "enacted by",
        ("LAW",),
        ("ORGANIZATION", "POLITICAL_GROUP", "PERSON"),
        "Law or policy was enacted by an actor.",
    ),
    RelationSchemaEntry(
        "SUPPORTED_BY",
        "supported by",
        ("LAW", "EVENT", "PROGRAM_INITIATIVE", "POLITICAL_GROUP"),
        ("PERSON", "ORGANIZATION", "POLITICAL_GROUP"),
        "Policy, event, program, or group is supported by an actor.",
    ),
    RelationSchemaEntry(
        "OPPOSED_BY",
        "opposed by",
        ("LAW", "EVENT", "PROGRAM_INITIATIVE", "POLITICAL_GROUP"),
        ("PERSON", "ORGANIZATION", "POLITICAL_GROUP"),
        "Policy, event, program, or group is opposed by an actor.",
    ),
)
RELATION_BY_GLIREL_LABEL = {
    entry.glirel_label: entry.graph_relation for entry in RELATION_SCHEMA
}
RELATION_BY_GRAPH_TYPE = {entry.graph_relation: entry for entry in RELATION_SCHEMA}
RELATION_FAMILY_BY_TYPE = {
    "AFFILIATED_WITH": "actor_affiliation",
    "WORKS_FOR": "actor_affiliation",
    "LEADS": "actor_affiliation",
    "MEMBER_OF": "actor_affiliation",
    "LOCATED_IN": "location",
    "HEADQUARTERED_IN": "location",
    "OCCURRED_IN": "location",
    "PARTICIPATED_IN": "event_program",
    "ORGANIZED_BY": "event_program",
    "CREATED_BY": "creation",
    "PUBLISHED_BY": "creation",
    "PRODUCED_BY": "creation",
    "FOUNDED_BY": "creation",
    "GOVERNS": "law_policy",
    "ENACTED_BY": "law_policy",
    "SUPPORTED_BY": "law_policy",
    "OPPOSED_BY": "law_policy",
    "PART_OF": "structure",
}
RELATION_SCORE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("very_low", float("-inf"), 0.15),
    ("low", 0.15, 0.40),
    ("medium", 0.40, 0.70),
    ("high", 0.70, float("inf")),
)


RESOLVED_RELATION_MENTION_COLUMNS = [
    "mention_id",
    "chunk_id",
    "document_id",
    "dataset",
    "modality",
    "mention_text",
    "mention_type",
    "start_char",
    "end_char",
    "entity_id",
    "canonical_name",
    "canonicalization_status",
    "canonicalization_method",
    "canonicalizer",
    "model_name",
    "model_version",
    "source",
    "cluster_size",
]


RELATION_PREDICTION_COLUMNS = [
    "relation_id:ID(Relation)",
    ":START_ID(Entity)",
    ":END_ID(Entity)",
    "relation_type",
    "glirel_label",
    "confidence:float",
    "chunk_id",
    "document_id",
    "dataset",
    "modality",
    "head_mention_id",
    "tail_mention_id",
    "head_text",
    "tail_text",
    "head_type",
    "tail_type",
    "evidence_text",
    "extractor",
    "model_name",
    "model_version",
    ":TYPE",
]

RELATION_GOLD_RAW_PREDICTION_COLUMNS = [
    "sample_id",
    "split",
    "chunk_id",
    "document_id",
    "dataset",
    "modality",
    "text",
    "head_mention_id",
    "tail_mention_id",
    "head_entity_id",
    "tail_entity_id",
    "head_text",
    "tail_text",
    "head_canonical_name",
    "tail_canonical_name",
    "head_type",
    "tail_type",
    "head_source",
    "tail_source",
    "entity_source_mix",
    "predicted_relation_type",
    "predicted_relation_family",
    "predicted_glirel_label",
    "predicted_score",
    "score_band",
    "allowed_relation_types",
    "gold_relation_type",
    "gold_is_correct",
    "correction_notes",
]

RELATION_GOLD_CHUNK_COLUMNS = [
    "sample_id",
    "split",
    "chunk_id",
    "document_id",
    "dataset",
    "modality",
    "text",
    "entity_count",
    "valid_schema_pair_count",
    "entity_types",
    "relation_families",
    "has_refined_entity",
    "has_local_entity",
    "has_refined_local_pair",
]


def glirel_label_constraints(
    schema: Sequence[RelationSchemaEntry] = RELATION_SCHEMA,
) -> Dict[str, Dict[str, Dict[str, list[str]]]]:
    """Return constrained GLiREL labels in the documented spaCy format."""
    return {
        "glirel_labels": {
            entry.glirel_label: entry.to_glirel_constraint() for entry in schema
        }
    }


def glirel_label_list(
    schema: Sequence[RelationSchemaEntry] = RELATION_SCHEMA,
) -> list[str]:
    """Return plain GLiREL labels for direct model calls."""
    return [entry.glirel_label for entry in schema]


def relation_labels_for_type_pair(
    head_type: Any,
    tail_type: Any,
    schema: Sequence[RelationSchemaEntry] = RELATION_SCHEMA,
) -> list[str]:
    """Return graph relation types allowed for a directed entity type pair."""
    head = str(head_type or "").upper()
    tail = str(tail_type or "").upper()
    return [
        entry.graph_relation
        for entry in schema
        if head in entry.allowed_head and tail in entry.allowed_tail
    ]


def relation_families_for_type_pair(
    head_type: Any,
    tail_type: Any,
    schema: Sequence[RelationSchemaEntry] = RELATION_SCHEMA,
) -> list[str]:
    """Return relation families allowed for a directed entity type pair."""
    families = {
        RELATION_FAMILY_BY_TYPE[relation]
        for relation in relation_labels_for_type_pair(head_type, tail_type, schema)
        if relation in RELATION_FAMILY_BY_TYPE
    }
    return sorted(families)


def relation_score_band(score: Any) -> str:
    """Return the score band used for raw relation-gold sampling."""
    value = float(score or 0.0)
    for name, lower, upper in RELATION_SCORE_BANDS:
        if lower <= value < upper:
            return name
    return "unknown"


def is_allowed_relation_type_pair(
    head_type: Any,
    tail_type: Any,
    graph_relation: str,
    schema_by_type: Mapping[str, RelationSchemaEntry] = RELATION_BY_GRAPH_TYPE,
) -> bool:
    """Return whether a graph relation is valid for a directed type pair."""
    entry = schema_by_type.get(str(graph_relation or "").upper())
    if entry is None:
        return False
    return (
        str(head_type or "").upper() in entry.allowed_head
        and str(tail_type or "").upper() in entry.allowed_tail
    )


def _standardize_mentions_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "mention_id:ID(Mention)": "mention_id",
        "start_char:int": "start_char",
        "end_char:int": "end_char",
    }
    output = df.rename(columns=rename_map).copy()
    for column in ["mention_id", "chunk_id", "mention_text", "mention_type"]:
        if column in output.columns:
            output[column] = output[column].fillna("").astype(str)
    if "mention_type" in output.columns:
        output["mention_type"] = output["mention_type"].str.upper().str.strip()
    for column in ["start_char", "end_char"]:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _standardize_chunks_columns(
    df: pd.DataFrame,
    *,
    text_column: str = DEFAULT_RELATION_TEXT_COLUMN,
) -> pd.DataFrame:
    rename_map = {
        "chunk_id:ID(Chunk)": "chunk_id",
        "chunk_index:int": "chunk_index",
    }
    output = df.rename(columns=rename_map).copy()
    if text_column not in output.columns:
        raise ValueError(f"Chunk dataframe is missing text column: {text_column}")
    keep_columns = [
        "chunk_id",
        "document_id",
        "dataset",
        "modality",
        "title",
        text_column,
        "sensitivity_level",
        "access_level",
    ]
    keep_columns = [column for column in keep_columns if column in output.columns]
    output = output[keep_columns].copy()
    output = output.rename(columns={text_column: "chunk_text"})
    output["chunk_id"] = output["chunk_id"].fillna("").astype(str)
    output["chunk_text"] = output["chunk_text"].fillna("").astype(str)
    return output


def _filter_chunk_ids(df: pd.DataFrame, chunk_ids: Optional[set[str]]) -> pd.DataFrame:
    if chunk_ids is None:
        return df
    if "chunk_id" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["chunk_id"].astype(str).isin(chunk_ids)].copy()


def read_chunks_for_relation_extraction(
    chunks_csv: str | Path,
    *,
    chunk_ids: Optional[Iterable[str]] = None,
    text_column: str = DEFAULT_RELATION_TEXT_COLUMN,
    chunksize: int = 250_000,
    verbose: bool = False,
) -> pd.DataFrame:
    """Read chunk text and metadata for relation extraction."""
    requested = {str(value) for value in chunk_ids} if chunk_ids is not None else None
    frames: list[pd.DataFrame] = []
    reader = pd.read_csv(chunks_csv, chunksize=chunksize, low_memory=False)
    for chunk in tqdm(
        reader,
        desc="Loading relation chunks",
        unit="csv chunk",
        disable=not verbose,
    ):
        chunk = _standardize_chunks_columns(chunk, text_column=text_column)
        chunk = _filter_chunk_ids(chunk, requested)
        if not chunk.empty:
            frames.append(chunk)
        if requested is not None:
            found = set(chunk["chunk_id"].astype(str))
            requested.difference_update(found)
            if not requested:
                break
    if not frames:
        return pd.DataFrame(
            columns=[
                "chunk_id",
                "document_id",
                "dataset",
                "modality",
                "title",
                "chunk_text",
                "sensitivity_level",
                "access_level",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def read_mention_spans(
    mentions_csv: str | Path,
    *,
    mention_ids: Optional[Iterable[str]] = None,
    chunk_ids: Optional[Iterable[str]] = None,
    chunksize: int = 250_000,
    verbose: bool = False,
) -> pd.DataFrame:
    """Read original mention spans needed by GLiREL input construction."""
    requested_mentions = (
        {str(value) for value in mention_ids} if mention_ids is not None else None
    )
    requested_chunks = {str(value) for value in chunk_ids} if chunk_ids else None
    usecols = [
        "mention_id:ID(Mention)",
        "mention_text",
        "mention_type",
        "chunk_id",
        "document_id",
        "dataset",
        "modality",
        "start_char:int",
        "end_char:int",
        "confidence:float",
    ]
    frames: list[pd.DataFrame] = []
    reader = pd.read_csv(
        mentions_csv,
        usecols=lambda column: column in usecols,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in tqdm(
        reader,
        desc="Loading mention spans",
        unit="csv chunk",
        disable=not verbose,
    ):
        chunk = _standardize_mentions_columns(chunk)
        if requested_mentions is not None:
            chunk = chunk[chunk["mention_id"].isin(requested_mentions)].copy()
        if requested_chunks is not None:
            chunk = chunk[chunk["chunk_id"].isin(requested_chunks)].copy()
        if not chunk.empty:
            frames.append(chunk)
        if requested_mentions is not None:
            requested_mentions.difference_update(set(chunk["mention_id"]))
            if not requested_mentions:
                break
    if not frames:
        return pd.DataFrame(
            columns=[
                "mention_id",
                "mention_text",
                "mention_type",
                "chunk_id",
                "document_id",
                "dataset",
                "modality",
                "start_char",
                "end_char",
                "confidence:float",
            ]
        )
    return pd.concat(frames, ignore_index=True).drop_duplicates("mention_id")


def _read_refined_resolved_mentions(
    refined_canonicalization_csv: str | Path,
    *,
    chunk_ids: Optional[Iterable[str]] = None,
    chunksize: int = 250_000,
    verbose: bool = False,
) -> pd.DataFrame:
    requested_chunks = {str(value) for value in chunk_ids} if chunk_ids else None
    usecols = [
        "mention_id",
        "chunk_id",
        "document_id",
        "dataset",
        "modality",
        "mention_text",
        "mention_type",
        "start_char:int",
        "end_char:int",
        "entity_id",
        "canonical_name",
        "canonicalization_status",
        "canonicalization_method",
        "canonicalizer",
        "model_name",
        "model_version",
    ]
    frames: list[pd.DataFrame] = []
    reader = pd.read_csv(
        refined_canonicalization_csv,
        usecols=lambda column: column in usecols,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in tqdm(
        reader,
        desc="Loading ReFinED resolved mentions",
        unit="csv chunk",
        disable=not verbose,
    ):
        chunk = _standardize_mentions_columns(chunk)
        chunk = _filter_chunk_ids(chunk, requested_chunks)
        chunk["entity_id"] = chunk["entity_id"].fillna("").astype(str).str.strip()
        chunk["canonicalization_status"] = (
            chunk["canonicalization_status"].fillna("").astype(str).str.strip()
        )
        chunk = chunk[
            (chunk["entity_id"] != "")
            & (chunk["canonicalization_status"] == "LINKED")
        ].copy()
        if chunk.empty:
            continue
        chunk["source"] = "refined"
        chunk["cluster_size"] = pd.NA
        frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=RESOLVED_RELATION_MENTION_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _read_local_resolved_mentions(
    local_coreference_predictions_csv: str | Path,
    *,
    mentions_csv: str | Path,
    chunk_ids: Optional[Iterable[str]] = None,
    include_singletons: bool = False,
    chunksize: int = 250_000,
    verbose: bool = False,
) -> pd.DataFrame:
    requested_chunks = {str(value) for value in chunk_ids} if chunk_ids else None
    accepted_statuses = set(LOCAL_RELATION_STATUSES)
    if include_singletons:
        accepted_statuses.add("SINGLETON")
    usecols = [
        "mention_id",
        "chunk_id",
        "document_id",
        "dataset",
        "modality",
        "mention_text",
        "mention_type",
        "entity_id",
        "canonical_name",
        "cluster_size:int",
        "canonicalization_status",
        "canonicalization_method",
        "canonicalizer",
        "model_name",
        "model_version",
    ]
    frames: list[pd.DataFrame] = []
    mention_ids: set[str] = set()
    reader = pd.read_csv(
        local_coreference_predictions_csv,
        usecols=lambda column: column in usecols,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in tqdm(
        reader,
        desc="Loading local resolved mentions",
        unit="csv chunk",
        disable=not verbose,
    ):
        chunk = chunk.rename(columns={"cluster_size:int": "cluster_size"})
        chunk = _standardize_mentions_columns(chunk)
        chunk = _filter_chunk_ids(chunk, requested_chunks)
        chunk["entity_id"] = chunk["entity_id"].fillna("").astype(str).str.strip()
        chunk["canonicalization_status"] = (
            chunk["canonicalization_status"].fillna("").astype(str).str.strip()
        )
        chunk = chunk[
            (chunk["entity_id"] != "")
            & chunk["canonicalization_status"].isin(accepted_statuses)
        ].copy()
        if chunk.empty:
            continue
        chunk["source"] = "local_coreference"
        frames.append(chunk)
        mention_ids.update(chunk["mention_id"].dropna().astype(str).tolist())
    if not frames:
        return pd.DataFrame(columns=RESOLVED_RELATION_MENTION_COLUMNS)

    local_mentions = pd.concat(frames, ignore_index=True)
    span_df = read_mention_spans(
        mentions_csv,
        mention_ids=mention_ids,
        chunksize=chunksize,
        verbose=verbose,
    )
    span_columns = ["mention_id", "start_char", "end_char"]
    local_mentions = local_mentions.merge(
        span_df[span_columns],
        on="mention_id",
        how="left",
    )
    return local_mentions


def _clean_resolved_relation_mentions(
    df: pd.DataFrame,
    *,
    allowed_types: Iterable[str] = DEFAULT_RELATION_CANDIDATE_TYPES,
    exclude_generic: bool = True,
) -> pd.DataFrame:
    allowed = {str(value).upper() for value in allowed_types}
    output = df.copy()
    for column in RESOLVED_RELATION_MENTION_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    output = output[RESOLVED_RELATION_MENTION_COLUMNS].copy()
    output["mention_type"] = output["mention_type"].fillna("").astype(str).str.upper()
    output["mention_text"] = output["mention_text"].fillna("").astype(str).str.strip()
    output["canonical_name"] = (
        output["canonical_name"].fillna("").astype(str).str.strip()
    )
    output["entity_id"] = output["entity_id"].fillna("").astype(str).str.strip()
    output["chunk_id"] = output["chunk_id"].fillna("").astype(str).str.strip()
    output = output[
        (output["entity_id"] != "")
        & (output["chunk_id"] != "")
        & output["mention_type"].isin(allowed)
        & ~output["mention_type"].isin(CONTACT_DETAIL_TYPES)
    ].copy()
    output["start_char"] = pd.to_numeric(output["start_char"], errors="coerce")
    output["end_char"] = pd.to_numeric(output["end_char"], errors="coerce")
    output = output[
        output["start_char"].notna()
        & output["end_char"].notna()
        & (output["end_char"] > output["start_char"])
    ].copy()
    if exclude_generic and not output.empty:
        generic_mask = output.apply(
            lambda row: is_obvious_generic_local_coreference_mention(row),
            axis=1,
        )
        output = output[~generic_mask].copy()
    output["entity_label"] = output["canonical_name"].where(
        output["canonical_name"] != "",
        output["mention_text"],
    )
    return output.drop_duplicates(["chunk_id", "mention_id", "entity_id"])


def load_resolved_relation_mentions(
    refined_canonicalization_csv: str | Path,
    local_coreference_predictions_csv: str | Path,
    mentions_csv: str | Path,
    *,
    chunk_ids: Optional[Iterable[str]] = None,
    allowed_types: Iterable[str] = DEFAULT_RELATION_CANDIDATE_TYPES,
    include_local_singletons: bool = False,
    exclude_generic: bool = True,
    chunksize: int = 250_000,
    verbose: bool = False,
) -> pd.DataFrame:
    """Load ReFinED and local canonicalized mentions for relation extraction."""
    requested_chunks = {str(value) for value in chunk_ids} if chunk_ids else None
    refined = _read_refined_resolved_mentions(
        refined_canonicalization_csv,
        chunk_ids=requested_chunks,
        chunksize=chunksize,
        verbose=verbose,
    )
    local = _read_local_resolved_mentions(
        local_coreference_predictions_csv,
        mentions_csv=mentions_csv,
        chunk_ids=requested_chunks,
        include_singletons=include_local_singletons,
        chunksize=chunksize,
        verbose=verbose,
    )
    combined = pd.concat([refined, local], ignore_index=True)
    return _clean_resolved_relation_mentions(
        combined,
        allowed_types=allowed_types,
        exclude_generic=exclude_generic,
    )


def load_relation_extraction_batch(
    chunks_csv: str | Path,
    refined_canonicalization_csv: str | Path,
    local_coreference_predictions_csv: str | Path,
    mentions_csv: str | Path,
    *,
    chunk_ids: Iterable[str],
    text_column: str = DEFAULT_RELATION_TEXT_COLUMN,
    allowed_types: Iterable[str] = DEFAULT_RELATION_CANDIDATE_TYPES,
    include_local_singletons: bool = False,
    exclude_generic: bool = True,
    chunksize: int = 250_000,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load chunk text and resolved entity mentions for a specific chunk batch."""
    chunk_id_list = [str(value) for value in chunk_ids]
    chunks_df = read_chunks_for_relation_extraction(
        chunks_csv,
        chunk_ids=chunk_id_list,
        text_column=text_column,
        chunksize=chunksize,
        verbose=verbose,
    )
    mentions_df = load_resolved_relation_mentions(
        refined_canonicalization_csv,
        local_coreference_predictions_csv,
        mentions_csv,
        chunk_ids=chunk_id_list,
        allowed_types=allowed_types,
        include_local_singletons=include_local_singletons,
        exclude_generic=exclude_generic,
        chunksize=chunksize,
        verbose=verbose,
    )
    return chunks_df, mentions_df


def iter_chunk_id_batches(
    chunks_csv: str | Path,
    *,
    batch_size: int = 500,
    chunksize: int = 250_000,
) -> Iterator[list[str]]:
    """Yield chunk IDs in stable CSV order for later resumable processing."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    chunk_id_col = "chunk_id:ID(Chunk)"
    buffer: list[str] = []
    for chunk in pd.read_csv(
        chunks_csv,
        usecols=lambda column: column == chunk_id_col or column == "chunk_id",
        chunksize=chunksize,
        low_memory=False,
    ):
        if chunk_id_col in chunk.columns:
            ids = chunk[chunk_id_col]
        else:
            ids = chunk["chunk_id"]
        for chunk_id in ids.dropna().astype(str):
            buffer.append(chunk_id)
            if len(buffer) >= batch_size:
                yield buffer
                buffer = []
    if buffer:
        yield buffer


def _empty_chunk_stats_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "chunk_id",
            "document_id",
            "dataset",
            "modality",
            "entity_count",
            "valid_schema_pair_count",
            "entity_types",
            "relation_families",
            "has_refined_entity",
            "has_local_entity",
            "has_refined_local_pair",
            "promising_score",
        ]
    )


def _iter_resolved_rows_for_chunk_stats(
    csv_path: str | Path,
    *,
    source: str,
    allowed_types: set[str],
    include_local_singletons: bool,
    chunksize: int,
    max_rows: Optional[int],
) -> Iterator[Dict[str, Any]]:
    usecols = [
        "chunk_id",
        "document_id",
        "dataset",
        "modality",
        "mention_text",
        "mention_type",
        "entity_id",
        "canonical_name",
        "canonicalization_status",
    ]
    accepted_local_statuses = set(LOCAL_RELATION_STATUSES)
    if include_local_singletons:
        accepted_local_statuses.add("SINGLETON")
    rows_seen = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=lambda column: column in usecols,
        chunksize=chunksize,
        low_memory=False,
    ):
        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        rows_seen += len(chunk)
        chunk["entity_id"] = chunk["entity_id"].fillna("").astype(str).str.strip()
        chunk["mention_type"] = chunk["mention_type"].fillna("").astype(str).str.upper()
        chunk["canonicalization_status"] = (
            chunk["canonicalization_status"].fillna("").astype(str).str.strip()
        )
        if source == "refined":
            mask = (
                (chunk["entity_id"] != "")
                & (chunk["canonicalization_status"] == "LINKED")
            )
        else:
            mask = (
                (chunk["entity_id"] != "")
                & chunk["canonicalization_status"].isin(accepted_local_statuses)
            )
        chunk = chunk[mask & chunk["mention_type"].isin(allowed_types)].copy()
        if chunk.empty:
            continue
        if source == "local_coreference":
            generic_mask = chunk.apply(
                lambda row: is_obvious_generic_local_coreference_mention(row),
                axis=1,
            )
            chunk = chunk[~generic_mask].copy()
        for row in chunk.to_dict("records"):
            row["source"] = source
            yield row


def collect_relation_candidate_chunk_stats(
    refined_canonicalization_csv: str | Path,
    local_coreference_predictions_csv: str | Path,
    *,
    allowed_types: Iterable[str] = DEFAULT_RELATION_CANDIDATE_TYPES,
    include_local_singletons: bool = False,
    min_entities_per_chunk: int = 2,
    max_entities_per_chunk: int = 30,
    chunksize: int = 250_000,
    max_rows_per_source: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Collect lightweight chunk statistics for relation-gold candidate sampling.

    This streams the canonicalization exports and keeps only capped per-chunk
    entity/type/source summaries, so the later GLiREL pass can load full text and
    spans only for promising chunk IDs.
    """
    allowed = {str(value).upper() for value in allowed_types}
    allowed -= set(CONTACT_DETAIL_TYPES)
    chunk_entities: dict[str, dict[str, Any]] = {}

    def update(row: Mapping[str, Any]) -> None:
        chunk_id = str(row.get("chunk_id") or "").strip()
        entity_id = str(row.get("entity_id") or "").strip()
        entity_type = str(row.get("mention_type") or "").upper().strip()
        if not chunk_id or not entity_id or entity_type not in allowed:
            return
        summary = chunk_entities.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "document_id": row.get("document_id"),
                "dataset": row.get("dataset"),
                "modality": row.get("modality"),
                "entities": {},
                "truncated": False,
            },
        )
        entities = summary["entities"]
        if entity_id not in entities and len(entities) >= max_entities_per_chunk + 1:
            summary["truncated"] = True
            return
        entity = entities.setdefault(
            entity_id,
            {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "sources": set(),
            },
        )
        entity["sources"].add(str(row.get("source") or ""))

    if verbose:
        tqdm.write("Collecting relation candidate chunk stats from ReFinED links...")
    refined_rows = _iter_resolved_rows_for_chunk_stats(
        refined_canonicalization_csv,
        source="refined",
        allowed_types=allowed,
        include_local_singletons=include_local_singletons,
        chunksize=chunksize,
        max_rows=max_rows_per_source,
    )
    for row in tqdm(
        refined_rows,
        desc="Scanning ReFinED relation candidates",
        unit="mention",
        disable=not verbose,
    ):
        update(row)

    if verbose:
        tqdm.write("Collecting relation candidate chunk stats from local coreference links...")
    local_rows = _iter_resolved_rows_for_chunk_stats(
        local_coreference_predictions_csv,
        source="local_coreference",
        allowed_types=allowed,
        include_local_singletons=include_local_singletons,
        chunksize=chunksize,
        max_rows=max_rows_per_source,
    )
    for row in tqdm(
        local_rows,
        desc="Scanning local relation candidates",
        unit="mention",
        disable=not verbose,
    ):
        update(row)

    records: list[Dict[str, Any]] = []
    if verbose:
        tqdm.write(
            f"Scored raw relation candidate mentions into {len(chunk_entities):,} chunks; "
            "computing schema-pair summaries..."
        )
    for summary in tqdm(
        chunk_entities.values(),
        desc="Scoring relation candidate chunks",
        unit="chunk",
        disable=not verbose,
    ):
        entities = list(summary["entities"].values())
        entity_count = len(entities)
        if entity_count < min_entities_per_chunk or entity_count > max_entities_per_chunk:
            continue
        valid_pair_count = 0
        families: set[str] = set()
        has_refined = False
        has_local = False
        has_mixed_pair = False
        for entity in entities:
            has_refined = has_refined or "refined" in entity["sources"]
            has_local = has_local or "local_coreference" in entity["sources"]
        for head in entities:
            for tail in entities:
                if head["entity_id"] == tail["entity_id"]:
                    continue
                relation_types = relation_labels_for_type_pair(
                    head["entity_type"],
                    tail["entity_type"],
                )
                if not relation_types:
                    continue
                valid_pair_count += 1
                families.update(
                    RELATION_FAMILY_BY_TYPE[relation]
                    for relation in relation_types
                    if relation in RELATION_FAMILY_BY_TYPE
                )
                if head["sources"] != tail["sources"]:
                    has_mixed_pair = True
        if valid_pair_count == 0:
            continue
        entity_types = sorted({entity["entity_type"] for entity in entities})
        promising_score = (
            min(valid_pair_count, 20)
            + min(len(entity_types), 6)
            + (2 if has_refined else 0)
            + (1 if has_local else 0)
            + (1 if has_mixed_pair else 0)
        )
        records.append(
            {
                "chunk_id": summary["chunk_id"],
                "document_id": summary.get("document_id"),
                "dataset": summary.get("dataset"),
                "modality": summary.get("modality"),
                "entity_count": entity_count,
                "valid_schema_pair_count": valid_pair_count,
                "entity_types": entity_types,
                "relation_families": sorted(families),
                "has_refined_entity": has_refined,
                "has_local_entity": has_local,
                "has_refined_local_pair": has_mixed_pair,
                "promising_score": promising_score,
            }
        )
    if not records:
        return _empty_chunk_stats_dataframe()
    return pd.DataFrame(records).sort_values(
        ["promising_score", "valid_schema_pair_count", "entity_count"],
        ascending=False,
    )


def sample_relation_candidate_chunks(
    chunk_stats_df: pd.DataFrame,
    *,
    target_dev_chunks: int = 500,
    target_test_chunks: int = 500,
    random_state: int = 42,
    dev_fraction: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample promising relation chunks with document-level dev/test separation."""
    if chunk_stats_df.empty:
        return chunk_stats_df.copy(), chunk_stats_df.copy()
    rng = random.Random(random_state)
    stats = chunk_stats_df.copy()
    stats["document_id"] = stats["document_id"].fillna("").astype(str)
    stats["primary_family"] = stats["relation_families"].map(
        lambda values: list(values)[0] if isinstance(values, list) and values else "background_schema_valid"
    )
    mixed_local = stats["has_local_entity"].fillna(False).astype(bool)
    stats.loc[mixed_local, "primary_family"] = stats.loc[
        mixed_local, "primary_family"
    ].map(lambda value: "mixed_local" if rng.random() < 0.35 else value)

    documents = list(stats["document_id"].dropna().unique())
    rng.shuffle(documents)
    dev_doc_count = max(1, int(round(len(documents) * dev_fraction)))
    dev_docs = set(documents[:dev_doc_count])
    stats["split"] = stats["document_id"].map(
        lambda document_id: "dev" if document_id in dev_docs else "test"
    )

    def sample_split(split: str, target: int) -> pd.DataFrame:
        split_df = stats[stats["split"] == split].copy()
        if split_df.empty:
            return split_df
        sampled_frames: list[pd.DataFrame] = []
        families = sorted(split_df["primary_family"].dropna().unique())
        per_family = max(1, target // max(1, len(families)))
        for family in families:
            family_df = split_df[split_df["primary_family"] == family].sort_values(
                "promising_score",
                ascending=False,
            )
            take = min(len(family_df), per_family)
            if take:
                sampled_frames.append(family_df.head(take))
        sampled = (
            pd.concat(sampled_frames, ignore_index=True)
            if sampled_frames
            else split_df.iloc[0:0].copy()
        )
        remaining = target - len(sampled)
        if remaining > 0:
            used = set(sampled["chunk_id"])
            filler = split_df[~split_df["chunk_id"].isin(used)].sort_values(
                "promising_score",
                ascending=False,
            )
            sampled = pd.concat([sampled, filler.head(remaining)], ignore_index=True)
        return sampled.head(target).copy()

    return (
        sample_split("dev", target_dev_chunks),
        sample_split("test", target_test_chunks),
    )


def align_char_span_to_token_span(
    doc: Any,
    start_char: int,
    end_char: int,
    *,
    alignment_mode: str = "expand",
) -> Optional[tuple[int, int]]:
    """Map character offsets to GLiREL inclusive token offsets."""
    span = doc.char_span(
        int(start_char),
        int(end_char),
        alignment_mode=alignment_mode,
    )
    if span is None or span.start == span.end:
        return None
    return span.start, span.end - 1


def build_glirel_chunk_input(
    chunk_row: Mapping[str, Any],
    mentions_df: pd.DataFrame,
    nlp: Any,
    *,
    max_entities_per_chunk: int = 40,
    dedupe_entity_spans: bool = True,
) -> Optional[Dict[str, Any]]:
    """Build one direct-call GLiREL input from chunk text and resolved mentions."""
    chunk_id = str(chunk_row.get("chunk_id") or "")
    text = str(chunk_row.get("chunk_text") or "")
    if not chunk_id or not text:
        return None

    doc = nlp.make_doc(text) if hasattr(nlp, "make_doc") else nlp(text)
    tokens = [token.text for token in doc]
    mentions = mentions_df[mentions_df["chunk_id"].astype(str) == chunk_id].copy()
    if mentions.empty:
        return None
    mentions = mentions.sort_values(["start_char", "end_char", "mention_id"])

    seen_entity_spans: set[tuple[str, int, int]] = set()
    ner: list[list[Any]] = []
    mention_records: list[Dict[str, Any]] = []
    for row in mentions.to_dict("records"):
        token_span = align_char_span_to_token_span(
            doc,
            int(row["start_char"]),
            int(row["end_char"]),
        )
        if token_span is None:
            continue
        start_token, end_token = token_span
        dedupe_key = (str(row["entity_id"]), start_token, end_token)
        if dedupe_entity_spans and dedupe_key in seen_entity_spans:
            continue
        seen_entity_spans.add(dedupe_key)
        mention_text = str(row.get("mention_text") or "")
        mention_type = str(row.get("mention_type") or "").upper()
        ner.append([start_token, end_token, mention_type, mention_text])
        record = dict(row)
        record["ner_index"] = len(ner) - 1
        record["token_start"] = start_token
        record["token_end"] = end_token
        mention_records.append(record)

    if len(ner) < 2:
        return None
    if len(ner) > max_entities_per_chunk:
        return None
    if not _chunk_has_allowed_relation_pair(mention_records):
        return None

    return {
        "chunk_id": chunk_id,
        "document_id": chunk_row.get("document_id"),
        "dataset": chunk_row.get("dataset"),
        "modality": chunk_row.get("modality"),
        "text": text,
        "tokens": tokens,
        "ner": ner,
        "mention_records": mention_records,
        "labels": glirel_label_constraints(),
    }


def build_glirel_batch_inputs(
    chunks_df: pd.DataFrame,
    mentions_df: pd.DataFrame,
    nlp: Any,
    *,
    max_entities_per_chunk: int = 40,
    verbose: bool = True,
) -> list[Dict[str, Any]]:
    """Build GLiREL inputs for a loaded chunk batch."""
    inputs: list[Dict[str, Any]] = []
    for row in tqdm(
        chunks_df.to_dict("records"),
        desc="Building GLiREL chunk inputs",
        unit="chunk",
        disable=not verbose,
    ):
        item = build_glirel_chunk_input(
            row,
            mentions_df,
            nlp,
            max_entities_per_chunk=max_entities_per_chunk,
        )
        if item is not None:
            inputs.append(item)
    return inputs


def _chunk_has_allowed_relation_pair(mention_records: Sequence[Mapping[str, Any]]) -> bool:
    for head in mention_records:
        for tail in mention_records:
            if head is tail:
                continue
            if str(head.get("entity_id")) == str(tail.get("entity_id")):
                continue
            if relation_labels_for_type_pair(
                head.get("mention_type"),
                tail.get("mention_type"),
            ):
                return True
    return False


def predict_glirel_relations_for_input(
    model: Any,
    glirel_input: Mapping[str, Any],
    *,
    threshold: float = 0.8,
    top_k: int = 1,
) -> list[Dict[str, Any]]:
    """Run a GLiREL direct-call prediction for one prepared chunk input."""
    return model.predict_relations(
        glirel_input["tokens"],
        glirel_input["labels"],
        threshold=threshold,
        ner=glirel_input["ner"],
        top_k=top_k,
    )


def predict_glirel_relations_for_batch(
    model: Any,
    glirel_inputs: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.8,
    top_k: int = 1,
    verbose: bool = True,
) -> list[Dict[str, Any]]:
    """Run GLiREL on a prepared batch and attach chunk context to predictions."""
    predictions: list[Dict[str, Any]] = []
    for item in tqdm(
        glirel_inputs,
        desc="Running GLiREL relation prediction",
        unit="chunk",
        disable=not verbose,
    ):
        for prediction in predict_glirel_relations_for_input(
            model,
            item,
            threshold=threshold,
            top_k=top_k,
        ):
            enriched = dict(prediction)
            enriched["chunk_id"] = item.get("chunk_id")
            enriched["document_id"] = item.get("document_id")
            enriched["dataset"] = item.get("dataset")
            enriched["modality"] = item.get("modality")
            predictions.append(enriched)
    return predictions


def validate_and_normalize_relation_predictions(
    raw_predictions: Sequence[Mapping[str, Any]],
    glirel_inputs: Sequence[Mapping[str, Any]],
    *,
    extractor: str = "GLiREL",
    model_name: str = "jackboyla/glirel-large-v0",
    model_version: str = "",
) -> pd.DataFrame:
    """Validate raw GLiREL predictions against schema and return graph edges."""
    input_by_chunk = {str(item["chunk_id"]): item for item in glirel_inputs}
    rows: list[Dict[str, Any]] = []
    for prediction in raw_predictions:
        chunk_id = str(prediction.get("chunk_id") or "")
        item = input_by_chunk.get(chunk_id)
        if item is None:
            continue
        head = _find_prediction_endpoint(prediction, item, "head")
        tail = _find_prediction_endpoint(prediction, item, "tail")
        if head is None or tail is None:
            continue
        if str(head.get("entity_id")) == str(tail.get("entity_id")):
            continue
        glirel_label = str(prediction.get("label") or "").strip()
        graph_relation = RELATION_BY_GLIREL_LABEL.get(glirel_label)
        if not graph_relation:
            continue
        if not is_allowed_relation_type_pair(
            head.get("mention_type"),
            tail.get("mention_type"),
            graph_relation,
        ):
            continue
        score = float(prediction.get("score") or 0.0)
        relation_id = stable_id(
            "relation",
            chunk_id,
            head.get("entity_id"),
            graph_relation,
            tail.get("entity_id"),
            head.get("mention_id"),
            tail.get("mention_id"),
        )
        rows.append(
            {
                "relation_id:ID(Relation)": relation_id,
                ":START_ID(Entity)": head.get("entity_id"),
                ":END_ID(Entity)": tail.get("entity_id"),
                "relation_type": graph_relation,
                "glirel_label": glirel_label,
                "confidence:float": score,
                "chunk_id": chunk_id,
                "document_id": item.get("document_id"),
                "dataset": item.get("dataset"),
                "modality": item.get("modality"),
                "head_mention_id": head.get("mention_id"),
                "tail_mention_id": tail.get("mention_id"),
                "head_text": head.get("mention_text"),
                "tail_text": tail.get("mention_text"),
                "head_type": head.get("mention_type"),
                "tail_type": tail.get("mention_type"),
                "evidence_text": item.get("text"),
                "extractor": extractor,
                "model_name": model_name,
                "model_version": model_version,
                ":TYPE": graph_relation,
            }
        )
    if not rows:
        return pd.DataFrame(columns=RELATION_PREDICTION_COLUMNS)
    return pd.DataFrame(rows, columns=RELATION_PREDICTION_COLUMNS).drop_duplicates(
        ["relation_id:ID(Relation)"]
    )


def raw_glirel_predictions_to_gold_records(
    raw_predictions: Sequence[Mapping[str, Any]],
    glirel_inputs: Sequence[Mapping[str, Any]],
    *,
    split: str,
    verbose: bool = True,
) -> list[Dict[str, Any]]:
    """Convert raw GLiREL predictions to annotation-ready gold JSONL records."""
    input_by_chunk = {str(item["chunk_id"]): item for item in glirel_inputs}
    records: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for prediction in tqdm(
        raw_predictions,
        desc=f"Converting {split} GLiREL predictions",
        unit="prediction",
        disable=not verbose,
    ):
        chunk_id = str(prediction.get("chunk_id") or "")
        item = input_by_chunk.get(chunk_id)
        if item is None:
            continue
        head = _find_prediction_endpoint(prediction, item, "head")
        tail = _find_prediction_endpoint(prediction, item, "tail")
        if head is None or tail is None:
            continue
        if str(head.get("entity_id")) == str(tail.get("entity_id")):
            continue
        glirel_label = str(prediction.get("label") or "").strip()
        graph_relation = RELATION_BY_GLIREL_LABEL.get(glirel_label)
        if not graph_relation:
            continue
        if not is_allowed_relation_type_pair(
            head.get("mention_type"),
            tail.get("mention_type"),
            graph_relation,
        ):
            continue

        allowed_relation_types = relation_labels_for_type_pair(
            head.get("mention_type"),
            tail.get("mention_type"),
        )
        score = float(prediction.get("score") or 0.0)
        source_mix = "-".join(
            sorted(
                [
                    str(head.get("source") or "unknown"),
                    str(tail.get("source") or "unknown"),
                ]
            )
        )
        sample_id = stable_id(
            "relation_gold_raw",
            split,
            chunk_id,
            head.get("mention_id"),
            tail.get("mention_id"),
            graph_relation,
        )
        if sample_id in seen:
            continue
        seen.add(sample_id)
        records.append(
            {
                "sample_id": sample_id,
                "split": split,
                "chunk_id": chunk_id,
                "document_id": item.get("document_id"),
                "dataset": item.get("dataset"),
                "modality": item.get("modality"),
                "text": item.get("text"),
                "head_mention_id": head.get("mention_id"),
                "tail_mention_id": tail.get("mention_id"),
                "head_entity_id": head.get("entity_id"),
                "tail_entity_id": tail.get("entity_id"),
                "head_text": head.get("mention_text"),
                "tail_text": tail.get("mention_text"),
                "head_canonical_name": head.get("entity_label")
                or head.get("canonical_name"),
                "tail_canonical_name": tail.get("entity_label")
                or tail.get("canonical_name"),
                "head_type": head.get("mention_type"),
                "tail_type": tail.get("mention_type"),
                "head_source": head.get("source"),
                "tail_source": tail.get("source"),
                "entity_source_mix": source_mix,
                "predicted_relation_type": graph_relation,
                "predicted_relation_family": RELATION_FAMILY_BY_TYPE.get(
                    graph_relation,
                    "other",
                ),
                "predicted_glirel_label": glirel_label,
                "predicted_score": score,
                "score_band": relation_score_band(score),
                "allowed_relation_types": allowed_relation_types,
                "gold_relation_type": "",
                "gold_is_correct": "",
                "correction_notes": "",
            }
        )
    return records


def sample_relation_gold_prediction_records(
    records: Sequence[Mapping[str, Any]],
    *,
    target_records: int = 1_000,
    random_state: int = 42,
    verbose: bool = True,
) -> list[Dict[str, Any]]:
    """Sample raw relation prediction records across score bands and families."""
    if not records:
        return []
    df = pd.DataFrame([dict(record) for record in records])
    if len(df) <= target_records:
        if verbose:
            tqdm.write(
                f"Keeping all {len(df):,} raw relation records; target is {target_records:,}."
            )
        return df[RELATION_GOLD_RAW_PREDICTION_COLUMNS].to_dict("records")
    if verbose:
        tqdm.write(
            f"Sampling {target_records:,} raw relation records from {len(df):,} candidates..."
        )
    rng = random.Random(random_state)
    df["_rand"] = [rng.random() for _ in range(len(df))]
    df["score_band"] = df["predicted_score"].map(relation_score_band)
    desired_band_fractions = {
        "high": 0.20,
        "medium": 0.35,
        "low": 0.35,
        "very_low": 0.10,
    }
    sampled_frames: list[pd.DataFrame] = []
    for band, fraction in desired_band_fractions.items():
        band_df = df[df["score_band"] == band].copy()
        if band_df.empty:
            continue
        band_target = max(1, int(round(target_records * fraction)))
        families = sorted(band_df["predicted_relation_family"].dropna().unique())
        per_family = max(1, band_target // max(1, len(families)))
        family_frames: list[pd.DataFrame] = []
        for family in families:
            family_df = band_df[band_df["predicted_relation_family"] == family]
            family_df = family_df.sort_values(["_rand"])
            family_frames.append(family_df.head(per_family))
        band_sample = pd.concat(family_frames, ignore_index=True)
        if len(band_sample) < band_target:
            used = set(band_sample["sample_id"])
            filler = band_df[~band_df["sample_id"].isin(used)].sort_values("_rand")
            band_sample = pd.concat(
                [band_sample, filler.head(band_target - len(band_sample))],
                ignore_index=True,
            )
        sampled_frames.append(band_sample.head(band_target))

    sampled = (
        pd.concat(sampled_frames, ignore_index=True)
        if sampled_frames
        else df.iloc[0:0].copy()
    )
    if len(sampled) < target_records:
        used = set(sampled["sample_id"])
        filler = df[~df["sample_id"].isin(used)].sort_values("_rand")
        sampled = pd.concat(
            [sampled, filler.head(target_records - len(sampled))],
            ignore_index=True,
        )
    sampled = sampled.head(target_records).copy()
    return sampled[RELATION_GOLD_RAW_PREDICTION_COLUMNS].to_dict("records")


def build_relation_gold_chunk_records(
    sampled_chunks_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    *,
    split: str,
) -> list[Dict[str, Any]]:
    """Build chunk-frame JSONL records for future recall-oriented annotation."""
    if sampled_chunks_df.empty:
        return []
    chunks = chunks_df.set_index("chunk_id").to_dict("index")
    records: list[Dict[str, Any]] = []
    for row in sampled_chunks_df.to_dict("records"):
        chunk_id = str(row.get("chunk_id") or "")
        chunk = chunks.get(chunk_id, {})
        records.append(
            {
                "sample_id": stable_id("relation_gold_chunk", split, chunk_id),
                "split": split,
                "chunk_id": chunk_id,
                "document_id": row.get("document_id") or chunk.get("document_id"),
                "dataset": row.get("dataset") or chunk.get("dataset"),
                "modality": row.get("modality") or chunk.get("modality"),
                "text": chunk.get("chunk_text", ""),
                "entity_count": int(row.get("entity_count") or 0),
                "valid_schema_pair_count": int(
                    row.get("valid_schema_pair_count") or 0
                ),
                "entity_types": row.get("entity_types") or [],
                "relation_families": row.get("relation_families") or [],
                "has_refined_entity": bool(row.get("has_refined_entity")),
                "has_local_entity": bool(row.get("has_local_entity")),
                "has_refined_local_pair": bool(row.get("has_refined_local_pair")),
            }
        )
    return records


def write_jsonl_records(
    records: Sequence[Mapping[str, Any]],
    output_jsonl: str | Path,
) -> Path:
    """Write records to JSONL with stable UTF-8 encoding."""
    path = Path(output_jsonl)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    return path


def summarize_relation_gold_records(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Summarize raw relation gold candidates for quality checks."""
    if not records:
        return {
            "records": 0,
            "chunks": 0,
            "documents": 0,
            "relation_types": 0,
            "relation_families": 0,
            "datasets": 0,
            "modalities": 0,
            "score_bands": {},
            "entity_source_mixes": {},
        }
    df = pd.DataFrame([dict(record) for record in records])
    return {
        "records": int(len(df)),
        "chunks": int(df["chunk_id"].nunique(dropna=True)),
        "documents": int(df["document_id"].nunique(dropna=True)),
        "relation_types": int(df["predicted_relation_type"].nunique(dropna=True)),
        "relation_families": int(
            df["predicted_relation_family"].nunique(dropna=True)
        ),
        "datasets": int(df["dataset"].nunique(dropna=True)),
        "modalities": int(df["modality"].nunique(dropna=True)),
        "score_bands": df["score_band"].value_counts(dropna=False).to_dict(),
        "entity_source_mixes": df["entity_source_mix"]
        .value_counts(dropna=False)
        .to_dict(),
    }


def _find_prediction_endpoint(
    prediction: Mapping[str, Any],
    glirel_input: Mapping[str, Any],
    prefix: str,
) -> Optional[Mapping[str, Any]]:
    """Map GLiREL token positions back to our canonical mention record."""
    pos = prediction.get(f"{prefix}_pos")
    if isinstance(pos, (str, bytes)) or not isinstance(pos, Sequence) or len(pos) < 2:
        return None
    start_token = int(pos[0])
    raw_end_token = int(pos[1])
    candidate_end_tokens = [raw_end_token]
    if raw_end_token > start_token:
        candidate_end_tokens.append(raw_end_token - 1)
    for record in glirel_input.get("mention_records", []):
        if int(record.get("token_start")) == start_token and int(record.get("token_end")) in candidate_end_tokens:
            return record
    return None
