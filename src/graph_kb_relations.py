"""Typed relation extraction helpers for the graph knowledge-base pipeline.

The helpers in this module recover chunk-local, canonicalized entity mentions
from the existing CSV exports and prepare constrained GLiREL inputs. They are
designed so notebook cells can run a small sample first, while later production
cells can reuse the same loaders and predictors in resumable chunk batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
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
RELATION_GLIREL_LABEL_BY_GRAPH_TYPE = {
    entry.graph_relation: entry.glirel_label for entry in RELATION_SCHEMA
}
ALLOWED_DIRECTED_TYPE_PAIRS = {
    (head_type, tail_type)
    for entry in RELATION_SCHEMA
    for head_type in entry.allowed_head
    for tail_type in entry.allowed_tail
}
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
    "input_id",
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
    "input_id",
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


def align_char_span_to_token_span_fast(
    token_starts: Sequence[int],
    token_ends: Sequence[int],
    start_char: int,
    end_char: int,
) -> Optional[tuple[int, int]]:
    """Map character offsets to inclusive token offsets using precomputed token bounds."""
    if end_char <= start_char or not token_starts:
        return None
    start_token = bisect_right(token_ends, int(start_char))
    end_token = bisect_left(token_starts, int(end_char)) - 1
    if start_token < 0 or end_token < start_token or end_token >= len(token_starts):
        return None
    return start_token, end_token


def build_glirel_chunk_input(
    chunk_row: Mapping[str, Any],
    mentions_df: pd.DataFrame,
    nlp: Any,
    *,
    max_entities_per_chunk: int = 40,
    dedupe_entity_spans: bool = True,
    mentions_are_pre_filtered: bool = False,
) -> Optional[Dict[str, Any]]:
    """Build one direct-call GLiREL input from chunk text and resolved mentions."""
    chunk_id = str(chunk_row.get("chunk_id") or "")
    text = str(chunk_row.get("chunk_text") or "")
    if not chunk_id or not text:
        return None

    doc = nlp.make_doc(text) if hasattr(nlp, "make_doc") else nlp(text)
    tokens = [token.text for token in doc]
    token_starts = [int(token.idx) for token in doc]
    token_ends = [int(token.idx) + len(token.text) for token in doc]
    if mentions_are_pre_filtered:
        mentions = mentions_df
    else:
        mentions = mentions_df[mentions_df["chunk_id"].astype(str) == chunk_id].copy()
    if mentions.empty:
        return None

    seen_entity_spans: set[tuple[str, int, int]] = set()
    ner: list[list[Any]] = []
    mention_records: list[Dict[str, Any]] = []
    for row in mentions.itertuples(index=False):
        start_char = int(getattr(row, "start_char"))
        end_char = int(getattr(row, "end_char"))
        token_span = align_char_span_to_token_span_fast(
            token_starts,
            token_ends,
            start_char,
            end_char,
        )
        if token_span is None:
            continue
        start_token, end_token = token_span
        entity_id = str(getattr(row, "entity_id"))
        dedupe_key = (entity_id, start_token, end_token)
        if dedupe_entity_spans and dedupe_key in seen_entity_spans:
            continue
        seen_entity_spans.add(dedupe_key)
        mention_text = str(getattr(row, "mention_text") or "")
        mention_type = str(getattr(row, "mention_type") or "").upper()
        ner.append([start_token, end_token, mention_type, mention_text])
        record = row._asdict()
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
    if chunks_df.empty or mentions_df.empty:
        return inputs

    sorted_mentions = mentions_df.sort_values(
        ["chunk_id", "start_char", "end_char", "mention_id"],
        kind="mergesort",
    )
    mention_groups = {
        str(chunk_id): group
        for chunk_id, group in sorted_mentions.groupby("chunk_id", sort=False)
    }
    for row in tqdm(
        chunks_df.to_dict("records"),
        desc="Building GLiREL chunk inputs",
        unit="chunk",
        disable=not verbose,
    ):
        chunk_mentions = mention_groups.get(str(row.get("chunk_id") or ""))
        if chunk_mentions is None or chunk_mentions.empty:
            continue
        item = build_glirel_chunk_input(
            row,
            chunk_mentions,
            nlp,
            max_entities_per_chunk=max_entities_per_chunk,
            mentions_are_pre_filtered=True,
        )
        if item is not None:
            inputs.append(item)
    return inputs


def _token_gap(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    """Return the number of tokens between two mention spans."""
    if int(left["token_end"]) < int(right["token_start"]):
        return int(right["token_start"]) - int(left["token_end"]) - 1
    if int(right["token_end"]) < int(left["token_start"]):
        return int(left["token_start"]) - int(right["token_end"]) - 1
    return 0


def _align_mentions_for_relation_chunk(
    mentions: pd.DataFrame,
    doc: Any,
    *,
    dedupe_entity_spans: bool = True,
) -> list[Dict[str, Any]]:
    token_starts = [int(token.idx) for token in doc]
    token_ends = [int(token.idx) + len(token.text) for token in doc]
    seen_entity_spans: set[tuple[str, int, int]] = set()
    mention_records: list[Dict[str, Any]] = []
    for row in mentions.itertuples(index=False):
        token_span = align_char_span_to_token_span_fast(
            token_starts,
            token_ends,
            int(getattr(row, "start_char")),
            int(getattr(row, "end_char")),
        )
        if token_span is None:
            continue
        start_token, end_token = token_span
        entity_id = str(getattr(row, "entity_id"))
        dedupe_key = (entity_id, start_token, end_token)
        if dedupe_entity_spans and dedupe_key in seen_entity_spans:
            continue
        seen_entity_spans.add(dedupe_key)
        record = row._asdict()
        record["mention_type"] = str(record.get("mention_type") or "").upper()
        record["token_start"] = start_token
        record["token_end"] = end_token
        mention_records.append(record)
    return mention_records


def build_glirel_pair_window_inputs(
    chunks_df: pd.DataFrame,
    mentions_df: pd.DataFrame,
    nlp: Any,
    *,
    max_pair_token_distance: int = 96,
    window_token_radius: int = 64,
    max_pairs_per_chunk: int = 80,
    max_entities_per_window: int = 8,
    max_windows: Optional[int] = None,
    verbose: bool = True,
) -> list[Dict[str, Any]]:
    """Build small pair-window GLiREL inputs for scalable relation extraction.

    Each output contains only one directed candidate pair, only the relation
    labels allowed by that pair's entity types, and a cropped token window around
    the two mentions. This avoids the quadratic all-entities-in-full-chunk
    payload used by the diagnostic/gold-sampling chunk builder.
    """
    inputs: list[Dict[str, Any]] = []
    if chunks_df.empty or mentions_df.empty:
        return inputs

    sorted_mentions = mentions_df.sort_values(
        ["chunk_id", "start_char", "end_char", "mention_id"],
        kind="mergesort",
    )
    mention_groups = {
        str(chunk_id): group
        for chunk_id, group in sorted_mentions.groupby("chunk_id", sort=False)
    }
    skipped_no_mentions = 0
    skipped_no_aligned_mentions = 0
    skipped_no_near_pairs = 0
    candidate_pairs_seen = 0

    for row in tqdm(
        chunks_df.to_dict("records"),
        desc="Building GLiREL pair-window inputs",
        unit="chunk",
        disable=not verbose,
    ):
        if max_windows is not None and len(inputs) >= max_windows:
            break
        chunk_id = str(row.get("chunk_id") or "")
        text = str(row.get("chunk_text") or "")
        chunk_mentions = mention_groups.get(chunk_id)
        if not text or chunk_mentions is None or chunk_mentions.empty:
            skipped_no_mentions += 1
            continue

        doc = nlp.make_doc(text) if hasattr(nlp, "make_doc") else nlp(text)
        tokens = [token.text for token in doc]
        mention_records = _align_mentions_for_relation_chunk(chunk_mentions, doc)
        if len(mention_records) < 2:
            skipped_no_aligned_mentions += 1
            continue

        candidate_pairs: list[tuple[int, Dict[str, Any], Dict[str, Any], list[str]]] = []
        for head in mention_records:
            head_type = str(head.get("mention_type") or "").upper()
            head_entity_id = str(head.get("entity_id"))
            for tail in mention_records:
                if head is tail or head_entity_id == str(tail.get("entity_id")):
                    continue
                tail_type = str(tail.get("mention_type") or "").upper()
                relation_types = relation_labels_for_type_pair(head_type, tail_type)
                if not relation_types:
                    continue
                gap = _token_gap(head, tail)
                if gap > max_pair_token_distance:
                    continue
                labels = [
                    RELATION_GLIREL_LABEL_BY_GRAPH_TYPE[relation_type]
                    for relation_type in relation_types
                    if relation_type in RELATION_GLIREL_LABEL_BY_GRAPH_TYPE
                ]
                if not labels:
                    continue
                candidate_pairs.append((gap, head, tail, labels))

        if not candidate_pairs:
            skipped_no_near_pairs += 1
            continue
        candidate_pairs.sort(
            key=lambda item: (
                item[0],
                int(item[1]["token_start"]),
                int(item[2]["token_start"]),
            )
        )
        candidate_pairs = candidate_pairs[:max_pairs_per_chunk]
        candidate_pairs_seen += len(candidate_pairs)

        windows: list[Dict[str, Any]] = []
        for gap, head, tail, labels in candidate_pairs:
            pair_start = min(int(head["token_start"]), int(tail["token_start"]))
            pair_end = max(int(head["token_end"]), int(tail["token_end"]))
            window_start = max(0, pair_start - window_token_radius)
            window_end = min(len(tokens) - 1, pair_end + window_token_radius)

            assigned = False
            for window in windows:
                # Bundle pairs whose desired windows overlap heavily. This
                # amortizes GLiREL's fixed per-call overhead without going back
                # to full dense chunks.
                if window_start <= window["end"] and window_end >= window["start"]:
                    existing_keys = {
                        str(record.get("mention_id"))
                        for record in window["mentions"]
                    }
                    new_mentions = [
                        record
                        for record in (head, tail)
                        if str(record.get("mention_id")) not in existing_keys
                    ]
                    if len(window["mentions"]) + len(new_mentions) <= max_entities_per_window:
                        window["start"] = min(window["start"], window_start)
                        window["end"] = max(window["end"], window_end)
                        window["labels"].update(labels)
                        window["pair_gaps"].append(gap)
                        window["pair_count"] += 1
                        window["mentions"].extend(dict(record) for record in new_mentions)
                        assigned = True
                        break
            if not assigned:
                windows.append(
                    {
                        "start": window_start,
                        "end": window_end,
                        "labels": set(labels),
                        "pair_gaps": [gap],
                        "pair_count": 1,
                        "mentions": [dict(head), dict(tail)],
                    }
                )

        for window in windows:
            if max_windows is not None and len(inputs) >= max_windows:
                break
            window_start = int(window["start"])
            window_end = int(window["end"])
            window_tokens = tokens[window_start : window_end + 1]
            if not window_tokens:
                continue

            mention_records: list[Dict[str, Any]] = []
            ner: list[list[Any]] = []
            for record in sorted(
                window["mentions"],
                key=lambda item: (int(item["token_start"]), int(item["token_end"])),
            ):
                adjusted_record = dict(record)
                adjusted_record["token_start"] = int(adjusted_record["token_start"]) - window_start
                adjusted_record["token_end"] = int(adjusted_record["token_end"]) - window_start
                adjusted_record["ner_index"] = len(ner)
                mention_records.append(adjusted_record)
                ner.append(
                    [
                        int(adjusted_record["token_start"]),
                        int(adjusted_record["token_end"]),
                        str(adjusted_record.get("mention_type") or "").upper(),
                        str(adjusted_record.get("mention_text") or ""),
                    ]
                )
            input_id = stable_id(
                "relation_pair_bundle_window",
                chunk_id,
                ",".join(str(record.get("mention_id")) for record in mention_records),
                ",".join(sorted(window["labels"])),
                window_start,
                window_end,
            )
            inputs.append(
                {
                    "input_id": input_id,
                    "chunk_id": chunk_id,
                    "document_id": row.get("document_id"),
                    "dataset": row.get("dataset"),
                    "modality": row.get("modality"),
                    "text": " ".join(window_tokens),
                    "source_text": text,
                    "window_start_token": window_start,
                    "window_end_token": window_end,
                    "pair_token_gap": min(window["pair_gaps"]) if window["pair_gaps"] else None,
                    "candidate_pair_count": int(window["pair_count"]),
                    "tokens": window_tokens,
                    "ner": ner,
                    "mention_records": mention_records,
                    "labels": sorted(window["labels"]),
                }
            )

    if verbose:
        tqdm.write(
            "Built GLiREL pair-window inputs: "
            f"{len(inputs):,} windows from {candidate_pairs_seen:,} candidate pairs; "
            f"{skipped_no_mentions:,} chunks without mentions/text, "
            f"{skipped_no_aligned_mentions:,} without aligned mentions, "
            f"{skipped_no_near_pairs:,} without nearby schema-valid pairs"
        )
    return inputs


def _chunk_has_allowed_relation_pair(mention_records: Sequence[Mapping[str, Any]]) -> bool:
    for head in mention_records:
        head_type = str(head.get("mention_type") or "").upper()
        head_entity_id = str(head.get("entity_id"))
        for tail in mention_records:
            if head is tail:
                continue
            if head_entity_id == str(tail.get("entity_id")):
                continue
            if (head_type, str(tail.get("mention_type") or "").upper()) in ALLOWED_DIRECTED_TYPE_PAIRS:
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
    labels = glirel_input.get("labels")
    if isinstance(labels, Mapping):
        labels = list(labels.get("glirel_labels", labels).keys())
    elif not labels:
        labels = glirel_label_list()
    return model.predict_relations(
        glirel_input["tokens"],
        list(labels),
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
    batch_size: int = 1,
) -> list[Dict[str, Any]]:
    """Run GLiREL on prepared inputs and attach chunk context to predictions.

    ``batch_size=1`` is the default because GLiREL's batch API can be slower for
    archive chunks with highly variable lengths: the padded batch work may cost
    more than repeated single-example calls. Increase only after measuring on the
    current hardware/model.
    """
    predictions: list[Dict[str, Any]] = []
    if not glirel_inputs:
        return predictions
    if batch_size <= 1 or not hasattr(model, "batch_predict_relations"):
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
                enriched["input_id"] = item.get("input_id")
                enriched["chunk_id"] = item.get("chunk_id")
                enriched["document_id"] = item.get("document_id")
                enriched["dataset"] = item.get("dataset")
                enriched["modality"] = item.get("modality")
                predictions.append(enriched)
        return predictions

    grouped_inputs: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for item in glirel_inputs:
        labels = item.get("labels")
        if isinstance(labels, Mapping):
            label_values = tuple(labels.get("glirel_labels", labels).keys())
        elif labels:
            label_values = tuple(str(label) for label in labels)
        else:
            label_values = tuple(glirel_label_list())
        grouped_inputs[tuple(sorted(label_values))].append(item)

    batch_specs: list[tuple[tuple[str, ...], list[Mapping[str, Any]]]] = []
    for label_values, items in grouped_inputs.items():
        sorted_items = sorted(
            items,
            key=lambda item: (len(item.get("tokens") or []), len(item.get("ner") or [])),
        )
        for start in range(0, len(sorted_items), batch_size):
            batch_specs.append((label_values, sorted_items[start : start + batch_size]))
    batch_specs.sort(
        key=lambda spec: (
            max(len(item.get("tokens") or []) for item in spec[1]),
            max(len(item.get("ner") or []) for item in spec[1]),
            len(spec[0]),
        )
    )

    if verbose:
        tqdm.write(
            "Running GLiREL batched prediction with "
            f"{len(grouped_inputs):,} label groups and {len(batch_specs):,} batches "
            f"(batch_size={batch_size})"
        )

    for label_values, batch_items in tqdm(
        batch_specs,
        desc="Running GLiREL relation prediction",
        unit="batch",
        disable=not verbose,
    ):
        batch_outputs = model.batch_predict_relations(
            [item["tokens"] for item in batch_items],
            list(label_values),
            threshold=threshold,
            ner=[item["ner"] for item in batch_items],
            top_k=top_k,
        )
        for item, item_predictions in zip(batch_items, batch_outputs):
            for prediction in item_predictions:
                glirel_label = str(prediction.get("label") or "")
                if glirel_label not in RELATION_BY_GLIREL_LABEL:
                    continue
                enriched = dict(prediction)
                enriched["input_id"] = item.get("input_id")
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
    input_by_id = {
        str(item.get("input_id")): item
        for item in glirel_inputs
        if item.get("input_id")
    }
    input_by_chunk: dict[str, Mapping[str, Any]] = {}
    for item in glirel_inputs:
        input_by_chunk.setdefault(str(item["chunk_id"]), item)
    rows: list[Dict[str, Any]] = []
    for prediction in raw_predictions:
        chunk_id = str(prediction.get("chunk_id") or "")
        input_id = str(prediction.get("input_id") or "")
        item = input_by_id.get(input_id) if input_id else None
        if item is None:
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
            input_id,
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
                "input_id": input_id,
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
    input_by_id = {
        str(item.get("input_id")): item
        for item in glirel_inputs
        if item.get("input_id")
    }
    input_by_chunk: dict[str, Mapping[str, Any]] = {}
    for item in glirel_inputs:
        input_by_chunk.setdefault(str(item["chunk_id"]), item)
    records: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for prediction in tqdm(
        raw_predictions,
        desc=f"Converting {split} GLiREL predictions",
        unit="prediction",
        disable=not verbose,
    ):
        chunk_id = str(prediction.get("chunk_id") or "")
        input_id = str(prediction.get("input_id") or "")
        item = input_by_id.get(input_id) if input_id else None
        if item is None:
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
            input_id,
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
                "input_id": input_id,
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
    for column in RELATION_GOLD_RAW_PREDICTION_COLUMNS:
        if column not in df.columns:
            df[column] = "" if column not in {"allowed_relation_types"} else [[] for _ in range(len(df))]
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


def build_relation_gold_annotation_chunk_records(
    sampled_chunks_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    glirel_inputs: Sequence[Mapping[str, Any]],
    raw_prediction_records: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> list[Dict[str, Any]]:
    """Build chunk-centered, model-assisted relation gold annotation records.

    The exported records use the selected chunk as the annotation unit. GLiREL
    predictions are grouped into ``model_predictions`` and copied into
    ``gold_relations`` as a draft that can be manually corrected by deleting
    false positives, changing labels/directions, and adding missed relations.
    """
    if sampled_chunks_df.empty:
        return []

    chunk_lookup = chunks_df.set_index("chunk_id").to_dict("index")
    inputs_by_chunk: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in glirel_inputs:
        inputs_by_chunk[str(item.get("chunk_id") or "")].append(item)

    predictions_by_chunk: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in raw_prediction_records:
        predictions_by_chunk[str(record.get("chunk_id") or "")].append(record)

    records: list[Dict[str, Any]] = []
    for row in sampled_chunks_df.to_dict("records"):
        chunk_id = str(row.get("chunk_id") or "")
        chunk = chunk_lookup.get(chunk_id, {})

        mentions_by_id: dict[str, Dict[str, Any]] = {}
        window_records: list[Dict[str, Any]] = []
        for item in inputs_by_chunk.get(chunk_id, []):
            window_records.append(
                {
                    "input_id": item.get("input_id"),
                    "text": item.get("text"),
                    "window_start_token": item.get("window_start_token"),
                    "window_end_token": item.get("window_end_token"),
                    "candidate_pair_count": item.get("candidate_pair_count"),
                    "labels": item.get("labels") or [],
                }
            )
            for mention in item.get("mention_records", []) or []:
                mention_id = str(mention.get("mention_id") or "")
                if not mention_id or mention_id in mentions_by_id:
                    continue
                mentions_by_id[mention_id] = {
                    "mention_id": mention.get("mention_id"),
                    "entity_id": mention.get("entity_id"),
                    "canonical_name": mention.get("entity_label")
                    or mention.get("canonical_name"),
                    "mention_text": mention.get("mention_text"),
                    "mention_type": mention.get("mention_type"),
                    "source": mention.get("source"),
                    "start_char": mention.get("start_char"),
                    "end_char": mention.get("end_char"),
                    "token_start": mention.get("token_start"),
                    "token_end": mention.get("token_end"),
                }

        model_predictions: list[Dict[str, Any]] = []
        seen_predictions: set[str] = set()
        for prediction in predictions_by_chunk.get(chunk_id, []):
            prediction_id = stable_id(
                "relation_gold_prediction",
                split,
                prediction.get("input_id"),
                chunk_id,
                prediction.get("head_mention_id"),
                prediction.get("tail_mention_id"),
                prediction.get("predicted_relation_type"),
            )
            if prediction_id in seen_predictions:
                continue
            seen_predictions.add(prediction_id)
            model_predictions.append(
                {
                    "relation_id": prediction_id,
                    "input_id": prediction.get("input_id"),
                    "head_mention_id": prediction.get("head_mention_id"),
                    "tail_mention_id": prediction.get("tail_mention_id"),
                    "head_entity_id": prediction.get("head_entity_id"),
                    "tail_entity_id": prediction.get("tail_entity_id"),
                    "head_text": prediction.get("head_text"),
                    "tail_text": prediction.get("tail_text"),
                    "head_canonical_name": prediction.get("head_canonical_name"),
                    "tail_canonical_name": prediction.get("tail_canonical_name"),
                    "head_type": prediction.get("head_type"),
                    "tail_type": prediction.get("tail_type"),
                    "relation_type": prediction.get("predicted_relation_type"),
                    "relation_family": prediction.get("predicted_relation_family"),
                    "glirel_label": prediction.get("predicted_glirel_label"),
                    "score": prediction.get("predicted_score"),
                    "score_band": prediction.get("score_band"),
                    "allowed_relation_types": prediction.get("allowed_relation_types")
                    or [],
                    "evidence_text": prediction.get("text"),
                    "correction_notes": "",
                }
            )

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
                "mentions": sorted(
                    mentions_by_id.values(),
                    key=lambda mention: (
                        int(mention.get("start_char") or 0),
                        str(mention.get("mention_id") or ""),
                    ),
                ),
                "glirel_windows": window_records,
                "model_predictions": model_predictions,
                "gold_relations": [dict(prediction) for prediction in model_predictions],
                "annotation_notes": "",
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


def glirel_inputs_to_diagnostic_records(
    glirel_inputs: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> list[Dict[str, Any]]:
    """Convert prepared GLiREL inputs to compact diagnostic JSONL records."""
    records: list[Dict[str, Any]] = []
    for item in glirel_inputs:
        mention_records = list(item.get("mention_records") or [])
        entity_ids = {str(record.get("entity_id")) for record in mention_records}
        type_pairs: set[str] = set()
        allowed_relation_types: set[str] = set()
        for head in mention_records:
            for tail in mention_records:
                if str(head.get("entity_id")) == str(tail.get("entity_id")):
                    continue
                head_type = str(head.get("mention_type") or "").upper()
                tail_type = str(tail.get("mention_type") or "").upper()
                relation_types = relation_labels_for_type_pair(head_type, tail_type)
                if not relation_types:
                    continue
                type_pairs.add(f"{head_type}->{tail_type}")
                allowed_relation_types.update(relation_types)
        records.append(
            {
                "sample_id": stable_id(
                    "relation_glirel_input_diagnostic",
                    split,
                    item.get("input_id"),
                    item.get("chunk_id"),
                ),
                "split": split,
                "input_id": item.get("input_id"),
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "dataset": item.get("dataset"),
                "modality": item.get("modality"),
                "token_count": len(item.get("tokens") or []),
                "ner_count": len(item.get("ner") or []),
                "unique_entity_count": len(entity_ids),
                "window_start_token": item.get("window_start_token"),
                "window_end_token": item.get("window_end_token"),
                "pair_token_gap": item.get("pair_token_gap"),
                "allowed_type_pairs": sorted(type_pairs),
                "allowed_relation_types": sorted(allowed_relation_types),
                "text": item.get("text"),
                "tokens": item.get("tokens") or [],
                "ner": item.get("ner") or [],
                "mentions": [
                    {
                        "mention_id": record.get("mention_id"),
                        "entity_id": record.get("entity_id"),
                        "canonical_name": record.get("entity_label")
                        or record.get("canonical_name"),
                        "mention_text": record.get("mention_text"),
                        "mention_type": record.get("mention_type"),
                        "source": record.get("source"),
                        "start_char": record.get("start_char"),
                        "end_char": record.get("end_char"),
                        "token_start": record.get("token_start"),
                        "token_end": record.get("token_end"),
                    }
                    for record in mention_records
                ],
            }
        )
    return records


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


def read_relation_gold_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    """Read chunk-centered relation gold JSONL records."""
    records: list[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _relation_edge_key(relation: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the graph-edge key used for relation threshold evaluation."""
    return (
        str(relation.get("head_entity_id") or ""),
        str(relation.get("tail_entity_id") or ""),
        str(relation.get("relation_type") or "").upper(),
    )


def _relation_group_value(
    record: Mapping[str, Any],
    relation: Mapping[str, Any],
    group_field: str,
) -> Any:
    if group_field in relation:
        return relation.get(group_field)
    if group_field == "relation_family":
        return relation.get("relation_family") or RELATION_FAMILY_BY_TYPE.get(
            str(relation.get("relation_type") or "").upper(),
            "other",
        )
    return record.get(group_field)


def relation_gold_summary(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Summarize corrected chunk-centered relation gold records."""
    gold = [
        relation
        for record in records
        for relation in (record.get("gold_relations") or [])
    ]
    predictions = [
        relation
        for record in records
        for relation in (record.get("model_predictions") or [])
    ]
    gold_families = Counter(
        relation.get("relation_family")
        or RELATION_FAMILY_BY_TYPE.get(str(relation.get("relation_type") or ""), "other")
        for relation in gold
    )
    gold_types = Counter(str(relation.get("relation_type") or "") for relation in gold)
    return {
        "chunks": int(len(records)),
        "chunks_with_gold": int(
            sum(bool(record.get("gold_relations")) for record in records)
        ),
        "chunks_with_model_predictions": int(
            sum(bool(record.get("model_predictions")) for record in records)
        ),
        "gold_relations": int(len(gold)),
        "model_predictions": int(len(predictions)),
        "gold_relation_types": int(len(gold_types)),
        "gold_relation_families": int(len(gold_families)),
        "gold_by_type": dict(gold_types.most_common()),
        "gold_by_family": dict(gold_families.most_common()),
        "datasets": dict(Counter(record.get("dataset") for record in records).most_common()),
        "modalities": dict(
            Counter(record.get("modality") for record in records).most_common()
        ),
    }


def _deduplicate_relation_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    """Keep the highest-score prediction per directed entity/relation key."""
    best_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for prediction in predictions:
        try:
            score = float(prediction.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score < threshold:
            continue
        key = _relation_edge_key(prediction)
        if not all(key):
            continue
        current = best_by_key.get(key)
        current_score = float(current.get("score") or 0.0) if current else -1.0
        if current is None or score > current_score:
            best_by_key[key] = prediction
    return best_by_key


def evaluate_relation_gold_threshold(
    records: Sequence[Mapping[str, Any]],
    threshold: float,
    *,
    group_field: str | None = None,
) -> Dict[str, Any]:
    """Evaluate GLiREL predictions against corrected relation gold at one threshold."""
    gold_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    prediction_candidates: list[Mapping[str, Any]] = []
    gold_group_by_key: dict[tuple[str, str, str], Any] = {}
    prediction_group_by_key: dict[tuple[str, str, str], Any] = {}

    for record in records:
        for relation in record.get("gold_relations") or []:
            key = _relation_edge_key(relation)
            if all(key):
                gold_by_key.setdefault(key, relation)
                if group_field is not None:
                    gold_group_by_key.setdefault(
                        key,
                        _relation_group_value(record, relation, group_field),
                    )
        for prediction in record.get("model_predictions") or []:
            prediction_candidates.append(prediction)
            if group_field is not None:
                key = _relation_edge_key(prediction)
                if all(key):
                    prediction_group_by_key.setdefault(
                        key,
                        _relation_group_value(record, prediction, group_field),
                    )

    predictions_by_key = _deduplicate_relation_predictions(
        prediction_candidates,
        threshold=threshold,
    )
    gold_keys = set(gold_by_key)
    predicted_keys = set(predictions_by_key)
    tp_keys = gold_keys & predicted_keys
    fp_keys = predicted_keys - gold_keys
    fn_keys = gold_keys - predicted_keys

    if group_field is not None:
        group_values = set()
        for key in gold_keys:
            group_values.add(gold_group_by_key.get(key))
        for key in predicted_keys:
            group_values.add(prediction_group_by_key.get(key))
        rows: list[Dict[str, Any]] = []
        for group_value in sorted(group_values, key=lambda value: str(value)):
            group_gold = {
                key
                for key in gold_keys
                if gold_group_by_key.get(key) == group_value
            }
            group_predicted = {
                key
                for key in predicted_keys
                if prediction_group_by_key.get(key) == group_value
            }
            group_tp = group_gold & group_predicted
            group_fp = group_predicted - group_gold
            group_fn = group_gold - group_predicted
            precision = len(group_tp) / len(group_predicted) if group_predicted else 0.0
            recall = len(group_tp) / len(group_gold) if group_gold else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            rows.append(
                {
                    "threshold": float(threshold),
                    group_field: group_value,
                    "predicted": int(len(group_predicted)),
                    "gold": int(len(group_gold)),
                    "tp": int(len(group_tp)),
                    "fp": int(len(group_fp)),
                    "fn": int(len(group_fn)),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
            )
        return {"groups": rows}

    precision = len(tp_keys) / len(predicted_keys) if predicted_keys else 0.0
    recall = len(tp_keys) / len(gold_keys) if gold_keys else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "chunks": int(len(records)),
        "gold": int(len(gold_keys)),
        "predicted": int(len(predicted_keys)),
        "tp": int(len(tp_keys)),
        "fp": int(len(fp_keys)),
        "fn": int(len(fn_keys)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def tune_relation_extraction_thresholds(
    records: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
    *,
    group_fields: Sequence[str] = ("relation_family", "relation_type", "dataset", "modality"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate relation extraction over thresholds overall and by group."""
    overall_rows = [
        evaluate_relation_gold_threshold(records, float(threshold))
        for threshold in thresholds
    ]
    group_rows: list[Dict[str, Any]] = []
    for threshold in thresholds:
        for group_field in group_fields:
            grouped = evaluate_relation_gold_threshold(
                records,
                float(threshold),
                group_field=group_field,
            )
            for row in grouped["groups"]:
                row["group_field"] = group_field
                row["group_value"] = row.get(group_field)
                group_rows.append(row)
    return pd.DataFrame(overall_rows), pd.DataFrame(group_rows)


def select_relation_threshold(
    metrics_df: pd.DataFrame,
    *,
    precision_floor: float = 0.80,
) -> pd.Series:
    """Select highest-recall relation threshold meeting the precision floor."""
    eligible = metrics_df[metrics_df["precision"] >= precision_floor].copy()
    if not eligible.empty:
        return eligible.sort_values(
            ["recall", "f1", "precision", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
    return metrics_df.sort_values(
        ["f1", "precision", "recall", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]


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
