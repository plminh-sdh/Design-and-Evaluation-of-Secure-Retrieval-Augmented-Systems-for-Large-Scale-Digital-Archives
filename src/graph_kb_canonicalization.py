"""Entity canonicalization table/export helpers for the graph knowledge base."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm

from src.archive_schema import append_dataframe_to_csv, stable_id
from src.graph_kb_skeleton import GRAPH_JSON_COLUMNS


CANONICALIZATION_EXPORT_DIR = Path("data") / "graph_kb_exports" / "step_03_canonicalization"
CANONICALIZATION_PROGRESS_FILENAME = "mention_canonicalization_progress.csv"
ENTITY_TABLE_COLUMNS = [
    "entity_id:ID(Entity)",
    "canonical_name",
    "entity_type",
    "external_kb_id",
    "external_kb",
    "wikipedia_entity_title",
    "aliases",
    "link_score:float",
    "canonicalization_method",
    "canonicalizer",
    "model_name",
    "model_version",
    ":LABEL",
]
MENTION_ENTITY_EDGE_COLUMNS = [
    ":START_ID(Mention)",
    ":END_ID(Entity)",
    "confidence:float",
    "canonicalization_method",
    "canonicalizer",
    "model_name",
    ":TYPE",
]
MENTION_CANONICALIZATION_COLUMNS = [
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
    "external_kb_id",
    "wikipedia_entity_title",
    "canonical_name",
    "link_score:float",
    "canonicalization_status",
    "canonicalization_method",
    "canonicalizer",
    "model_name",
    "model_version",
]
CANONICALIZATION_PROGRESS_COLUMNS = [
    "mention_id",
    "canonicalizer",
    "model_name",
    "model_version",
]


def _first_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            value = obj[name]
            return value() if callable(value) else value
        if hasattr(obj, name):
            value = getattr(obj, name)
            return value() if callable(value) else value
    return default


def _is_nil(value: Any) -> bool:
    return value is None or str(value).strip().upper() in {"", "NIL", "NONE", "Q-1"}


def _clean_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mention_id_column(mentions_df: pd.DataFrame) -> str:
    if "mention_id:ID(Mention)" in mentions_df.columns:
        return "mention_id:ID(Mention)"
    if "mention_id" in mentions_df.columns:
        return "mention_id"
    raise KeyError("mentions table must contain mention_id:ID(Mention) or mention_id")


def _chunk_id_column(chunks_df: pd.DataFrame) -> str:
    if "chunk_id:ID(Chunk)" in chunks_df.columns:
        return "chunk_id:ID(Chunk)"
    if "chunk_id" in chunks_df.columns:
        return "chunk_id"
    raise KeyError("chunks table must contain chunk_id:ID(Chunk) or chunk_id")


def _remaining_mention_export_stats(
    mentions_csv: Path,
    *,
    completed_mention_ids: set[str],
    chunksize: int,
) -> Dict[str, int]:
    remaining_mentions = 0
    remaining_chunk_ids: set[str] = set()

    for mentions_df in pd.read_csv(
        mentions_csv,
        usecols=lambda column: column in {
            "mention_id:ID(Mention)",
            "mention_id",
            "chunk_id",
        },
        chunksize=chunksize,
    ):
        mention_id_col = _mention_id_column(mentions_df)
        mentions_df = mentions_df.rename(columns={mention_id_col: "mention_id"})
        mentions_df["mention_id"] = mentions_df["mention_id"].astype(str)

        if completed_mention_ids:
            mentions_df = mentions_df[
                ~mentions_df["mention_id"].isin(completed_mention_ids)
            ]
        if mentions_df.empty:
            continue

        remaining_mentions += len(mentions_df)
        remaining_chunk_ids.update(mentions_df["chunk_id"].dropna().astype(str))

    return {
        "remaining_mentions": remaining_mentions,
        "remaining_archive_chunks": len(remaining_chunk_ids),
    }


def _span_for_mention(row: Mapping[str, Any]) -> Any:
    from refined.data_types.base_types import Span

    start = int(row["start_char:int"])
    end = int(row["end_char:int"])
    return Span(
        text=str(row.get("mention_text") or ""),
        start=start,
        ln=end - start,
        coarse_type="MENTION",
        coarse_mention_type=row.get("mention_type"),
    )


def _entity_from_refined_span(span: Any) -> Dict[str, Any]:
    entity = _first_attr(span, "predicted_entity", "entity", "linked_entity")
    qid = _first_attr(span, "wikidata_entity_id", "wikidata_id", "qid")
    title = _first_attr(span, "wikipedia_entity_title", "wikipedia_title", "title")

    if entity is not None:
        qid = qid or _first_attr(entity, "wikidata_entity_id", "wikidata_id", "qid")
        title = title or _first_attr(entity, "wikipedia_entity_title", "wikipedia_title", "title")

    score = _first_attr(
        span,
        "entity_linking_model_confidence_score",
        "score",
        "confidence",
        "link_score",
    )
    return {
        "external_kb_id": None if _is_nil(qid) else str(qid),
        "wikipedia_entity_title": title,
        "link_score": _clean_score(score),
    }


def load_completed_canonicalization_mention_ids(
    export_dir: str | Path = CANONICALIZATION_EXPORT_DIR,
    *,
    canonicalizer: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
) -> set[str]:
    """Load mention IDs already checkpointed by incremental canonicalization."""
    progress_path = Path(export_dir) / CANONICALIZATION_PROGRESS_FILENAME
    if not progress_path.exists():
        return set()

    progress_df = pd.read_csv(progress_path, dtype={"mention_id": "string"})
    for column, expected_value in {
        "canonicalizer": canonicalizer,
        "model_name": model_name,
        "model_version": model_version,
    }.items():
        if expected_value is not None and column in progress_df.columns:
            progress_df = progress_df[progress_df[column] == expected_value]
    return set(progress_df["mention_id"].dropna().astype(str))


def load_existing_entity_ids(export_dir: str | Path = CANONICALIZATION_EXPORT_DIR) -> set[str]:
    """Load entity IDs already written to the entity export."""
    entity_path = Path(export_dir) / "entities.csv"
    if not entity_path.exists():
        return set()
    entity_df = pd.read_csv(entity_path, usecols=["entity_id:ID(Entity)"])
    return set(entity_df["entity_id:ID(Entity)"].dropna().astype(str))


def load_canonicalization_export_tables(
    export_dir: str | Path = CANONICALIZATION_EXPORT_DIR,
) -> Dict[str, pd.DataFrame]:
    """Load existing canonicalization export tables from CSV files."""
    export_path = Path(export_dir)
    return {
        table_name: pd.read_csv(export_path / f"{table_name}.csv")
        if (export_path / f"{table_name}.csv").exists()
        else pd.DataFrame()
        for table_name in ["entities", "mention_refers_to_entity", "mention_canonicalization"]
    }


def build_canonicalization_tables(
    mention_rows: Sequence[Mapping[str, Any]],
    refined_spans: Sequence[Any],
    *,
    canonicalizer: str,
    model_name: str,
    model_version: str,
    canonicalization_method: str = "ReFinED",
    min_link_score: float = 0.0,
    existing_entity_ids: Optional[set[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Build Neo4j-ready Entity nodes and Mention-REFERS_TO relationships."""
    existing_entity_ids = existing_entity_ids or set()
    batch_entity_ids: set[str] = set()
    entity_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    for mention_row, span in zip(mention_rows, refined_spans):
        mention_id = str(mention_row["mention_id"])
        entity_prediction = _entity_from_refined_span(span)
        qid = entity_prediction["external_kb_id"]
        title = entity_prediction["wikipedia_entity_title"]
        link_score = entity_prediction["link_score"]
        linked = qid is not None and (link_score is None or link_score >= min_link_score)
        entity_id = stable_id("entity", "wikidata", qid) if linked else None
        canonical_name = title or (str(mention_row.get("mention_text") or "") if linked else None)

        if linked and entity_id not in existing_entity_ids and entity_id not in batch_entity_ids:
            entity_rows.append(
                {
                    "entity_id:ID(Entity)": entity_id,
                    "canonical_name": canonical_name,
                    "entity_type": mention_row.get("mention_type"),
                    "external_kb_id": qid,
                    "external_kb": "wikidata",
                    "wikipedia_entity_title": title,
                    "aliases": [mention_row.get("mention_text")],
                    "link_score:float": link_score,
                    "canonicalization_method": canonicalization_method,
                    "canonicalizer": canonicalizer,
                    "model_name": model_name,
                    "model_version": model_version,
                    ":LABEL": "Entity",
                }
            )
            batch_entity_ids.add(entity_id)

        if linked:
            edge_rows.append(
                {
                    ":START_ID(Mention)": mention_id,
                    ":END_ID(Entity)": entity_id,
                    "confidence:float": link_score,
                    "canonicalization_method": canonicalization_method,
                    "canonicalizer": canonicalizer,
                    "model_name": model_name,
                    ":TYPE": "REFERS_TO",
                }
            )

        audit_rows.append(
            {
                "mention_id": mention_id,
                "chunk_id": mention_row.get("chunk_id"),
                "document_id": mention_row.get("document_id"),
                "dataset": mention_row.get("dataset"),
                "modality": mention_row.get("modality"),
                "mention_text": mention_row.get("mention_text"),
                "mention_type": mention_row.get("mention_type"),
                "start_char:int": mention_row.get("start_char:int"),
                "end_char:int": mention_row.get("end_char:int"),
                "entity_id": entity_id,
                "external_kb_id": qid,
                "wikipedia_entity_title": title,
                "canonical_name": canonical_name,
                "link_score:float": link_score,
                "canonicalization_status": "LINKED" if linked else "NIL",
                "canonicalization_method": canonicalization_method,
                "canonicalizer": canonicalizer,
                "model_name": model_name,
                "model_version": model_version,
            }
        )

    return {
        "entities": pd.DataFrame(entity_rows, columns=ENTITY_TABLE_COLUMNS),
        "mention_refers_to_entity": pd.DataFrame(edge_rows, columns=MENTION_ENTITY_EDGE_COLUMNS),
        "mention_canonicalization": pd.DataFrame(audit_rows, columns=MENTION_CANONICALIZATION_COLUMNS),
    }


def append_canonicalization_tables(
    tables: Mapping[str, pd.DataFrame],
    export_dir: str | Path = CANONICALIZATION_EXPORT_DIR,
) -> Dict[str, Path]:
    """Append canonicalization tables to disk."""
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    exported_paths: Dict[str, Path] = {}

    for table_name, table in tables.items():
        csv_path = export_path / f"{table_name}.csv"
        exported_paths[table_name] = append_dataframe_to_csv(
            table,
            csv_path,
            json_columns=GRAPH_JSON_COLUMNS,
        )

    return exported_paths


def export_incremental_canonicalization_tables(
    mention_rows: Sequence[Mapping[str, Any]],
    refined_spans: Sequence[Any],
    *,
    canonicalizer: str,
    model_name: str,
    model_version: str,
    export_dir: str | Path = CANONICALIZATION_EXPORT_DIR,
    min_link_score: float = 0.0,
    existing_entity_ids: Optional[set[str]] = None,
) -> Dict[str, Path]:
    """Append one completed canonicalization batch and progress checkpoint."""
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    existing_entity_ids = existing_entity_ids or load_existing_entity_ids(export_path)

    tables = build_canonicalization_tables(
        mention_rows,
        refined_spans,
        canonicalizer=canonicalizer,
        model_name=model_name,
        model_version=model_version,
        min_link_score=min_link_score,
        existing_entity_ids=existing_entity_ids,
    )

    exported_paths: Dict[str, Path] = {}
    for table_name, table in tables.items():
        csv_path = export_path / f"{table_name}.csv"
        exported_paths[table_name] = append_dataframe_to_csv(
            table,
            csv_path,
            json_columns=GRAPH_JSON_COLUMNS,
        )

    progress_path = export_path / CANONICALIZATION_PROGRESS_FILENAME
    progress_rows = [
        {
            "mention_id": row["mention_id"],
            "canonicalizer": canonicalizer,
            "model_name": model_name,
            "model_version": model_version,
        }
        for row in mention_rows
    ]
    exported_paths["mention_canonicalization_progress"] = append_dataframe_to_csv(
        pd.DataFrame(progress_rows, columns=CANONICALIZATION_PROGRESS_COLUMNS),
        progress_path,
    )
    return exported_paths


def prepare_mentions_for_canonicalization(
    mentions_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    *,
    completed_mention_ids: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Join mention rows to chunk text and remove completed mention IDs."""
    completed_mention_ids = completed_mention_ids or set()
    mention_id_col = _mention_id_column(mentions_df)
    chunk_id_col = _chunk_id_column(chunks_df)
    chunk_columns = [
        chunk_id_col,
        "masked_text",
        "title",
        "summary",
    ]
    chunk_columns = [column for column in chunk_columns if column in chunks_df.columns]

    mentions = mentions_df.rename(columns={mention_id_col: "mention_id"}).copy()
    mentions["mention_id"] = mentions["mention_id"].astype(str)
    mentions = mentions[~mentions["mention_id"].isin(completed_mention_ids)].copy()

    chunks = chunks_df[chunk_columns].rename(columns={chunk_id_col: "chunk_id"}).copy()
    return mentions.merge(chunks, on="chunk_id", how="left")


def canonicalize_mentions_batch(
    refined_model: Any,
    mention_rows: Sequence[Mapping[str, Any]],
    *,
    text_field: str = "masked_text",
    max_batch_size: int = 16,
    ner_threshold: float = 0.5,
) -> List[Any]:
    """Canonicalize mention rows by passing their spans to ReFinED with chunk context."""
    if not mention_rows:
        return []

    text = str(mention_rows[0].get(text_field) or "")
    spans = [_span_for_mention(row) for row in mention_rows]
    predicted_spans = refined_model.process_text(
        text,
        spans=spans,
        ner_threshold=ner_threshold,
        max_batch_size=max_batch_size,
        return_special_spans=True,
    )
    predicted_by_position = {
        (_first_attr(span, "start"), _first_attr(span, "end")): span
        for span in predicted_spans
    }
    return [
        predicted_by_position.get(
            (int(row["start_char:int"]), int(row["end_char:int"])),
            spans[index],
        )
        for index, row in enumerate(mention_rows)
    ]


def run_refined_canonicalization_export(
    *,
    refined_model: Any,
    mentions_csv: str | Path,
    chunk_has_mention_csv: str | Path,
    chunks_csv: str | Path,
    export_dir: str | Path = CANONICALIZATION_EXPORT_DIR,
    canonicalizer: str = "ReFinED",
    model_name: str,
    model_version: str,
    min_link_score: float = 0.0,
    mentions_chunksize: int = 50_000,
    max_mentions: Optional[int] = None,
    resume_from_exports: bool = True,
    max_batch_size: int = 16,
    ner_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Canonicalize mention exports with ReFinED and append Neo4j-ready CSVs.

    The run is resumable at mention granularity via
    ``mention_canonicalization_progress.csv``.
    """
    mentions_csv = Path(mentions_csv)
    chunk_has_mention_csv = Path(chunk_has_mention_csv)
    chunks_csv = Path(chunks_csv)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src.refined_canonicalization import patch_refined_for_py312_windows

        patch_refined_for_py312_windows()
    except Exception:
        pass

    completed_mention_ids = (
        load_completed_canonicalization_mention_ids(
            export_dir,
            canonicalizer=canonicalizer,
            model_name=model_name,
            model_version=model_version,
        )
        if resume_from_exports
        else set()
    )
    existing_entity_ids = load_existing_entity_ids(export_dir)

    chunk_id_col = "chunk_id:ID(Chunk)"
    chunks_df = pd.read_csv(
        chunks_csv,
        usecols=lambda column: column in {
            chunk_id_col,
            "chunk_id",
            "masked_text",
            "title",
            "summary",
        },
    )
    if chunk_id_col not in chunks_df.columns and "chunk_id" in chunks_df.columns:
        chunks_df = chunks_df.rename(columns={"chunk_id": chunk_id_col})
    chunk_text_by_id = dict(
        zip(
            chunks_df[chunk_id_col].astype(str),
            chunks_df.get("masked_text", pd.Series(index=chunks_df.index, dtype="object")).fillna("").astype(str),
        )
    )

    chunk_has_mention_count: Optional[int] = None

    processed_mentions = 0
    linked_mentions = 0
    nil_mentions = 0
    error_chunks = 0
    processed_archive_chunks = 0
    last_exported_paths: Dict[str, Path] = {}

    remaining_stats = _remaining_mention_export_stats(
        mentions_csv,
        completed_mention_ids=completed_mention_ids,
        chunksize=mentions_chunksize,
    )
    remaining_mention_estimate = remaining_stats["remaining_mentions"]
    remaining_archive_chunk_estimate = remaining_stats["remaining_archive_chunks"]
    progress_total = (
        min(max_mentions, remaining_mention_estimate)
        if max_mentions is not None
        else remaining_mention_estimate
    )

    mention_reader = pd.read_csv(mentions_csv, chunksize=mentions_chunksize)
    with tqdm(
        total=progress_total,
        desc="Canonicalizing mentions with ReFinED",
        unit="mention",
    ) as progress:
        progress.set_postfix(
            archive_chunks=f"0/{remaining_archive_chunk_estimate:,}",
            linked=linked_mentions,
            nil=nil_mentions,
            errors=error_chunks,
        )
        for mentions_chunk_df in mention_reader:
            mention_id_col = _mention_id_column(mentions_chunk_df)
            mentions_chunk_df = mentions_chunk_df.rename(columns={mention_id_col: "mention_id"})
            mentions_chunk_df["mention_id"] = mentions_chunk_df["mention_id"].astype(str)

            if completed_mention_ids:
                mentions_chunk_df = mentions_chunk_df[
                    ~mentions_chunk_df["mention_id"].isin(completed_mention_ids)
                ].copy()
            if mentions_chunk_df.empty:
                continue

            if max_mentions is not None:
                remaining = max_mentions - processed_mentions
                if remaining <= 0:
                    break
                mentions_chunk_df = mentions_chunk_df.head(remaining)

            mentions_chunk_df["masked_text"] = mentions_chunk_df["chunk_id"].astype(str).map(chunk_text_by_id)
            mentions_chunk_df = mentions_chunk_df[mentions_chunk_df["masked_text"].notna()].copy()

            for chunk_id, group_df in mentions_chunk_df.groupby("chunk_id", sort=False):
                mention_rows = group_df.to_dict(orient="records")
                try:
                    refined_spans = canonicalize_mentions_batch(
                        refined_model,
                        mention_rows,
                        max_batch_size=max_batch_size,
                        ner_threshold=ner_threshold,
                    )
                    tables = build_canonicalization_tables(
                        mention_rows,
                        refined_spans,
                        canonicalizer=canonicalizer,
                        model_name=model_name,
                        model_version=model_version,
                        min_link_score=min_link_score,
                        existing_entity_ids=existing_entity_ids,
                    )
                    last_exported_paths = append_canonicalization_tables(tables, export_dir)
                    progress_path = export_dir / CANONICALIZATION_PROGRESS_FILENAME
                    progress_rows = [
                        {
                            "mention_id": row["mention_id"],
                            "canonicalizer": canonicalizer,
                            "model_name": model_name,
                            "model_version": model_version,
                        }
                        for row in mention_rows
                    ]
                    append_dataframe_to_csv(
                        pd.DataFrame(progress_rows, columns=CANONICALIZATION_PROGRESS_COLUMNS),
                        progress_path,
                    )
                    last_exported_paths["mention_canonicalization_progress"] = progress_path

                    if not tables["entities"].empty:
                        existing_entity_ids.update(
                            tables["entities"]["entity_id:ID(Entity)"].dropna().astype(str)
                        )
                    linked_mentions += int(
                        (tables["mention_canonicalization"]["canonicalization_status"] == "LINKED").sum()
                    )
                    nil_mentions += int(
                        (tables["mention_canonicalization"]["canonicalization_status"] == "NIL").sum()
                    )
                    processed_mentions += len(mention_rows)
                    processed_archive_chunks += 1
                    completed_mention_ids.update(str(row["mention_id"]) for row in mention_rows)
                    progress.update(len(mention_rows))
                    progress.set_postfix(
                        archive_chunks=f"{processed_archive_chunks:,}/{remaining_archive_chunk_estimate:,}",
                        linked=linked_mentions,
                        nil=nil_mentions,
                        errors=error_chunks,
                    )
                except Exception:
                    error_chunks += 1
                    progress_path = export_dir / CANONICALIZATION_PROGRESS_FILENAME
                    append_dataframe_to_csv(
                        pd.DataFrame(
                            [
                                {
                                    "mention_id": row["mention_id"],
                                    "canonicalizer": canonicalizer,
                                    "model_name": model_name,
                                    "model_version": model_version,
                                }
                                for row in mention_rows
                            ],
                            columns=CANONICALIZATION_PROGRESS_COLUMNS,
                        ),
                        progress_path,
                    )
                    processed_mentions += len(mention_rows)
                    processed_archive_chunks += 1
                    completed_mention_ids.update(str(row["mention_id"]) for row in mention_rows)
                    progress.update(len(mention_rows))
                    progress.set_postfix(
                        archive_chunks=f"{processed_archive_chunks:,}/{remaining_archive_chunk_estimate:,}",
                        linked=linked_mentions,
                        nil=nil_mentions,
                        errors=error_chunks,
                    )

            if max_mentions is not None and processed_mentions >= max_mentions:
                break

    return {
        "processed_mentions": processed_mentions,
        "linked_mentions": linked_mentions,
        "nil_mentions": nil_mentions,
        "error_chunks": error_chunks,
        "completed_mentions_total": len(completed_mention_ids),
        "chunk_has_mention_rows": chunk_has_mention_count,
        "exported_paths": last_exported_paths,
    }
