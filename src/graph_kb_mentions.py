"""Mention extraction table/export helpers for the graph knowledge base."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from src.archive_schema import ArchiveChunk, dataframe_to_csv, stable_id
from src.graph_kb_skeleton import GRAPH_JSON_COLUMNS


MENTION_EXPORT_DIR = Path("data") / "graph_kb_exports" / "step_02_mentions"
MENTION_PROGRESS_FILENAME = "mention_extraction_progress.csv"
MENTION_TABLE_COLUMNS = [
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
    "extractor",
    "model_name",
    "label_set_version",
    "sensitivity_level",
    "access_level",
    ":LABEL",
]
CHUNK_MENTION_EDGE_COLUMNS = [
    ":START_ID(Chunk)",
    ":END_ID(Mention)",
    "confidence:float",
    "extractor",
    ":TYPE",
]


def _chunk_value(chunk: ArchiveChunk | Mapping[str, Any], field_name: str) -> Any:
    if isinstance(chunk, Mapping):
        return chunk.get(field_name)
    return getattr(chunk, field_name)


def _prediction_value(prediction: Mapping[str, Any], *field_names: str) -> Any:
    for field_name in field_names:
        if field_name in prediction:
            return prediction[field_name]
    return None


def build_mention_tables(
    chunks: Sequence[ArchiveChunk | Mapping[str, Any]],
    predictions_by_chunk_id: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    extractor: str,
    model_name: str,
    label_set_version: str,
) -> Dict[str, pd.DataFrame]:
    """Build Neo4j-ready Mention nodes and Chunk-HAS_MENTION relationships."""
    mention_rows: List[Dict[str, Any]] = []
    chunk_mention_edges: List[Dict[str, Any]] = []

    for chunk in chunks:
        chunk_id = _chunk_value(chunk, "chunk_id")
        document_id = _chunk_value(chunk, "document_id")
        dataset = _chunk_value(chunk, "dataset")
        modality = _chunk_value(chunk, "modality")
        sensitivity_level = _chunk_value(chunk, "sensitivity_level")
        access_level = _chunk_value(chunk, "access_level")

        for prediction in predictions_by_chunk_id.get(chunk_id, []):
            mention_text = _prediction_value(prediction, "text", "mention_text")
            mention_type = _prediction_value(prediction, "label", "type", "mention_type")
            start_char = _prediction_value(prediction, "start", "start_char")
            end_char = _prediction_value(prediction, "end", "end_char")
            confidence = _prediction_value(prediction, "score", "confidence")

            if mention_text is None or mention_type is None:
                continue

            mention_id = stable_id(
                "mention",
                chunk_id,
                start_char,
                end_char,
                mention_type,
                mention_text,
            )

            mention_rows.append(
                {
                    "mention_id:ID(Mention)": mention_id,
                    "mention_text": mention_text,
                    "mention_type": mention_type,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "dataset": dataset,
                    "modality": modality,
                    "start_char:int": start_char,
                    "end_char:int": end_char,
                    "confidence:float": confidence,
                    "extractor": extractor,
                    "model_name": model_name,
                    "label_set_version": label_set_version,
                    "sensitivity_level": sensitivity_level,
                    "access_level": access_level,
                    ":LABEL": "Mention",
                }
            )

            chunk_mention_edges.append(
                {
                    ":START_ID(Chunk)": chunk_id,
                    ":END_ID(Mention)": mention_id,
                    "confidence:float": confidence,
                    "extractor": extractor,
                    ":TYPE": "HAS_MENTION",
                }
            )

    return {
        "mentions": pd.DataFrame(mention_rows, columns=MENTION_TABLE_COLUMNS),
        "chunk_has_mention": pd.DataFrame(
            chunk_mention_edges,
            columns=CHUNK_MENTION_EDGE_COLUMNS,
        ),
    }


def export_mention_tables(
    tables: Mapping[str, pd.DataFrame],
    export_dir: str | Path = MENTION_EXPORT_DIR,
) -> Dict[str, Path]:
    """Export Mention node and relationship tables to CSV files."""
    export_path = Path(export_dir)
    exported_paths: Dict[str, Path] = {}

    for table_name, table in tables.items():
        csv_path = export_path / f"{table_name}.csv"
        exported_paths[table_name] = dataframe_to_csv(
            table,
            csv_path,
            json_columns=GRAPH_JSON_COLUMNS,
        )

    return exported_paths


def load_completed_mention_chunk_ids(
    export_dir: str | Path = MENTION_EXPORT_DIR,
    *,
    extractor: str | None = None,
    model_name: str | None = None,
    label_set_version: str | None = None,
) -> set[str]:
    """Load chunk IDs already checkpointed by incremental mention extraction."""
    export_path = Path(export_dir)
    progress_path = export_path / MENTION_PROGRESS_FILENAME

    if progress_path.exists():
        progress_df = pd.read_csv(progress_path, dtype={"chunk_id": "string"})
        for column, expected_value in {
            "extractor": extractor,
            "model_name": model_name,
            "label_set_version": label_set_version,
        }.items():
            if expected_value is not None and column in progress_df.columns:
                progress_df = progress_df[progress_df[column] == expected_value]
        if "chunk_id" in progress_df.columns:
            return set(progress_df["chunk_id"].dropna().astype(str))

    completed_chunk_ids: set[str] = set()
    for table_name, chunk_column in {
        "mentions": "chunk_id",
        "chunk_has_mention": ":START_ID(Chunk)",
    }.items():
        table_path = export_path / f"{table_name}.csv"
        if not table_path.exists():
            continue
        metadata_columns = {"extractor", "model_name", "label_set_version"}
        table_df = pd.read_csv(
            table_path,
            usecols=lambda column: column == chunk_column or column in metadata_columns,
        )
        for column, expected_value in {
            "extractor": extractor,
            "model_name": model_name,
            "label_set_version": label_set_version,
        }.items():
            if expected_value is not None and column in table_df.columns:
                table_df = table_df[table_df[column] == expected_value]
        if chunk_column in table_df.columns:
            completed_chunk_ids.update(table_df[chunk_column].dropna().astype(str))
    return completed_chunk_ids


def load_mention_export_tables(
    export_dir: str | Path = MENTION_EXPORT_DIR,
) -> Dict[str, pd.DataFrame]:
    """Load existing mention export tables from CSV files."""
    export_path = Path(export_dir)
    return {
        table_name: pd.read_csv(export_path / f"{table_name}.csv")
        if (export_path / f"{table_name}.csv").exists()
        else pd.DataFrame()
        for table_name in ["mentions", "chunk_has_mention"]
    }


def export_incremental_mention_tables(
    chunks: Sequence[ArchiveChunk | Mapping[str, Any]],
    predictions_by_chunk_id: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    extractor: str,
    model_name: str,
    label_set_version: str,
    export_dir: str | Path = MENTION_EXPORT_DIR,
) -> Dict[str, Path]:
    """Merge one completed extraction batch into the mention CSV exports.

    The progress CSV records every completed chunk, including chunks with zero
    mentions, so interrupted runs can resume without reprocessing them.
    """
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    batch_chunk_ids = {str(_chunk_value(chunk, "chunk_id")) for chunk in chunks}
    batch_tables = build_mention_tables(
        chunks,
        predictions_by_chunk_id,
        extractor=extractor,
        model_name=model_name,
        label_set_version=label_set_version,
    )

    exported_paths: Dict[str, Path] = {}
    for table_name, chunk_column in {
        "mentions": "chunk_id",
        "chunk_has_mention": ":START_ID(Chunk)",
    }.items():
        csv_path = export_path / f"{table_name}.csv"
        batch_table = batch_tables[table_name]

        if csv_path.exists():
            existing_table = pd.read_csv(csv_path)
            if chunk_column in existing_table.columns:
                existing_table = existing_table[
                    ~existing_table[chunk_column].astype(str).isin(batch_chunk_ids)
                ]
            combined_table = pd.concat(
                [existing_table, batch_table],
                ignore_index=True,
                sort=False,
            )
        else:
            combined_table = batch_table

        exported_paths[table_name] = dataframe_to_csv(
            combined_table,
            csv_path,
            json_columns=GRAPH_JSON_COLUMNS,
        )

    progress_path = export_path / MENTION_PROGRESS_FILENAME
    progress_rows = [
        {
            "chunk_id": _chunk_value(chunk, "chunk_id"),
            "document_id": _chunk_value(chunk, "document_id"),
            "dataset": _chunk_value(chunk, "dataset"),
            "modality": _chunk_value(chunk, "modality"),
            "mention_count": len(predictions_by_chunk_id.get(_chunk_value(chunk, "chunk_id"), [])),
            "extractor": extractor,
            "model_name": model_name,
            "label_set_version": label_set_version,
        }
        for chunk in chunks
    ]
    batch_progress = pd.DataFrame(progress_rows)

    if progress_path.exists():
        existing_progress = pd.read_csv(progress_path)
        if "chunk_id" in existing_progress.columns:
            existing_progress = existing_progress[
                ~existing_progress["chunk_id"].astype(str).isin(batch_chunk_ids)
            ]
        progress_table = pd.concat(
            [existing_progress, batch_progress],
            ignore_index=True,
            sort=False,
        )
    else:
        progress_table = batch_progress

    exported_paths["mention_extraction_progress"] = dataframe_to_csv(
        progress_table,
        progress_path,
        json_columns=GRAPH_JSON_COLUMNS,
    )
    return exported_paths
