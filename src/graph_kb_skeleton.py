"""Deterministic graph knowledge-base skeleton export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd

from src.archive_schema import ArchiveChunk, dataframe_to_csv


GRAPH_EXPORT_DIR = Path("data") / "graph_kb_exports" / "step_01_archive_skeleton"
GRAPH_JSON_COLUMNS = [
    "aliases",
    "source_metadata",
    "sensitive_entities",
]


def _chunk_value(chunk: ArchiveChunk | Mapping[str, Any], field_name: str) -> Any:
    if isinstance(chunk, Mapping):
        return chunk.get(field_name)
    return getattr(chunk, field_name)


def _iter_unique_chunks(
    chunk_groups: Mapping[str, Sequence[ArchiveChunk | Mapping[str, Any]]],
) -> Iterable[Tuple[str, ArchiveChunk | Mapping[str, Any]]]:
    seen_chunk_ids = set()

    for group_name, chunks in chunk_groups.items():
        for chunk in chunks:
            chunk_id = _chunk_value(chunk, "chunk_id")
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(chunk_id)
            yield group_name, chunk


def build_archive_skeleton_tables(
    chunk_groups: Mapping[str, Sequence[ArchiveChunk | Mapping[str, Any]]],
) -> Dict[str, pd.DataFrame]:
    """Build deterministic Neo4j-ready node and relationship tables.

    The output is intentionally derived from chunk payloads only, so it can be
    regenerated from cached CSVs or in-memory notebook chunk lists.
    """
    dataset_rows: Dict[str, Dict[str, Any]] = {}
    document_rows: Dict[str, Dict[str, Any]] = {}
    modality_rows: Dict[str, Dict[str, Any]] = {}
    chunk_rows: Dict[str, Dict[str, Any]] = {}

    dataset_document_edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
    document_chunk_edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
    document_modality_edges: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for source_group, chunk in _iter_unique_chunks(chunk_groups):
        chunk_id = _chunk_value(chunk, "chunk_id")
        document_id = _chunk_value(chunk, "document_id")
        dataset = _chunk_value(chunk, "dataset")
        modality = _chunk_value(chunk, "modality")
        source_metadata = _chunk_value(chunk, "source_metadata") or {}

        dataset_rows.setdefault(
            dataset,
            {
                "dataset_id:ID(Dataset)": dataset,
                "name": dataset,
                "source_group": source_group,
                ":LABEL": "Dataset",
            },
        )

        modality_rows.setdefault(
            modality,
            {
                "modality_id:ID(Modality)": modality,
                "name": modality,
                ":LABEL": "Modality",
            },
        )

        document_rows.setdefault(
            document_id,
            {
                "document_id:ID(Document)": document_id,
                "source_id": _chunk_value(chunk, "source_id"),
                "dataset": dataset,
                "modality": modality,
                "title": _chunk_value(chunk, "title"),
                "summary": _chunk_value(chunk, "summary"),
                "source_metadata": source_metadata,
                ":LABEL": "Document",
            },
        )

        chunk_rows[chunk_id] = {
            "chunk_id:ID(Chunk)": chunk_id,
            "document_id": document_id,
            "source_id": _chunk_value(chunk, "source_id"),
            "dataset": dataset,
            "modality": modality,
            "chunk_index:int": _chunk_value(chunk, "chunk_index"),
            "title": _chunk_value(chunk, "title"),
            "masked_text": _chunk_value(chunk, "masked_text"),
            "embedding_text": _chunk_value(chunk, "embedding_text"),
            "summary": _chunk_value(chunk, "summary"),
            "sensitivity_level": _chunk_value(chunk, "sensitivity_level"),
            "access_level": _chunk_value(chunk, "access_level"),
            "sensitive_entities": _chunk_value(chunk, "sensitive_entities") or [],
            "source_metadata": source_metadata,
            ":LABEL": "Chunk",
        }

        dataset_document_edges.setdefault(
            (dataset, document_id),
            {
                ":START_ID(Dataset)": dataset,
                ":END_ID(Document)": document_id,
                ":TYPE": "HAS_DOCUMENT",
            },
        )

        document_chunk_edges.setdefault(
            (document_id, chunk_id),
            {
                ":START_ID(Document)": document_id,
                ":END_ID(Chunk)": chunk_id,
                "chunk_index:int": _chunk_value(chunk, "chunk_index"),
                ":TYPE": "HAS_CHUNK",
            },
        )

        document_modality_edges.setdefault(
            (document_id, modality),
            {
                ":START_ID(Document)": document_id,
                ":END_ID(Modality)": modality,
                ":TYPE": "HAS_MODALITY",
            },
        )

    return {
        "datasets": pd.DataFrame(dataset_rows.values()).sort_values("dataset_id:ID(Dataset)"),
        "documents": pd.DataFrame(document_rows.values()).sort_values("document_id:ID(Document)"),
        "chunks": pd.DataFrame(chunk_rows.values()).sort_values("chunk_id:ID(Chunk)"),
        "modalities": pd.DataFrame(modality_rows.values()).sort_values("modality_id:ID(Modality)"),
        "dataset_has_document": pd.DataFrame(dataset_document_edges.values()).sort_values(
            [":START_ID(Dataset)", ":END_ID(Document)"]
        ),
        "document_has_chunk": pd.DataFrame(document_chunk_edges.values()).sort_values(
            [":START_ID(Document)", "chunk_index:int", ":END_ID(Chunk)"]
        ),
        "document_has_modality": pd.DataFrame(document_modality_edges.values()).sort_values(
            [":START_ID(Document)", ":END_ID(Modality)"]
        ),
    }


def export_archive_skeleton_tables(
    tables: Mapping[str, pd.DataFrame],
    export_dir: str | Path = GRAPH_EXPORT_DIR,
) -> Dict[str, Path]:
    """Export archive skeleton node and relationship tables to CSV files."""
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
