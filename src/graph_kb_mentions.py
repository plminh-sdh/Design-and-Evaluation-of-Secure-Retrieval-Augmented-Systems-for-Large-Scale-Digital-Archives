"""Mention extraction table/export helpers for the graph knowledge base."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

from src.archive_schema import ArchiveChunk, dataframe_to_csv, stable_id
from src.graph_kb_skeleton import GRAPH_JSON_COLUMNS


MENTION_EXPORT_DIR = Path("data") / "graph_kb_exports" / "step_02_mentions"


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
        "mentions": pd.DataFrame(mention_rows),
        "chunk_has_mention": pd.DataFrame(chunk_mention_edges),
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
