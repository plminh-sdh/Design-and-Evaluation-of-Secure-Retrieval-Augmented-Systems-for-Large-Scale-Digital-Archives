"""Neo4j loading helpers for the compact graph knowledge base export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pandas as pd
from tqdm.auto import tqdm

from src.graph_kb_canonicalization import CANONICALIZATION_EXPORT_DIR
from src.graph_kb_local_coreference import LOCAL_COREFERENCE_EXPORT_DIR
from src.graph_kb_mentions import MENTION_EXPORT_DIR
from src.graph_kb_relations import RELATION_EXPORT_DIR
from src.graph_kb_skeleton import GRAPH_EXPORT_DIR


NEO4J_EXPORT_DIR = Path("data") / "graph_kb_exports" / "step_05_neo4j"
CHUNK_ENTITY_EDGES_FILENAME = "chunk_mentions_entity.csv"


@dataclass(frozen=True)
class GraphKbNeo4jPaths:
    """Resolved CSV paths used by the compact Neo4j loader."""

    graph_dir: Path = GRAPH_EXPORT_DIR
    mention_dir: Path = MENTION_EXPORT_DIR
    canonicalization_dir: Path = CANONICALIZATION_EXPORT_DIR
    local_coreference_dir: Path = LOCAL_COREFERENCE_EXPORT_DIR
    relation_dir: Path = RELATION_EXPORT_DIR
    neo4j_export_dir: Path = NEO4J_EXPORT_DIR

    @property
    def datasets(self) -> Path:
        return self.graph_dir / "datasets.csv"

    @property
    def documents(self) -> Path:
        return self.graph_dir / "documents.csv"

    @property
    def chunks(self) -> Path:
        return self.graph_dir / "chunks.csv"

    @property
    def modalities(self) -> Path:
        return self.graph_dir / "modalities.csv"

    @property
    def dataset_has_document(self) -> Path:
        return self.graph_dir / "dataset_has_document.csv"

    @property
    def document_has_chunk(self) -> Path:
        return self.graph_dir / "document_has_chunk.csv"

    @property
    def document_has_modality(self) -> Path:
        return self.graph_dir / "document_has_modality.csv"

    @property
    def mentions(self) -> Path:
        return self.mention_dir / "mentions.csv"

    @property
    def chunk_has_mention(self) -> Path:
        return self.mention_dir / "chunk_has_mention.csv"

    @property
    def refined_entities(self) -> Path:
        return self.canonicalization_dir / "entities.csv"

    @property
    def refined_mention_refers(self) -> Path:
        return self.canonicalization_dir / "mention_refers_to_entity.csv"

    @property
    def local_entities(self) -> Path:
        return self.local_coreference_dir / "local_entities.csv"

    @property
    def local_mention_refers(self) -> Path:
        return self.local_coreference_dir / "local_mention_refers_to_entity.csv"

    @property
    def typed_relations(self) -> Path:
        return self.relation_dir / "typed_relations.csv"

    @property
    def chunk_entity_edges(self) -> Path:
        return self.neo4j_export_dir / CHUNK_ENTITY_EDGES_FILENAME

    def required_inputs(self) -> list[Path]:
        return [
            self.datasets,
            self.documents,
            self.chunks,
            self.modalities,
            self.dataset_has_document,
            self.document_has_chunk,
            self.document_has_modality,
            self.mentions,
            self.chunk_has_mention,
            self.refined_entities,
            self.refined_mention_refers,
            self.local_entities,
            self.local_mention_refers,
            self.typed_relations,
        ]


def missing_neo4j_inputs(paths: GraphKbNeo4jPaths) -> list[Path]:
    """Return missing CSV inputs for the Neo4j loader."""
    return [path for path in paths.required_inputs() if not path.exists()]


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _records_from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in df.to_dict("records"):
        records.append({key: _clean_value(value) for key, value in record.items()})
    return records


def iter_csv_records(
    csv_path: str | Path,
    *,
    usecols: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
    chunksize: int = 50_000,
    verbose: bool = True,
    desc: str | None = None,
    total_rows: int | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Yield cleaned CSV records in batches."""
    path = Path(csv_path)
    reader = pd.read_csv(
        path,
        usecols=(lambda column: column in set(usecols)) if usecols else None,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in tqdm(
        reader,
        desc=desc or f"Reading {path.name}",
        unit="csv chunk",
        total=math.ceil(total_rows / chunksize) if total_rows is not None else None,
        disable=not verbose,
    ):
        if rename:
            chunk = chunk.rename(columns=rename)
        yield _records_from_dataframe(chunk)


def count_csv_rows(
    csv_path: str | Path,
    *,
    usecols: Sequence[str] | None = None,
    chunksize: int = 250_000,
    verbose: bool = True,
    desc: str | None = None,
) -> int:
    """Count CSV rows with pandas so embedded newlines do not corrupt totals."""
    path = Path(csv_path)
    total = 0
    reader = pd.read_csv(
        path,
        usecols=(lambda column: column in set(usecols)) if usecols else None,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in tqdm(
        reader,
        desc=desc or f"Counting {path.name}",
        unit="csv chunk",
        disable=not verbose,
    ):
        total += len(chunk)
    return int(total)


def run_write_batch(driver: Any, query: str, rows: Sequence[Mapping[str, Any]]) -> int:
    """Run one Neo4j UNWIND write batch."""
    if not rows:
        return 0
    with driver.session() as session:
        session.execute_write(lambda tx: tx.run(query, rows=list(rows)).consume())
    return len(rows)


def create_graph_kb_constraints(driver: Any) -> None:
    """Create compact graph constraints and lookup indexes."""
    statements = [
        "CREATE CONSTRAINT dataset_id IF NOT EXISTS FOR (n:Dataset) REQUIRE n.dataset_id IS UNIQUE",
        "CREATE CONSTRAINT modality_id IF NOT EXISTS FOR (n:Modality) REQUIRE n.modality_id IS UNIQUE",
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.document_id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT mention_id IF NOT EXISTS FOR (n:Mention) REQUIRE n.mention_id IS UNIQUE",
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.entity_id IS UNIQUE",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.canonical_name)",
        "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.entity_type)",
        "CREATE INDEX chunk_dataset IF NOT EXISTS FOR (n:Chunk) ON (n.dataset)",
        "CREATE INDEX chunk_document IF NOT EXISTS FOR (n:Chunk) ON (n.document_id)",
        "CREATE INDEX mention_type IF NOT EXISTS FOR (n:Mention) ON (n.mention_type)",
        "CREATE INDEX mention_chunk IF NOT EXISTS FOR (n:Mention) ON (n.chunk_id)",
    ]
    with driver.session() as session:
        for statement in statements:
            session.execute_write(lambda tx, q: tx.run(q).consume(), statement)


def upload_csv_batches(
    driver: Any,
    csv_path: str | Path,
    query: str,
    *,
    usecols: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
    batch_size: int = 10_000,
    verbose: bool = True,
    desc: str | None = None,
    count_first: bool = True,
) -> int:
    """Stream a CSV into Neo4j with one Cypher query."""
    total = 0
    total_rows = (
        count_csv_rows(
            csv_path,
            usecols=usecols,
            chunksize=max(batch_size, 50_000),
            verbose=verbose,
            desc=f"Counting {Path(csv_path).name}",
        )
        if count_first
        else None
    )
    progress = tqdm(
        total=total_rows,
        desc=desc or f"Uploading {Path(csv_path).name}",
        unit="row",
        disable=not verbose or total_rows is None,
    )
    for rows in iter_csv_records(
        csv_path,
        usecols=usecols,
        rename=rename,
        chunksize=batch_size,
        verbose=False,
        desc=desc,
        total_rows=total_rows,
    ):
        total += run_write_batch(driver, query, rows)
        if total_rows is not None:
            progress.update(len(rows))
            progress.set_postfix(uploaded=f"{total:,}")
        elif verbose:
            tqdm.write(f"{desc or Path(csv_path).name}: uploaded {total:,} rows")
    progress.close()
    return total


DATASET_QUERY = """
UNWIND $rows AS row
MERGE (d:Dataset {dataset_id: row.dataset_id})
SET d.name = row.name,
    d.source_group = row.source_group
"""

MODALITY_QUERY = """
UNWIND $rows AS row
MERGE (m:Modality {modality_id: row.modality_id})
SET m.name = row.name
"""

DOCUMENT_QUERY = """
UNWIND $rows AS row
MERGE (d:Document {document_id: row.document_id})
SET d.source_id = row.source_id,
    d.dataset = row.dataset,
    d.modality = row.modality,
    d.title = row.title,
    d.summary = row.summary
"""

CHUNK_QUERY = """
UNWIND $rows AS row
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c.document_id = row.document_id,
    c.source_id = row.source_id,
    c.dataset = row.dataset,
    c.modality = row.modality,
    c.chunk_index = row.chunk_index,
    c.title = row.title,
    c.summary = row.summary,
    c.sensitivity_level = row.sensitivity_level,
    c.access_level = row.access_level
"""

MENTION_QUERY = """
UNWIND $rows AS row
MERGE (m:Mention {mention_id: row.mention_id})
SET m.mention_text = row.mention_text,
    m.mention_type = row.mention_type,
    m.chunk_id = row.chunk_id,
    m.document_id = row.document_id,
    m.dataset = row.dataset,
    m.modality = row.modality,
    m.start_char = row.start_char,
    m.end_char = row.end_char,
    m.confidence = row.confidence,
    m.extractor = row.extractor,
    m.label_set_version = row.label_set_version,
    m.sensitivity_level = row.sensitivity_level,
    m.access_level = row.access_level
"""

ENTITY_QUERY = """
UNWIND $rows AS row
MERGE (e:Entity {entity_id: row.entity_id})
SET e.canonical_name = coalesce(row.canonical_name, e.canonical_name),
    e.entity_type = coalesce(row.entity_type, e.entity_type),
    e.external_kb_id = coalesce(row.external_kb_id, e.external_kb_id),
    e.external_kb = coalesce(row.external_kb, e.external_kb),
    e.wikipedia_entity_title = coalesce(row.wikipedia_entity_title, e.wikipedia_entity_title),
    e.canonicalization_method = coalesce(row.canonicalization_method, e.canonicalization_method)
"""

DATASET_DOCUMENT_QUERY = """
UNWIND $rows AS row
MATCH (d:Dataset {dataset_id: row.start_id})
MATCH (doc:Document {document_id: row.end_id})
MERGE (d)-[:HAS_DOCUMENT]->(doc)
"""

DOCUMENT_CHUNK_QUERY = """
UNWIND $rows AS row
MATCH (doc:Document {document_id: row.start_id})
MATCH (c:Chunk {chunk_id: row.end_id})
MERGE (doc)-[r:HAS_CHUNK]->(c)
SET r.chunk_index = row.chunk_index
"""

DOCUMENT_MODALITY_QUERY = """
UNWIND $rows AS row
MATCH (doc:Document {document_id: row.start_id})
MATCH (m:Modality {modality_id: row.end_id})
MERGE (doc)-[:HAS_MODALITY]->(m)
"""

CHUNK_MENTION_QUERY = """
UNWIND $rows AS row
MATCH (c:Chunk {chunk_id: row.start_id})
MATCH (m:Mention {mention_id: row.end_id})
MERGE (c)-[r:MENTIONS]->(m)
SET r.confidence = row.confidence,
    r.extractor = row.extractor
"""

MENTION_ENTITY_QUERY = """
UNWIND $rows AS row
MATCH (m:Mention {mention_id: row.start_id})
MATCH (e:Entity {entity_id: row.end_id})
MERGE (m)-[r:REFERS_TO]->(e)
SET r.confidence = row.confidence,
    r.canonicalization_method = row.canonicalization_method,
    r.canonicalizer = row.canonicalizer,
    r.model_name = row.model_name
"""

CHUNK_ENTITY_QUERY = """
UNWIND $rows AS row
MATCH (c:Chunk {chunk_id: row.chunk_id})
MATCH (e:Entity {entity_id: row.entity_id})
MERGE (c)-[r:MENTIONS_ENTITY]->(e)
SET r.mention_count = coalesce(r.mention_count, 0) + coalesce(row.mention_count, 0),
    r.best_confidence = CASE
        WHEN r.best_confidence IS NULL THEN row.best_confidence
        WHEN row.best_confidence IS NULL THEN r.best_confidence
        WHEN row.best_confidence > r.best_confidence THEN row.best_confidence
        ELSE r.best_confidence
    END
"""


def upload_archive_skeleton_to_neo4j(
    driver: Any,
    paths: GraphKbNeo4jPaths,
    *,
    batch_size: int = 10_000,
    verbose: bool = True,
) -> dict[str, int]:
    """Upload compact Dataset/Document/Chunk/Modality nodes and hierarchy edges."""
    summary: dict[str, int] = {}
    summary["datasets"] = upload_csv_batches(
        driver,
        paths.datasets,
        DATASET_QUERY,
        usecols=["dataset_id:ID(Dataset)", "name", "source_group"],
        rename={"dataset_id:ID(Dataset)": "dataset_id"},
        batch_size=batch_size,
        verbose=verbose,
        desc="Uploading Dataset nodes",
    )
    summary["modalities"] = upload_csv_batches(
        driver,
        paths.modalities,
        MODALITY_QUERY,
        usecols=["modality_id:ID(Modality)", "name"],
        rename={"modality_id:ID(Modality)": "modality_id"},
        batch_size=batch_size,
        verbose=verbose,
        desc="Uploading Modality nodes",
    )
    summary["documents"] = upload_csv_batches(
        driver,
        paths.documents,
        DOCUMENT_QUERY,
        usecols=[
            "document_id:ID(Document)",
            "source_id",
            "dataset",
            "modality",
            "title",
            "summary",
        ],
        rename={"document_id:ID(Document)": "document_id"},
        batch_size=batch_size,
        verbose=verbose,
        desc="Uploading Document nodes",
    )
    summary["chunks"] = upload_csv_batches(
        driver,
        paths.chunks,
        CHUNK_QUERY,
        usecols=[
            "chunk_id:ID(Chunk)",
            "document_id",
            "source_id",
            "dataset",
            "modality",
            "chunk_index:int",
            "title",
            "summary",
            "sensitivity_level",
            "access_level",
        ],
        rename={
            "chunk_id:ID(Chunk)": "chunk_id",
            "chunk_index:int": "chunk_index",
        },
        batch_size=batch_size,
        verbose=verbose,
        desc="Uploading Chunk nodes",
    )
    summary["dataset_has_document"] = upload_csv_batches(
        driver,
        paths.dataset_has_document,
        DATASET_DOCUMENT_QUERY,
        usecols=[":START_ID(Dataset)", ":END_ID(Document)"],
        rename={":START_ID(Dataset)": "start_id", ":END_ID(Document)": "end_id"},
        batch_size=batch_size,
        verbose=verbose,
        desc="Uploading Dataset-HAS_DOCUMENT-Document",
    )
    summary["document_has_modality"] = upload_csv_batches(
        driver,
        paths.document_has_modality,
        DOCUMENT_MODALITY_QUERY,
        usecols=[":START_ID(Document)", ":END_ID(Modality)"],
        rename={":START_ID(Document)": "start_id", ":END_ID(Modality)": "end_id"},
        batch_size=batch_size,
        verbose=verbose,
        desc="Uploading Document-HAS_MODALITY-Modality",
    )
    summary["document_has_chunk"] = upload_csv_batches(
        driver,
        paths.document_has_chunk,
        DOCUMENT_CHUNK_QUERY,
        usecols=[":START_ID(Document)", ":END_ID(Chunk)", "chunk_index:int"],
        rename={
            ":START_ID(Document)": "start_id",
            ":END_ID(Chunk)": "end_id",
            "chunk_index:int": "chunk_index",
        },
        batch_size=batch_size,
        verbose=verbose,
        desc="Uploading Document-HAS_CHUNK-Chunk",
    )
    return summary


def upload_mentions_to_neo4j(
    driver: Any,
    paths: GraphKbNeo4jPaths,
    *,
    batch_size: int = 10_000,
    verbose: bool = True,
) -> int:
    """Upload compact Mention nodes from GLiNER mention exports."""
    return upload_csv_batches(
        driver,
        paths.mentions,
        MENTION_QUERY,
        usecols=[
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
            "label_set_version",
            "sensitivity_level",
            "access_level",
        ],
        rename={
            "mention_id:ID(Mention)": "mention_id",
            "start_char:int": "start_char",
            "end_char:int": "end_char",
            "confidence:float": "confidence",
        },
        batch_size=batch_size,
        verbose=verbose,
        desc="Uploading Mention nodes",
    )


def upload_chunk_mentions_to_neo4j(
    driver: Any,
    paths: GraphKbNeo4jPaths,
    *,
    batch_size: int = 25_000,
    verbose: bool = True,
) -> int:
    """Upload (:Chunk)-[:MENTIONS]->(:Mention) relationships."""
    return upload_csv_batches(
        driver,
        paths.chunk_has_mention,
        CHUNK_MENTION_QUERY,
        usecols=[
            ":START_ID(Chunk)",
            ":END_ID(Mention)",
            "confidence:float",
            "extractor",
        ],
        rename={
            ":START_ID(Chunk)": "start_id",
            ":END_ID(Mention)": "end_id",
            "confidence:float": "confidence",
        },
        batch_size=batch_size,
        verbose=verbose,
        desc="Uploading Chunk-MENTIONS-Mention",
    )


def upload_entities_to_neo4j(
    driver: Any,
    paths: GraphKbNeo4jPaths,
    *,
    batch_size: int = 10_000,
    verbose: bool = True,
) -> dict[str, int]:
    """Upload public ReFinED and local NIL Entity nodes."""
    usecols = [
        "entity_id:ID(Entity)",
        "canonical_name",
        "entity_type",
        "external_kb_id",
        "external_kb",
        "wikipedia_entity_title",
        "canonicalization_method",
    ]
    rename = {"entity_id:ID(Entity)": "entity_id"}
    return {
        "refined_entities": upload_csv_batches(
            driver,
            paths.refined_entities,
            ENTITY_QUERY,
            usecols=usecols,
            rename=rename,
            batch_size=batch_size,
            verbose=verbose,
            desc="Uploading ReFinED Entity nodes",
        ),
        "local_entities": upload_csv_batches(
            driver,
            paths.local_entities,
            ENTITY_QUERY,
            usecols=usecols,
            rename=rename,
            batch_size=batch_size,
            verbose=verbose,
            desc="Uploading local Entity nodes",
        ),
    }


def upload_mention_entity_edges_to_neo4j(
    driver: Any,
    paths: GraphKbNeo4jPaths,
    *,
    batch_size: int = 25_000,
    verbose: bool = True,
) -> dict[str, int]:
    """Upload public and local (:Mention)-[:REFERS_TO]->(:Entity) edges."""
    usecols = [
        ":START_ID(Mention)",
        ":END_ID(Entity)",
        "confidence:float",
        "canonicalization_method",
        "canonicalizer",
        "model_name",
    ]
    rename = {
        ":START_ID(Mention)": "start_id",
        ":END_ID(Entity)": "end_id",
        "confidence:float": "confidence",
    }
    return {
        "refined_refers_to": upload_csv_batches(
            driver,
            paths.refined_mention_refers,
            MENTION_ENTITY_QUERY,
            usecols=usecols,
            rename=rename,
            batch_size=batch_size,
            verbose=verbose,
            desc="Uploading ReFinED Mention-REFERS_TO-Entity",
        ),
        "local_refers_to": upload_csv_batches(
            driver,
            paths.local_mention_refers,
            MENTION_ENTITY_QUERY,
            usecols=usecols,
            rename=rename,
            batch_size=batch_size,
            verbose=verbose,
            desc="Uploading local Mention-REFERS_TO-Entity",
        ),
    }


def build_chunk_entity_edges_csv(
    mentions_csv: str | Path,
    mention_refers_csvs: Sequence[str | Path],
    output_csv: str | Path,
    *,
    chunksize: int = 250_000,
    verbose: bool = True,
) -> Path:
    """Build compact Chunk-MENTIONS_ENTITY-Entity edges from mention exports."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    mention_to_chunk: dict[str, str] = {}
    mention_reader = pd.read_csv(
        mentions_csv,
        usecols=lambda column: column in {"mention_id:ID(Mention)", "chunk_id"},
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in tqdm(
        mention_reader,
        desc="Indexing mention->chunk IDs",
        unit="csv chunk",
        disable=not verbose,
    ):
        chunk = chunk.rename(columns={"mention_id:ID(Mention)": "mention_id"})
        mention_to_chunk.update(
            zip(chunk["mention_id"].astype(str), chunk["chunk_id"].astype(str))
        )

    for refers_csv in mention_refers_csvs:
        path = Path(refers_csv)
        if not path.exists():
            continue
        reader = pd.read_csv(
            path,
            usecols=lambda column: column
            in {":START_ID(Mention)", ":END_ID(Entity)", "confidence:float"},
            chunksize=chunksize,
            low_memory=False,
        )
        for chunk in tqdm(
            reader,
            desc=f"Building chunk-entity edges from {path.name}",
            unit="csv chunk",
            disable=not verbose,
        ):
            chunk = chunk.rename(
                columns={
                    ":START_ID(Mention)": "mention_id",
                    ":END_ID(Entity)": "entity_id",
                    "confidence:float": "confidence",
                }
            )
            chunk["chunk_id"] = chunk["mention_id"].astype(str).map(mention_to_chunk)
            chunk = chunk.dropna(subset=["chunk_id", "entity_id"])
            if chunk.empty:
                continue
            grouped = (
                chunk.groupby(["chunk_id", "entity_id"], dropna=False)
                .agg(
                    mention_count=("mention_id", "count"),
                    best_confidence=("confidence", "max"),
                )
                .reset_index()
            )
            grouped.to_csv(
                output_path,
                mode="a",
                header=not output_path.exists(),
                index=False,
            )
    return output_path


def upload_chunk_entity_edges_to_neo4j(
    driver: Any,
    chunk_entity_edges_csv: str | Path,
    *,
    batch_size: int = 25_000,
    verbose: bool = True,
) -> int:
    """Upload compact Chunk-MENTIONS_ENTITY-Entity relationships."""
    return upload_csv_batches(
        driver,
        chunk_entity_edges_csv,
        CHUNK_ENTITY_QUERY,
        usecols=["chunk_id", "entity_id", "mention_count", "best_confidence"],
        batch_size=batch_size,
        verbose=verbose,
        desc="Uploading Chunk-MENTIONS_ENTITY-Entity",
    )


def _safe_relation_type(value: Any) -> str | None:
    relation_type = str(value or "").upper().strip()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", relation_type):
        return None
    return relation_type


def upload_typed_relations_to_neo4j(
    driver: Any,
    typed_relations_csv: str | Path,
    *,
    batch_size: int = 25_000,
    verbose: bool = True,
    count_first: bool = True,
) -> int:
    """Upload typed Entity-to-Entity relationships using dynamic relation types."""
    path = Path(typed_relations_csv)
    total = 0
    total_rows = (
        count_csv_rows(
            path,
            chunksize=max(batch_size, 50_000),
            verbose=verbose,
            desc=f"Counting {path.name}",
        )
        if count_first
        else None
    )
    reader = pd.read_csv(path, chunksize=batch_size, low_memory=False)
    progress = tqdm(
        reader,
        desc="Uploading typed Entity relationships",
        unit="csv chunk",
        total=math.ceil(total_rows / batch_size) if total_rows is not None else None,
        disable=not verbose,
    )
    for chunk in progress:
        uploaded_before_chunk = total
        chunk = chunk.rename(
            columns={
                "relation_id:ID(Relation)": "relation_id",
                ":START_ID(Entity)": "start_id",
                ":END_ID(Entity)": "end_id",
            }
        )
        for relation_type, group in chunk.groupby("relation_type", dropna=True):
            safe_type = _safe_relation_type(relation_type)
            if safe_type is None:
                continue
            rows = _records_from_dataframe(
                group[
                    [
                        "relation_id",
                        "start_id",
                        "end_id",
                        "relation_type",
                        "glirel_label",
                        "input_id",
                        "chunk_id",
                        "document_id",
                        "dataset",
                        "modality",
                    ]
                ]
            )
            query = f"""
            UNWIND $rows AS row
            MATCH (head:Entity {{entity_id: row.start_id}})
            MATCH (tail:Entity {{entity_id: row.end_id}})
            MERGE (head)-[r:{safe_type} {{relation_id: row.relation_id}}]->(tail)
            SET r.relation_type = row.relation_type,
                r.glirel_label = row.glirel_label,
                r.input_id = row.input_id,
                r.chunk_id = row.chunk_id,
                r.document_id = row.document_id,
                r.dataset = row.dataset,
                r.modality = row.modality
            """
            total += run_write_batch(driver, query, rows)
        progress.set_postfix(
            rows=f"{min(total_rows or total, total):,}"
            if total_rows is not None
            else f"{total:,}",
            uploaded_delta=f"{total - uploaded_before_chunk:,}",
        )
    return total


def compact_graph_upload_summary(paths: GraphKbNeo4jPaths) -> pd.DataFrame:
    """Summarize compact upload inputs and generated derived files."""
    rows = []
    for name, path in [
        ("datasets", paths.datasets),
        ("documents", paths.documents),
        ("chunks", paths.chunks),
        ("modalities", paths.modalities),
        ("dataset_has_document", paths.dataset_has_document),
        ("document_has_chunk", paths.document_has_chunk),
        ("document_has_modality", paths.document_has_modality),
        ("mentions", paths.mentions),
        ("chunk_has_mention", paths.chunk_has_mention),
        ("refined_entities", paths.refined_entities),
        ("local_entities", paths.local_entities),
        ("refined_mention_refers", paths.refined_mention_refers),
        ("local_mention_refers", paths.local_mention_refers),
        ("typed_relations", paths.typed_relations),
    ]:
        rows.append(
            {
                "table": name,
                "path": str(path),
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2)
                if path.exists()
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def neo4j_upload_preview_tables(
    paths: GraphKbNeo4jPaths,
    *,
    nrows: int = 5,
) -> dict[str, pd.DataFrame]:
    """Return compact previews matching the Neo4j upload shape."""
    previews: dict[str, pd.DataFrame] = {}
    previews["Dataset nodes"] = pd.read_csv(
        paths.datasets,
        usecols=["dataset_id:ID(Dataset)", "name", "source_group"],
        nrows=nrows,
    ).rename(columns={"dataset_id:ID(Dataset)": "dataset_id"})
    previews["Modality nodes"] = pd.read_csv(
        paths.modalities,
        usecols=["modality_id:ID(Modality)", "name"],
        nrows=nrows,
    ).rename(columns={"modality_id:ID(Modality)": "modality_id"})
    previews["Document nodes"] = pd.read_csv(
        paths.documents,
        usecols=[
            "document_id:ID(Document)",
            "source_id",
            "dataset",
            "modality",
            "title",
            "summary",
        ],
        nrows=nrows,
    ).rename(columns={"document_id:ID(Document)": "document_id"})
    previews["Chunk nodes"] = pd.read_csv(
        paths.chunks,
        usecols=[
            "chunk_id:ID(Chunk)",
            "document_id",
            "source_id",
            "dataset",
            "modality",
            "chunk_index:int",
            "title",
            "summary",
            "sensitivity_level",
            "access_level",
        ],
        nrows=nrows,
    ).rename(
        columns={
            "chunk_id:ID(Chunk)": "chunk_id",
            "chunk_index:int": "chunk_index",
        }
    )
    previews["Mention nodes"] = pd.read_csv(
        paths.mentions,
        usecols=[
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
            "label_set_version",
            "sensitivity_level",
            "access_level",
        ],
        nrows=nrows,
    ).rename(
        columns={
            "mention_id:ID(Mention)": "mention_id",
            "start_char:int": "start_char",
            "end_char:int": "end_char",
            "confidence:float": "confidence",
        }
    )
    entity_usecols = [
        "entity_id:ID(Entity)",
        "canonical_name",
        "entity_type",
        "external_kb_id",
        "external_kb",
        "wikipedia_entity_title",
        "canonicalization_method",
    ]
    previews["ReFinED Entity nodes"] = pd.read_csv(
        paths.refined_entities,
        usecols=entity_usecols,
        nrows=nrows,
    ).rename(columns={"entity_id:ID(Entity)": "entity_id"})
    previews["Local Entity nodes"] = pd.read_csv(
        paths.local_entities,
        usecols=entity_usecols,
        nrows=nrows,
    ).rename(columns={"entity_id:ID(Entity)": "entity_id"})
    previews["Dataset-HAS_DOCUMENT-Document"] = pd.read_csv(
        paths.dataset_has_document,
        usecols=[":START_ID(Dataset)", ":END_ID(Document)"],
        nrows=nrows,
    ).rename(columns={":START_ID(Dataset)": "start_id", ":END_ID(Document)": "end_id"})
    previews["Document-HAS_CHUNK-Chunk"] = pd.read_csv(
        paths.document_has_chunk,
        usecols=[":START_ID(Document)", ":END_ID(Chunk)", "chunk_index:int"],
        nrows=nrows,
    ).rename(
        columns={
            ":START_ID(Document)": "start_id",
            ":END_ID(Chunk)": "end_id",
            "chunk_index:int": "chunk_index",
        }
    )
    previews["Chunk-MENTIONS-Mention"] = pd.read_csv(
        paths.chunk_has_mention,
        usecols=[
            ":START_ID(Chunk)",
            ":END_ID(Mention)",
            "confidence:float",
            "extractor",
        ],
        nrows=nrows,
    ).rename(
        columns={
            ":START_ID(Chunk)": "start_id",
            ":END_ID(Mention)": "end_id",
            "confidence:float": "confidence",
        }
    )
    previews["ReFinED Mention-REFERS_TO-Entity"] = pd.read_csv(
        paths.refined_mention_refers,
        usecols=[
            ":START_ID(Mention)",
            ":END_ID(Entity)",
            "confidence:float",
            "canonicalization_method",
            "canonicalizer",
            "model_name",
        ],
        nrows=nrows,
    ).rename(
        columns={
            ":START_ID(Mention)": "start_id",
            ":END_ID(Entity)": "end_id",
            "confidence:float": "confidence",
        }
    )
    previews["Local Mention-REFERS_TO-Entity"] = pd.read_csv(
        paths.local_mention_refers,
        usecols=[
            ":START_ID(Mention)",
            ":END_ID(Entity)",
            "confidence:float",
            "canonicalization_method",
            "canonicalizer",
            "model_name",
        ],
        nrows=nrows,
    ).rename(
        columns={
            ":START_ID(Mention)": "start_id",
            ":END_ID(Entity)": "end_id",
            "confidence:float": "confidence",
        }
    )
    previews["Typed Entity relations"] = pd.read_csv(
        paths.typed_relations,
        nrows=nrows,
    ).rename(
        columns={
            "relation_id:ID(Relation)": "relation_id",
            ":START_ID(Entity)": "start_id",
            ":END_ID(Entity)": "end_id",
        }
    )
    return previews
