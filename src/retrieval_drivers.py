"""Driver boundaries for retrieval backends used by retrieval strategies."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client import models


def _load_dotenv_if_available(env_path: str | Path = ".env") -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path)


def _userdata_get(name: str) -> str | None:
    try:
        from google.colab import userdata
    except Exception:
        return None
    try:
        return userdata.get(name)
    except Exception:
        return None


def _get_config_value(name: str, *, use_colab_userdata: bool = True) -> str | None:
    if use_colab_userdata:
        value = _userdata_get(name)
        if value:
            return value
    return os.getenv(name)


@dataclass(frozen=True)
class QdrantRetrievalDriver:
    """Thin Qdrant access boundary for retrieval strategies.

    All dense/vector strategies should call Qdrant through this class so future
    access-control or auditing logic has a single place to live.
    """

    url: str | None
    api_key: str | None
    collection_name: str

    @classmethod
    def from_environment(
        cls,
        *,
        env_path: str | Path = ".env",
        use_colab_userdata: bool = True,
        url_env: str = "QDRANT_CLUSTER_URL",
        api_key_env: str = "QDRANT_CLUSTER_API_KEY",
        collection_name_env: str = "QDRANT_COLLECTION_NAME",
        default_collection_name: str = "archive-chunks",
    ) -> "QdrantRetrievalDriver":
        """Build a Qdrant driver from Colab userdata or local environment variables."""

        _load_dotenv_if_available(env_path)
        return cls(
            url=_get_config_value(url_env, use_colab_userdata=use_colab_userdata),
            api_key=_get_config_value(api_key_env, use_colab_userdata=use_colab_userdata),
            collection_name=_get_config_value(
                collection_name_env,
                use_colab_userdata=use_colab_userdata,
            )
            or default_collection_name,
        )

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("Qdrant URL is required.")
        object.__setattr__(
            self,
            "client",
            QdrantClient(url=self.url, api_key=self.api_key),
        )

    def verify_connectivity(self) -> Any:
        return self.client.get_collections()

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        query_filter: Any | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[Any]:
        """Search the configured Qdrant collection with the default dense vector."""

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return list(response.points)

    def search_dense(
        self,
        query_vector: Sequence[float],
        *,
        vector_name: str = "dense",
        top_k: int,
        query_filter: Any | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[Any]:
        """Search a named dense vector in the configured collection."""

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(query_vector),
            using=vector_name,
            limit=top_k,
            query_filter=query_filter,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return list(response.points)

    def search_sparse(
        self,
        query_sparse_vector: models.SparseVector,
        *,
        vector_name: str = "sparse",
        top_k: int,
        query_filter: Any | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[Any]:
        """Search a named sparse vector in the configured collection."""

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_sparse_vector,
            using=vector_name,
            limit=top_k,
            query_filter=query_filter,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return list(response.points)

    def search_hybrid_rrf(
        self,
        *,
        dense_query_vector: Sequence[float],
        sparse_query_vector: models.SparseVector,
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
        top_k: int,
        prefetch_limit: int | None = None,
        query_filter: Any | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        rrf_k: int | None = None,
    ) -> list[Any]:
        """Search dense+sparse vectors with Qdrant's server-side RRF fusion."""

        prefetch_limit = prefetch_limit or max(top_k, 20)
        if rrf_k is None:
            query = models.RrfQuery(rrf=models.Rrf())
        else:
            query = models.RrfQuery(rrf=models.Rrf(k=rrf_k))

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=sparse_query_vector,
                    using=sparse_vector_name,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=list(dense_query_vector),
                    using=dense_vector_name,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=query,
            limit=top_k,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return list(response.points)

    def search_prefetch_rrf(
        self,
        prefetches: Sequence[models.Prefetch],
        *,
        top_k: int,
        with_payload: bool = True,
        with_vectors: bool = False,
        rrf_k: int | None = None,
    ) -> list[Any]:
        """Search with caller-built Qdrant prefetches and server-side RRF fusion."""

        if rrf_k is None:
            query = models.RrfQuery(rrf=models.Rrf())
        else:
            query = models.RrfQuery(rrf=models.Rrf(k=rrf_k))

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=list(prefetches),
            query=query,
            limit=top_k,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return list(response.points)


@dataclass(frozen=True)
class Neo4jGraphRetrievalDriver:
    """Thin Neo4j access boundary for graph-expanded retrieval strategies."""

    uri: str | None
    username: str | None
    password: str | None
    database: str | None = None

    @classmethod
    def from_environment(
        cls,
        *,
        env_path: str | Path = ".env",
        use_colab_userdata: bool = True,
        uri_env: str = "NEO4J_URI",
        username_env: str = "NEO4J_USERNAME",
        password_env: str = "NEO4J_PASSWORD",
        database_env: str = "NEO4J_DATABASE",
    ) -> "Neo4jGraphRetrievalDriver":
        """Build a Neo4j driver from Colab userdata or local environment variables."""

        _load_dotenv_if_available(env_path)
        return cls(
            uri=_get_config_value(uri_env, use_colab_userdata=use_colab_userdata),
            username=_get_config_value(username_env, use_colab_userdata=use_colab_userdata),
            password=_get_config_value(password_env, use_colab_userdata=use_colab_userdata),
            database=_get_config_value(database_env, use_colab_userdata=use_colab_userdata)
            or None,
        )

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("Neo4j URI is required.")
        object.__setattr__(
            self,
            "driver",
            GraphDatabase.driver(self.uri, auth=(self.username, self.password)),
        )

    def close(self) -> None:
        self.driver.close()

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    def run_read_query(
        self,
        cypher: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a read query and return dictionaries."""

        parameters = dict(parameters or {})
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, parameters)
            return [dict(record) for record in result]

    def entity_surface_hints(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return simple entity/mention surface hints for a query string."""

        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.canonical_name) CONTAINS toLower($query)
           OR toLower($query) CONTAINS toLower(e.canonical_name)
        RETURN e.entity_id AS entity_id,
               e.canonical_name AS entity_name,
               e.entity_type AS entity_type
        LIMIT $limit
        """
        return self.run_read_query(cypher, {"query": query, "limit": limit})

    def graph_expansion_hints(
        self,
        *,
        surface_terms: Sequence[str],
        entity_ids: Sequence[str] | None = None,
        entity_limit: int = 8,
        alias_limit: int = 8,
        neighbor_limit: int = 12,
        include_mention_surface_lookup: bool = False,
    ) -> list[dict[str, Any]]:
        """Return entity aliases and one-hop relation neighbors for query expansion."""

        cleaned_surface_terms = [
            term.strip()
            for term in surface_terms
            if term is not None and len(term.strip()) >= 2
        ]
        cleaned_entity_ids = [
            str(entity_id).strip()
            for entity_id in (entity_ids or [])
            if entity_id is not None and str(entity_id).strip()
        ]
        if not cleaned_surface_terms and not cleaned_entity_ids:
            return []

        if include_mention_surface_lookup:
            cypher = """
            MATCH (e:Entity)
            WHERE e.entity_id IN $entity_ids
               OR any(term IN $surface_terms
                      WHERE toLower(e.canonical_name) CONTAINS toLower(term)
                         OR toLower(term) CONTAINS toLower(e.canonical_name))
               OR EXISTS {
                    MATCH (m:Mention)-[:REFERS_TO]->(e)
                    WHERE any(term IN $surface_terms
                              WHERE toLower(m.mention_text) CONTAINS toLower(term)
                                 OR toLower(term) CONTAINS toLower(m.mention_text))
               }
            WITH collect(DISTINCT e)[0..$entity_limit] AS matched_entities
            UNWIND matched_entities AS e
            CALL (e) {
                MATCH (m:Mention)-[:REFERS_TO]->(e)
                WHERE m.mention_text IS NOT NULL
                RETURN collect(DISTINCT m.mention_text)[0..$alias_limit] AS aliases
            }
            CALL (e) {
                MATCH (e)-[r]-(neighbor:Entity)
                WHERE neighbor.canonical_name IS NOT NULL
                RETURN collect(DISTINCT {
                    relation_type: type(r),
                    neighbor_entity_id: neighbor.entity_id,
                    neighbor_name: neighbor.canonical_name,
                    neighbor_type: neighbor.entity_type
                })[0..$neighbor_limit] AS relation_neighbors
            }
            RETURN e.entity_id AS entity_id,
                   e.canonical_name AS canonical_name,
                   e.entity_type AS entity_type,
                   aliases AS aliases,
                   relation_neighbors AS relation_neighbors
            """
        else:
            cypher = """
            MATCH (e:Entity)
            WHERE e.entity_id IN $entity_ids
               OR any(term IN $surface_terms
                      WHERE toLower(e.canonical_name) CONTAINS toLower(term)
                         OR toLower(term) CONTAINS toLower(e.canonical_name))
            WITH collect(DISTINCT e)[0..$entity_limit] AS matched_entities
            UNWIND matched_entities AS e
            CALL (e) {
                MATCH (m:Mention)-[:REFERS_TO]->(e)
                WHERE m.mention_text IS NOT NULL
                RETURN collect(DISTINCT m.mention_text)[0..$alias_limit] AS aliases
            }
            CALL (e) {
                MATCH (e)-[r]-(neighbor:Entity)
                WHERE neighbor.canonical_name IS NOT NULL
                RETURN collect(DISTINCT {
                    relation_type: type(r),
                    neighbor_entity_id: neighbor.entity_id,
                    neighbor_name: neighbor.canonical_name,
                    neighbor_type: neighbor.entity_type
                })[0..$neighbor_limit] AS relation_neighbors
            }
            RETURN e.entity_id AS entity_id,
                   e.canonical_name AS canonical_name,
                   e.entity_type AS entity_type,
                   aliases AS aliases,
                   relation_neighbors AS relation_neighbors
            """
        return self.run_read_query(
            cypher,
            {
                "surface_terms": cleaned_surface_terms,
                "entity_ids": cleaned_entity_ids,
                "entity_limit": entity_limit,
                "alias_limit": alias_limit,
                "neighbor_limit": neighbor_limit,
            },
        )

    def graph_expansion_hints_by_entity_ids(
        self,
        *,
        entity_ids: Sequence[str],
        alias_limit: int = 8,
        neighbor_limit: int = 12,
    ) -> list[dict[str, Any]]:
        """Return graph expansion hints for already-canonicalized entity IDs."""

        cleaned_entity_ids = [
            str(entity_id).strip()
            for entity_id in entity_ids
            if entity_id is not None and str(entity_id).strip()
        ]
        if not cleaned_entity_ids:
            return []

        cypher = """
        MATCH (e:Entity)
        WHERE e.entity_id IN $entity_ids
        CALL (e) {
            MATCH (m:Mention)-[:REFERS_TO]->(e)
            WHERE m.mention_text IS NOT NULL
            RETURN collect(DISTINCT m.mention_text)[0..$alias_limit] AS aliases
        }
        CALL (e) {
            MATCH (e)-[r]-(neighbor:Entity)
            WHERE neighbor.canonical_name IS NOT NULL
            RETURN collect(DISTINCT {
                relation_type: type(r),
                neighbor_entity_id: neighbor.entity_id,
                neighbor_name: neighbor.canonical_name,
                neighbor_type: neighbor.entity_type
            })[0..$neighbor_limit] AS relation_neighbors
        }
        RETURN e.entity_id AS entity_id,
               e.canonical_name AS canonical_name,
               e.entity_type AS entity_type,
               aliases AS aliases,
               relation_neighbors AS relation_neighbors
        """
        return self.run_read_query(
            cypher,
            {
                "entity_ids": cleaned_entity_ids,
                "alias_limit": alias_limit,
                "neighbor_limit": neighbor_limit,
            },
        )

    def graph_candidate_chunk_features(
        self,
        *,
        chunk_ids: Sequence[str],
        matched_entity_ids: Sequence[str],
        neighbor_entity_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Return bounded graph reranking features for candidate chunk IDs."""

        cleaned_chunk_ids = [
            str(chunk_id).strip()
            for chunk_id in chunk_ids
            if chunk_id is not None and str(chunk_id).strip()
        ]
        if not cleaned_chunk_ids:
            return []
        cleaned_matched_entity_ids = [
            str(entity_id).strip()
            for entity_id in matched_entity_ids
            if entity_id is not None and str(entity_id).strip()
        ]
        cleaned_neighbor_entity_ids = [
            str(entity_id).strip()
            for entity_id in neighbor_entity_ids
            if entity_id is not None and str(entity_id).strip()
        ]

        cypher = """
        MATCH (c:Chunk)
        WHERE c.chunk_id IN $chunk_ids
        OPTIONAL MATCH (c)-[:MENTIONS]->(:Mention)-[:REFERS_TO]->(matched:Entity)
        WHERE matched.entity_id IN $matched_entity_ids
        WITH c, collect(DISTINCT matched.entity_id) AS matched_entities
        OPTIONAL MATCH (c)-[:MENTIONS]->(:Mention)-[:REFERS_TO]->(neighbor:Entity)
        WHERE neighbor.entity_id IN $neighbor_entity_ids
        WITH c,
             matched_entities,
             collect(DISTINCT neighbor.entity_id) AS neighbor_entities
        OPTIONAL MATCH (matched_entity:Entity)-[r]-(other:Entity)
        WHERE matched_entity.entity_id IN matched_entities
          AND (
              other.entity_id IN neighbor_entities
              OR other.entity_id IN $matched_entity_ids
              OR other.entity_id IN $neighbor_entity_ids
          )
        RETURN c.chunk_id AS chunk_id,
               matched_entities AS matched_entity_ids,
               neighbor_entities AS neighbor_entity_ids,
               collect(DISTINCT type(r)) AS typed_relation_types,
               size(matched_entities) AS matched_entity_count,
               size(neighbor_entities) AS neighbor_entity_count,
               count(DISTINCT r) AS typed_relation_count
        """
        return self.run_read_query(
            cypher,
            {
                "chunk_ids": cleaned_chunk_ids,
                "matched_entity_ids": cleaned_matched_entity_ids,
                "neighbor_entity_ids": cleaned_neighbor_entity_ids,
            },
        )
