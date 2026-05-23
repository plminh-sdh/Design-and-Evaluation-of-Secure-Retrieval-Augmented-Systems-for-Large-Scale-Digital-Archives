"""Driver boundaries for retrieval backends used by retrieval strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from neo4j import GraphDatabase
from qdrant_client import QdrantClient


@dataclass(frozen=True)
class QdrantRetrievalDriver:
    """Thin Qdrant access boundary for retrieval strategies.

    All dense/vector strategies should call Qdrant through this class so future
    access-control or auditing logic has a single place to live.
    """

    url: str | None
    api_key: str | None
    collection_name: str

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
        """Search the configured Qdrant collection."""

        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )


@dataclass(frozen=True)
class Neo4jGraphRetrievalDriver:
    """Thin Neo4j access boundary for graph-expanded retrieval strategies."""

    uri: str | None
    username: str | None
    password: str | None
    database: str | None = None

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
        WHERE toLower(e.name) CONTAINS toLower($query)
           OR toLower($query) CONTAINS toLower(e.name)
        RETURN e.entity_id AS entity_id,
               e.name AS entity_name,
               e.entity_type AS entity_type
        LIMIT $limit
        """
        return self.run_read_query(cypher, {"query": query, "limit": limit})
