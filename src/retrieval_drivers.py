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
        WHERE toLower(e.name) CONTAINS toLower($query)
           OR toLower($query) CONTAINS toLower(e.name)
        RETURN e.entity_id AS entity_id,
               e.name AS entity_name,
               e.entity_type AS entity_type
        LIMIT $limit
        """
        return self.run_read_query(cypher, {"query": query, "limit": limit})
