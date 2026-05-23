"""Graph-expanded retrieval helpers for the retrieval component notebook."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import time
from typing import Any, Mapping, Sequence

import pandas as pd
from qdrant_client import models

from src.archive_schema import stable_id
from src.graph_kb_canonicalization import _entity_from_refined_span
from src.retrieval_drivers import Neo4jGraphRetrievalDriver
from src.retrieval_hybrid import QdrantHybridRrfRetriever
from src.retrieval_sparse import qdrant_points_to_retrieval_results


DEFAULT_GRAPH_QUERY_MENTION_LABELS = [
    "person",
    "organization",
    "location",
    "date",
    "event",
    "product or brand",
    "creative work title",
    "law or regulation",
    "nationality",
    "religious group",
    "political group",
    "language",
    "program or initiative",
]
DEFAULT_MENTION_LABEL_ALIASES = {
    "PRODUCT_OR_BRAND": "PRODUCT",
    "CREATIVE_WORK_TITLE": "CREATIVE_WORK",
    "LAW_OR_REGULATION": "LAW",
    "RELIGIOUS_GROUP": "RELIGION",
    "POLITICAL_GROUP": "POLITICAL_GROUP",
    "PROGRAM_OR_INITIATIVE": "PROGRAM_INITIATIVE",
}
QUERY_SURFACE_STOPWORDS = {
    "A",
    "An",
    "And",
    "Are",
    "Did",
    "Do",
    "Does",
    "How",
    "In",
    "Of",
    "On",
    "Or",
    "The",
    "To",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Why",
}


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000


def _debug_print(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def model_device_summary(model: Any) -> str:
    """Return a compact best-effort device summary for model-like objects."""

    if model is None:
        return "not loaded"
    if hasattr(model, "device"):
        try:
            return str(model.device)
        except Exception:
            pass
    for attr_name in ("model", "encoder", "span_encoder"):
        child = getattr(model, attr_name, None)
        if child is None:
            continue
        if hasattr(child, "device"):
            try:
                return str(child.device)
            except Exception:
                pass
        try:
            parameter = next(child.parameters())
            return str(parameter.device)
        except Exception:
            pass
    try:
        parameter = next(model.parameters())
        return str(parameter.device)
    except Exception:
        return type(model).__name__


def require_model_on_cuda(model: Any, *, model_name: str) -> None:
    """Raise if a loaded model does not appear to have CUDA-resident parameters."""

    summary = model_device_summary(model)
    if "cuda" not in summary.lower():
        raise RuntimeError(
            f"{model_name} is not on CUDA. Device summary: {summary}. "
            "Move the model to CUDA or set the corresponding require-CUDA flag to False."
        )


def _normalize_label(label: Any) -> str:
    normalized = (
        str(label or "")
        .strip()
        .upper()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(",", "")
    )
    return DEFAULT_MENTION_LABEL_ALIASES.get(normalized, normalized)


def _compact_terms(terms: Sequence[Any], *, limit: int = 32) -> list[str]:
    seen: set[str] = set()
    compacted: list[str] = []
    for term in terms:
        text = re.sub(r"\s+", " ", str(term or "").strip())
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        compacted.append(text)
        if len(compacted) >= limit:
            break
    return compacted


def extract_query_mentions_with_gliner(
    query: str,
    gliner_model: Any,
    *,
    labels: Sequence[str] = DEFAULT_GRAPH_QUERY_MENTION_LABELS,
    threshold: float = 0.4,
) -> list[dict[str, Any]]:
    """Extract entity-like query mentions using the same GLiNER API as Database.ipynb."""

    if not query.strip():
        return []
    predictions = gliner_model.predict_entities(
        query,
        list(labels),
        threshold=threshold,
    )
    mentions = []
    for prediction in predictions or []:
        mention_text = str(prediction.get("text") or "").strip()
        if not mention_text:
            continue
        mentions.append(
            {
                "text": mention_text,
                "label": _normalize_label(prediction.get("label")),
                "start": prediction.get("start"),
                "end": prediction.get("end"),
                "score": prediction.get("score"),
            }
        )
    return mentions


def fallback_query_surface_terms(query: str, *, max_terms: int = 8) -> list[str]:
    """Return deterministic surface terms when model-based mention extraction is unavailable."""

    quoted_terms = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
    surfaces = [left or right for left, right in quoted_terms]
    surfaces.extend(re.findall(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*)*", query))
    surfaces.extend(re.findall(r"\b[A-Z]{2,}\b", query))
    surfaces = [
        surface
        for surface in surfaces
        if surface.strip() not in QUERY_SURFACE_STOPWORDS
    ]
    return _compact_terms(surfaces, limit=max_terms)


def canonicalize_query_mentions_with_refined(
    query: str,
    mentions: Sequence[Mapping[str, Any]],
    refined_model: Any,
    *,
    max_batch_size: int = 16,
    ner_threshold: float = 0.5,
    min_link_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Canonicalize query mentions with the same ReFinED span path used for chunks."""

    if not mentions:
        return []

    from refined.data_types.base_types import Span

    spans = []
    span_mentions = []
    for mention in mentions:
        start = mention.get("start")
        end = mention.get("end")
        if start is None or end is None:
            continue
        try:
            start_int = int(start)
            end_int = int(end)
        except Exception:
            continue
        if start_int < 0 or end_int <= start_int:
            continue
        spans.append(
            Span(
                text=str(mention.get("text") or ""),
                start=start_int,
                ln=end_int - start_int,
                coarse_type="MENTION",
                coarse_mention_type=mention.get("label"),
            )
        )
        span_mentions.append(mention)

    if not spans:
        return []

    predicted_spans = refined_model.process_text(
        query,
        spans=spans,
        ner_threshold=ner_threshold,
        max_batch_size=max_batch_size,
        return_special_spans=True,
    )
    by_position = {
        (getattr(span, "start", None), getattr(span, "end", None)): span
        for span in predicted_spans
    }
    canonicalized = []
    for mention, fallback_span in zip(span_mentions, spans):
        span = by_position.get(
            (int(mention["start"]), int(mention["end"])),
            fallback_span,
        )
        entity = _entity_from_refined_span(span)
        qid = entity.get("external_kb_id")
        link_score = entity.get("link_score")
        linked = qid and (link_score is None or float(link_score) >= min_link_score)
        canonicalized.append(
            {
                "mention_text": mention.get("text"),
                "mention_type": mention.get("label"),
                "start": mention.get("start"),
                "end": mention.get("end"),
                "mention_score": mention.get("score"),
                "external_kb_id": qid,
                "wikipedia_entity_title": entity.get("wikipedia_entity_title"),
                "link_score": link_score,
                "entity_id": stable_id("entity", "wikidata", qid) if linked else None,
                "canonical_name": entity.get("wikipedia_entity_title") if linked else None,
                "canonicalization_status": "LINKED" if linked else "NIL",
            }
        )
    return canonicalized


@dataclass
class GraphQueryExpander:
    """Build compact graph query-expansion variants from query entities."""

    neo4j_driver: Neo4jGraphRetrievalDriver
    gliner_model: Any | None = None
    refined_model: Any | None = None
    mention_labels: Sequence[str] = field(default_factory=lambda: DEFAULT_GRAPH_QUERY_MENTION_LABELS)
    mention_threshold: float = 0.4
    refined_min_link_score: float = 0.0
    entity_limit: int = 8
    alias_limit: int = 8
    neighbor_limit: int = 12
    max_hint_terms: int = 40
    max_expanded_queries: int = 2
    include_mention_surface_lookup: bool = False

    def expand(self, query: str, *, debug: bool = False) -> dict[str, Any]:
        started_at = time.perf_counter()
        _debug_print(
            debug,
            "[graph-expand] start "
            f"gliner={model_device_summary(self.gliner_model)} "
            f"refined={model_device_summary(self.refined_model)}",
        )

        step_started_at = time.perf_counter()
        mentions = (
            extract_query_mentions_with_gliner(
                query,
                self.gliner_model,
                labels=self.mention_labels,
                threshold=self.mention_threshold,
            )
            if self.gliner_model is not None
            else []
        )
        _debug_print(
            debug,
            f"[graph-expand] GLiNER mentions: {len(mentions)} in {_elapsed_ms(step_started_at):.1f} ms",
        )

        step_started_at = time.perf_counter()
        fallback_surfaces = fallback_query_surface_terms(query)
        mention_surfaces = _compact_terms(
            [mention.get("text") for mention in mentions] + fallback_surfaces,
            limit=16,
        )
        _debug_print(
            debug,
            "[graph-expand] surface terms: "
            f"{mention_surfaces or 'none'} in {_elapsed_ms(step_started_at):.1f} ms",
        )

        step_started_at = time.perf_counter()
        canonicalized_mentions = (
            canonicalize_query_mentions_with_refined(
                query,
                mentions,
                self.refined_model,
                min_link_score=self.refined_min_link_score,
            )
            if self.refined_model is not None and mentions
            else []
        )
        _debug_print(
            debug,
            "[graph-expand] ReFinED canonicalized mentions: "
            f"{len(canonicalized_mentions)} in {_elapsed_ms(step_started_at):.1f} ms",
        )
        linked_entity_ids = _compact_terms(
            [
                mention.get("entity_id")
                for mention in canonicalized_mentions
                if mention.get("entity_id")
            ],
            limit=16,
        )

        step_started_at = time.perf_counter()
        _debug_print(
            debug,
            "[graph-expand] Neo4j hint lookup start "
            f"linked_entity_ids={len(linked_entity_ids)} "
            f"surface_terms={len(mention_surfaces)} "
            f"mention_surface_lookup={self.include_mention_surface_lookup}",
        )
        if linked_entity_ids:
            graph_hints = self.neo4j_driver.graph_expansion_hints_by_entity_ids(
                entity_ids=linked_entity_ids,
                alias_limit=self.alias_limit,
                neighbor_limit=self.neighbor_limit,
            )
        else:
            graph_hints = self.neo4j_driver.graph_expansion_hints(
                surface_terms=mention_surfaces,
                entity_ids=linked_entity_ids,
                entity_limit=self.entity_limit,
                alias_limit=self.alias_limit,
                neighbor_limit=self.neighbor_limit,
                include_mention_surface_lookup=self.include_mention_surface_lookup,
            )
        _debug_print(
            debug,
            f"[graph-expand] Neo4j graph hints: {len(graph_hints)} in {_elapsed_ms(step_started_at):.1f} ms",
        )

        step_started_at = time.perf_counter()
        hint_terms, relation_phrases = self._hint_terms_from_graph(graph_hints)
        expanded_queries = self._build_expanded_queries(
            query,
            hint_terms=hint_terms,
            relation_phrases=relation_phrases,
        )
        _debug_print(
            debug,
            "[graph-expand] built "
            f"{len(expanded_queries)} query variants, {len(hint_terms)} hint terms, "
            f"{len(relation_phrases)} relation phrases in {_elapsed_ms(step_started_at):.1f} ms",
        )
        _debug_print(debug, f"[graph-expand] total expansion time: {_elapsed_ms(started_at):.1f} ms")
        return {
            "query": query,
            "mentions": mentions,
            "canonicalized_mentions": canonicalized_mentions,
            "fallback_surfaces": fallback_surfaces,
            "graph_hints": graph_hints,
            "hint_terms": hint_terms,
            "relation_phrases": relation_phrases,
            "expanded_queries": expanded_queries,
        }

    def _hint_terms_from_graph(
        self,
        graph_hints: Sequence[Mapping[str, Any]],
    ) -> tuple[list[str], list[str]]:
        terms: list[str] = []
        relation_phrases: list[str] = []
        for hint in graph_hints:
            canonical_name = hint.get("canonical_name")
            terms.append(canonical_name)
            terms.extend(hint.get("aliases") or [])
            for neighbor in hint.get("relation_neighbors") or []:
                neighbor_name = neighbor.get("neighbor_name")
                relation_type = str(neighbor.get("relation_type") or "").replace("_", " ").lower()
                terms.append(neighbor_name)
                if canonical_name and neighbor_name and relation_type:
                    relation_phrases.append(f"{canonical_name} {relation_type} {neighbor_name}")
        return (
            _compact_terms(terms, limit=self.max_hint_terms),
            _compact_terms(relation_phrases, limit=max(8, self.max_hint_terms // 3)),
        )

    def _build_expanded_queries(
        self,
        query: str,
        *,
        hint_terms: Sequence[str],
        relation_phrases: Sequence[str],
    ) -> list[str]:
        variants = [query]
        if hint_terms:
            variants.append(f"{query} Entities and aliases: {'; '.join(hint_terms[:24])}")
        if relation_phrases:
            variants.append(f"{query} Graph relation hints: {'; '.join(relation_phrases[:12])}")
        return _compact_terms(variants, limit=max(1, self.max_expanded_queries + 1))


@dataclass
class GraphExpandedHybridRetriever:
    """Graph query expansion followed by one Qdrant multi-prefetch RRF search."""

    graph_expander: GraphQueryExpander
    hybrid_retriever: QdrantHybridRrfRetriever
    method_name: str = "graph_expanded_hybrid"
    rrf_k: int | None = None
    prefetch_multiplier: int = 5
    min_prefetch_limit: int = 20
    last_debug: dict[str, Any] = field(default_factory=dict)

    def retrieve(
        self,
        query: str,
        *,
        query_id: str = "sample_query",
        top_k: int = 10,
        query_filter: Any | None = None,
        with_payload: bool = True,
        debug: bool = False,
    ) -> pd.DataFrame:
        started_at = time.perf_counter()
        _debug_print(debug, "[graph-retrieval] start graph-expanded hybrid retrieval")
        expansion = self.graph_expander.expand(query, debug=debug)
        expanded_queries = expansion.get("expanded_queries") or [query]
        _debug_print(
            debug,
            f"[graph-retrieval] expanded query variants ready: {len(expanded_queries)}",
        )

        step_started_at = time.perf_counter()
        prefetch_limit = max(self.min_prefetch_limit, top_k * self.prefetch_multiplier)
        prefetches = self._build_prefetches(
            expanded_queries,
            prefetch_limit=prefetch_limit,
            query_filter=query_filter,
            debug=debug,
        )
        _debug_print(
            debug,
            f"[graph-retrieval] built {len(prefetches)} Qdrant prefetches "
            f"in {_elapsed_ms(step_started_at):.1f} ms",
        )

        step_started_at = time.perf_counter()
        points = self.hybrid_retriever.qdrant_driver.search_prefetch_rrf(
            prefetches,
            top_k=top_k,
            with_payload=with_payload,
            with_vectors=False,
            rrf_k=self.rrf_k,
        )
        _debug_print(
            debug,
            f"[graph-retrieval] Qdrant multi-prefetch RRF returned {len(points)} points "
            f"in {_elapsed_ms(step_started_at):.1f} ms",
        )

        step_started_at = time.perf_counter()
        latency_ms = _elapsed_ms(started_at)
        results_df = qdrant_points_to_retrieval_results(
            points,
            query=query,
            query_id=query_id,
            retrieval_method=self.method_name,
            latency_ms=latency_ms,
            retrieval_stage="graph_expanded_hybrid",
        )
        _debug_print(
            debug,
            f"[graph-retrieval] converted results to dataframe in {_elapsed_ms(step_started_at):.1f} ms",
        )
        if not results_df.empty:
            results_df["expanded_query_count"] = len(expanded_queries)
            results_df["prefetch_count"] = len(prefetches)
            results_df["graph_hint_terms"] = "; ".join(expansion.get("hint_terms") or [])
            results_df["graph_relation_phrases"] = "; ".join(expansion.get("relation_phrases") or [])

        self.last_debug = expansion
        if debug:
            self.print_debug(expansion)
            print(f"\n[graph-retrieval] total retrieval time: {_elapsed_ms(started_at):.1f} ms")
        return results_df

    def _build_prefetches(
        self,
        expanded_queries: Sequence[str],
        *,
        prefetch_limit: int,
        query_filter: Any | None = None,
        debug: bool = False,
    ) -> list[models.Prefetch]:
        prefetches: list[models.Prefetch] = []
        for expanded_query in expanded_queries:
            step_started_at = time.perf_counter()
            dense_query_vector, sparse_query_vector = self.hybrid_retriever.encode_query_vectors(
                expanded_query
            )
            _debug_print(
                debug,
                "[graph-retrieval] embedded query variant "
                f"{len(prefetches) // 2}: dense={len(dense_query_vector)} "
                f"sparse={len(sparse_query_vector.indices)} "
                f"in {_elapsed_ms(step_started_at):.1f} ms",
            )
            prefetches.append(
                models.Prefetch(
                    query=sparse_query_vector,
                    using=self.hybrid_retriever.sparse_vector_name,
                    limit=prefetch_limit,
                    filter=query_filter,
                )
            )
            prefetches.append(
                models.Prefetch(
                    query=list(dense_query_vector),
                    using=self.hybrid_retriever.dense_vector_name,
                    limit=prefetch_limit,
                    filter=query_filter,
                )
            )
        return prefetches

    @staticmethod
    def print_debug(expansion: Mapping[str, Any]) -> None:
        print("Original query:")
        print(expansion.get("query", ""))
        print("\nExtracted mentions:")
        for mention in expansion.get("mentions") or []:
            print(f"- {mention.get('text')} [{mention.get('label')}] score={mention.get('score')}")
        print("\nCanonicalized mentions:")
        for mention in expansion.get("canonicalized_mentions") or []:
            print(
                "- "
                f"{mention.get('mention_text')} -> {mention.get('canonical_name') or mention.get('canonicalization_status')} "
                f"({mention.get('external_kb_id')})"
            )
        print("\nHint terms:")
        print("; ".join(expansion.get("hint_terms") or []))
        print("\nRelation phrases:")
        print("; ".join(expansion.get("relation_phrases") or []))
        print("\nExpanded query variants:")
        for index, expanded_query in enumerate(expansion.get("expanded_queries") or []):
            print(f"{index}. {expanded_query}")


@dataclass
class GraphExpandedHybridReranker:
    """Post-retrieval graph-feature reranker for graph-expanded hybrid results."""

    graph_expanded_retriever: GraphExpandedHybridRetriever
    neo4j_driver: Neo4jGraphRetrievalDriver
    method_name: str = "graph_expanded_hybrid_reranked"
    candidate_multiplier: int = 5
    matched_entity_boost: float = 0.15
    neighbor_entity_boost: float = 0.08
    typed_relation_boost: float = 0.10
    last_debug: dict[str, Any] = field(default_factory=dict)

    def retrieve(
        self,
        query: str,
        *,
        query_id: str = "sample_query",
        top_k: int = 10,
        query_filter: Any | None = None,
        with_payload: bool = True,
        debug: bool = False,
    ) -> pd.DataFrame:
        started_at = time.perf_counter()
        candidate_top_k = max(top_k, top_k * self.candidate_multiplier)
        _debug_print(
            debug,
            "[graph-rerank] retrieving candidates "
            f"candidate_top_k={candidate_top_k} final_top_k={top_k}",
        )
        candidates_df = self.graph_expanded_retriever.retrieve(
            query,
            query_id=query_id,
            top_k=candidate_top_k,
            query_filter=query_filter,
            with_payload=with_payload,
            debug=debug,
        )
        if candidates_df.empty:
            return candidates_df

        expansion = self.graph_expanded_retriever.last_debug or {}
        matched_entity_ids, neighbor_entity_ids = self._entity_sets_from_expansion(expansion)
        chunk_ids = candidates_df["chunk_id"].dropna().astype(str).tolist()

        step_started_at = time.perf_counter()
        feature_rows = self.neo4j_driver.graph_candidate_chunk_features(
            chunk_ids=chunk_ids,
            matched_entity_ids=matched_entity_ids,
            neighbor_entity_ids=neighbor_entity_ids,
        )
        _debug_print(
            debug,
            f"[graph-rerank] Neo4j candidate features: {len(feature_rows)} rows "
            f"in {_elapsed_ms(step_started_at):.1f} ms",
        )
        feature_df = pd.DataFrame(feature_rows)
        reranked_df = self._rerank_candidates(candidates_df, feature_df)
        reranked_df = reranked_df.head(top_k).copy()
        reranked_df["rank"] = range(1, len(reranked_df) + 1)
        reranked_df["retrieval_method"] = self.method_name
        reranked_df["retrieval_stage"] = "graph_expanded_hybrid_reranked"
        reranked_df["latency_ms"] = _elapsed_ms(started_at)

        self.last_debug = {
            "expansion": expansion,
            "matched_entity_ids": matched_entity_ids,
            "neighbor_entity_ids": neighbor_entity_ids,
            "candidate_features": feature_rows,
        }
        if debug:
            print("\n[graph-rerank] matched entity ids:", matched_entity_ids)
            print("[graph-rerank] neighbor entity ids:", neighbor_entity_ids)
            print(f"[graph-rerank] total reranked retrieval time: {_elapsed_ms(started_at):.1f} ms")
        return reranked_df

    def _entity_sets_from_expansion(
        self,
        expansion: Mapping[str, Any],
    ) -> tuple[list[str], list[str]]:
        matched_entity_ids = _compact_terms(
            [
                mention.get("entity_id")
                for mention in expansion.get("canonicalized_mentions") or []
                if mention.get("entity_id")
            ],
            limit=32,
        )
        neighbor_entity_ids: list[str] = []
        for hint in expansion.get("graph_hints") or []:
            entity_id = hint.get("entity_id")
            if entity_id:
                matched_entity_ids.append(entity_id)
            for neighbor in hint.get("relation_neighbors") or []:
                neighbor_id = neighbor.get("neighbor_entity_id")
                if neighbor_id:
                    neighbor_entity_ids.append(neighbor_id)
        return (
            _compact_terms(matched_entity_ids, limit=64),
            _compact_terms(neighbor_entity_ids, limit=128),
        )

    def _rerank_candidates(
        self,
        candidates_df: pd.DataFrame,
        feature_df: pd.DataFrame,
    ) -> pd.DataFrame:
        candidates = candidates_df.copy()
        if feature_df.empty:
            candidates["matched_entity_count"] = 0
            candidates["neighbor_entity_count"] = 0
            candidates["typed_relation_count"] = 0
            candidates["graph_boost"] = 0.0
            candidates["base_score"] = candidates["score"].astype(float)
            candidates["score"] = candidates["base_score"]
            return candidates.sort_values(["score", "rank"], ascending=[False, True])

        features = feature_df.copy()
        for column in [
            "matched_entity_count",
            "neighbor_entity_count",
            "typed_relation_count",
        ]:
            if column not in features.columns:
                features[column] = 0
            features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0)

        candidates = candidates.merge(features, on="chunk_id", how="left")
        for column in [
            "matched_entity_count",
            "neighbor_entity_count",
            "typed_relation_count",
        ]:
            candidates[column] = pd.to_numeric(candidates[column], errors="coerce").fillna(0)

        candidates["has_matched_query_entity"] = candidates["matched_entity_count"] > 0
        candidates["has_graph_neighbor_entity"] = candidates["neighbor_entity_count"] > 0
        candidates["has_typed_relation_evidence"] = candidates["typed_relation_count"] > 0
        candidates["base_score"] = candidates["score"].astype(float)
        candidates["graph_boost"] = (
            self.matched_entity_boost * candidates["has_matched_query_entity"].astype(float)
            + self.neighbor_entity_boost * candidates["has_graph_neighbor_entity"].astype(float)
            + self.typed_relation_boost * candidates["has_typed_relation_evidence"].astype(float)
        )
        candidates["score"] = candidates["base_score"] + candidates["graph_boost"]
        return candidates.sort_values(
            ["score", "base_score", "rank"],
            ascending=[False, False, True],
        )
