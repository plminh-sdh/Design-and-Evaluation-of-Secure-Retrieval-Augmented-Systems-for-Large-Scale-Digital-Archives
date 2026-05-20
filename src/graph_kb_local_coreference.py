"""Local NIL coreference helpers for graph knowledge-base canonicalization."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.archive_schema import append_dataframe_to_csv, stable_id


LOCAL_COREFERENCE_EXPORT_DIR = (
    Path("data") / "graph_kb_exports" / "step_03b_local_coreference"
)
LOCAL_NIL_MENTIONS_FILENAME = "nil_mentions.csv"
LOCAL_COREFERENCE_GOLD_DEV_FILENAME = "local_coreference_gold_dev.jsonl"
LOCAL_COREFERENCE_GOLD_TEST_FILENAME = "local_coreference_gold_test.jsonl"

DEFAULT_LOCAL_COREFERENCE_TYPES = (
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "EVENT",
    "PROGRAM_INITIATIVE",
    "PRODUCT",
    "CREATIVE_WORK",
    "LAW",
    "POLITICAL_GROUP",
    "RELIGION",
)

CONTACT_DETAIL_TYPES = {"CONTACT_DETAIL", "EMAIL", "PHONE", "PHONE_NUMBER"}
GENERIC_PRONOUN_SURFACES = {
    "i",
    "me",
    "my",
    "mine",
    "you",
    "your",
    "yours",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "we",
    "us",
    "our",
    "ours",
    "they",
    "them",
    "their",
    "theirs",
}
GENERIC_LOCAL_COREFERENCE_SURFACES = {
    *GENERIC_PRONOUN_SURFACES,
    "a boy",
    "a car",
    "a company",
    "a country",
    "a girl",
    "a group",
    "a guy",
    "a man",
    "a person",
    "a woman",
    "an official",
    "another person",
    "bystander",
    "bill",
    "company",
    "country",
    "friend of mine",
    "government",
    "here",
    "inside",
    "official",
    "other person",
    "outside",
    "person",
    "rule",
    "rules",
    "someone",
    "somebody",
    "there",
    "the company",
    "the country",
    "the government",
    "the group",
    "the official",
    "the organization",
    "the person",
    "your wife",
}
COMMON_ROLE_NOUNS = {
    "administration",
    "agency",
    "analyst",
    "association",
    "authorities",
    "authority",
    "board",
    "campaign",
    "chairman",
    "chairwoman",
    "city",
    "committee",
    "company",
    "corporation",
    "country",
    "department",
    "director",
    "doctor",
    "executive",
    "firm",
    "government",
    "governor",
    "group",
    "hospital",
    "lawyer",
    "leader",
    "manager",
    "mayor",
    "minister",
    "official",
    "officer",
    "president",
    "prime minister",
    "professor",
    "program",
    "reporter",
    "researcher",
    "school",
    "secretary",
    "senator",
    "spokesman",
    "spokesperson",
    "spokeswoman",
    "state",
    "university",
}
GENERIC_DESCRIPTOR_SUFFIXES = COMMON_ROLE_NOUNS | {
    "boy",
    "friend",
    "girl",
    "guy",
    "husband",
    "man",
    "people",
    "person",
    "wife",
    "woman",
}
GENERIC_ORGANIZATION_DESCRIPTOR_SUFFIXES = {
    "authorities",
    "center",
    "commission",
    "entities",
    "judges",
    "owners",
    "voters",
    "workers",
}
GENERIC_EVENT_DESCRIPTOR_SUFFIXES = {
    "demonstration",
    "discussion",
    "game",
    "issue",
    "meeting",
}
AMBIGUOUS_SHORT_NAMES = {
    "april",
    "august",
    "black",
    "brown",
    "china",
    "christian",
    "france",
    "georgia",
    "green",
    "hope",
    "jack",
    "james",
    "john",
    "jordan",
    "june",
    "king",
    "lee",
    "mark",
    "martin",
    "mary",
    "may",
    "paris",
    "rose",
    "smith",
    "spring",
    "turkey",
    "washington",
    "white",
}
BLOCK_DESCRIPTOR_STOPWORDS = {
    "campaign",
    "case",
    "event",
    "foundation",
    "group",
    "initiative",
    "law",
    "movement",
    "office",
    "operation",
    "organization",
    "party",
    "plan",
    "program",
    "project",
    "system",
    "team",
    "the",
}
DEFAULT_TYPE_SIMILARITY_THRESHOLDS = {
    "PERSON": 0.92,
    "ORGANIZATION": 0.88,
    "LOCATION": 0.90,
    "EVENT": 0.88,
    "PROGRAM_INITIATIVE": 0.86,
    "PRODUCT": 0.88,
    "CREATIVE_WORK": 0.88,
    "LAW": 0.90,
    "POLITICAL_GROUP": 0.90,
    "RELIGION": 0.92,
}

NIL_MENTION_COLUMNS = [
    "mention_id",
    "chunk_id",
    "document_id",
    "dataset",
    "modality",
    "mention_text",
    "mention_type",
    "start_char:int",
    "end_char:int",
    "link_score:float",
    "canonicalization_status",
    "canonicalization_method",
    "canonicalizer",
    "model_name",
    "model_version",
    "normalized_mention_text",
    "surface_block_key",
]
def normalize_local_coreference_surface(value: Any) -> str:
    """Normalize a mention surface for blocking and diagnostics."""
    if value is None:
        return ""
    text = str(value).strip().casefold()
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def local_coreference_surface_block_key(value: Any) -> str:
    """Create a conservative surface key used before contextual clustering."""
    normalized = normalize_local_coreference_surface(value)
    if not normalized:
        return ""
    tokens = normalized.split()
    if len(tokens) == 1:
        token = tokens[0]
        return token[:4] if len(token) > 4 else token
    informative_tokens = [
        token
        for token in tokens
        if token not in BLOCK_DESCRIPTOR_STOPWORDS and len(token) > 2
    ]
    if informative_tokens:
        return max(informative_tokens, key=len)
    return " ".join(tokens[:2])


def is_acronym(value: Any) -> bool:
    """Return whether a surface form looks like an acronym."""
    text = re.sub(r"[^A-Za-z]", "", str(value or ""))
    return 2 <= len(text) <= 8 and text.isupper()


def is_common_role_surface(value: Any) -> bool:
    """Return whether a surface is a generic role/group noun."""
    normalized = normalize_local_coreference_surface(value)
    return normalized in COMMON_ROLE_NOUNS


def is_obvious_generic_local_coreference_surface(
    value: Any,
    *,
    mention_type: str | None = None,
) -> bool:
    """Return whether a mention is too generic to be clustered as a local entity."""
    normalized = normalize_local_coreference_surface(value)
    tokens = normalized.split()
    mention_type = str(mention_type or "").upper()
    if not normalized:
        return True
    if normalized in GENERIC_LOCAL_COREFERENCE_SURFACES:
        return True
    if len(tokens) == 1 and tokens[0] in COMMON_ROLE_NOUNS:
        return True

    if mention_type == "PERSON":
        if normalized.startswith(
            (
                "a ",
                "an ",
                "another ",
                "her ",
                "his ",
                "my ",
                "other ",
                "our ",
                "some ",
                "their ",
                "your ",
            )
        ):
            return True
        if tokens and tokens[-1] in GENERIC_DESCRIPTOR_SUFFIXES:
            return True
    elif mention_type in {"ORGANIZATION", "POLITICAL_GROUP"}:
        if normalized.startswith(("a ", "an ", "another ", "other ", "some ")):
            return True
        if tokens and tokens[-1] in GENERIC_ORGANIZATION_DESCRIPTOR_SUFFIXES:
            return True
    elif mention_type == "EVENT":
        if tokens and tokens[-1] in GENERIC_EVENT_DESCRIPTOR_SUFFIXES:
            return True
    elif mention_type == "PRODUCT" and normalized.startswith(("a ", "an ", "some ")):
        return True
    elif mention_type == "LOCATION" and normalized in {"here", "there", "inside", "outside"}:
        return True
    elif mention_type == "LAW" and normalized in {"bill", "rule", "rules", "law", "laws"}:
        return True

    return False


def is_obvious_generic_local_coreference_mention(record: Mapping[str, Any]) -> bool:
    """Return whether a mention record is an obvious generic/non-entity reference."""
    mention_type = str(record.get("mention_type") or "").upper()
    surface = record.get("normalized_mention_text") or record.get("mention_text")
    if mention_type in CONTACT_DETAIL_TYPES:
        return True
    return is_obvious_generic_local_coreference_surface(
        surface,
        mention_type=mention_type,
    )


def is_ambiguous_short_surface(value: Any) -> bool:
    """Return whether a mention surface should not be globally merged."""
    normalized = normalize_local_coreference_surface(value)
    tokens = normalized.split()
    if not tokens:
        return True
    if normalized in AMBIGUOUS_SHORT_NAMES:
        return True
    if len(tokens) == 1 and len(tokens[0]) <= 3:
        return True
    return False


def is_local_coreference_candidate(
    row: Mapping[str, Any],
    *,
    eligible_types: Sequence[str] = DEFAULT_LOCAL_COREFERENCE_TYPES,
    allow_common_roles: bool = False,
    allow_generic_mentions: bool = False,
) -> bool:
    """Apply whitebox merge guards before a NIL mention can be clustered."""
    mention_type = str(row.get("mention_type") or "").upper()
    if mention_type in CONTACT_DETAIL_TYPES:
        return False
    if mention_type not in {str(item).upper() for item in eligible_types}:
        return False

    surface = row.get("normalized_mention_text") or normalize_local_coreference_surface(
        row.get("mention_text")
    )
    if not surface:
        return False
    if not allow_common_roles and is_common_role_surface(surface):
        return False
    if not allow_generic_mentions and is_obvious_generic_local_coreference_mention(row):
        return False
    return True


def guarded_local_coreference_block_key(row: Mapping[str, Any]) -> str:
    """Build a merge block that encodes type and ambiguity guardrails."""
    mention_type = str(row.get("mention_type") or "").upper()
    surface = row.get("normalized_mention_text") or normalize_local_coreference_surface(
        row.get("mention_text")
    )
    surface_block = row.get("surface_block_key") or local_coreference_surface_block_key(surface)

    if is_ambiguous_short_surface(surface):
        dataset = str(row.get("dataset") or "")
        document_id = str(row.get("document_id") or "")
        return f"{mention_type}|ambiguous|{dataset}|{document_id}|{surface_block}"

    if is_acronym(str(row.get("mention_text") or "")):
        dataset = str(row.get("dataset") or "")
        return f"{mention_type}|acronym|{dataset}|{surface_block}"

    return f"{mention_type}|surface|{surface_block}"


def build_local_mention_context(
    row: Mapping[str, Any],
    text: str,
    *,
    left_chars: int = 240,
    right_chars: int = 240,
) -> str:
    """Build a contextual embedding input around a NIL mention span."""
    start = int(row.get("start_char:int", row.get("start_char", 0)) or 0)
    end = int(row.get("end_char:int", row.get("end_char", start)) or start)
    text = str(text or "")
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))

    left_context = text[max(0, start - left_chars) : start].strip()
    mention_text = text[start:end].strip() or str(row.get("mention_text") or "")
    right_context = text[end : min(len(text), end + right_chars)].strip()

    return "\n".join(
        [
            f"Dataset: {row.get('dataset')}",
            f"Modality: {row.get('modality')}",
            f"Mention type: {row.get('mention_type')}",
            f"Mention: {row.get('mention_text')}",
            f"Context: {left_context} [MENTION] {mention_text} [/MENTION] {right_context}",
        ]
    )


def prepare_local_coreference_mentions(
    nil_mentions_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    *,
    eligible_types: Sequence[str] = DEFAULT_LOCAL_COREFERENCE_TYPES,
    text_column: str = "masked_text",
    allow_common_roles: bool = False,
    allow_generic_mentions: bool = False,
) -> pd.DataFrame:
    """Join NIL mentions to chunk text and create guarded block/context columns."""
    mentions = nil_mentions_df.copy()
    mentions["mention_type"] = mentions["mention_type"].astype(str).str.upper()
    if "normalized_mention_text" not in mentions.columns:
        mentions["normalized_mention_text"] = mentions["mention_text"].map(
            normalize_local_coreference_surface
        )
    if "surface_block_key" not in mentions.columns:
        mentions["surface_block_key"] = mentions["mention_text"].map(
            local_coreference_surface_block_key
        )

    candidate_mask = mentions.apply(
        lambda row: is_local_coreference_candidate(
            row,
            eligible_types=eligible_types,
            allow_common_roles=allow_common_roles,
            allow_generic_mentions=allow_generic_mentions,
        ),
        axis=1,
    )
    mentions = mentions[candidate_mask].copy()
    if mentions.empty:
        return mentions

    chunk_id_col = "chunk_id:ID(Chunk)" if "chunk_id:ID(Chunk)" in chunks_df.columns else "chunk_id"
    chunk_columns = [chunk_id_col, text_column]
    chunk_columns = [column for column in chunk_columns if column in chunks_df.columns]
    chunks = chunks_df[chunk_columns].rename(columns={chunk_id_col: "chunk_id"}).copy()
    mentions = mentions.merge(chunks, on="chunk_id", how="left")
    mentions[text_column] = mentions[text_column].fillna("")
    mentions["block_key"] = mentions.apply(guarded_local_coreference_block_key, axis=1)
    mentions["context_text"] = mentions.apply(
        lambda row: build_local_mention_context(row, row.get(text_column)),
        axis=1,
    )
    return mentions


def _as_numpy_embeddings(embeddings: Any) -> np.ndarray:
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    array = np.asarray(embeddings, dtype="float32")
    if array.ndim != 2:
        raise ValueError("embeddings must be a 2D array-like object")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def embed_local_mention_contexts(
    prepared_mentions_df: pd.DataFrame,
    embedder: Any,
    *,
    text_column: str = "context_text",
    batch_size: int = 8,
) -> np.ndarray:
    """Embed contextual NIL mention strings with a provided embedder/callable."""
    texts = prepared_mentions_df[text_column].fillna("").astype(str).tolist()
    if hasattr(embedder, "encode_texts"):
        embedded = embedder.encode_texts(texts, batch_size=batch_size)
        vectors = [item.dense if hasattr(item, "dense") else item for item in embedded]
    elif callable(embedder):
        vectors = embedder(texts)
    else:
        raise TypeError("embedder must expose encode_texts(...) or be callable")
    return _as_numpy_embeddings(vectors)


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    normalized = _as_numpy_embeddings(embeddings)
    return normalized @ normalized.T


def _cluster_block(
    block_embeddings: np.ndarray,
    *,
    similarity_threshold: float,
) -> np.ndarray:
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for agglomerative NIL coreference clustering."
        ) from exc

    distance_threshold = 1.0 - similarity_threshold
    try:
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
    return model.fit_predict(block_embeddings)


def _cluster_similarity_stats(embeddings: np.ndarray) -> Tuple[float, float]:
    if len(embeddings) < 2:
        return 1.0, 1.0
    similarities = _cosine_similarity_matrix(embeddings)
    upper = similarities[np.triu_indices(len(similarities), k=1)]
    if len(upper) == 0:
        return 1.0, 1.0
    return float(upper.min()), float(upper.mean())


def canonical_name_for_local_cluster(cluster_df: pd.DataFrame) -> str:
    """Choose a readable canonical name while preserving aliases separately."""
    surfaces = [
        str(value).strip()
        for value in cluster_df["mention_text"].dropna().tolist()
        if str(value).strip()
    ]
    if not surfaces:
        return ""

    counts = Counter(surfaces)

    def score(surface: str) -> Tuple[int, int, int, str]:
        all_caps_penalty = int(surface.isupper() and len(surface) > 3)
        return (
            counts[surface],
            -all_caps_penalty,
            len(normalize_local_coreference_surface(surface)),
            surface,
        )

    return max(surfaces, key=score)


def should_exact_surface_cluster(row: Mapping[str, Any]) -> bool:
    """Return whether a mention can safely participate in exact-surface clustering."""
    surface = row.get("normalized_mention_text") or normalize_local_coreference_surface(
        row.get("mention_text")
    )
    if not surface:
        return False
    if is_obvious_generic_local_coreference_mention(row):
        return False
    if is_ambiguous_short_surface(surface):
        return False
    if is_acronym(str(row.get("mention_text") or "")):
        return False
    return True


def _emit_local_cluster_predictions(
    cluster_df: pd.DataFrame,
    cluster_embeddings: np.ndarray,
    *,
    mention_type: str,
    model_name: str,
    model_version: str,
    status: str = "CLUSTERED",
    canonicalizer: str = "contextual_embedding_agglomerative",
) -> list[Dict[str, Any]]:
    canonical_name = canonical_name_for_local_cluster(cluster_df)
    entity_id = stable_id(
        "local_entity",
        mention_type,
        canonical_name,
        "|".join(sorted(cluster_df["mention_id"].astype(str))),
    )
    cluster_id = stable_id("local_cluster", entity_id)
    cluster_min_similarity, cluster_mean_similarity = _cluster_similarity_stats(
        cluster_embeddings
    )

    return [
        _local_prediction_row(
            row,
            entity_id,
            status,
            model_name,
            model_version,
            cluster_id=cluster_id,
            cluster_size=len(cluster_df),
            cluster_min_similarity=cluster_min_similarity,
            cluster_mean_similarity=cluster_mean_similarity,
            canonical_name=canonical_name,
            canonicalizer=canonicalizer,
        )
        for _, row in cluster_df.iterrows()
    ]


def cluster_local_nil_mentions(
    prepared_mentions_df: pd.DataFrame,
    embeddings: Any,
    *,
    similarity_threshold: float = 0.90,
    type_similarity_thresholds: Optional[Mapping[str, float]] = None,
    use_default_type_thresholds: bool = True,
    acronym_similarity_threshold: float = 0.96,
    min_cluster_size: int = 2,
    max_block_size: int = 500,
    exact_surface_first: bool = True,
    model_name: str = "BAAI/bge-m3",
    model_version: str = "unspecified",
    canonicalizer: str = "contextual_embedding_agglomerative",
) -> pd.DataFrame:
    """Cluster prepared NIL mentions with guarded agglomerative clustering.

    The function never emits singleton clusters by default. Mentions in blocks
    that are too small, too large, generic, ambiguous, or below threshold remain
    unresolved.
    """
    mentions = prepared_mentions_df.reset_index(drop=True).copy()
    embeddings_array = _as_numpy_embeddings(embeddings)
    if len(mentions) != len(embeddings_array):
        raise ValueError("prepared_mentions_df and embeddings must have the same length")

    type_thresholds = (
        dict(DEFAULT_TYPE_SIMILARITY_THRESHOLDS)
        if use_default_type_thresholds
        else {}
    )
    if type_similarity_thresholds:
        type_thresholds.update(dict(type_similarity_thresholds))
    prediction_rows: list[Dict[str, Any]] = []

    for block_key, block_df in tqdm(
        mentions.groupby("block_key", sort=False),
        desc="Clustering NIL blocks",
        unit="block",
    ):
        block_indexes = block_df.index.to_numpy()
        block_size = len(block_df)
        mention_type = str(block_df["mention_type"].iloc[0]).upper()

        exact_clustered_indexes: set[int] = set()
        if exact_surface_first and block_df["mention_type"].nunique(dropna=False) == 1:
            exact_candidate_mask = block_df.apply(should_exact_surface_cluster, axis=1)
            exact_candidate_df = block_df[exact_candidate_mask].copy()
            for _, exact_df in exact_candidate_df.groupby(
                "normalized_mention_text",
                sort=False,
                dropna=False,
            ):
                if len(exact_df) < min_cluster_size:
                    continue
                exact_indexes = exact_df.index.to_numpy()
                exact_clustered_indexes.update(int(index) for index in exact_indexes)
                prediction_rows.extend(
                    _emit_local_cluster_predictions(
                        exact_df,
                        embeddings_array[exact_indexes],
                        mention_type=mention_type,
                        model_name=model_name,
                        model_version=model_version,
                        status="EXACT_SURFACE_CLUSTERED",
                        canonicalizer="exact_surface_local_coreference",
                    )
                )

        if exact_clustered_indexes:
            block_df = block_df[~block_df.index.isin(exact_clustered_indexes)].copy()
            block_indexes = block_df.index.to_numpy()
            block_size = len(block_df)
            if block_df.empty:
                continue

        unresolved_reason = None

        if block_size < min_cluster_size:
            unresolved_reason = "BLOCK_TOO_SMALL"
        elif block_size > max_block_size:
            unresolved_reason = "BLOCK_TOO_LARGE"
        elif block_df["mention_type"].nunique(dropna=False) != 1:
            unresolved_reason = "INCOMPATIBLE_TYPES"

        if unresolved_reason:
            for _, row in block_df.iterrows():
                prediction_rows.append(
                    _local_prediction_row(row, None, unresolved_reason, model_name, model_version)
                )
            continue

        block_threshold = type_thresholds.get(mention_type, similarity_threshold)
        if block_key.split("|")[1:2] == ["acronym"]:
            block_threshold = max(block_threshold, acronym_similarity_threshold)

        block_embeddings = embeddings_array[block_indexes]
        labels = _cluster_block(block_embeddings, similarity_threshold=block_threshold)

        for label in sorted(set(labels)):
            cluster_mask = labels == label
            cluster_df = block_df.iloc[np.where(cluster_mask)[0]].copy()
            cluster_embeddings = block_embeddings[cluster_mask]
            if len(cluster_df) < min_cluster_size:
                for _, row in cluster_df.iterrows():
                    prediction_rows.append(
                        _local_prediction_row(
                            row,
                            None,
                            "CLUSTER_TOO_SMALL",
                            model_name,
                            model_version,
                        )
                    )
                continue

            cluster_min_similarity, cluster_mean_similarity = _cluster_similarity_stats(
                cluster_embeddings
            )
            required_similarity = block_threshold
            if any(is_acronym(value) for value in cluster_df["mention_text"].dropna()):
                required_similarity = max(required_similarity, acronym_similarity_threshold)

            if cluster_min_similarity < required_similarity:
                for _, row in cluster_df.iterrows():
                    prediction_rows.append(
                        _local_prediction_row(
                            row,
                            None,
                            "LOW_CLUSTER_SIMILARITY",
                            model_name,
                            model_version,
                        )
                    )
                continue

            prediction_rows.extend(
                _emit_local_cluster_predictions(
                    cluster_df,
                    cluster_embeddings,
                    mention_type=mention_type,
                    model_name=model_name,
                    model_version=model_version,
                    status="CLUSTERED",
                    canonicalizer=canonicalizer,
                )
            )

    return pd.DataFrame(prediction_rows)


def _local_prediction_row(
    row: Mapping[str, Any],
    entity_id: Optional[str],
    status: str,
    model_name: str,
    model_version: str,
    *,
    cluster_id: Optional[str] = None,
    cluster_size: int = 0,
    cluster_min_similarity: Optional[float] = None,
    cluster_mean_similarity: Optional[float] = None,
    canonical_name: Optional[str] = None,
    canonicalizer: str = "contextual_embedding_agglomerative",
) -> Dict[str, Any]:
    return {
        "mention_id": row.get("mention_id"),
        "predicted_local_entity_id": entity_id,
        "entity_id": entity_id,
        "cluster_id": cluster_id,
        "canonicalization_status": status,
        "chunk_id": row.get("chunk_id"),
        "document_id": row.get("document_id"),
        "dataset": row.get("dataset"),
        "modality": row.get("modality"),
        "mention_text": row.get("mention_text"),
        "mention_type": row.get("mention_type"),
        "normalized_mention_text": row.get("normalized_mention_text"),
        "surface_block_key": row.get("surface_block_key"),
        "block_key": row.get("block_key"),
        "cluster_size:int": cluster_size,
        "cluster_min_similarity:float": cluster_min_similarity,
        "cluster_mean_similarity:float": cluster_mean_similarity,
        "canonical_name": canonical_name,
        "canonicalization_method": "local_coreference_cluster" if entity_id else None,
        "canonicalizer": canonicalizer,
        "model_name": model_name,
        "model_version": model_version,
    }


def tune_local_coreference_thresholds(
    gold_records: Sequence[Mapping[str, Any]],
    prepared_mentions_df: pd.DataFrame,
    embeddings: Any,
    thresholds: Sequence[float],
    *,
    precision_floor: float = 0.90,
    min_cluster_size: int = 2,
    max_block_size: int = 500,
    exact_surface_first: bool = True,
    use_type_threshold_floors: bool = False,
    model_name: str = "BAAI/bge-m3",
    model_version: str = "unspecified",
) -> Tuple[pd.DataFrame, Dict[float, pd.DataFrame]]:
    """Run local clustering over dev thresholds and score against gold."""
    gold_mention_ids = {str(record["mention_id"]) for record in gold_records}
    gold_mask = prepared_mentions_df["mention_id"].astype(str).isin(gold_mention_ids)
    prepared = prepared_mentions_df[gold_mask].reset_index(drop=True)
    prepared_gold_mention_ids = set(prepared["mention_id"].astype(str))
    prepared_gold_records = [
        record
        for record in gold_records
        if str(record.get("mention_id")) in prepared_gold_mention_ids
    ]
    embeddings_array = _as_numpy_embeddings(embeddings)
    if len(embeddings_array) != len(prepared_mentions_df):
        raise ValueError("embeddings must align with prepared_mentions_df")
    selected_embeddings = embeddings_array[np.flatnonzero(gold_mask.to_numpy())]

    metrics_rows = []
    predictions_by_threshold: Dict[float, pd.DataFrame] = {}
    for threshold in thresholds:
        threshold_value = float(threshold)
        if use_type_threshold_floors:
            tuned_type_thresholds = {
                mention_type: max(threshold_value, default_threshold)
                for mention_type, default_threshold in DEFAULT_TYPE_SIMILARITY_THRESHOLDS.items()
            }
        else:
            tuned_type_thresholds = {
                mention_type: threshold_value
                for mention_type in DEFAULT_TYPE_SIMILARITY_THRESHOLDS
            }
        predictions_df = cluster_local_nil_mentions(
            prepared,
            selected_embeddings,
            similarity_threshold=threshold_value,
            type_similarity_thresholds=tuned_type_thresholds,
            use_default_type_thresholds=False,
            min_cluster_size=min_cluster_size,
            max_block_size=max_block_size,
            exact_surface_first=exact_surface_first,
            model_name=model_name,
            model_version=model_version,
        )
        predictions_by_threshold[float(threshold)] = predictions_df
        metrics = evaluate_local_coreference_predictions(
            prepared_gold_records,
            predictions_df.to_dict(orient="records"),
        )
        metrics["threshold"] = threshold_value
        metrics["filtered_gold_mentions"] = len(gold_records) - len(prepared_gold_records)
        metrics["exact_surface_first"] = exact_surface_first
        metrics["use_type_threshold_floors"] = use_type_threshold_floors
        metrics["meets_precision_floor"] = metrics["pairwise_precision"] >= precision_floor
        metrics_rows.append(metrics)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("threshold").reset_index(drop=True)
    return metrics_df, predictions_by_threshold


def export_nil_mentions_for_local_coreference(
    mention_canonicalization_csv: str | Path,
    output_csv: str | Path | None = None,
    *,
    eligible_types: Sequence[str] = DEFAULT_LOCAL_COREFERENCE_TYPES,
    chunksize: int = 250_000,
    max_mentions: Optional[int] = None,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Extract ReFinED NIL mentions into a smaller local-coreference candidate CSV.

    The ReFinED audit table can be very large, so this function streams it in
    chunks and writes only NIL rows for mention types that local coreference can
    reasonably handle. Dates are intentionally excluded by the default type set.
    """
    input_path = Path(mention_canonicalization_csv)
    if output_csv is None:
        output_path = LOCAL_COREFERENCE_EXPORT_DIR / LOCAL_NIL_MENTIONS_FILENAME
    else:
        output_path = Path(output_csv)

    if overwrite and output_path.exists():
        output_path.unlink()

    eligible_type_set = {str(mention_type).upper() for mention_type in eligible_types}
    total_rows = 0
    nil_rows = 0
    exported_rows = 0
    type_counts: Counter[str] = Counter()
    dataset_modality_counts: Counter[Tuple[str, str]] = Counter()

    reader = pd.read_csv(input_path, chunksize=chunksize)
    with tqdm(desc="Extracting NIL mentions", unit="row") as progress:
        for chunk_df in reader:
            total_rows += len(chunk_df)
            progress.update(len(chunk_df))

            filtered_df = chunk_df[
                chunk_df["canonicalization_status"].astype(str).str.upper().eq("NIL")
            ].copy()
            nil_rows += len(filtered_df)

            filtered_df["mention_type"] = filtered_df["mention_type"].astype(str).str.upper()
            filtered_df = filtered_df[filtered_df["mention_type"].isin(eligible_type_set)].copy()
            if filtered_df.empty:
                continue

            if max_mentions is not None:
                remaining = max_mentions - exported_rows
                if remaining <= 0:
                    break
                filtered_df = filtered_df.head(remaining)

            filtered_df["normalized_mention_text"] = filtered_df["mention_text"].map(
                normalize_local_coreference_surface
            )
            filtered_df["surface_block_key"] = filtered_df["mention_text"].map(
                local_coreference_surface_block_key
            )
            filtered_df = filtered_df[NIL_MENTION_COLUMNS]

            append_dataframe_to_csv(filtered_df, output_path)
            exported_rows += len(filtered_df)
            type_counts.update(filtered_df["mention_type"].dropna().astype(str))
            dataset_modality_counts.update(
                zip(
                    filtered_df["dataset"].fillna("").astype(str),
                    filtered_df["modality"].fillna("").astype(str),
                )
            )

            progress.set_postfix(exported=f"{exported_rows:,}", nil=f"{nil_rows:,}")

            if max_mentions is not None and exported_rows >= max_mentions:
                break

    summary = {
        "input_path": input_path,
        "output_path": output_path,
        "total_rows_scanned": total_rows,
        "nil_rows_seen": nil_rows,
        "exported_rows": exported_rows,
        "eligible_types": sorted(eligible_type_set),
        "mention_type_counts": dict(type_counts),
        "dataset_modality_counts": {
            f"{dataset}/{modality}": count
            for (dataset, modality), count in dataset_modality_counts.items()
        },
    }
    return summary


def summarize_nil_mentions(
    nil_mentions_csv: str | Path,
    *,
    chunksize: int = 250_000,
) -> Dict[str, pd.DataFrame]:
    """Summarize a NIL mention candidate export without loading it all at once."""
    type_counts: Counter[str] = Counter()
    dataset_modality_counts: Counter[Tuple[str, str]] = Counter()
    surface_counts: Counter[Tuple[str, str]] = Counter()
    rows = 0

    for chunk_df in pd.read_csv(nil_mentions_csv, chunksize=chunksize):
        rows += len(chunk_df)
        type_counts.update(chunk_df["mention_type"].dropna().astype(str))
        dataset_modality_counts.update(
            zip(
                chunk_df["dataset"].fillna("").astype(str),
                chunk_df["modality"].fillna("").astype(str),
            )
        )
        surface_counts.update(
            zip(
                chunk_df["mention_type"].fillna("").astype(str),
                chunk_df["normalized_mention_text"].fillna("").astype(str),
            )
        )

    return {
        "overview": pd.DataFrame([{"nil_candidate_mentions": rows}]),
        "by_type": pd.DataFrame(
            [
                {"mention_type": mention_type, "mentions": count}
                for mention_type, count in type_counts.most_common()
            ]
        ),
        "by_dataset_modality": pd.DataFrame(
            [
                {"dataset": dataset, "modality": modality, "mentions": count}
                for (dataset, modality), count in dataset_modality_counts.most_common()
            ]
        ),
        "top_surfaces": pd.DataFrame(
            [
                {
                    "mention_type": mention_type,
                    "normalized_mention_text": surface,
                    "mentions": count,
                }
                for (mention_type, surface), count in surface_counts.most_common(100)
                if surface
            ]
        ),
    }


def load_nil_mentions_by_ids(
    nil_mentions_csv: str | Path,
    mention_ids: Iterable[str],
    *,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Load selected NIL mention rows from the candidate CSV by mention ID."""
    mention_id_set = {str(mention_id) for mention_id in mention_ids}
    if not mention_id_set:
        return pd.DataFrame(columns=NIL_MENTION_COLUMNS)

    frames = []
    for chunk_df in pd.read_csv(nil_mentions_csv, chunksize=chunksize):
        chunk_df["mention_id"] = chunk_df["mention_id"].astype(str)
        matched_df = chunk_df[chunk_df["mention_id"].isin(mention_id_set)].copy()
        if not matched_df.empty:
            frames.append(matched_df)

    if not frames:
        return pd.DataFrame(columns=NIL_MENTION_COLUMNS)
    return pd.concat(frames, ignore_index=True, sort=False)


def load_chunks_for_nil_mentions(
    chunks_csv: str | Path,
    nil_mentions_df: pd.DataFrame,
    *,
    text_column: str = "masked_text",
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Load only the chunk text rows needed by selected NIL mentions."""
    if nil_mentions_df.empty:
        return pd.DataFrame(columns=["chunk_id", text_column])

    chunk_ids = set(nil_mentions_df["chunk_id"].dropna().astype(str))
    frames = []
    use_columns = {"chunk_id:ID(Chunk)", "chunk_id", text_column}

    for chunk_df in pd.read_csv(
        chunks_csv,
        usecols=lambda column: column in use_columns,
        chunksize=chunksize,
    ):
        chunk_id_col = (
            "chunk_id:ID(Chunk)"
            if "chunk_id:ID(Chunk)" in chunk_df.columns
            else "chunk_id"
        )
        chunk_df[chunk_id_col] = chunk_df[chunk_id_col].astype(str)
        matched_df = chunk_df[chunk_df[chunk_id_col].isin(chunk_ids)].copy()
        if not matched_df.empty:
            frames.append(matched_df)

    if not frames:
        return pd.DataFrame(columns=["chunk_id", text_column])
    return pd.concat(frames, ignore_index=True, sort=False)


def sample_local_coreference_gold_candidates(
    nil_mentions_csv: str | Path,
    *,
    dev_output_jsonl: str | Path,
    test_output_jsonl: str | Path,
    target_dev_mentions: int = 500,
    target_test_mentions: int = 500,
    cluster_rich_fraction: float = 0.70,
    min_group_size: int = 2,
    max_group_size: int = 8,
    group_key_fields: Sequence[str] = ("mention_type", "normalized_mention_text"),
    supplemental_group_key_fields: Sequence[str] = (
        "mention_type",
        "surface_block_key",
    ),
    random_state: int = 42,
    eligible_types: Sequence[str] = DEFAULT_LOCAL_COREFERENCE_TYPES,
    exclude_generic: bool = True,
    chunksize: int = 250_000,
) -> Dict[str, Any]:
    """Sample annotation-ready NIL mentions for local coreference dev/test gold.

    The sampler is hybrid:

    1. a cluster-rich sample that keeps whole repeated mention groups, so the
       gold set contains enough positive pairs for threshold tuning;
    2. a background stratified sample across dataset/modality/type, so the gold
       set still contains negative and singleton cases.

    The exported ``gold_local_entity_id`` is intentionally empty so the sampled
    mentions can be clustered manually or with an assisted workflow later.
    """
    rng = np.random.default_rng(random_state)
    eligible_type_set = {str(mention_type).upper() for mention_type in eligible_types}
    needed = target_dev_mentions + target_test_mentions
    if needed <= 0:
        raise ValueError("target_dev_mentions + target_test_mentions must be positive")
    if not 0.0 <= cluster_rich_fraction <= 1.0:
        raise ValueError("cluster_rich_fraction must be between 0 and 1")
    if min_group_size < 2:
        raise ValueError("min_group_size must be at least 2")
    if max_group_size < min_group_size:
        raise ValueError("max_group_size must be greater than or equal to min_group_size")

    sampled_by_stratum: Dict[Tuple[str, str, str], list[Dict[str, Any]]] = defaultdict(list)
    seen_by_stratum: Counter[Tuple[str, str, str]] = Counter()
    exact_groups: Dict[Tuple[Any, ...], list[Dict[str, Any]]] = defaultdict(list)
    supplemental_groups: Dict[Tuple[Any, ...], list[Dict[str, Any]]] = defaultdict(list)
    reservoir_multiplier = 4

    for chunk_df in pd.read_csv(nil_mentions_csv, chunksize=chunksize):
        chunk_df["mention_type"] = chunk_df["mention_type"].astype(str).str.upper()
        chunk_df = chunk_df[chunk_df["mention_type"].isin(eligible_type_set)].copy()
        if chunk_df.empty:
            continue

        for record in chunk_df.to_dict(orient="records"):
            if exclude_generic and _looks_generic_for_gold_sampling(record):
                continue

            stratum = (
                str(record.get("dataset") or ""),
                str(record.get("modality") or ""),
                str(record.get("mention_type") or ""),
            )
            seen_by_stratum[stratum] += 1
            reservoir = sampled_by_stratum[stratum]
            cap = max(20, int(np.ceil(needed * reservoir_multiplier / 20)))

            exact_key = _sampling_group_key(record, group_key_fields)
            if exact_key:
                exact_groups[exact_key].append(record)
            supplemental_key = _sampling_group_key(record, supplemental_group_key_fields)
            if supplemental_key:
                supplemental_groups[supplemental_key].append(record)

            if len(reservoir) < cap:
                reservoir.append(record)
                continue

            replacement_index = rng.integers(0, seen_by_stratum[stratum])
            if replacement_index < cap:
                reservoir[int(replacement_index)] = record

    strata = sorted(sampled_by_stratum)
    if not strata:
        raise ValueError(f"No eligible NIL mentions found in {nil_mentions_csv}")

    cluster_rich_target = int(round(needed * cluster_rich_fraction))
    background_target = needed - cluster_rich_target
    selected_records: list[Dict[str, Any]] = []
    selected_ids: set[str] = set()

    selected_cluster_groups = _sample_candidate_groups(
        exact_groups,
        target_mentions=cluster_rich_target,
        rng=rng,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        selected_ids=selected_ids,
    )
    for group in selected_cluster_groups:
        for record in group:
            mention_id = str(record.get("mention_id"))
            if mention_id not in selected_ids:
                selected_records.append(record)
                selected_ids.add(mention_id)

    if len(selected_records) < cluster_rich_target:
        selected_supplemental_groups = _sample_candidate_groups(
            supplemental_groups,
            target_mentions=cluster_rich_target - len(selected_records),
            rng=rng,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
            selected_ids=selected_ids,
        )
        for group in selected_supplemental_groups:
            for record in group:
                mention_id = str(record.get("mention_id"))
                if mention_id not in selected_ids:
                    selected_records.append(record)
                    selected_ids.add(mention_id)

    base_per_stratum = max(1, background_target // len(strata)) if background_target else 0
    for stratum in strata:
        candidates = sampled_by_stratum[stratum]
        candidates = [
            record
            for record in candidates
            if str(record.get("mention_id")) not in selected_ids
        ]
        take = min(base_per_stratum, len(candidates))
        if take:
            indexes = rng.choice(len(candidates), size=take, replace=False)
            for index in indexes:
                record = candidates[int(index)]
                selected_records.append(record)
                selected_ids.add(str(record.get("mention_id")))

    if len(selected_records) < needed:
        remaining_candidates = [
            record
            for records in sampled_by_stratum.values()
            for record in records
            if str(record.get("mention_id")) not in selected_ids
        ]
        rng.shuffle(remaining_candidates)
        selected_records.extend(remaining_candidates[: needed - len(selected_records)])

    selected_records = selected_records[:needed]
    rng.shuffle(selected_records)
    dev_records = selected_records[:target_dev_mentions]
    test_records = selected_records[target_dev_mentions : target_dev_mentions + target_test_mentions]

    dev_path = Path(dev_output_jsonl)
    test_path = Path(test_output_jsonl)
    dev_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    _write_local_coreference_gold_candidates(dev_records, dev_path, split="dev")
    _write_local_coreference_gold_candidates(test_records, test_path, split="test")

    return {
        "dev_output_jsonl": dev_path,
        "test_output_jsonl": test_path,
        "dev_records": len(dev_records),
        "test_records": len(test_records),
        "strata_seen": len(strata),
        "cluster_rich_fraction": cluster_rich_fraction,
        "cluster_rich_target": cluster_rich_target,
        "background_target": background_target,
        "selected_candidate_groups": len(selected_cluster_groups),
        "strata_counts": {
            f"{dataset}/{modality}/{mention_type}": count
            for (dataset, modality, mention_type), count in seen_by_stratum.items()
        },
    }


def _sampling_group_key(
    record: Mapping[str, Any],
    fields: Sequence[str],
) -> Optional[Tuple[Any, ...]]:
    key = tuple(str(record.get(field) or "").strip().lower() for field in fields)
    if any(not item for item in key):
        return None
    return key


def _sample_candidate_groups(
    groups: Mapping[Tuple[Any, ...], Sequence[Mapping[str, Any]]],
    *,
    target_mentions: int,
    rng: np.random.Generator,
    min_group_size: int,
    max_group_size: int,
    selected_ids: set[str],
) -> list[list[Dict[str, Any]]]:
    if target_mentions <= 0:
        return []

    candidates: list[list[Dict[str, Any]]] = []
    for records in groups.values():
        deduped_by_id = {
            str(record.get("mention_id")): dict(record)
            for record in records
            if str(record.get("mention_id")) not in selected_ids
        }
        deduped = list(deduped_by_id.values())
        if len(deduped) < min_group_size:
            continue
        rng.shuffle(deduped)
        candidates.append(deduped[:max_group_size])

    rng.shuffle(candidates)
    candidates.sort(key=lambda group: (min(len(group), max_group_size), rng.random()), reverse=True)

    selected_groups: list[list[Dict[str, Any]]] = []
    selected_count = 0
    for group in candidates:
        if selected_count >= target_mentions:
            break
        selected_groups.append(group)
        selected_count += len(group)
    return selected_groups


def _looks_generic_for_gold_sampling(record: Mapping[str, Any]) -> bool:
    return is_obvious_generic_local_coreference_mention(record)


def read_local_coreference_gold_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    """Read local-coreference gold JSONL records."""
    records: list[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_local_coreference_gold_jsonl(
    records: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """Write local-coreference gold JSONL records."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    return output_path


def merge_local_coreference_gold_annotations(
    candidate_records: Sequence[Mapping[str, Any]],
    annotated_records: Sequence[Mapping[str, Any]],
    *,
    id_field: str = "mention_id",
    gold_field: str = "gold_local_entity_id",
) -> list[Dict[str, Any]]:
    """Copy existing gold cluster IDs onto a fresh candidate set by mention ID."""
    annotation_by_id = {
        str(record.get(id_field)): str(record.get(gold_field) or "").strip()
        for record in annotated_records
        if str(record.get(gold_field) or "").strip()
    }
    merged_records: list[Dict[str, Any]] = []
    for record in candidate_records:
        merged_record = dict(record)
        mention_id = str(merged_record.get(id_field))
        if mention_id in annotation_by_id:
            merged_record[gold_field] = annotation_by_id[mention_id]
        else:
            merged_record.setdefault(gold_field, "")
        merged_records.append(merged_record)
    return merged_records


def suggest_local_coreference_gold_annotations(
    records: Sequence[Mapping[str, Any]],
    *,
    gold_field: str = "gold_local_entity_id",
    min_group_size: int = 2,
    overwrite: bool = False,
    include_ambiguous_short: bool = False,
) -> list[Dict[str, Any]]:
    """Conservatively assign reusable gold IDs for repeated exact NIL surfaces.

    This helper is meant for assisted annotation. It only fills empty gold IDs
    by default, skips obvious generic mentions, and avoids ambiguous short names
    unless explicitly requested.
    """
    output_records = [dict(record) for record in records]
    groups: Dict[Tuple[str, str], list[int]] = defaultdict(list)

    for index, record in enumerate(output_records):
        if not overwrite and str(record.get(gold_field) or "").strip():
            continue
        if is_obvious_generic_local_coreference_mention(record):
            output_records[index][gold_field] = ""
            continue

        mention_type = str(record.get("mention_type") or "").upper()
        surface = normalize_local_coreference_surface(
            record.get("normalized_mention_text") or record.get("mention_text")
        )
        if not mention_type or not surface:
            continue
        if not include_ambiguous_short and is_ambiguous_short_surface(surface):
            continue
        groups[(mention_type, surface)].append(index)

    for (mention_type, surface), indexes in groups.items():
        if len(indexes) < min_group_size:
            continue
        gold_id = f"GOLD_{mention_type}_{_gold_id_slug(surface)}"
        for index in indexes:
            output_records[index][gold_field] = gold_id

    return output_records


def annotate_local_coreference_gold_jsonl(
    input_jsonl: str | Path,
    output_jsonl: str | Path | None = None,
    *,
    annotated_jsonl: str | Path | None = None,
    gold_field: str = "gold_local_entity_id",
    min_group_size: int = 2,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Merge existing annotations, suggest exact-surface gold IDs, and write JSONL."""
    records = read_local_coreference_gold_jsonl(input_jsonl)
    if annotated_jsonl is not None:
        annotated_records = read_local_coreference_gold_jsonl(annotated_jsonl)
        records = merge_local_coreference_gold_annotations(
            records,
            annotated_records,
            gold_field=gold_field,
        )

    annotated_records = suggest_local_coreference_gold_annotations(
        records,
        gold_field=gold_field,
        min_group_size=min_group_size,
        overwrite=overwrite,
    )

    output_path = Path(output_jsonl) if output_jsonl is not None else Path(input_jsonl)
    write_local_coreference_gold_jsonl(annotated_records, output_path)
    summary = summarize_local_coreference_gold_records(
        annotated_records,
        gold_field=gold_field,
    )
    summary["input_jsonl"] = Path(input_jsonl)
    summary["output_jsonl"] = output_path
    if annotated_jsonl is not None:
        summary["merged_annotations_from"] = Path(annotated_jsonl)
    return summary


def summarize_local_coreference_gold_records(
    records: Sequence[Mapping[str, Any]],
    *,
    gold_field: str = "gold_local_entity_id",
) -> Dict[str, Any]:
    """Summarize local-coreference gold annotations and generic assignment risk."""
    cluster_sizes: Counter[str] = Counter()
    assigned_mentions = 0
    generic_assigned_mentions = 0
    for record in records:
        gold_id = str(record.get(gold_field) or "").strip()
        if not gold_id:
            continue
        assigned_mentions += 1
        cluster_sizes[gold_id] += 1
        if is_obvious_generic_local_coreference_mention(record):
            generic_assigned_mentions += 1

    positive_clusters = {
        cluster_id: size
        for cluster_id, size in cluster_sizes.items()
        if size >= 2
    }
    positive_pairs = sum(size * (size - 1) // 2 for size in positive_clusters.values())
    return {
        "records": len(records),
        "assigned_mentions": assigned_mentions,
        "unassigned_mentions": len(records) - assigned_mentions,
        "gold_clusters": len(cluster_sizes),
        "positive_clusters": len(positive_clusters),
        "positive_pairs": positive_pairs,
        "generic_assigned_mentions": generic_assigned_mentions,
    }


def _gold_id_slug(value: Any) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").upper()).strip("_")
    return slug or "UNSPECIFIED"


def _write_local_coreference_gold_candidates(
    records: Sequence[Mapping[str, Any]],
    output_jsonl: Path,
    *,
    split: str,
) -> None:
    import json

    fields = [
        "mention_id",
        "chunk_id",
        "document_id",
        "dataset",
        "modality",
        "mention_text",
        "mention_type",
        "start_char:int",
        "end_char:int",
        "normalized_mention_text",
        "surface_block_key",
    ]
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            output_record = {field: record.get(field) for field in fields}
            output_record["split"] = split
            output_record["gold_local_entity_id"] = ""
            handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")


def _cluster_id(record: Mapping[str, Any], *names: str) -> Optional[str]:
    for name in names:
        value = record.get(name)
        if value is None:
            continue
        value = str(value).strip()
        if value and value.upper() not in {"NONE", "NULL", "NIL", "UNRESOLVED"}:
            return value
    return None


def _positive_pairs(cluster_by_mention_id: Mapping[str, Optional[str]]) -> set[Tuple[str, str]]:
    mentions_by_cluster: Dict[str, list[str]] = defaultdict(list)
    for mention_id, cluster_id in cluster_by_mention_id.items():
        if cluster_id is not None:
            mentions_by_cluster[str(cluster_id)].append(str(mention_id))

    pairs: set[Tuple[str, str]] = set()
    for mention_ids in mentions_by_cluster.values():
        mention_ids = sorted(set(mention_ids))
        for left_index, left_id in enumerate(mention_ids):
            for right_id in mention_ids[left_index + 1 :]:
                pairs.add((left_id, right_id))
    return pairs


def pairwise_coreference_metrics(
    gold_cluster_by_mention_id: Mapping[str, Optional[str]],
    predicted_cluster_by_mention_id: Mapping[str, Optional[str]],
) -> Dict[str, Any]:
    """Compute pairwise precision, recall, and F1 for local coreference."""
    mention_ids = set(gold_cluster_by_mention_id) | set(predicted_cluster_by_mention_id)
    gold_pairs = _positive_pairs(
        {mention_id: gold_cluster_by_mention_id.get(mention_id) for mention_id in mention_ids}
    )
    predicted_pairs = _positive_pairs(
        {mention_id: predicted_cluster_by_mention_id.get(mention_id) for mention_id in mention_ids}
    )
    true_positive_pairs = gold_pairs & predicted_pairs

    precision = (
        len(true_positive_pairs) / len(predicted_pairs)
        if predicted_pairs
        else 0.0
    )
    recall = len(true_positive_pairs) / len(gold_pairs) if gold_pairs else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "gold_positive_pairs": len(gold_pairs),
        "predicted_positive_pairs": len(predicted_pairs),
        "true_positive_pairs": len(true_positive_pairs),
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
    }


def cluster_purity_metrics(
    gold_cluster_by_mention_id: Mapping[str, Optional[str]],
    predicted_cluster_by_mention_id: Mapping[str, Optional[str]],
) -> Dict[str, float]:
    """Compute simple purity and inverse purity over clustered mentions."""
    mention_ids = sorted(set(gold_cluster_by_mention_id) | set(predicted_cluster_by_mention_id))

    predicted_to_gold: Dict[str, Counter[str]] = defaultdict(Counter)
    gold_to_predicted: Dict[str, Counter[str]] = defaultdict(Counter)

    for mention_id in mention_ids:
        gold_id = gold_cluster_by_mention_id.get(mention_id)
        predicted_id = predicted_cluster_by_mention_id.get(mention_id)
        if gold_id is not None and predicted_id is not None:
            predicted_to_gold[str(predicted_id)][str(gold_id)] += 1
            gold_to_predicted[str(gold_id)][str(predicted_id)] += 1

    clustered_mentions = sum(sum(counter.values()) for counter in predicted_to_gold.values())
    gold_clustered_mentions = sum(sum(counter.values()) for counter in gold_to_predicted.values())

    purity = (
        sum(max(counter.values()) for counter in predicted_to_gold.values()) / clustered_mentions
        if clustered_mentions
        else 0.0
    )
    inverse_purity = (
        sum(max(counter.values()) for counter in gold_to_predicted.values()) / gold_clustered_mentions
        if gold_clustered_mentions
        else 0.0
    )

    return {
        "purity": purity,
        "inverse_purity": inverse_purity,
        "clustered_mention_coverage": (
            len([cluster_id for cluster_id in predicted_cluster_by_mention_id.values() if cluster_id])
            / len(mention_ids)
            if mention_ids
            else 0.0
        ),
    }


def evaluate_local_coreference_predictions(
    gold_records: Iterable[Mapping[str, Any]],
    predicted_records: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Evaluate predicted local NIL clusters against gold local cluster IDs."""
    gold_cluster_by_mention_id = {
        str(record["mention_id"]): _cluster_id(
            record,
            "gold_local_entity_id",
            "gold_cluster_id",
            "gold_entity_id",
        )
        for record in gold_records
    }
    predicted_cluster_by_mention_id = {
        str(record["mention_id"]): _cluster_id(
            record,
            "predicted_local_entity_id",
            "predicted_cluster_id",
            "entity_id",
        )
        for record in predicted_records
    }

    metrics = {
        "gold_mentions": len(gold_cluster_by_mention_id),
        "predicted_mentions": len(predicted_cluster_by_mention_id),
    }
    metrics.update(
        pairwise_coreference_metrics(
            gold_cluster_by_mention_id,
            predicted_cluster_by_mention_id,
        )
    )
    metrics.update(
        cluster_purity_metrics(
            gold_cluster_by_mention_id,
            predicted_cluster_by_mention_id,
        )
    )
    return metrics
