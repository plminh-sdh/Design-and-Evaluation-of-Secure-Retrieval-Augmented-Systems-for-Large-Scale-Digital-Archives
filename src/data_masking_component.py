"""Helpers for the Data Masking Component notebook."""

from __future__ import annotations

import json
import math
import zipfile
from collections import defaultdict
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from tqdm.auto import tqdm


TAB_DATASET_NAME = "ildpil/text-anonymization-benchmark"
TAB_DATASET_LABEL = "TAB"
TAB_MODALITY = "text"
SENSITIVE_IDENTIFIER_TYPES = {"DIRECT", "QUASI"}
PRESERVE_IDENTIFIER_TYPES = {"NO_MASK"}
MASKING_GLIREL_DEFAULT_LABELS = [
    "associated with",
    "located in",
    "lives in",
    "born in",
    "works for",
    "member of",
    "represented by",
    "has contact detail",
    "has identifier",
    "has date",
    "has nationality",
    "has address",
    "treated by",
    "employed by",
    "owns",
    "part of",
]
TAB_SEMANTIC_LABEL_MAP = {
    "PERSON": "person",
    "ORG": "organization",
    "LOC": "location",
    "DATETIME": "date/time",
    "CODE": "code",
    "DEM": "demographic",
    "QUANTITY": "quantity",
    "MISC": "miscellaneous",
}
TAB_SPLIT_ARCHIVES = {
    "train": "echr_train.zip",
    "dev": "echr_dev.zip",
    "test": "echr_test.zip",
}
DATABASE_STYLE_GLINER_MAX_WORDS = 350
DATABASE_STYLE_GLINER_OVERLAP_WORDS = 60
DATABASE_STYLE_GLINER_STRIDE_WORDS = (
    DATABASE_STYLE_GLINER_MAX_WORDS - DATABASE_STYLE_GLINER_OVERLAP_WORDS
)


def safe_json_dumps(value: Any) -> str:
    """Serialize nested metadata for compact dataframe display/export."""

    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def maybe_parse_json(value: Any) -> Any:
    """Parse JSON-like strings while leaving already-structured values intact."""

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def _is_missing_value(value: Any) -> bool:
    """Return True for scalar missing values without coercing arrays to bool."""

    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        return False
    if isinstance(value, (list, tuple, dict)):
        return False
    if hasattr(value, "size"):
        return getattr(value, "size", 0) == 0
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    if isinstance(missing, bool):
        return missing
    return False


def _as_text(value: Any, default: str = "") -> str:
    """Convert TAB scalar-ish values to text without array truth-value checks."""

    if _is_missing_value(value):
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_as_text(item) for item in value if not _is_missing_value(item))
    if hasattr(value, "tolist"):
        return _as_text(value.tolist(), default=default)
    return str(value)


def _as_bool(value: Any, default: bool = False) -> bool:
    """Convert scalar-ish TAB flags to bool without array truth-value checks."""

    if _is_missing_value(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n", ""}:
            return False
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return bool(value[0]) if value else default
    return bool(value)


def _as_list(value: Any) -> list[Any]:
    """Convert common pandas/numpy nested values to a plain Python list."""

    value = maybe_parse_json(value)
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _normalize_raw_mentions(value: Any) -> list[dict[str, Any]]:
    """Normalize TAB entity_mentions to list-of-dict records.

    Depending on the loader and pandas conversion path, nested TAB mentions can
    appear as the intended list of mention dictionaries or as a columnar mapping
    of field names to equally sized lists. This helper supports both layouts.
    """

    value = maybe_parse_json(value)
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, MappingABC):
        field_values = {key: _as_list(field_value) for key, field_value in value.items()}
        lengths = [len(field_value) for field_value in field_values.values()]
        if not lengths:
            return []
        mention_count = max(lengths)
        mentions: list[dict[str, Any]] = []
        for mention_index in range(mention_count):
            mention: dict[str, Any] = {}
            for key, field_value in field_values.items():
                mention[key] = field_value[mention_index] if mention_index < len(field_value) else None
            mentions.append(mention)
        return mentions

    mentions = _as_list(value)
    return [
        dict(mention)
        for mention in mentions
        if isinstance(mention, MappingABC)
    ]


def _repair_tab_span_offsets(
    text: str,
    start: int,
    end: int,
    span_text: str,
    *,
    search_window: int = 80,
) -> tuple[int, int, str, bool]:
    """Return validated TAB offsets, repairing obvious loader/offset variants."""

    if start >= 0 and end > start and text[start:end] == span_text:
        return start, end, text[start:end], True

    # Some annotation formats use inclusive end offsets. TAB should not, but
    # accepting this variant makes diagnostics robust without changing valid spans.
    if start >= 0 and end >= start and text[start : end + 1] == span_text:
        return start, end + 1, text[start : end + 1], True

    # Defensive handling for one-based offsets.
    if start > 0 and end > start and text[start - 1 : end - 1] == span_text:
        return start - 1, end - 1, text[start - 1 : end - 1], True

    if span_text:
        window_start = max(0, start - search_window)
        window_end = min(len(text), max(end, start) + search_window)
        local_index = text.find(span_text, window_start, window_end)
        if local_index >= 0:
            repaired_end = local_index + len(span_text)
            return local_index, repaired_end, text[local_index:repaired_end], True

    text_slice = text[start:end] if start >= 0 and end > start else ""
    return start, end, text_slice, text_slice == span_text


def load_tab_dataset(
    dataset_name: str = TAB_DATASET_NAME,
    *,
    prefer_direct_zip_load: bool = True,
    prefer_direct_zip_fallback: bool = True,
) -> Any:
    """Load TAB from Hugging Face datasets, with a robust zip-file fallback.

    Some local `datasets`/`fsspec` combinations fail on TAB's zipped JSON files
    with `ValueError: seek of closed file`. The fallback downloads the same
    split archives from the Hugging Face Hub and parses the JSON directly.
    """

    if prefer_direct_zip_load:
        try:
            return load_tab_dataset_from_zip_archives(dataset_name)
        except Exception:
            if not prefer_direct_zip_fallback:
                raise

    try:
        from datasets import load_dataset
    except ImportError as exc:
        if prefer_direct_zip_load:
            raise
        raise ImportError(
            "Install datasets first: pip install datasets"
        ) from exc

    try:
        return load_dataset(dataset_name)
    except Exception as primary_exc:
        if not prefer_direct_zip_fallback:
            raise
        try:
            return load_tab_dataset_from_zip_archives(dataset_name)
        except Exception as fallback_exc:
            raise RuntimeError(
                "Failed to load TAB with datasets.load_dataset and with the "
                "direct zip fallback. Try clearing the Hugging Face datasets "
                "cache or upgrading datasets/fsspec."
            ) from fallback_exc


def load_tab_dataset_from_zip_archives(
    dataset_name: str = TAB_DATASET_NAME,
    *,
    archives: Mapping[str, str] = TAB_SPLIT_ARCHIVES,
) -> dict[str, list[dict[str, Any]]]:
    """Download TAB split zip files from Hugging Face Hub and parse JSON records."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Install huggingface_hub first: pip install huggingface_hub"
        ) from exc

    split_records: dict[str, list[dict[str, Any]]] = {}
    for split_name, archive_name in archives.items():
        archive_path = hf_hub_download(
            repo_id=dataset_name,
            filename=archive_name,
            repo_type="dataset",
        )
        with zipfile.ZipFile(archive_path) as archive:
            json_members = [
                member
                for member in archive.namelist()
                if member.lower().endswith(".json")
            ]
            if not json_members:
                raise ValueError(f"No JSON file found in {archive_name}.")
            with archive.open(json_members[0]) as handle:
                records = json.load(handle)

        if isinstance(records, MappingABC):
            for candidate_key in ("data", "documents", "records"):
                candidate_records = records.get(candidate_key)
                if isinstance(candidate_records, list):
                    records = candidate_records
                    break
        if not isinstance(records, list):
            raise ValueError(
                f"Expected a list of records in {archive_name}, got {type(records)!r}."
            )

        normalized_records = []
        for record in records:
            if not isinstance(record, MappingABC):
                continue
            row = dict(record)
            row.setdefault("hf_split", split_name)
            row.setdefault("dataset_type", split_name)
            normalized_records.append(row)
        split_records[split_name] = normalized_records

    return split_records


def dataset_to_dataframe(dataset: Any) -> pd.DataFrame:
    """Convert a Hugging Face Dataset/DatasetDict or record iterable to a dataframe."""

    frames: list[pd.DataFrame] = []
    if isinstance(dataset, MappingABC):
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            if hasattr(split_data, "to_pandas"):
                split_df = split_data.to_pandas()
            elif isinstance(split_data, pd.DataFrame):
                split_df = split_data.copy()
            else:
                split_df = pd.DataFrame(list(split_data))
            split_df["hf_split"] = split_name
            frames.append(split_df)
    elif hasattr(dataset, "to_pandas"):
        frames.append(dataset.to_pandas())
    else:
        frames.append(pd.DataFrame(list(dataset)))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_identifier_type(value: Any) -> str:
    return _as_text(value).strip().upper()


def normalize_entity_type(value: Any) -> str:
    return (
        _as_text(value)
        .strip()
        .upper()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _coerce_int(value: Any, default: int = -1) -> int:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except Exception:
        return default


def normalize_tab_mentions(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize TAB entity_mentions into a compact span schema."""

    raw_mentions = _normalize_raw_mentions(record.get("entity_mentions", []))
    text = _as_text(record.get("text"))
    mentions: list[dict[str, Any]] = []
    for mention_index, mention in enumerate(raw_mentions):
        start = _coerce_int(mention.get("start_offset"))
        end = _coerce_int(mention.get("end_offset"))
        if start < 0 or end <= start:
            continue
        span_text = _as_text(mention.get("span_text"), default=text[start:end])
        start, end, expected_text, offset_text_matches = _repair_tab_span_offsets(
            text,
            start,
            end,
            span_text,
        )
        identifier_type = normalize_identifier_type(mention.get("identifier_type"))
        entity_type = normalize_entity_type(mention.get("entity_type"))
        mentions.append(
            {
                "mention_index": mention_index,
                "entity_mention_id": _as_text(mention.get("entity_mention_id")),
                "entity_id": _as_text(mention.get("entity_id")),
                "entity_type": entity_type,
                "identifier_type": identifier_type,
                "policy_action": "MASK"
                if identifier_type in SENSITIVE_IDENTIFIER_TYPES
                else "KEEP",
                "start": start,
                "end": end,
                "span_text": span_text,
                "text_slice": expected_text,
                "offset_text_matches": offset_text_matches,
                "edit_type": _as_text(mention.get("edit_type")),
                "confidential_status": _as_text(mention.get("confidential_status")),
            }
        )
    return mentions


def _tab_record_id(record: Mapping[str, Any], row_index: int) -> str:
    doc_id = _as_text(record.get("doc_id"), default=_as_text(record.get("document_id"), default=f"row_{row_index}"))
    annotator_id = _as_text(record.get("annotator_id"), default="annotator_unknown")
    dataset_type = _as_text(record.get("dataset_type"), default=_as_text(record.get("hf_split"), default="split_unknown"))
    return f"tab::{dataset_type}::{doc_id}::{annotator_id}"


def normalize_tab_records(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw TAB dataframe rows into the masking experiment schema."""

    records: list[dict[str, Any]] = []
    for row_index, row in raw_df.reset_index(drop=True).iterrows():
        record = row.to_dict()
        text = _as_text(record.get("text"))
        mentions = normalize_tab_mentions(record)
        sensitive_spans = [
            mention
            for mention in mentions
            if mention["identifier_type"] in SENSITIVE_IDENTIFIER_TYPES
        ]
        no_mask_spans = [
            mention
            for mention in mentions
            if mention["identifier_type"] in PRESERVE_IDENTIFIER_TYPES
        ]
        dataset_type = _as_text(record.get("dataset_type"), default=_as_text(record.get("hf_split"))).strip()
        normalized_record = {
            "chunk_id": _tab_record_id(record, row_index),
            "dataset": TAB_DATASET_LABEL,
            "modality": TAB_MODALITY,
            "document_id": _as_text(record.get("doc_id")),
            "annotator_id": _as_text(record.get("annotator_id")),
            "tab_dataset_type": dataset_type.lower(),
            "hf_split": _as_text(record.get("hf_split")),
            "quality_checked": _as_bool(record.get("quality_checked")),
            "task": _as_text(record.get("task")),
            "raw_text": text,
            "masked_text": "",
            "text_char_count": len(text),
            "text_token_count": len(text.split()),
            "meta_json": safe_json_dumps(maybe_parse_json(record.get("meta"))),
            "entity_mentions": mentions,
            "gold_sensitive_spans": sensitive_spans,
            "gold_no_mask_spans": no_mask_spans,
            "gold_span_count": len(mentions),
            "gold_sensitive_span_count": len(sensitive_spans),
            "gold_no_mask_span_count": len(no_mask_spans),
            "offset_text_mismatch_count": sum(
                not mention["offset_text_matches"] for mention in mentions
            ),
        }
        records.append(normalized_record)

    return pd.DataFrame(records)


def clean_normalized_tab_records(records_df: pd.DataFrame) -> pd.DataFrame:
    """Apply light cleaning and derive inspection columns."""

    if records_df.empty:
        return records_df.copy()

    df = records_df.copy()
    df["raw_text"] = df["raw_text"].fillna("").astype(str)
    df = df[df["raw_text"].str.strip().ne("")].copy()
    df = df.drop_duplicates(subset=["chunk_id"]).reset_index(drop=True)
    df["has_sensitive_spans"] = df["gold_sensitive_span_count"].fillna(0).astype(int) > 0
    df["has_no_mask_spans"] = df["gold_no_mask_span_count"].fillna(0).astype(int) > 0
    df["usable_for_masking_eval"] = (
        (df["gold_span_count"].fillna(0).astype(int) > 0)
        & df["has_sensitive_spans"]
        & (df["offset_text_mismatch_count"].fillna(0).astype(int) == 0)
    )
    return df


def split_tab_records(
    records_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return TAB's built-in train/dev/test splits.

    TAB ships with official split labels. Keep those labels intact instead of
    making a random split in the notebook.
    """

    if records_df.empty:
        return records_df.copy(), records_df.copy(), records_df.copy()

    df = records_df.copy().reset_index(drop=True)
    split_values = df["tab_dataset_type"].fillna("").str.lower()
    expected_splits = {"train", "dev", "test"}
    missing_splits = sorted(expected_splits - set(split_values.dropna().unique()))
    if missing_splits:
        raise ValueError(
            "TAB records must contain the built-in train/dev/test split labels. "
            f"Missing splits: {missing_splits}"
        )

    train_df = df[split_values.eq("train")].copy()
    dev_df = df[split_values.eq("dev")].copy()
    test_df = df[split_values.eq("test")].copy()

    return (
        train_df.reset_index(drop=True),
        dev_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def summarize_tab_records(records_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build compact data-preparation inspection tables."""

    if records_df.empty:
        empty = pd.DataFrame(columns=["records"])
        return {
            "split": empty,
            "span_counts": empty,
            "quality": empty,
            "entity_types": empty,
            "identifier_types": empty,
        }

    mention_rows: list[dict[str, Any]] = []
    for _, row in records_df.iterrows():
        for mention in row.get("entity_mentions") or []:
            mention_rows.append(
                {
                    "chunk_id": row["chunk_id"],
                    "tab_dataset_type": row["tab_dataset_type"],
                    "entity_type": mention.get("entity_type"),
                    "identifier_type": mention.get("identifier_type"),
                }
            )
    mentions_df = pd.DataFrame(mention_rows)
    if mentions_df.empty:
        entity_types = pd.DataFrame(columns=["entity_type", "mentions"])
        identifier_types = pd.DataFrame(columns=["identifier_type", "mentions"])
    else:
        entity_types = (
            mentions_df.groupby("entity_type", dropna=False)
            .size()
            .reset_index(name="mentions")
            .sort_values("mentions", ascending=False)
        )
        identifier_types = (
            mentions_df.groupby("identifier_type", dropna=False)
            .size()
            .reset_index(name="mentions")
            .sort_values("mentions", ascending=False)
        )

    return {
        "split": records_df.groupby("tab_dataset_type", dropna=False)
        .size()
        .reset_index(name="records"),
        "span_counts": records_df[
            [
                "gold_span_count",
                "gold_sensitive_span_count",
                "gold_no_mask_span_count",
                "offset_text_mismatch_count",
            ]
        ].describe(),
        "quality": records_df.groupby(
            ["tab_dataset_type", "usable_for_masking_eval"],
            dropna=False,
        )
        .size()
        .reset_index(name="records"),
        "entity_types": entity_types,
        "identifier_types": identifier_types,
    }


def semantic_gliner_label(entity_type: Any) -> str:
    """Return a GLiNER-friendly semantic label for a TAB entity type."""

    normalized = normalize_entity_type(entity_type)
    return TAB_SEMANTIC_LABEL_MAP.get(
        normalized,
        normalized.lower().replace("_", " "),
    )


def policy_composite_gliner_label(
    entity_type: Any,
    identifier_type: Any,
    *,
    compact: bool = False,
) -> str:
    """Return a policy-aware GLiNER label.

    ``compact=True`` returns labels like ``DIRECT_PERSON`` for tabulation.
    ``compact=False`` returns natural labels like ``direct person`` for GLiNER.
    """

    normalized_identifier = normalize_identifier_type(identifier_type)
    normalized_entity = normalize_entity_type(entity_type)
    if compact:
        return f"{normalized_identifier}_{normalized_entity}".strip("_")
    semantic_label = semantic_gliner_label(normalized_entity)
    prefix = normalized_identifier.lower().replace("_", " ")
    return f"{prefix} {semantic_label}".strip()


def tab_mentions_to_dataframe(records_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten normalized TAB records into one mention-level dataframe."""

    rows: list[dict[str, Any]] = []
    if records_df.empty:
        return pd.DataFrame(rows)

    for _, row in records_df.iterrows():
        for mention in row.get("entity_mentions") or []:
            entity_type = mention.get("entity_type")
            identifier_type = mention.get("identifier_type")
            rows.append(
                {
                    "chunk_id": row.get("chunk_id"),
                    "tab_dataset_type": row.get("tab_dataset_type"),
                    "document_id": row.get("document_id"),
                    "mention_index": mention.get("mention_index"),
                    "entity_mention_id": mention.get("entity_mention_id"),
                    "entity_id": mention.get("entity_id"),
                    "entity_type": entity_type,
                    "identifier_type": identifier_type,
                    "semantic_label": semantic_gliner_label(entity_type),
                    "policy_label": policy_composite_gliner_label(
                        entity_type,
                        identifier_type,
                        compact=True,
                    ),
                    "gliner_policy_label": policy_composite_gliner_label(
                        entity_type,
                        identifier_type,
                        compact=False,
                    ),
                    "confidential_status": mention.get("confidential_status"),
                    "start": mention.get("start"),
                    "end": mention.get("end"),
                    "span_text": mention.get("span_text"),
                    "policy_action": mention.get("policy_action"),
                    "offset_text_matches": mention.get("offset_text_matches"),
                }
            )
    return pd.DataFrame(rows)


def summarize_tab_label_support(records_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build label-support tables for section 3 GLiNER setup."""

    mentions_df = tab_mentions_to_dataframe(records_df)
    if mentions_df.empty:
        empty = pd.DataFrame(columns=["mentions"])
        return {
            "entity_type": empty,
            "identifier_type": empty,
            "entity_policy": empty,
            "semantic_gliner_labels": empty,
            "policy_gliner_labels": empty,
            "confidential_status": empty,
        }

    def _count(columns: list[str]) -> pd.DataFrame:
        return (
            mentions_df.groupby(columns, dropna=False)
            .size()
            .reset_index(name="mentions")
            .sort_values("mentions", ascending=False)
            .reset_index(drop=True)
        )

    return {
        "entity_type": _count(["entity_type"]),
        "identifier_type": _count(["identifier_type"]),
        "entity_policy": _count(["entity_type", "identifier_type"]),
        "semantic_gliner_labels": _count(["semantic_label"]),
        "policy_gliner_labels": _count(["policy_label", "gliner_policy_label"]),
        "confidential_status": _count(["confidential_status"]),
    }


def tokenize_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Whitespace-tokenize text while preserving character offsets."""

    import re

    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def char_span_to_token_span(
    offsets: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> tuple[int, int] | None:
    """Convert a character span to GLiNER's inclusive token span."""

    overlapping_indices = [
        token_index
        for token_index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end
    ]
    if not overlapping_indices:
        return None
    return overlapping_indices[0], overlapping_indices[-1]


def build_gliner_training_records(
    records_df: pd.DataFrame,
    *,
    label_mode: str = "semantic",
    include_identifier_types: Iterable[str] | None = None,
    require_offset_match: bool = True,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Convert normalized TAB rows to GLiNER training records.

    GLiNER expects ``{"tokenized_text": [...], "ner": [[start, end, label], ...]}``
    where token spans are inclusive.
    """

    if label_mode not in {"semantic", "policy_composite"}:
        raise ValueError("label_mode must be 'semantic' or 'policy_composite'.")

    allowed_identifier_types = (
        {normalize_identifier_type(value) for value in include_identifier_types}
        if include_identifier_types is not None
        else None
    )

    examples: list[dict[str, Any]] = []
    source_df = records_df.head(max_records) if max_records is not None else records_df
    for _, row in source_df.iterrows():
        text = _as_text(row.get("raw_text"))
        tokens, offsets = tokenize_with_offsets(text)
        if not tokens:
            continue

        ner_spans: list[list[Any]] = []
        seen_spans: set[tuple[int, int, str]] = set()
        for mention in row.get("entity_mentions") or []:
            if require_offset_match and not mention.get("offset_text_matches", False):
                continue
            identifier_type = normalize_identifier_type(mention.get("identifier_type"))
            if allowed_identifier_types is not None and identifier_type not in allowed_identifier_types:
                continue

            token_span = char_span_to_token_span(
                offsets,
                int(mention.get("start", -1)),
                int(mention.get("end", -1)),
            )
            if token_span is None:
                continue

            if label_mode == "semantic":
                label = semantic_gliner_label(mention.get("entity_type"))
            else:
                label = policy_composite_gliner_label(
                    mention.get("entity_type"),
                    identifier_type,
                    compact=False,
                )
            span_key = (token_span[0], token_span[1], label)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            ner_spans.append([token_span[0], token_span[1], label])

        if ner_spans:
            examples.append(
                {
                    "id": row.get("chunk_id"),
                    "tokenized_text": tokens,
                    "ner": ner_spans,
                }
            )

    return examples


def gliner_label_list(training_records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return sorted unique labels from GLiNER training records."""

    labels = {
        str(span[2])
        for record in training_records
        for span in record.get("ner", [])
        if len(span) >= 3
    }
    return sorted(labels)


def write_jsonl_records(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write JSONL records and return the resolved path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL records, returning an empty list when the file is absent."""

    input_path = Path(path)
    if not input_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def write_json_record(record: Mapping[str, Any], path: str | Path) -> Path:
    """Write one JSON record and return the output path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(record), handle, ensure_ascii=False, indent=2, default=_json_default)
    return output_path


def read_json_record(path: str | Path) -> dict[str, Any]:
    """Read one JSON record, returning an empty dict when absent."""

    input_path = Path(path)
    if not input_path.exists():
        return {}
    with input_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def move_model_to_cuda_if_available(model: Any) -> Any:
    """Move a model-like object to CUDA when PyTorch reports an available GPU."""

    try:
        import torch

        if torch.cuda.is_available() and hasattr(model, "to"):
            return model.to("cuda")
    except Exception:
        pass
    return model


def move_model_to_device(model: Any, device: str | None) -> Any:
    """Move a model-like object to an explicit device when possible."""

    if device is None:
        return move_model_to_cuda_if_available(model)
    try:
        if hasattr(model, "to"):
            return model.to(device)
    except Exception:
        pass
    return model


def _stable_masking_id(*parts: Any) -> str:
    import hashlib

    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _span_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def _is_sensitive_identifier(identifier_type: Any) -> bool:
    return normalize_identifier_type(identifier_type) in SENSITIVE_IDENTIFIER_TYPES


def policy_label_is_sensitive(label: Any) -> bool:
    """Return True when a policy-composite GLiNER label implies masking."""

    normalized = str(label or "").strip().lower().replace("_", " ")
    return normalized.startswith("direct ") or normalized.startswith("quasi ")


def policy_label_semantic_type(label: Any) -> str:
    """Strip a policy prefix from a policy-composite GLiNER label."""

    normalized = str(label or "").strip().lower().replace("_", " ")
    for prefix in ("direct ", "quasi ", "no mask ", "no_mask "):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :].strip()
    return normalized


def _gold_mentions_for_row(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(mention) for mention in row.get("entity_mentions") or []]


def _mention_gold_target(
    prediction: Mapping[str, Any],
    gold_mentions: Sequence[Mapping[str, Any]],
    *,
    min_overlap: int = 1,
) -> int | None:
    """Map a predicted span to TAB gold mask target by relaxed overlap."""

    start = int(prediction.get("start", -1))
    end = int(prediction.get("end", -1))
    best_overlap = 0
    best_target: int | None = None
    for mention in gold_mentions:
        overlap = _span_overlap(
            start,
            end,
            int(mention.get("start", -1)),
            int(mention.get("end", -1)),
        )
        if overlap < min_overlap or overlap < best_overlap:
            continue
        best_overlap = overlap
        best_target = 1 if _is_sensitive_identifier(mention.get("identifier_type")) else 0
    return best_target


def database_style_word_windows(
    text: str,
    *,
    max_words: int = DATABASE_STYLE_GLINER_MAX_WORDS,
    overlap_words: int = DATABASE_STYLE_GLINER_OVERLAP_WORDS,
) -> list[dict[str, Any]]:
    """Split text like Database.ipynb archive chunks while preserving offsets."""

    tokens, offsets = tokenize_with_offsets(text)
    if not tokens:
        return []
    max_words = max(1, int(max_words))
    overlap_words = max(0, min(int(overlap_words), max_words - 1))
    stride = max_words - overlap_words

    windows: list[dict[str, Any]] = []
    start_token = 0
    window_index = 0
    while start_token < len(tokens):
        end_token = min(len(tokens), start_token + max_words)
        char_start = offsets[start_token][0]
        char_end = offsets[end_token - 1][1]
        windows.append(
            {
                "window_index": window_index,
                "start_token": start_token,
                "end_token": end_token,
                "char_start": char_start,
                "char_end": char_end,
                "tokens": tokens[start_token:end_token],
                "offsets": [
                    (start - char_start, end - char_start)
                    for start, end in offsets[start_token:end_token]
                ],
                "text": text[char_start:char_end],
            }
        )
        if end_token >= len(tokens):
            break
        start_token += stride
        window_index += 1
    return windows


def _gold_sensitive_spans(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(mention) for mention in row.get("gold_sensitive_spans") or []]


def evaluate_sensitive_span_predictions(
    records_df: pd.DataFrame,
    predictions_by_chunk: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    threshold: float = 0.5,
    score_field: str = "sensitivity_score",
    verbose: bool = False,
) -> dict[str, Any]:
    """Evaluate predicted sensitive spans against TAB DIRECT/QUASI spans."""

    tp = fp = fn = 0
    iterator = records_df.to_dict(orient="records")
    for row in tqdm(
        iterator,
        desc="Evaluating sensitive spans",
        unit="chunk",
        disable=not verbose,
    ):
        chunk_id = str(row.get("chunk_id") or "")
        gold_spans = _gold_sensitive_spans(row)
        matched_gold: set[int] = set()
        sensitive_predictions = [
            prediction
            for prediction in predictions_by_chunk.get(chunk_id, [])
            if float(prediction.get(score_field, 0.0) or 0.0) >= threshold
        ]
        for prediction in sensitive_predictions:
            pred_start = int(prediction.get("start", -1))
            pred_end = int(prediction.get("end", -1))
            matched_index = None
            for gold_index, gold in enumerate(gold_spans):
                if gold_index in matched_gold:
                    continue
                if _span_overlap(
                    pred_start,
                    pred_end,
                    int(gold.get("start", -1)),
                    int(gold.get("end", -1)),
                ) > 0:
                    matched_index = gold_index
                    break
            if matched_index is None:
                fp += 1
            else:
                tp += 1
                matched_gold.add(matched_index)
        fn += max(0, len(gold_spans) - len(matched_gold))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    loss = 1.0 - f1
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loss": loss,
    }


def tune_sensitive_threshold(
    records_df: pd.DataFrame,
    predictions_by_chunk: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    thresholds: Sequence[float] | None = None,
    score_field: str = "sensitivity_score",
    verbose: bool = True,
) -> dict[str, Any]:
    """Find the score threshold that maximizes relaxed Detection F1."""

    thresholds = list(thresholds or [round(value / 100, 2) for value in range(0, 101, 5)])
    rows = []
    for threshold in tqdm(
        thresholds,
        desc="Tuning sensitive-span threshold",
        unit="threshold",
        disable=not verbose,
    ):
        rows.append(
            evaluate_sensitive_span_predictions(
                records_df,
                predictions_by_chunk,
                threshold=float(threshold),
                score_field=score_field,
                verbose=False,
            )
        )
    best = sorted(
        rows,
        key=lambda row: (row["f1"], row["precision"], row["recall"], -row["threshold"]),
        reverse=True,
    )[0] if rows else {}
    return {"best": best, "metrics_by_threshold": rows}


def predict_gliner_mentions_for_records(
    records_df: pd.DataFrame,
    gliner_model: Any,
    labels: Sequence[str],
    *,
    threshold: float = 0.3,
    gliner_device: str | None = None,
    max_words: int = DATABASE_STYLE_GLINER_MAX_WORDS,
    overlap_words: int = DATABASE_STYLE_GLINER_OVERLAP_WORDS,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run GLiNER over Database.ipynb-style word windows."""

    gliner_model = move_model_to_device(gliner_model, gliner_device)
    predictions: list[dict[str, Any]] = []
    for row in tqdm(
        records_df.to_dict(orient="records"),
        desc="Running GLiNER masking spans",
        unit="chunk",
        disable=not verbose,
    ):
        text = _as_text(row.get("raw_text"))
        chunk_id = str(row.get("chunk_id") or "")
        if not text.strip() or not chunk_id:
            continue
        gold_mentions = _gold_mentions_for_row(row)
        seen_predictions: set[tuple[int, int, str]] = set()
        windows = database_style_word_windows(
            text,
            max_words=max_words,
            overlap_words=overlap_words,
        )
        for window in windows:
            raw_predictions = gliner_model.predict_entities(
                str(window["text"]),
                list(labels),
                threshold=threshold,
            )
            for index, prediction in enumerate(raw_predictions or []):
                start = _coerce_int(prediction.get("start")) + int(window["char_start"])
                end = _coerce_int(prediction.get("end")) + int(window["char_start"])
                if start < 0 or end <= start or end > len(text):
                    continue
                label = str(prediction.get("label") or "")
                dedupe_key = (start, end, label)
                if dedupe_key in seen_predictions:
                    continue
                seen_predictions.add(dedupe_key)
                score = float(prediction.get("score") or 0.0)
                record = {
                    "prediction_id": _stable_masking_id(
                        chunk_id,
                        start,
                        end,
                        label,
                        int(window["window_index"]),
                        index,
                    ),
                    "chunk_id": chunk_id,
                    "document_id": row.get("document_id"),
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                    "label": label,
                    "score": score,
                    "window_index": int(window["window_index"]),
                    "window_start_token": int(window["start_token"]),
                    "window_end_token": int(window["end_token"]) - 1,
                    "gold_target": _mention_gold_target(
                        {"start": start, "end": end},
                        gold_mentions,
                    ),
                }
                predictions.append(record)
    return predictions


def train_policy_gliner_masking_strategy(
    records_df: pd.DataFrame,
    policy_gliner_model: Any,
    labels: Sequence[str],
    output_json: str | Path,
    *,
    gliner_threshold: float = 0.3,
    gliner_device: str | None = None,
    max_words: int = DATABASE_STYLE_GLINER_MAX_WORDS,
    overlap_words: int = DATABASE_STYLE_GLINER_OVERLAP_WORDS,
    tune_thresholds: Sequence[float] | None = None,
    force_retrain: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train/tune Strategy 1: policy GLiNER alone.

    The function loads a saved JSON result when available. Otherwise it runs
    policy-composite GLiNER, converts DIRECT/QUASI labels into sensitivity
    scores, tunes a masking threshold by Detection F1, and saves the result.
    """

    output_path = Path(output_json)
    if output_path.exists() and not force_retrain:
        return read_json_record(output_path)

    predictions = predict_gliner_mentions_for_records(
        records_df,
        policy_gliner_model,
        labels,
        threshold=gliner_threshold,
        gliner_device=gliner_device,
        max_words=max_words,
        overlap_words=overlap_words,
        verbose=verbose,
    )
    predictions_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        sensitive = policy_label_is_sensitive(prediction.get("label"))
        prediction["semantic_label"] = policy_label_semantic_type(prediction.get("label"))
        prediction["policy_sensitive"] = bool(sensitive)
        prediction["sensitivity_score"] = float(prediction.get("score") or 0.0) if sensitive else 0.0
        predictions_by_chunk[str(prediction["chunk_id"])].append(prediction)

    tuning = tune_sensitive_threshold(
        records_df,
        predictions_by_chunk,
        thresholds=tune_thresholds,
        score_field="sensitivity_score",
        verbose=verbose,
    )
    scored_predictions = [
        dict(prediction)
        for predictions_for_chunk in predictions_by_chunk.values()
        for prediction in predictions_for_chunk
    ]
    result = {
        "strategy": "policy_gliner_alone",
        "gliner_threshold": float(gliner_threshold),
        "gliner_device": gliner_device or "auto",
        "max_words": int(max_words),
        "overlap_words": int(overlap_words),
        "objective": "maximize_detection_f1_or_minimize_1_minus_f1",
        "best": tuning["best"],
        "metrics_by_threshold": tuning["metrics_by_threshold"],
        "predictions": scored_predictions,
    }
    write_json_record(result, output_path)
    return result


def build_gliner_predictions_for_glirel(
    records_df: pd.DataFrame,
    gliner_model: Any,
    labels: Sequence[str],
    *,
    label_mode: str = "semantic",
    gliner_threshold: float = 0.3,
    gliner_device: str | None = None,
    max_words: int = DATABASE_STYLE_GLINER_MAX_WORDS,
    overlap_words: int = DATABASE_STYLE_GLINER_OVERLAP_WORDS,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run GLiNER and convert predictions into Database-style GLiREL windows."""

    predictions = predict_gliner_mentions_for_records(
        records_df,
        gliner_model,
        labels,
        threshold=gliner_threshold,
        gliner_device=gliner_device,
        max_words=max_words,
        overlap_words=overlap_words,
        verbose=verbose,
    )
    rows_by_chunk = {
        str(row.get("chunk_id") or ""): row
        for row in records_df.to_dict(orient="records")
    }
    predictions_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        label = str(prediction.get("label") or "")
        if label_mode == "policy_composite":
            mention_type = policy_label_semantic_type(label).upper().replace(" ", "_").replace("/", "_")
            prediction["policy_sensitive"] = policy_label_is_sensitive(label)
        else:
            mention_type = label.upper().replace(" ", "_").replace("/", "_")
        prediction["mention_type"] = mention_type
        prediction["entity_id"] = prediction["prediction_id"]
        predictions_by_chunk[str(prediction["chunk_id"])].append(prediction)

    glirel_inputs: list[dict[str, Any]] = []
    for chunk_id, chunk_predictions in tqdm(
        list(predictions_by_chunk.items()),
        desc="Building GLiNER-to-GLiREL inputs",
        unit="chunk",
        disable=not verbose,
    ):
        row = rows_by_chunk.get(chunk_id)
        if row is None or len(chunk_predictions) < 2:
            continue
        text = _as_text(row.get("raw_text"))
        windows = database_style_word_windows(
            text,
            max_words=max_words,
            overlap_words=overlap_words,
        )
        if not windows:
            continue
        for window in windows:
            window_char_start = int(window["char_start"])
            window_char_end = int(window["char_end"])
            window_predictions = [
                prediction
                for prediction in chunk_predictions
                if int(prediction["start"]) >= window_char_start
                and int(prediction["end"]) <= window_char_end
            ]
            if len(window_predictions) < 2:
                continue

            ner: list[list[Any]] = []
            mention_records: list[dict[str, Any]] = []
            for prediction in sorted(window_predictions, key=lambda item: (int(item["start"]), int(item["end"]))):
                local_start = int(prediction["start"]) - window_char_start
                local_end = int(prediction["end"]) - window_char_start
                token_span = char_span_to_token_span(
                    window["offsets"],
                    local_start,
                    local_end,
                )
                if token_span is None:
                    continue
                start_token, end_token = token_span
                mention_record = dict(prediction)
                mention_record["mention_id"] = prediction["prediction_id"]
                mention_record["mention_text"] = prediction["text"]
                mention_record["token_start"] = start_token
                mention_record["token_end"] = end_token
                mention_record["ner_index"] = len(ner)
                mention_records.append(mention_record)
                ner.append([start_token, end_token, mention_record["mention_type"], prediction["text"]])
            if len(ner) < 2:
                continue
            glirel_inputs.append(
                {
                    "input_id": _stable_masking_id(
                        "masking_glirel",
                        chunk_id,
                        int(window["window_index"]),
                        len(ner),
                    ),
                    "chunk_id": chunk_id,
                    "document_id": row.get("document_id"),
                    "dataset": row.get("dataset", TAB_DATASET_LABEL),
                    "modality": row.get("modality", TAB_MODALITY),
                    "text": window["text"],
                    "tokens": window["tokens"],
                    "ner": ner,
                    "mention_records": mention_records,
                    "labels": list(MASKING_GLIREL_DEFAULT_LABELS),
                    "window_index": int(window["window_index"]),
                    "window_start_token": int(window["start_token"]),
                    "window_end_token": int(window["end_token"]) - 1,
                }
            )
    return predictions, glirel_inputs


def predict_glirel_for_masking_inputs(
    glirel_model: Any,
    glirel_inputs: Sequence[Mapping[str, Any]],
    *,
    relation_labels: Sequence[str] | None = None,
    threshold: float = 0.05,
    top_k: int = 3,
    batch_size: int = 4,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run GLiREL on GLiNER-derived inputs and attach endpoint mention ids."""

    glirel_model = move_model_to_cuda_if_available(glirel_model)
    relation_labels = list(relation_labels or MASKING_GLIREL_DEFAULT_LABELS)
    predictions: list[dict[str, Any]] = []
    if not glirel_inputs:
        return predictions

    def endpoint(item: Mapping[str, Any], pos: Any) -> Mapping[str, Any] | None:
        if isinstance(pos, (str, bytes)) or not isinstance(pos, Sequence) or len(pos) < 2:
            return None
        start_token = int(pos[0])
        raw_end = int(pos[1])
        possible_ends = {raw_end, raw_end - 1}
        for mention in item.get("mention_records") or []:
            if int(mention.get("token_start")) == start_token and int(mention.get("token_end")) in possible_ends:
                return mention
        return None

    if batch_size <= 1 or not hasattr(glirel_model, "batch_predict_relations"):
        for item in tqdm(
            glirel_inputs,
            desc="Running GLiREL masking relations",
            unit="chunk",
            disable=not verbose,
        ):
            raw = glirel_model.predict_relations(
                item["tokens"],
                relation_labels,
                threshold=threshold,
                ner=item["ner"],
                top_k=top_k,
            )
            for prediction in raw or []:
                head = endpoint(item, prediction.get("head_pos"))
                tail = endpoint(item, prediction.get("tail_pos"))
                if head is None or tail is None:
                    continue
                enriched = dict(prediction)
                enriched.update(
                    {
                        "input_id": item.get("input_id"),
                        "chunk_id": item.get("chunk_id"),
                        "head_mention_id": head.get("mention_id"),
                        "tail_mention_id": tail.get("mention_id"),
                        "head_type": head.get("mention_type"),
                        "tail_type": tail.get("mention_type"),
                    }
                )
                predictions.append(enriched)
        return predictions

    for start in tqdm(
        range(0, len(glirel_inputs), batch_size),
        desc="Running GLiREL masking relations",
        unit="batch",
        disable=not verbose,
    ):
        batch = list(glirel_inputs[start : start + batch_size])
        batch_outputs = glirel_model.batch_predict_relations(
            [item["tokens"] for item in batch],
            relation_labels,
            threshold=threshold,
            ner=[item["ner"] for item in batch],
            top_k=top_k,
        )
        for item, raw_predictions in zip(batch, batch_outputs):
            for prediction in raw_predictions or []:
                head = endpoint(item, prediction.get("head_pos"))
                tail = endpoint(item, prediction.get("tail_pos"))
                if head is None or tail is None:
                    continue
                enriched = dict(prediction)
                enriched.update(
                    {
                        "input_id": item.get("input_id"),
                        "chunk_id": item.get("chunk_id"),
                        "head_mention_id": head.get("mention_id"),
                        "tail_mention_id": tail.get("mention_id"),
                        "head_type": head.get("mention_type"),
                        "tail_type": tail.get("mention_type"),
                    }
                )
                predictions.append(enriched)
    return predictions


def _relation_side_weights(
    gliner_predictions: Sequence[Mapping[str, Any]],
    relation_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Estimate P(sensitive | relation label, side) from prediction/gold overlap."""

    prediction_by_id = {
        str(prediction.get("prediction_id")): prediction
        for prediction in gliner_predictions
    }
    buckets: dict[str, list[int]] = defaultdict(list)
    for relation in relation_predictions:
        label = str(relation.get("label") or "")
        for side, key in (("head", "head_mention_id"), ("tail", "tail_mention_id")):
            mention = prediction_by_id.get(str(relation.get(key) or ""))
            if mention is None:
                continue
            target = mention.get("gold_target")
            if target is None:
                continue
            type_name = str(relation.get(f"{side}_type") or "")
            buckets[f"{label}|{side}|{type_name}"].append(int(target))
    return {
        key: sum(values) / len(values)
        for key, values in buckets.items()
        if values
    }


def _relation_scores_by_mention(
    gliner_predictions: Sequence[Mapping[str, Any]],
    relation_predictions: Sequence[Mapping[str, Any]],
    weights: Mapping[str, float],
    *,
    fallback_by_type: Mapping[str, float] | None = None,
) -> dict[str, float]:
    fallback_by_type = fallback_by_type or {}
    scores: dict[str, list[float]] = defaultdict(list)
    prediction_by_id = {
        str(prediction.get("prediction_id")): prediction
        for prediction in gliner_predictions
    }
    for relation in relation_predictions:
        label = str(relation.get("label") or "")
        for side, key in (("head", "head_mention_id"), ("tail", "tail_mention_id")):
            mention_id = str(relation.get(key) or "")
            mention_type = str(relation.get(f"{side}_type") or "")
            weight = weights.get(f"{label}|{side}|{mention_type}")
            if weight is not None:
                scores[mention_id].append(float(weight))
    result: dict[str, float] = {}
    for prediction in gliner_predictions:
        mention_id = str(prediction.get("prediction_id"))
        values = scores.get(mention_id) or []
        if values:
            result[mention_id] = sum(values) / len(values)
        else:
            result[mention_id] = float(fallback_by_type.get(str(prediction.get("mention_type") or ""), 0.0))
    return result


def _type_priors(gliner_predictions: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for prediction in gliner_predictions:
        target = prediction.get("gold_target")
        if target is None:
            continue
        buckets[str(prediction.get("mention_type") or "")].append(int(target))
    return {key: sum(values) / len(values) for key, values in buckets.items() if values}


def _predictions_by_chunk_with_scores(
    gliner_predictions: Sequence[Mapping[str, Any]],
    scores_by_mention: Mapping[str, float],
    *,
    score_field: str = "sensitivity_score",
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in gliner_predictions:
        record = dict(prediction)
        record[score_field] = float(scores_by_mention.get(str(record.get("prediction_id")), 0.0))
        grouped[str(record.get("chunk_id") or "")].append(record)
    return grouped


def optimize_relation_side_weights_for_f1(
    records_df: pd.DataFrame,
    gliner_predictions: Sequence[Mapping[str, Any]],
    relation_predictions: Sequence[Mapping[str, Any]],
    initial_weights: Mapping[str, float],
    *,
    fallback_by_type: Mapping[str, float] | None = None,
    thresholds: Sequence[float] | None = None,
    weight_candidates: Sequence[float] | None = None,
    max_passes: int = 2,
    verbose: bool = True,
) -> dict[str, Any]:
    """Coordinate-search relation-side weights for the Detection F1 objective.

    Empirical relation sensitivity rates are useful starting values, but the
    thesis formula defines the training objective as Detection F1. This function
    performs a small coordinate search over observed relation-side weights and
    chooses updates only when they improve the best threshold-tuned F1.
    """

    weights = {str(key): float(value) for key, value in initial_weights.items()}
    thresholds = list(thresholds or [round(value / 100, 2) for value in range(0, 101, 5)])
    weight_candidates = list(weight_candidates or [0.0, 0.25, 0.5, 0.75, 1.0])

    def evaluate(weights_to_eval: Mapping[str, float]) -> dict[str, Any]:
        scores = _relation_scores_by_mention(
            gliner_predictions,
            relation_predictions,
            weights_to_eval,
            fallback_by_type=fallback_by_type,
        )
        grouped = _predictions_by_chunk_with_scores(gliner_predictions, scores)
        tuning = tune_sensitive_threshold(
            records_df,
            grouped,
            thresholds=thresholds,
            score_field="sensitivity_score",
            verbose=False,
        )
        return {
            "best": tuning["best"],
            "scores_by_mention": scores,
            "predictions_by_chunk": grouped,
        }

    current = evaluate(weights)
    history = [
        {
            "pass": 0,
            "relation_side": "__initial__",
            **current["best"],
        }
    ]
    if not weights:
        return {
            "weights": weights,
            "best": current["best"],
            "scores_by_mention": current["scores_by_mention"],
            "predictions_by_chunk": current["predictions_by_chunk"],
            "history": history,
        }

    for pass_index in range(1, max(1, int(max_passes)) + 1):
        improved_this_pass = False
        for key in tqdm(
            list(weights.keys()),
            desc=f"Optimizing relation weights pass {pass_index}",
            unit="weight",
            disable=not verbose,
        ):
            original = weights[key]
            best_candidate_value = original
            best_candidate = current
            for candidate_value in weight_candidates:
                candidate_value = float(candidate_value)
                if candidate_value == original:
                    continue
                candidate_weights = dict(weights)
                candidate_weights[key] = candidate_value
                candidate = evaluate(candidate_weights)
                if (
                    candidate["best"]["f1"],
                    candidate["best"]["precision"],
                    candidate["best"]["recall"],
                ) > (
                    best_candidate["best"]["f1"],
                    best_candidate["best"]["precision"],
                    best_candidate["best"]["recall"],
                ):
                    best_candidate_value = candidate_value
                    best_candidate = candidate
            weights[key] = best_candidate_value
            if best_candidate_value != original:
                current = best_candidate
                improved_this_pass = True
                history.append(
                    {
                        "pass": pass_index,
                        "relation_side": key,
                        "weight": best_candidate_value,
                        **current["best"],
                    }
                )
        if not improved_this_pass:
            break

    return {
        "weights": weights,
        "best": current["best"],
        "scores_by_mention": current["scores_by_mention"],
        "predictions_by_chunk": current["predictions_by_chunk"],
        "history": history,
    }


def train_semantic_gliner_glirel_strategy(
    records_df: pd.DataFrame,
    semantic_gliner_model: Any,
    glirel_model: Any,
    labels: Sequence[str],
    output_json: str | Path,
    *,
    relation_labels: Sequence[str] | None = None,
    gliner_threshold: float = 0.3,
    gliner_device: str | None = None,
    max_words: int = DATABASE_STYLE_GLINER_MAX_WORDS,
    overlap_words: int = DATABASE_STYLE_GLINER_OVERLAP_WORDS,
    glirel_threshold: float = 0.05,
    glirel_top_k: int = 3,
    glirel_batch_size: int = 4,
    tune_thresholds: Sequence[float] | None = None,
    optimize_relation_weights: bool = True,
    relation_weight_candidates: Sequence[float] | None = None,
    relation_weight_max_passes: int = 2,
    force_retrain: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train/tune Strategy 2: semantic GLiNER plus GLiREL relation weights."""

    output_path = Path(output_json)
    if output_path.exists() and not force_retrain:
        return read_json_record(output_path)

    gliner_predictions, glirel_inputs = build_gliner_predictions_for_glirel(
        records_df,
        semantic_gliner_model,
        labels,
        label_mode="semantic",
        gliner_threshold=gliner_threshold,
        gliner_device=gliner_device,
        max_words=max_words,
        overlap_words=overlap_words,
        verbose=verbose,
    )
    relation_predictions = predict_glirel_for_masking_inputs(
        glirel_model,
        glirel_inputs,
        relation_labels=relation_labels,
        threshold=glirel_threshold,
        top_k=glirel_top_k,
        batch_size=glirel_batch_size,
        verbose=verbose,
    )
    weights = _relation_side_weights(gliner_predictions, relation_predictions)
    fallback = _type_priors(gliner_predictions)
    if optimize_relation_weights:
        optimized = optimize_relation_side_weights_for_f1(
            records_df,
            gliner_predictions,
            relation_predictions,
            weights,
            fallback_by_type=fallback,
            thresholds=tune_thresholds,
            weight_candidates=relation_weight_candidates,
            max_passes=relation_weight_max_passes,
            verbose=verbose,
        )
        weights = optimized["weights"]
        predictions_by_chunk = optimized["predictions_by_chunk"]
        best = optimized["best"]
        metrics_by_threshold = []
        optimization_history = optimized["history"]
    else:
        scores_by_mention = _relation_scores_by_mention(
            gliner_predictions,
            relation_predictions,
            weights,
            fallback_by_type=fallback,
        )
        predictions_by_chunk = _predictions_by_chunk_with_scores(
            gliner_predictions,
            scores_by_mention,
        )
        tuning = tune_sensitive_threshold(
            records_df,
            predictions_by_chunk,
            thresholds=tune_thresholds,
            score_field="sensitivity_score",
            verbose=verbose,
        )
        best = tuning["best"]
        metrics_by_threshold = tuning["metrics_by_threshold"]
        optimization_history = []
    scored_predictions = [
        dict(prediction)
        for predictions_for_chunk in predictions_by_chunk.values()
        for prediction in predictions_for_chunk
    ]
    result = {
        "strategy": "semantic_gliner_glirel_relation_weights",
        "gliner_threshold": float(gliner_threshold),
        "gliner_device": gliner_device or "auto",
        "max_words": int(max_words),
        "overlap_words": int(overlap_words),
        "glirel_threshold": float(glirel_threshold),
        "glirel_top_k": int(glirel_top_k),
        "objective": "maximize_detection_f1_or_minimize_1_minus_f1",
        "best": best,
        "metrics_by_threshold": metrics_by_threshold,
        "relation_weight_optimization_history": optimization_history,
        "relation_side_weights": dict(weights),
        "semantic_type_priors": dict(fallback),
        "gliner_predictions": scored_predictions,
        "glirel_inputs": list(glirel_inputs),
        "relation_predictions": relation_predictions,
    }
    write_json_record(result, output_path)
    return result


def _policy_prior_scores(
    gliner_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for prediction in gliner_predictions:
        sensitive = policy_label_is_sensitive(prediction.get("label"))
        scores[str(prediction.get("prediction_id"))] = (
            float(prediction.get("score") or 0.0) if sensitive else 0.0
        )
    return scores


def _combine_scores(
    policy_score: float,
    relation_score: float,
    *,
    alpha: float = 0.5,
    beta: float = 1.0,
    mode: str = "weighted_sum",
) -> float:
    if mode == "harmonic":
        epsilon = 1e-9
        numerator = (1 + beta**2) * policy_score * relation_score
        denominator = beta**2 * policy_score + relation_score + epsilon
        return numerator / denominator if denominator else 0.0
    return alpha * policy_score + (1 - alpha) * relation_score


def _merged_char_length(spans: Sequence[tuple[int, int]]) -> int:
    cleaned = sorted((int(start), int(end)) for start, end in spans if int(end) > int(start))
    if not cleaned:
        return 0
    merged: list[list[int]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _intersection_char_length(
    left_spans: Sequence[tuple[int, int]],
    right_spans: Sequence[tuple[int, int]],
) -> int:
    intersections: list[tuple[int, int]] = []
    for left_start, left_end in left_spans:
        for right_start, right_end in right_spans:
            start = max(int(left_start), int(right_start))
            end = min(int(left_end), int(right_end))
            if end > start:
                intersections.append((start, end))
    return _merged_char_length(intersections)


def compute_masking_metrics(
    records_df: pd.DataFrame,
    predictions_by_chunk: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    threshold: float = 0.5,
    score_field: str = "sensitivity_score",
    verbose: bool = False,
) -> dict[str, Any]:
    """Compute the six masking metrics used for final strategy comparison.

    SSR/SSP/Detection F1 are relaxed span-level detection metrics. RSTR measures
    retained sensitive character rate, while MCR and MCR* report total and
    non-sensitive character masking rates.
    """

    detection = evaluate_sensitive_span_predictions(
        records_df,
        predictions_by_chunk,
        threshold=threshold,
        score_field=score_field,
        verbose=verbose,
    )
    sensitive_chars = 0
    covered_sensitive_chars = 0
    total_chars = 0
    masked_chars = 0
    nonsensitive_chars = 0
    masked_nonsensitive_chars = 0

    for row in records_df.to_dict(orient="records"):
        chunk_id = str(row.get("chunk_id") or "")
        text = _as_text(row.get("raw_text"))
        total_chars += len(text)
        sensitive_spans = [
            (int(span.get("start", -1)), int(span.get("end", -1)))
            for span in row.get("gold_sensitive_spans") or []
        ]
        predicted_spans = [
            (int(prediction.get("start", -1)), int(prediction.get("end", -1)))
            for prediction in predictions_by_chunk.get(chunk_id, [])
            if float(prediction.get(score_field, 0.0) or 0.0) >= threshold
        ]
        chunk_sensitive_chars = _merged_char_length(sensitive_spans)
        chunk_masked_chars = _merged_char_length(predicted_spans)
        chunk_covered_sensitive_chars = _intersection_char_length(predicted_spans, sensitive_spans)
        sensitive_chars += chunk_sensitive_chars
        covered_sensitive_chars += chunk_covered_sensitive_chars
        masked_chars += chunk_masked_chars
        chunk_nonsensitive_chars = max(len(text) - chunk_sensitive_chars, 0)
        nonsensitive_chars += chunk_nonsensitive_chars
        masked_nonsensitive_chars += max(chunk_masked_chars - chunk_covered_sensitive_chars, 0)

    ssr = detection["recall"]
    ssp = detection["precision"]
    detection_f1 = detection["f1"]
    rstr = 1.0 - (covered_sensitive_chars / sensitive_chars) if sensitive_chars else 0.0
    mcr = masked_chars / total_chars if total_chars else 0.0
    mcr_star = masked_nonsensitive_chars / nonsensitive_chars if nonsensitive_chars else 0.0
    return {
        "threshold": float(threshold),
        "tp": detection["tp"],
        "fp": detection["fp"],
        "fn": detection["fn"],
        "SSR": ssr,
        "SSP": ssp,
        "Detection_F1": detection_f1,
        "RSTR": rstr,
        "MCR": mcr,
        "MCR_star": mcr_star,
        "loss": 1.0 - detection_f1,
        "sensitive_chars": int(sensitive_chars),
        "covered_sensitive_chars": int(covered_sensitive_chars),
        "total_chars": int(total_chars),
        "masked_chars": int(masked_chars),
        "nonsensitive_chars": int(nonsensitive_chars),
        "masked_nonsensitive_chars": int(masked_nonsensitive_chars),
    }


def evaluate_policy_gliner_strategy_on_records(
    records_df: pd.DataFrame,
    policy_gliner_model: Any,
    labels: Sequence[str],
    strategy_result: Mapping[str, Any],
    *,
    gliner_device: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Apply saved Strategy 1 settings to evaluation records."""

    best = strategy_result.get("best") or {}
    threshold = float(best.get("threshold", 0.5))
    predictions = predict_gliner_mentions_for_records(
        records_df,
        policy_gliner_model,
        labels,
        threshold=float(strategy_result.get("gliner_threshold", 0.3)),
        gliner_device=gliner_device or strategy_result.get("gliner_device"),
        max_words=int(strategy_result.get("max_words", DATABASE_STYLE_GLINER_MAX_WORDS)),
        overlap_words=int(strategy_result.get("overlap_words", DATABASE_STYLE_GLINER_OVERLAP_WORDS)),
        verbose=verbose,
    )
    predictions_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        record = dict(prediction)
        sensitive = policy_label_is_sensitive(record.get("label"))
        record["semantic_label"] = policy_label_semantic_type(record.get("label"))
        record["policy_sensitive"] = bool(sensitive)
        record["sensitivity_score"] = float(record.get("score") or 0.0) if sensitive else 0.0
        predictions_by_chunk[str(record.get("chunk_id") or "")].append(record)
    metrics = compute_masking_metrics(records_df, predictions_by_chunk, threshold=threshold, verbose=False)
    return {
        "strategy": "Policy GLiNER",
        "threshold": threshold,
        "metrics": metrics,
        "predictions_by_chunk": predictions_by_chunk,
    }


def evaluate_semantic_gliner_glirel_strategy_on_records(
    records_df: pd.DataFrame,
    semantic_gliner_model: Any,
    glirel_model: Any,
    labels: Sequence[str],
    strategy_result: Mapping[str, Any],
    *,
    relation_labels: Sequence[str] | None = None,
    gliner_device: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Apply saved Strategy 2 settings to evaluation records."""

    best = strategy_result.get("best") or {}
    threshold = float(best.get("threshold", 0.5))
    gliner_predictions, glirel_inputs = build_gliner_predictions_for_glirel(
        records_df,
        semantic_gliner_model,
        labels,
        label_mode="semantic",
        gliner_threshold=float(strategy_result.get("gliner_threshold", 0.3)),
        gliner_device=gliner_device or strategy_result.get("gliner_device"),
        max_words=int(strategy_result.get("max_words", DATABASE_STYLE_GLINER_MAX_WORDS)),
        overlap_words=int(strategy_result.get("overlap_words", DATABASE_STYLE_GLINER_OVERLAP_WORDS)),
        verbose=verbose,
    )
    relation_predictions = predict_glirel_for_masking_inputs(
        glirel_model,
        glirel_inputs,
        relation_labels=relation_labels,
        threshold=float(strategy_result.get("glirel_threshold", 0.05)),
        top_k=int(strategy_result.get("glirel_top_k", 3)),
        batch_size=int(strategy_result.get("glirel_batch_size", 4)),
        verbose=verbose,
    )
    scores = _relation_scores_by_mention(
        gliner_predictions,
        relation_predictions,
        strategy_result.get("relation_side_weights") or {},
        fallback_by_type=strategy_result.get("semantic_type_priors") or {},
    )
    predictions_by_chunk = _predictions_by_chunk_with_scores(gliner_predictions, scores)
    metrics = compute_masking_metrics(records_df, predictions_by_chunk, threshold=threshold, verbose=False)
    return {
        "strategy": "Semantic GLiNER + GLiREL",
        "threshold": threshold,
        "metrics": metrics,
        "predictions_by_chunk": predictions_by_chunk,
        "glirel_inputs": glirel_inputs,
        "relation_predictions": relation_predictions,
    }


def evaluate_policy_gliner_glirel_strategy_on_records(
    records_df: pd.DataFrame,
    policy_gliner_model: Any,
    glirel_model: Any,
    labels: Sequence[str],
    strategy_result: Mapping[str, Any],
    *,
    relation_labels: Sequence[str] | None = None,
    gliner_device: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Apply saved Strategy 3 settings to evaluation records."""

    best = strategy_result.get("best") or {}
    threshold = float(best.get("threshold", 0.5))
    combination_mode = str(strategy_result.get("combination_mode") or "weighted_sum")
    alpha = float(best.get("alpha", 0.5) if best.get("alpha") is not None else 0.5)
    beta = float(best.get("beta", 1.0) if best.get("beta") is not None else 1.0)
    gliner_predictions, glirel_inputs = build_gliner_predictions_for_glirel(
        records_df,
        policy_gliner_model,
        labels,
        label_mode="policy_composite",
        gliner_threshold=float(strategy_result.get("gliner_threshold", 0.3)),
        gliner_device=gliner_device or strategy_result.get("gliner_device"),
        max_words=int(strategy_result.get("max_words", DATABASE_STYLE_GLINER_MAX_WORDS)),
        overlap_words=int(strategy_result.get("overlap_words", DATABASE_STYLE_GLINER_OVERLAP_WORDS)),
        verbose=verbose,
    )
    relation_predictions = predict_glirel_for_masking_inputs(
        glirel_model,
        glirel_inputs,
        relation_labels=relation_labels,
        threshold=float(strategy_result.get("glirel_threshold", 0.05)),
        top_k=int(strategy_result.get("glirel_top_k", 3)),
        batch_size=int(strategy_result.get("glirel_batch_size", 4)),
        verbose=verbose,
    )
    relation_scores = _relation_scores_by_mention(
        gliner_predictions,
        relation_predictions,
        strategy_result.get("relation_side_weights") or {},
        fallback_by_type=strategy_result.get("semantic_type_priors") or {},
    )
    policy_scores = _policy_prior_scores(gliner_predictions)
    scores = {
        mention_id: _combine_scores(
            policy_scores.get(mention_id, 0.0),
            relation_scores.get(mention_id, 0.0),
            alpha=alpha,
            beta=beta,
            mode=combination_mode,
        )
        for mention_id in policy_scores.keys() | relation_scores.keys()
    }
    predictions_by_chunk = _predictions_by_chunk_with_scores(gliner_predictions, scores)
    metrics = compute_masking_metrics(records_df, predictions_by_chunk, threshold=threshold, verbose=False)
    return {
        "strategy": "Policy GLiNER + GLiREL",
        "threshold": threshold,
        "metrics": metrics,
        "predictions_by_chunk": predictions_by_chunk,
        "glirel_inputs": glirel_inputs,
        "relation_predictions": relation_predictions,
    }


def train_policy_gliner_glirel_strategy(
    records_df: pd.DataFrame,
    policy_gliner_model: Any,
    glirel_model: Any,
    labels: Sequence[str],
    output_json: str | Path,
    *,
    relation_labels: Sequence[str] | None = None,
    gliner_threshold: float = 0.3,
    gliner_device: str | None = None,
    max_words: int = DATABASE_STYLE_GLINER_MAX_WORDS,
    overlap_words: int = DATABASE_STYLE_GLINER_OVERLAP_WORDS,
    glirel_threshold: float = 0.05,
    glirel_top_k: int = 3,
    glirel_batch_size: int = 4,
    combination_mode: str = "weighted_sum",
    alphas: Sequence[float] | None = None,
    betas: Sequence[float] | None = None,
    tune_thresholds: Sequence[float] | None = None,
    optimize_relation_weights: bool = True,
    relation_weight_candidates: Sequence[float] | None = None,
    relation_weight_max_passes: int = 2,
    force_retrain: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train/tune Strategy 3: policy GLiNER plus GLiREL relation enhancement."""

    output_path = Path(output_json)
    if output_path.exists() and not force_retrain:
        return read_json_record(output_path)

    gliner_predictions, glirel_inputs = build_gliner_predictions_for_glirel(
        records_df,
        policy_gliner_model,
        labels,
        label_mode="policy_composite",
        gliner_threshold=gliner_threshold,
        gliner_device=gliner_device,
        max_words=max_words,
        overlap_words=overlap_words,
        verbose=verbose,
    )
    relation_predictions = predict_glirel_for_masking_inputs(
        glirel_model,
        glirel_inputs,
        relation_labels=relation_labels,
        threshold=glirel_threshold,
        top_k=glirel_top_k,
        batch_size=glirel_batch_size,
        verbose=verbose,
    )
    weights = _relation_side_weights(gliner_predictions, relation_predictions)
    fallback = _type_priors(gliner_predictions)
    relation_weight_optimization_history: list[dict[str, Any]] = []
    if optimize_relation_weights:
        optimized = optimize_relation_side_weights_for_f1(
            records_df,
            gliner_predictions,
            relation_predictions,
            weights,
            fallback_by_type=fallback,
            thresholds=tune_thresholds,
            weight_candidates=relation_weight_candidates,
            max_passes=relation_weight_max_passes,
            verbose=verbose,
        )
        weights = optimized["weights"]
        relation_weight_optimization_history = optimized["history"]
    relation_scores = _relation_scores_by_mention(
        gliner_predictions,
        relation_predictions,
        weights,
        fallback_by_type=fallback,
    )
    policy_scores = _policy_prior_scores(gliner_predictions)
    alphas = list(alphas or [round(value / 10, 1) for value in range(0, 11)])
    betas = list(betas or [0.5, 1.0, 2.0])
    tune_thresholds = list(tune_thresholds or [round(value / 100, 2) for value in range(0, 101, 5)])

    candidate_rows: list[dict[str, Any]] = []
    if combination_mode == "harmonic":
        iterator = [(None, beta) for beta in betas]
    else:
        iterator = [(alpha, None) for alpha in alphas]
    best_result: dict[str, Any] | None = None
    best_scores_by_mention: dict[str, float] = {}
    best_predictions: list[dict[str, Any]] = []
    for alpha, beta in tqdm(
        iterator,
        desc="Tuning policy+relation combination",
        unit="combo",
        disable=not verbose,
    ):
        scores_by_mention = {
            mention_id: _combine_scores(
                policy_scores.get(mention_id, 0.0),
                relation_scores.get(mention_id, 0.0),
                alpha=float(alpha if alpha is not None else 0.5),
                beta=float(beta if beta is not None else 1.0),
                mode=combination_mode,
            )
            for mention_id in policy_scores.keys() | relation_scores.keys()
        }
        predictions_by_chunk = _predictions_by_chunk_with_scores(
            gliner_predictions,
            scores_by_mention,
        )
        tuning = tune_sensitive_threshold(
            records_df,
            predictions_by_chunk,
            thresholds=tune_thresholds,
            score_field="sensitivity_score",
            verbose=False,
        )
        row = dict(tuning["best"])
        row["alpha"] = alpha
        row["beta"] = beta
        candidate_rows.append(row)
        if best_result is None or (row["f1"], row["precision"], row["recall"]) > (
            best_result["f1"],
            best_result["precision"],
            best_result["recall"],
        ):
            best_result = row
            best_scores_by_mention = scores_by_mention
            best_predictions = [
                dict(prediction)
                for predictions_for_chunk in predictions_by_chunk.values()
                for prediction in predictions_for_chunk
            ]

    result = {
        "strategy": "policy_gliner_glirel_relation_enhancement",
        "combination_mode": combination_mode,
        "gliner_threshold": float(gliner_threshold),
        "gliner_device": gliner_device or "auto",
        "max_words": int(max_words),
        "overlap_words": int(overlap_words),
        "glirel_threshold": float(glirel_threshold),
        "glirel_top_k": int(glirel_top_k),
        "objective": "maximize_detection_f1_or_minimize_1_minus_f1",
        "best": best_result or {},
        "metrics_by_combination": candidate_rows,
        "relation_weight_optimization_history": relation_weight_optimization_history,
        "relation_side_weights": dict(weights),
        "semantic_type_priors": dict(fallback),
        "policy_scores_by_mention": dict(policy_scores),
        "relation_scores_by_mention": dict(relation_scores),
        "combined_scores_by_mention": dict(best_scores_by_mention),
        "gliner_predictions": best_predictions or list(gliner_predictions),
        "glirel_inputs": list(glirel_inputs),
        "relation_predictions": relation_predictions,
    }
    write_json_record(result, output_path)
    return result


def download_hf_snapshot_without_symlinks(
    repo_id: str,
    local_dir: str | Path,
    *,
    repo_type: str = "model",
    token: str | None = None,
) -> Path:
    """Download a Hugging Face snapshot into a normal local directory.

    This avoids Windows cache symlink failures such as WinError 1314 when the
    current account does not have symlink privileges.
    """

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Install huggingface_hub first: pip install huggingface_hub"
        ) from exc

    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
            token=token,
        )
    except TypeError:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(local_path),
            token=token,
        )
    return local_path
