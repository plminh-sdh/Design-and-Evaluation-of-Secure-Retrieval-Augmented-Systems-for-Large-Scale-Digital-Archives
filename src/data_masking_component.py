"""Helpers for the Data Masking Component notebook."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


TAB_DATASET_NAME = "ildpil/text-anonymization-benchmark"
TAB_DATASET_LABEL = "TAB"
TAB_MODALITY = "text"
SENSITIVE_IDENTIFIER_TYPES = {"DIRECT", "QUASI"}
PRESERVE_IDENTIFIER_TYPES = {"NO_MASK"}
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
