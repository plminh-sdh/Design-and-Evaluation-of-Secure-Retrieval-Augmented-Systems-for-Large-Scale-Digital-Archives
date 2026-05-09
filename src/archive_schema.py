"""
    Common archive schema definitions for the database layer.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class DatasetName(str, Enum):
    MEDIASUM = "mediasum"
    CNN_DAILYMAIL = "cnn_dailymail"
    MSR_VTT = "msr_vtt"
    DOCVQA = "docvqa"


class Modality(str, Enum):
    TRANSCRIPT = "transcript"
    ARTICLE = "article"
    VIDEO = "video"
    OCR_DOCUMENT = "ocr_document"


class SensitivityLevel(str, Enum):
    S0 = "S0"  # Public / no sensitive content detected
    S1 = "S1"  # Direct identifiers, e.g. email, phone, ID
    S2 = "S2"  # Quasi-identifiers, e.g. organization, role, location
    S3 = "S3"  # Confidential business/academic content
    S4 = "S4"  # Contextual or temporal sensitive information


class AccessLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"


@dataclass
class SensitiveEntity:
    entity_type: str
    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    sensitivity_level: str = SensitivityLevel.S1.value
    masking_strategy: str = "placeholder"


@dataclass
class ArchiveDocument:
    document_id: str
    source_id: str
    dataset: str
    modality: str
    raw_text: str
    title: Optional[str] = None
    summary: Optional[str] = None
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ArchiveChunk:
    chunk_id: str
    document_id: str
    source_id: str
    dataset: str
    modality: str
    chunk_index: int
    raw_text: str
    masked_text: str
    embedding_text: str
    title: Optional[str] = None
    summary: Optional[str] = None
    topic: Optional[str] = None
    sensitivity_level: str = SensitivityLevel.S0.value
    sensitive_entities: List[Dict[str, Any]] = field(default_factory=list)
    access_level: str = AccessLevel.PUBLIC.value
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    dense_vector_ready: bool = False
    sparse_vector_ready: bool = False


def stable_id(*parts: Any, length: int = 24) -> str:
    """Create a deterministic ID from several stable values."""
    text = "::".join(str(part) for part in parts if part is not None)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def normalize_text(text: Any) -> str:
    """Normalize text while keeping the original meaning intact."""
    if text is None:
        return ""

    if isinstance(text, list):
        text = "\n".join(str(item) for item in text)

    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text_by_words(
    text: str,
    max_words: int = 350,
    overlap_words: int = 60,
) -> List[str]:
    """Simple word-based chunking baseline.

    TODO: replace or extend this with modality-aware chunking:
    - transcript turn-aware chunking for MediaSum
    - paragraph-aware chunking for CNN/DailyMail
    - caption/segment-aware chunking for MSR-VTT
    - layout-aware chunking for DocVQA
    """
    words = normalize_text(text).split()
    if not words:
        return []

    chunks = []
    start = 0
    step = max(max_words - overlap_words, 1)

    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks


def apply_security_masking_placeholder(text: str) -> Tuple[str, List[Dict[str, Any]], str, str]:
    """Temporary masking function.

    This is intentionally a no-op for now so we can build the database layer first.
    TODO: replace this with the rule-based sensitive data detector and masker.
    """
    masked_text = text
    sensitive_entities: List[Dict[str, Any]] = []
    sensitivity_level = SensitivityLevel.S0.value
    access_level = AccessLevel.PUBLIC.value
    return masked_text, sensitive_entities, sensitivity_level, access_level


def build_embedding_text(
    *,
    dataset: str,
    modality: str,
    masked_text: str,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    source_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the future embedding input from masked content plus useful context."""
    lines = [
        f"Dataset: {dataset}",
        f"Modality: {modality}",
    ]

    if title:
        lines.append(f"Title: {title}")

    if summary:
        lines.append(f"Summary: {summary}")

    if source_metadata:
        compact_metadata = {
            key: value
            for key, value in source_metadata.items()
            if value is not None and key not in {"raw_text", "masked_text"}
        }
        if compact_metadata:
            lines.append(f"Metadata: {json.dumps(compact_metadata, ensure_ascii=False)}")

    lines.append(f"Content: {masked_text}")
    return "\n".join(lines)


def archive_chunk_to_qdrant_payload(chunk: ArchiveChunk) -> Dict[str, Any]:
    """Convert an ArchiveChunk to a Qdrant payload.

    Embedding vectors will be attached later when we implement dense/sparse embedding.
    """
    return asdict(chunk)
