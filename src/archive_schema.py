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
    text = text.replace("\\'", "'").replace('\\"', '"')
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _word_count(text: str) -> int:
    return len(normalize_text(text).split())


def _chunk_words(
    text: str,
    max_words: int,
    overlap_words: int,
) -> List[str]:
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


def _is_transcript_turn(line: str) -> bool:
    """Return whether a line looks like a speaker turn."""
    if ":" not in line:
        return False

    speaker, utterance = line.split(":", 1)
    speaker = speaker.strip()
    utterance = utterance.strip()

    if not speaker or not utterance:
        return False

    if len(speaker.split()) > 8:
        return False

    return bool(re.search(r"[A-Za-z0-9]", speaker))


def _extract_transcript_turns(text: str) -> List[str]:
    turns = [
        normalize_text(line)
        for line in normalize_text(text).split("\n")
        if normalize_text(line)
    ]

    if len(turns) < 2:
        return []

    turn_like_count = sum(1 for turn in turns if _is_transcript_turn(turn))
    if turn_like_count / len(turns) < 0.5:
        return []

    return turns


def _extract_paragraphs(text: str) -> List[str]:
    paragraphs = [
        normalize_text(paragraph)
        for paragraph in re.split(r"\n\s*\n", normalize_text(text))
        if normalize_text(paragraph)
    ]

    if len(paragraphs) < 2:
        return []

    return paragraphs


def _is_caption_segment(line: str) -> bool:
    normalized_line = normalize_text(line)
    return bool(
        re.match(
            r"^(segment\s+\d+\s+)?\[\d+(?:\.\d+)?s?-\d+(?:\.\d+)?s?\]\s*:",
            normalized_line,
            flags=re.IGNORECASE,
        )
    )


def _extract_caption_segments(text: str) -> List[str]:
    segments = [
        normalize_text(line)
        for line in normalize_text(text).split("\n")
        if normalize_text(line)
    ]

    if len(segments) < 2:
        return []

    segment_like_count = sum(1 for segment in segments if _is_caption_segment(segment))
    if segment_like_count / len(segments) < 0.5:
        return []

    return segments


def _chunk_segments_by_words(
    segments: List[str],
    max_words: int,
    overlap_words: int,
    separator: str,
) -> List[str]:
    chunks: List[str] = []
    start = 0

    while start < len(segments):
        current_segments: List[str] = []
        current_words = 0
        index = start

        while index < len(segments):
            segment = segments[index]
            segment_words = _word_count(segment)

            if segment_words > max_words and not current_segments:
                chunks.extend(_chunk_words(segment, max_words=max_words, overlap_words=overlap_words))
                index += 1
                break

            if current_segments and current_words + segment_words > max_words:
                break

            current_segments.append(segment)
            current_words += segment_words
            index += 1

            if current_words >= max_words:
                break

        if current_segments:
            chunks.append(separator.join(current_segments))

        if index >= len(segments):
            break

        overlap_count = 0
        overlap_total = 0
        for segment in reversed(current_segments):
            if overlap_total >= overlap_words:
                break
            overlap_total += _word_count(segment)
            overlap_count += 1

        next_start = index - overlap_count
        start = next_start if next_start > start else index

    return chunks


def _chunk_turns_by_words(
    turns: List[str],
    max_words: int,
    overlap_words: int,
) -> List[str]:
    return _chunk_segments_by_words(
        turns,
        max_words=max_words,
        overlap_words=overlap_words,
        separator="\n",
    )


def _chunk_paragraphs_by_words(
    paragraphs: List[str],
    max_words: int,
    overlap_words: int,
) -> List[str]:
    return _chunk_segments_by_words(
        paragraphs,
        max_words=max_words,
        overlap_words=overlap_words,
        separator="\n\n",
    )


def _chunk_captions_by_words(
    captions: List[str],
    max_words: int,
    overlap_words: int,
) -> List[str]:
    return _chunk_segments_by_words(
        captions,
        max_words=max_words,
        overlap_words=overlap_words,
        separator="\n",
    )


def _looks_like_flattened_article(text: str) -> bool:
    return bool(re.search(r"(?<=[.!?])\s+(?=[A-Z][a-z])", normalize_text(text)))


def _split_flattened_article_paragraphs(text: str) -> List[str]:
    normalized_text = normalize_text(text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z][a-z])", normalized_text)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if len(sentences) < 4:
        return []

    paragraphs: List[str] = []
    current_sentences: List[str] = []
    current_words = 0
    target_words = 120

    for sentence in sentences:
        sentence_words = _word_count(sentence)
        if current_sentences and current_words + sentence_words > target_words:
            paragraphs.append(" ".join(current_sentences))
            current_sentences = []
            current_words = 0

        current_sentences.append(sentence)
        current_words += sentence_words

    if current_sentences:
        paragraphs.append(" ".join(current_sentences))

    return paragraphs if len(paragraphs) > 1 else []


def chunk_text_by_words(
    text: str,
    max_words: int = 350,
    overlap_words: int = 60,
) -> List[str]:
    """Chunk text by words while preserving detected source structure.

    If the text looks like timestamped caption segments, e.g.
    ``Segment 1 [0.00s-10.00s]: a person is cooking``, chunks are built
    from whole caption segments. If the text looks like newline-separated
    transcript turns, e.g.
    ``Speaker: utterance``, chunks are built from whole turns so speaker
    labels and utterances stay together. Otherwise, it uses paragraph-aware
    article chunking when paragraphs can be detected. If no structure is
    available, it falls back to a plain sliding word window.

    TODO: extend this with additional modality-aware chunking:
    - layout-aware chunking for DocVQA
    """
    if max_words <= 0:
        raise ValueError("max_words must be greater than 0")

    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")

    normalized_text = normalize_text(text)
    if not normalized_text:
        return []

    captions = _extract_caption_segments(normalized_text)
    if captions:
        return _chunk_captions_by_words(
            captions,
            max_words=max_words,
            overlap_words=overlap_words,
        )

    turns = _extract_transcript_turns(normalized_text)
    if turns:
        return _chunk_turns_by_words(
            turns,
            max_words=max_words,
            overlap_words=overlap_words,
        )

    paragraphs = _extract_paragraphs(normalized_text)
    if not paragraphs and _looks_like_flattened_article(normalized_text):
        paragraphs = _split_flattened_article_paragraphs(normalized_text)

    if paragraphs:
        return _chunk_paragraphs_by_words(
            paragraphs,
            max_words=max_words,
            overlap_words=overlap_words,
        )

    return _chunk_words(
        normalized_text,
        max_words=max_words,
        overlap_words=overlap_words,
    )


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
