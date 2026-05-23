"""Sparse lexical retrieval helpers for the retrieval component notebook."""

from __future__ import annotations

import re
import time
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm


CHUNK_ID_COLUMN = "chunk_id"
NEO4J_CHUNK_ID_COLUMN = "chunk_id:ID(Chunk)"
WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")
BM25_CACHE_VERSION = 1
DEFAULT_RETRIEVAL_INDEX_DIR = Path("data") / "evaluation" / "retrieval_indexes"


def normalize_chunk_columns(chunks_df: pd.DataFrame) -> pd.DataFrame:
    df = chunks_df.copy()
    if NEO4J_CHUNK_ID_COLUMN in df.columns and CHUNK_ID_COLUMN not in df.columns:
        df = df.rename(columns={NEO4J_CHUNK_ID_COLUMN: CHUNK_ID_COLUMN})
    if "chunk_index:int" in df.columns and "chunk_index" not in df.columns:
        df = df.rename(columns={"chunk_index:int": "chunk_index"})
    return df


def tokenize_for_bm25(text: Any) -> list[str]:
    """Small lexical tokenizer for BM25 over archive chunks."""

    if pd.isna(text):
        return []
    return WORD_PATTERN.findall(str(text).lower())


def build_retrieval_text(row: pd.Series) -> str:
    """Build retrieval-safe chunk text from exported chunk fields."""

    parts = [
        str(row.get("title", "") or ""),
        str(row.get("summary", "") or ""),
        str(row.get("masked_text", "") or ""),
    ]
    return "\n".join(part for part in parts if part.strip())


@dataclass
class ArchiveChunkCorpus:
    """In-memory retrieval corpus loaded from exported archive chunks."""

    chunks_df: pd.DataFrame
    text_column: str = "retrieval_text"

    @classmethod
    def from_chunks_csv(
        cls,
        chunks_csv: str | Path,
        *,
        max_chunks: int | None = None,
        chunksize: int | None = None,
        verbose: bool = True,
    ) -> "ArchiveChunkCorpus":
        path = Path(chunks_csv)
        usecols = [
            NEO4J_CHUNK_ID_COLUMN,
            "document_id",
            "dataset",
            "modality",
            "title",
            "masked_text",
            "embedding_text",
            "summary",
            "access_level",
            "sensitivity_level",
        ]

        if chunksize is None:
            if verbose:
                tqdm.write(f"Loading retrieval corpus from {path}")
            df = pd.read_csv(path, usecols=usecols, nrows=max_chunks)
        else:
            frames = []
            remaining = max_chunks
            reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize)
            for chunk in tqdm(reader, desc="Loading retrieval corpus", disable=not verbose):
                if remaining is not None:
                    if remaining <= 0:
                        break
                    chunk = chunk.head(remaining)
                    remaining -= len(chunk)
                frames.append(chunk)
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=usecols)

        df = normalize_chunk_columns(df)
        df["retrieval_text"] = df.apply(build_retrieval_text, axis=1)
        df = df[df["retrieval_text"].str.strip().ne("")].reset_index(drop=True)
        return cls(chunks_df=df)


class BM25SparseRetriever:
    """BM25 sparse retriever over archive chunks."""

    method_name = "bm25"

    def __init__(
        self,
        corpus: ArchiveChunkCorpus,
        *,
        tokenized_corpus: list[list[str]] | None = None,
        index: BM25Okapi | None = None,
        verbose: bool = True,
    ) -> None:
        self.corpus = corpus
        if tokenized_corpus is None:
            token_iter = tqdm(
                corpus.chunks_df[corpus.text_column],
                total=len(corpus.chunks_df),
                desc="Tokenizing BM25 corpus",
                disable=not verbose,
            )
            tokenized_corpus = [tokenize_for_bm25(text) for text in token_iter]
        self.tokenized_corpus = tokenized_corpus
        self.index = index if index is not None else BM25Okapi(self.tokenized_corpus)

    @classmethod
    def from_cache_or_build(
        cls,
        chunks_csv: str | Path,
        *,
        cache_path: str | Path | None = None,
        max_chunks: int | None = None,
        chunksize: int | None = 50_000,
        force_rebuild: bool = False,
        verbose: bool = True,
    ) -> "BM25SparseRetriever":
        """Load a cached BM25 retriever or build and persist it."""

        chunks_csv = Path(chunks_csv)
        if cache_path is None:
            cache_path = DEFAULT_RETRIEVAL_INDEX_DIR / "bm25_sparse_retriever.pkl"
        cache_path = Path(cache_path)

        if cache_path.exists() and not force_rebuild:
            if verbose:
                tqdm.write(f"Loading BM25 sparse retriever cache from {cache_path}")
            return cls.load_cache(cache_path)

        corpus = ArchiveChunkCorpus.from_chunks_csv(
            chunks_csv,
            max_chunks=max_chunks,
            chunksize=chunksize,
            verbose=verbose,
        )
        retriever = cls(corpus, verbose=verbose)
        retriever.save_cache(
            cache_path,
            source_chunks_csv=chunks_csv,
            max_chunks=max_chunks,
        )
        return retriever

    def save_cache(
        self,
        cache_path: str | Path,
        *,
        source_chunks_csv: str | Path | None = None,
        max_chunks: int | None = None,
    ) -> Path:
        """Persist BM25 index, tokenized corpus, and chunk metadata to disk."""

        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_version": BM25_CACHE_VERSION,
            "method_name": self.method_name,
            "source_chunks_csv": str(source_chunks_csv) if source_chunks_csv else None,
            "max_chunks": max_chunks,
            "text_column": self.corpus.text_column,
            "chunks_df": self.corpus.chunks_df,
            "tokenized_corpus": self.tokenized_corpus,
            "index": self.index,
            "created_at": time.time(),
        }
        with cache_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return cache_path

    @classmethod
    def load_cache(cls, cache_path: str | Path) -> "BM25SparseRetriever":
        """Load a persisted BM25 retriever cache."""

        cache_path = Path(cache_path)
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)

        cache_version = payload.get("cache_version")
        if cache_version != BM25_CACHE_VERSION:
            raise ValueError(
                f"Unsupported BM25 cache version {cache_version}; "
                f"expected {BM25_CACHE_VERSION}. Rebuild the cache."
            )

        corpus = ArchiveChunkCorpus(
            chunks_df=payload["chunks_df"],
            text_column=payload.get("text_column", "retrieval_text"),
        )
        return cls(
            corpus,
            tokenized_corpus=payload.get("tokenized_corpus"),
            index=payload.get("index"),
            verbose=False,
        )

    def retrieve(
        self,
        query: str,
        *,
        query_id: str = "sample_query",
        top_k: int = 10,
    ) -> pd.DataFrame:
        started_at = time.perf_counter()
        query_tokens = tokenize_for_bm25(query)
        scores = self.index.get_scores(query_tokens)
        top_k = min(top_k, len(scores))
        if top_k <= 0:
            return pd.DataFrame()

        top_indices = scores.argsort()[-top_k:][::-1]
        latency_ms = (time.perf_counter() - started_at) * 1000
        rows = []
        for rank, index in enumerate(top_indices, start=1):
            chunk = self.corpus.chunks_df.iloc[int(index)]
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "retrieval_method": self.method_name,
                    "chunk_id": chunk.get(CHUNK_ID_COLUMN, ""),
                    "rank": rank,
                    "score": float(scores[index]),
                    "document_id": chunk.get("document_id", ""),
                    "dataset": chunk.get("dataset", ""),
                    "modality": chunk.get("modality", ""),
                    "title": chunk.get("title", ""),
                    "latency_ms": latency_ms,
                    "retrieval_stage": "sparse",
                    "retrieval_text_preview": str(chunk.get(self.corpus.text_column, ""))[:500],
                }
            )
        return pd.DataFrame(rows)
