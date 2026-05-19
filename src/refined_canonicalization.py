"""Helpers for ReFinED entity canonicalization fine-tuning."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm


def patch_refined_for_py312_windows() -> Dict[str, str]:
    """Patch known ReFinED/Transformers compatibility issues."""
    status: Dict[str, str] = {}

    try:
        from refined.resource_management.aws import S3Manager, tqdm_hook

        def download_file_if_needed_py312_windows(
            self: Any,
            s3_bucket: str,
            s3_key: str,
            output_file_path: str,
            progress_bar: bool = True,
        ) -> None:
            s3_obj = self._s3.Object(s3_bucket, s3_key)
            s3_obj_size = s3_obj.content_length
            s3_last_modified = int(s3_obj.last_modified.timestamp())

            if (
                os.path.isfile(output_file_path)
                and int(os.stat(output_file_path).st_mtime) > s3_last_modified
            ):
                self._log.debug(f"File already downloaded: {output_file_path}.")
                return

            self._log.debug(
                f"Downloading {output_file_path} file from S3 bucket: "
                f"{s3_bucket}, key: {s3_key}"
            )
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

            with tqdm(
                total=s3_obj_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {output_file_path}",
                disable=not progress_bar,
            ) as progress:
                s3_obj.download_file(output_file_path, Callback=tqdm_hook(progress))

            self._log.debug("Download complete.")

        S3Manager.download_file_if_needed = download_file_if_needed_py312_windows
        status["refined_s3"] = "patched"
    except Exception as exc:  # pragma: no cover - depends on optional package.
        status["refined_s3"] = f"not_patched: {exc}"

    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        if not getattr(
            PreTrainedTokenizerBase,
            "_refined_add_special_tokens_patch",
            False,
        ):
            original_tokenizer_base_init = PreTrainedTokenizerBase.__init__

            def patched_tokenizer_base_init(self: Any, *args: Any, **kwargs: Any) -> Any:
                kwargs.pop("add_special_tokens", None)
                return original_tokenizer_base_init(self, *args, **kwargs)

            PreTrainedTokenizerBase.__init__ = patched_tokenizer_base_init
            PreTrainedTokenizerBase._refined_add_special_tokens_patch = True

        status["transformers_tokenizer"] = "patched"
    except Exception as exc:  # pragma: no cover - depends on optional package.
        status["transformers_tokenizer"] = f"not_patched: {exc}"

    try:
        import refined.training.fine_tune.fine_tune as refined_fine_tune
        from torch.utils.data import DataLoader as TorchDataLoader

        def single_worker_dataloader(*args: Any, **kwargs: Any) -> Any:
            # ReFinED's default num_workers=1 fails on Windows because the
            # spawned worker must pickle the DocDataset/preprocessor graph.
            kwargs["num_workers"] = 0
            return TorchDataLoader(*args, **kwargs)

        refined_fine_tune.DataLoader = single_worker_dataloader
        status["refined_fine_tune_dataloader"] = "patched_num_workers_0"
    except Exception as exc:  # pragma: no cover - depends on optional package.
        status["refined_fine_tune_dataloader"] = f"not_patched: {exc}"

    return status


def load_refined_model(
    *,
    model_name: str,
    entity_set: str,
    use_precomputed_descriptions: bool = True,
    device: Optional[str] = None,
    patch_compatibility: bool = True,
) -> Any:
    """Load a ReFinED model after applying local compatibility patches."""
    if patch_compatibility:
        patch_refined_for_py312_windows()

    from refined.inference.processor import Refined

    return Refined.from_pretrained(
        model_name=model_name,
        entity_set=entity_set,
        use_precomputed_descriptions=use_precomputed_descriptions,
        device=device,
    )


def load_jsonl_records(path: Path | str) -> List[Dict[str, Any]]:
    """Load JSONL records from disk."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonicalization_records_to_gold_summary(
    records: Iterable[Mapping[str, Any]],
) -> Dict[str, int]:
    """Summarize linkable vs NIL gold spans for quick notebook diagnostics."""
    record_count = 0
    linkable = 0
    nil = 0
    mentions = 0

    for record in records:
        record_count += 1
        for span in record.get("spans", []):
            mentions += 1
            if str(span.get("gold_entity")).upper() == "NIL":
                nil += 1
            else:
                linkable += 1

    return {
        "records": record_count,
        "mentions": mentions,
        "linkable": linkable,
        "nil": nil,
    }


def _is_nil_entity(entity_id: Any) -> bool:
    return entity_id is None or str(entity_id).strip().upper() in {"", "NIL", "NONE"}


def _span_to_refined_span(span: Mapping[str, Any], *, include_gold: bool) -> Any:
    from refined.data_types.base_types import Entity, Span

    start = int(span["start"])
    end = int(span["end"])
    gold_entity_id = span.get("gold_entity") or span.get("wikidata_entity_id")
    title = span.get("wikipedia_entity_title")
    gold_entity = None

    if include_gold and not _is_nil_entity(gold_entity_id):
        gold_entity = Entity(
            wikidata_entity_id=str(gold_entity_id),
            wikipedia_entity_title=title,
            human_readable_name=title,
        )

    return Span(
        text=str(span.get("text") or ""),
        start=start,
        ln=end - start,
        gold_entity=gold_entity,
        coarse_type=span.get("coarse_type") or "MENTION",
    )


def canonicalization_record_to_refined_doc(
    record: Mapping[str, Any],
    preprocessor: Any,
    *,
    add_candidate: bool = True,
    sample_k_candidates: Optional[int] = None,
) -> Any:
    """Convert one canonicalization JSONL record to a ReFinED training Doc."""
    from refined.data_types.doc_types import Doc

    text = str(record.get("text") or "")
    gold_spans = [
        _span_to_refined_span(span, include_gold=True)
        for span in record.get("spans", [])
        if not _is_nil_entity(span.get("gold_entity") or span.get("wikidata_entity_id"))
    ]
    md_spans = [
        _span_to_refined_span(span, include_gold=False)
        for span in record.get("md_spans") or record.get("spans", [])
    ]

    try:
        doc_id = int(record.get("doc_id") or record.get("chunk_id"))
    except Exception:
        doc_id = None

    return Doc.from_text_with_spans(
        text=text,
        spans=gold_spans,
        preprocessor=preprocessor,
        add_candidate=add_candidate,
        md_spans=md_spans,
        sample_k_candidates=sample_k_candidates,
        doc_id=doc_id,
    )


def canonicalization_records_to_refined_docs(
    records: Sequence[Mapping[str, Any]],
    preprocessor: Any,
    *,
    add_candidate: bool = True,
    sample_k_candidates: Optional[int] = None,
) -> List[Any]:
    """Convert canonicalization JSONL records to ReFinED Doc objects."""
    return [
        canonicalization_record_to_refined_doc(
            record,
            preprocessor,
            add_candidate=add_candidate,
            sample_k_candidates=sample_k_candidates,
        )
        for record in tqdm(records, desc="Building ReFinED Docs", unit="doc")
    ]


def fine_tune_refined_on_canonicalization_records(
    refined_model: Any,
    *,
    train_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    fine_tuning_args: Optional[Any] = None,
    add_candidate: bool = True,
) -> Any:
    """Fine-tune ReFinED with its built-in ``fine_tune_on_docs`` API."""
    patch_refined_for_py312_windows()

    from refined.training.fine_tune.fine_tune import fine_tune_on_docs

    sample_k_candidates = getattr(fine_tuning_args, "num_candidates_train", None)
    train_docs = canonicalization_records_to_refined_docs(
        train_records,
        refined_model.preprocessor,
        add_candidate=add_candidate,
        sample_k_candidates=sample_k_candidates,
    )
    eval_docs = canonicalization_records_to_refined_docs(
        eval_records,
        refined_model.preprocessor,
        add_candidate=add_candidate,
        sample_k_candidates=getattr(fine_tuning_args, "num_candidates_eval", None),
    )

    return fine_tune_on_docs(
        refined=refined_model,
        train_docs=train_docs,
        eval_docs=eval_docs,
        fine_tuning_args=fine_tuning_args,
    )


def refined_metrics_to_dataframe(metrics_by_name: Mapping[str, Any]) -> pd.DataFrame:
    """Flatten ReFinED Metrics objects into a notebook-friendly DataFrame."""
    rows = []

    for dataset_name, metrics in metrics_by_name.items():
        row = {
            "dataset": dataset_name,
            "mode": "EL" if metrics.el else "ED",
            "num_docs": metrics.num_docs,
            "num_gold_spans": metrics.num_gold_spans,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "precision": metrics.get_precision(),
            "recall": metrics.get_recall(),
            "f1": metrics.get_f1(),
            "accuracy": metrics.get_accuracy(),
            "gold_recall": metrics.get_gold_recall(),
        }
        if metrics.el:
            row.update(
                {
                    "tp_md": metrics.tp_md,
                    "fp_md": metrics.fp_md,
                    "fn_md": metrics.fn_md,
                    "md_precision": metrics.get_precision_md(),
                    "md_recall": metrics.get_recall_md(),
                    "md_f1": metrics.get_f1_md(),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_refined_on_canonicalization_records(
    refined_model: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str = "canonicalization_test",
    ed_threshold: float = 0.15,
    el: bool = True,
    ed: bool = True,
    print_errors: bool = False,
    return_special_spans: bool = True,
    add_candidate: bool = True,
    num_candidates_eval: Optional[int] = None,
) -> tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate a ReFinED model on canonicalization JSONL records."""
    from refined.evaluation.evaluation import evaluate

    eval_docs = canonicalization_records_to_refined_docs(
        records,
        refined_model.preprocessor,
        add_candidate=add_candidate,
        sample_k_candidates=num_candidates_eval,
    )
    metrics_by_name = evaluate(
        evaluation_dataset_name_to_docs={dataset_name: eval_docs},
        refined=refined_model,
        ed_threshold=ed_threshold,
        el=el,
        ed=ed,
        print_errors=print_errors,
        return_special_spans=return_special_spans,
    )

    return metrics_by_name, refined_metrics_to_dataframe(metrics_by_name)
