"""Reusable evaluation metrics for extraction-style experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


def spans_overlap(
    a_start: int,
    a_end: int,
    b_start: int,
    b_end: int,
) -> bool:
    """Return whether two half-open character spans overlap."""
    return max(a_start, b_start) < min(a_end, b_end)


def precision_recall_f1(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> Dict[str, float]:
    """Compute precision, recall, and F1 from counts."""
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _mention_label(mention: Mapping[str, Any]) -> Optional[str]:
    label = mention.get("label", mention.get("type", mention.get("mention_type")))
    return str(label).upper() if label is not None else None


def _mention_start(mention: Mapping[str, Any]) -> Optional[int]:
    value = mention.get("start", mention.get("start_char"))
    return int(value) if value is not None else None


def _mention_end(mention: Mapping[str, Any]) -> Optional[int]:
    value = mention.get("end", mention.get("end_char"))
    return int(value) if value is not None else None


def _valid_mentions(
    mentions: Sequence[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    valid_mentions = []
    for mention in mentions:
        if _mention_label(mention) is None:
            continue
        if _mention_start(mention) is None or _mention_end(mention) is None:
            continue
        valid_mentions.append(mention)
    return valid_mentions


def mention_matches(
    prediction: Mapping[str, Any],
    gold: Mapping[str, Any],
    *,
    mode: str = "strict",
    require_label_match: bool = True,
) -> bool:
    """Return whether a predicted mention matches a gold mention."""
    if require_label_match and _mention_label(prediction) != _mention_label(gold):
        return False

    pred_start = _mention_start(prediction)
    pred_end = _mention_end(prediction)
    gold_start = _mention_start(gold)
    gold_end = _mention_end(gold)

    if None in {pred_start, pred_end, gold_start, gold_end}:
        return False

    if mode == "strict":
        return pred_start == gold_start and pred_end == gold_end

    if mode == "relaxed":
        return spans_overlap(pred_start, pred_end, gold_start, gold_end)

    raise ValueError(f"Unknown mention matching mode: {mode}")


def evaluate_mention_lists(
    predictions: Sequence[Mapping[str, Any]],
    gold_mentions: Sequence[Mapping[str, Any]],
    *,
    mode: str = "strict",
    require_label_match: bool = True,
) -> Dict[str, Any]:
    """Evaluate one predicted mention list against one gold mention list."""
    predictions = _valid_mentions(predictions)
    gold_mentions = _valid_mentions(gold_mentions)

    matched_gold_indexes = set()
    true_positive = 0

    for prediction in predictions:
        for gold_index, gold in enumerate(gold_mentions):
            if gold_index in matched_gold_indexes:
                continue
            if mention_matches(
                prediction,
                gold,
                mode=mode,
                require_label_match=require_label_match,
            ):
                true_positive += 1
                matched_gold_indexes.add(gold_index)
                break

    false_positive = len(predictions) - true_positive
    false_negative = len(gold_mentions) - true_positive
    metrics = precision_recall_f1(true_positive, false_positive, false_negative)

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "predicted": len(predictions),
        "gold": len(gold_mentions),
        **metrics,
    }


def evaluate_mention_records(
    records: Iterable[Mapping[str, Any]],
    predictions_by_chunk_id: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    mode: str = "strict",
    group_fields: Sequence[str] = (),
    require_label_match: bool = True,
) -> pd.DataFrame:
    """Evaluate mention predictions over records, optionally grouped by metadata fields."""
    grouped_counts: Dict[Tuple[Any, ...], Dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "predicted": 0, "gold": 0, "chunks": 0}
    )

    for record in records:
        chunk_id = record["chunk_id"]
        group_key = tuple(record.get(field) for field in group_fields)
        counts = evaluate_mention_lists(
            predictions_by_chunk_id.get(chunk_id, []),
            record.get("mentions", []),
            mode=mode,
            require_label_match=require_label_match,
        )

        grouped_counts[group_key]["chunks"] += 1
        for field in ["tp", "fp", "fn", "predicted", "gold"]:
            grouped_counts[group_key][field] += int(counts[field])

    rows = []
    for group_key, counts in grouped_counts.items():
        metrics = precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
        row = {field: value for field, value in zip(group_fields, group_key)}
        row.update(counts)
        row.update(metrics)
        row["mode"] = mode
        rows.append(row)

    return pd.DataFrame(rows)


def filter_predictions_by_score(
    predictions_by_chunk_id: Mapping[str, Sequence[Mapping[str, Any]]],
    threshold: float,
    *,
    score_field: str = "score",
) -> Dict[str, List[Mapping[str, Any]]]:
    """Filter per-chunk predictions by confidence score."""
    return {
        chunk_id: [
            prediction
            for prediction in predictions
            if float(prediction.get(score_field, 0.0)) >= threshold
        ]
        for chunk_id, predictions in predictions_by_chunk_id.items()
    }


def tune_mention_thresholds(
    records: Iterable[Mapping[str, Any]],
    predictions_by_chunk_id: Mapping[str, Sequence[Mapping[str, Any]]],
    thresholds: Sequence[float],
    *,
    group_fields: Sequence[str] = (),
    modes: Sequence[str] = ("strict", "relaxed"),
    require_label_match: bool = True,
    score_field: str = "score",
) -> Tuple[pd.DataFrame, Dict[float, Dict[str, List[Mapping[str, Any]]]]]:
    """Evaluate mention extraction metrics over several confidence thresholds."""
    records = list(records)
    result_frames = []
    filtered_predictions_by_threshold = {}

    for threshold in thresholds:
        filtered_predictions = filter_predictions_by_score(
            predictions_by_chunk_id,
            threshold,
            score_field=score_field,
        )
        filtered_predictions_by_threshold[float(threshold)] = filtered_predictions

        for mode in modes:
            overall_df = evaluate_mention_records(
                records,
                filtered_predictions,
                mode=mode,
                require_label_match=require_label_match,
            )
            overall_df["threshold"] = float(threshold)
            overall_df["scope"] = "overall"
            result_frames.append(overall_df)

            if group_fields:
                grouped_df = evaluate_mention_records(
                    records,
                    filtered_predictions,
                    mode=mode,
                    group_fields=group_fields,
                    require_label_match=require_label_match,
                )
                grouped_df["threshold"] = float(threshold)
                grouped_df["scope"] = "_".join(group_fields)
                result_frames.append(grouped_df)

    if not result_frames:
        return pd.DataFrame(), filtered_predictions_by_threshold

    return (
        pd.concat(result_frames, ignore_index=True, sort=False),
        filtered_predictions_by_threshold,
    )
