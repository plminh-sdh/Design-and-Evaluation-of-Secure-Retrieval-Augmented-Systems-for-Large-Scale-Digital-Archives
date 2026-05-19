"""Reusable plotting helpers for notebook evaluation sections."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def plot_threshold_metrics(
    results_df: pd.DataFrame,
    *,
    threshold_column: str = "threshold",
    metrics: Sequence[str] = ("precision", "recall", "f1"),
    mode_column: str | None = "mode",
    modes: Sequence[str] = ("strict", "relaxed"),
    title_prefix: str = "",
    x_label: str = "Confidence threshold",
    y_label: str = "Score",
    figsize: tuple[int, int] = (14, 5),
):
    """Plot metric curves over thresholds.

    When ``mode_column`` is present, one subplot is created per requested mode.
    Otherwise a single axis is used.
    """
    import matplotlib.pyplot as plt

    if mode_column and mode_column in results_df.columns:
        fig, axes = plt.subplots(1, len(modes), figsize=figsize, sharey=True)
        if len(modes) == 1:
            axes = [axes]

        for axis, mode in zip(axes, modes):
            mode_df = results_df[results_df[mode_column] == mode].sort_values(
                threshold_column
            )
            for metric in metrics:
                if metric in mode_df.columns:
                    axis.plot(
                        mode_df[threshold_column],
                        mode_df[metric],
                        marker="o",
                        label=metric,
                    )
            axis.set_title(f"{title_prefix}{mode.title()}")
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.set_ylim(0, 1)
            axis.grid(True, alpha=0.3)
            axis.legend()
    else:
        fig, axis = plt.subplots(figsize=figsize)
        plot_df = results_df.sort_values(threshold_column)
        for metric in metrics:
            if metric in plot_df.columns:
                axis.plot(
                    plot_df[threshold_column],
                    plot_df[metric],
                    marker="o",
                    label=metric,
                )
        axis.set_title(title_prefix.rstrip() or "Threshold Metrics")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_ylim(0, 1)
        axis.grid(True, alpha=0.3)
        axis.legend()

    plt.tight_layout()
    return fig


def plot_group_metric_by_threshold(
    results_df: pd.DataFrame,
    *,
    group_fields: Sequence[str] = ("dataset", "modality"),
    metric: str = "f1",
    threshold_column: str = "threshold",
    title: str = "Metric by Group",
    x_label: str = "Confidence threshold",
    y_label: str | None = None,
    figsize: tuple[int, int] = (12, 6),
):
    """Plot one metric over thresholds for each dataset/modality group."""
    import matplotlib.pyplot as plt

    plot_df = results_df.copy()
    group_column = "_group"
    plot_df[group_column] = plot_df[list(group_fields)].astype(str).agg(" / ".join, axis=1)

    fig, axis = plt.subplots(figsize=figsize)
    for name, group_df in plot_df.groupby(group_column):
        group_df = group_df.sort_values(threshold_column)
        axis.plot(
            group_df[threshold_column],
            group_df[metric],
            marker="o",
            label=name,
        )

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label or metric)
    axis.set_ylim(0, 1)
    axis.grid(True, alpha=0.3)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig


def plot_mismatch_summary(
    mismatch_summary_df: pd.DataFrame,
    *,
    label_column: str,
    count_column: str = "count",
    title: str,
    top_n: int = 25,
    figsize: tuple[int, int] = (10, 6),
    color: str = "#4477AA",
):
    """Plot a horizontal bar chart for mismatch summary rows."""
    import matplotlib.pyplot as plt

    plot_df = mismatch_summary_df.head(top_n).copy()
    fig, axis = plt.subplots(figsize=figsize)
    axis.barh(plot_df[label_column].astype(str), plot_df[count_column], color=color)
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel("Count")
    plt.tight_layout()
    return fig
