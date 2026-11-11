"""Phase 5 figures: rolling-origin performance and calibration curves."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from queueiq.style import (  # noqa: E402
    AXIS,
    COMPARISON_COLORS,
    GRID,
    OUTCOME_COLORS,
    OUTCOME_LABELS,
    TEXT_SECONDARY,
    apply_matplotlib_style,
)


def rolling_origin(metrics: pd.DataFrame, lines: dict[str, tuple[str, str, str]]) -> plt.Figure:
    """2x2: rows AUC / skill at 36 months, columns outcomes; x = origin year.

    lines: model name -> (label, color key, linestyle). Blocks are shaded and labelled.
    """
    apply_matplotlib_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    m = metrics[metrics["tau"] == 36]
    for col, outcome in enumerate(("operational", "withdrawn")):
        for row, (metric, ylabel) in enumerate(
            (("auc", "AUC at 3 years"), ("skill", "Brier skill vs null"))
        ):
            ax = axes[row, col]
            for name, (label, color, style) in lines.items():
                d = m[(m["model"] == name) & (m["outcome"] == outcome)].sort_values("origin")
                ax.plot(
                    d["origin"],
                    d[metric],
                    linestyle=style,
                    marker="o",
                    markersize=4,
                    color=COMPARISON_COLORS[color],
                    label=label,
                )
            ref = 0.5 if metric == "auc" else 0.0
            ax.axhline(ref, color=AXIS, linewidth=1)
            if row == 0:  # headroom so the block labels never sit on a data point
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi + 0.18 * (hi - lo))
            ax.axvspan(2013.5, 2017.5, color=GRID, alpha=0.35, linewidth=0)
            ax.axvspan(2020.5, 2021.5, color=GRID, alpha=0.6, linewidth=0)
            if row == 0:
                ax.set_title(OUTCOME_LABELS[outcome], fontsize=10)
                ax.text(
                    2015.5,
                    0.98,
                    "development (A)",
                    ha="center",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                    color=TEXT_SECONDARY,
                    fontsize=9,
                )
                ax.text(
                    2019,
                    0.98,
                    "extension (B)",
                    ha="center",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                    color=TEXT_SECONDARY,
                    fontsize=9,
                )
                ax.text(
                    2021,
                    0.98,
                    "final (C)",
                    ha="center",
                    va="top",
                    transform=ax.get_xaxis_transform(),
                    color=TEXT_SECONDARY,
                    fontsize=9,
                )
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("Origin year (test cohort entered that year; final = 2021-22)")
            ax.set_xticks(range(2014, 2022))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=len(lines))
    fig.suptitle("Rolling-origin performance at 3 years", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def calibration(bins: dict[str, dict[str, list]], title: str) -> plt.Figure:
    """bins[outcome][variant label] = [(pred, obs, n), ...]; first variant filled, second hollow."""
    apply_matplotlib_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, outcome in zip(axes, ("operational", "withdrawn"), strict=True):
        color = OUTCOME_COLORS[outcome]
        top = 0.0
        for i, (label, b) in enumerate(bins[outcome].items()):
            pred = np.array([p for p, _, _ in b]) * 100
            obs = np.array([o for _, o, _ in b]) * 100
            top = max(top, pred.max(), obs.max())
            ax.plot(
                pred,
                obs,
                marker="o",
                markersize=6,
                linewidth=1.2,
                color=color,
                markerfacecolor=color if i == 0 else "white",
                label=label,
            )
        lim = top * 1.1 + 1
        ax.plot([0, lim], [0, lim], color=AXIS, linewidth=1, linestyle="--")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=10)
        ax.set_xlabel("Predicted share by 3 years (%)")
        ax.set_ylabel("Observed share by 3 years (%)")
        ax.legend(loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig
