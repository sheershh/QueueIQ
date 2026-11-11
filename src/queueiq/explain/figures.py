"""Phase 6 figures: grouped permutation importance and one local explanation."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from queueiq.explain.groups import TARGETS  # noqa: E402
from queueiq.style import (  # noqa: E402
    AXIS,
    NEUTRAL_SERIES,
    OUTCOME_COLORS,
    OUTCOME_LABELS,
    TEXT,
    apply_matplotlib_style,
)


def permutation_importance(table: pd.DataFrame) -> plt.Figure:
    """Drop in Brier skill when a group is shuffled: block A mean with min-max range (bar +
    whisker) and the final test (marker), one panel per outcome."""
    apply_matplotlib_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    groups = list(table.index)
    y = np.arange(len(groups))[::-1]
    for ax, outcome in zip(axes, ("withdrawn", "operational"), strict=True):
        col = f"{outcome}_skill"
        mean, lo, hi = (table[(k, col)].to_numpy() for k in ("A mean (2014-17)", "A min", "A max"))
        final = table[("C (2021-22)", col)].to_numpy()
        ax.barh(
            y, mean, height=0.55, color=OUTCOME_COLORS[outcome], label="Development mean (2014-17)"
        )
        ax.errorbar(
            mean,
            y,
            xerr=[mean - lo, hi - mean],
            fmt="none",
            ecolor=NEUTRAL_SERIES,
            elinewidth=1,
            capsize=3,
            label="Development range",
        )
        ax.scatter(final, y, marker="D", s=28, color=TEXT, zorder=3, label="Final test (2021-22)")
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_yticks(y, groups)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=10)
        ax.set_xlabel("Drop in Brier skill when shuffled")
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=3)
    fig.suptitle("What the model relies on out of sample", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def local_explanation(table: pd.DataFrame, title: str) -> plt.Figure:
    """Contribution of each group (percentage points) for both headline numbers."""
    apply_matplotlib_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    groups = list(table.index)
    y = np.arange(len(groups))[::-1]
    for ax, (target, (outcome, _, label)) in zip(axes, TARGETS.items(), strict=True):
        v = table[target].to_numpy() * 100
        colors = [OUTCOME_COLORS[outcome] if x > 0 else NEUTRAL_SERIES for x in v]
        ax.barh(y, v, height=0.55, color=colors)
        for yi, x in zip(y, v, strict=True):
            ax.annotate(
                "0.0" if abs(x) < 0.05 else f"{x:+.1f}",  # avoid printing "-0.0"
                (x, yi),
                xytext=(4 if x >= 0 else -4, 0),
                textcoords="offset points",
                ha="left" if x >= 0 else "right",
                va="center",
                fontsize=9,
                color=TEXT,
            )
        ax.axvline(0, color=AXIS, linewidth=1)
        p, b = table.attrs["prediction"][target] * 100, table.attrs["baseline"][target] * 100
        ax.set_title(f"{label}\n{p:.1f}% vs typical {b:.1f}%", fontsize=10)
        ax.set_yticks(y, groups)
        ax.set_xlabel("Contribution (percentage points)")
        ax.grid(axis="y", visible=False)
        lim = max(abs(v).max() * 1.35, 1)
        ax.set_xlim(-lim, lim)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig
