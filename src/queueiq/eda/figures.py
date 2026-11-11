"""EDA figures. Each function returns a matplotlib Figure; build.py saves them."""

import matplotlib

matplotlib.use("Agg")  # render to files, no window

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from queueiq.eda.tables import MIN_AT_RISK  # noqa: E402
from queueiq.style import (  # noqa: E402
    NEUTRAL_SERIES,
    OUTCOME_COLORS,
    OUTCOME_LABELS,
    SURFACE,
    TEXT_SECONDARY,
    apply_matplotlib_style,
)

CURVE_EVENTS = ("operational", "withdrawn")


def _trim(curve: pd.DataFrame) -> pd.DataFrame:
    """Plot only while enough projects are still observed (same rule as the tables)."""
    return curve[curve["n_at_risk"] >= MIN_AT_RISK]


def _step_xy(curve: pd.DataFrame, column: str):
    years = pd.concat([pd.Series([0.0]), curve["time"] / 12], ignore_index=True)
    values = pd.concat([pd.Series([0.0]), curve[column] * 100], ignore_index=True)
    return years, values


def _draw_cif(ax, curve: pd.DataFrame, label_lines: bool) -> None:
    curve = _trim(curve)
    for k in CURVE_EVENTS:
        x, y = _step_xy(curve, f"cif_{k}")
        ax.step(x, y, where="post", color=OUTCOME_COLORS[k], label=OUTCOME_LABELS[k])
        if label_lines and len(x):
            ax.annotate(
                f"{OUTCOME_LABELS[k]} {y.iloc[-1]:.0f}%",
                (x.iloc[-1], y.iloc[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=TEXT_SECONDARY,
            )
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    ax.set_yticks(range(0, 101, 25))


def cif_overall(curve: pd.DataFrame, title: str) -> plt.Figure:
    apply_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _draw_cif(ax, curve, label_lines=True)
    ax.set_title(title)
    ax.set_xlabel("Years since entering the queue")
    ax.set_ylabel("Cumulative share of requests (%)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def cif_small_multiples(curves: dict[str, pd.DataFrame], title: str, ncols: int) -> plt.Figure:
    apply_matplotlib_style()
    n = len(curves)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows + 0.6), sharex=True, sharey=True
    )
    axes = list(axes.flat) if n > 1 else [axes]
    for ax, (name, curve) in zip(axes, curves.items(), strict=False):
        _draw_cif(ax, curve, label_lines=False)
        ax.set_title(f"{name}  (n={int(curve['n_at_risk'].iloc[0]):,})", fontsize=10)
    for ax in axes[n:]:
        ax.set_visible(False)
    # The lowest visible panel in each column carries the x-axis labels.
    for col in range(ncols):
        visible = [i for i in range(col, n, ncols)]
        if visible:
            bottom = axes[visible[-1]]
            bottom.xaxis.set_tick_params(labelbottom=True)
            bottom.set_xlabel("Years since entry")
    for ax in axes[::ncols]:
        ax.set_ylabel("Cumulative %")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def cohort_outcomes(cohorts: pd.DataFrame, title: str) -> plt.Figure:
    """100% stacked bars: status at the cutoff for each entry year (raw, not censoring-aware)."""
    apply_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    years = cohorts.index.astype(str)
    bottom = pd.Series(0.0, index=cohorts.index)
    for k in ("operational", "withdrawn", "in_queue"):
        ax.bar(
            years,
            cohorts[k],
            bottom=bottom,
            width=0.75,
            color=OUTCOME_COLORS[k],
            label=OUTCOME_LABELS[k],
            edgecolor=SURFACE,
            linewidth=1,  # thin surface gap between stacked segments
        )
        bottom += cohorts[k]
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of requests (%)")
    ax.set_xlabel("Entry year")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncols=3)
    fig.tight_layout()
    return fig


def requests_per_year(counts: pd.Series, title: str) -> plt.Figure:
    apply_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(counts.index.astype(str), counts.values, width=0.75, color=NEUTRAL_SERIES)
    ax.set_title(title)
    ax.set_xlabel("Entry year")
    ax.set_ylabel("Requests")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig
