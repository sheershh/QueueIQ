"""One visual system for every chart (reports now, the dashboard in Phase 7).

The two outcome colors are fixed: an outcome is the same color everywhere. They are
slots 1-2 of a palette checked for color-blind separation and contrast (operational =
blue, withdrawn = orange). 'Still in queue' is a neutral gray, because it is the absence
of an outcome, not a third outcome.
"""

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

OUTCOME_COLORS = {
    "operational": "#2a78d6",
    "withdrawn": "#eb6834",
    "in_queue": "#b9b8b0",
}
# For charts that are not about outcomes (e.g. request counts), so blue/orange keep their meaning.
NEUTRAL_SERIES = "#6f6e69"

OUTCOME_LABELS = {
    "operational": "Reached operation",
    "withdrawn": "Withdrawn",
    "in_queue": "Still in queue",
}

FONT_FAMILY = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
FONT_SIZES = {"title": 12, "body": 10, "small": 9}  # no more than three sizes


def apply_matplotlib_style() -> None:
    """Calm, recessive chrome: hairline grid, no top/right spines, text in ink colors."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": FONT_SIZES["body"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.titleweight": "regular",
            "axes.titlelocation": "left",
            "axes.labelsize": FONT_SIZES["body"],
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": FONT_SIZES["small"],
            "ytick.labelsize": FONT_SIZES["small"],
            "legend.fontsize": FONT_SIZES["small"],
            "legend.frameon": False,
            "lines.linewidth": 2,
            "text.color": TEXT,
            "figure.dpi": 100,
            "savefig.dpi": 150,
        }
    )
