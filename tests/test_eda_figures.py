"""Smoke tests: each figure builds from small inputs without errors."""

import matplotlib.pyplot as plt
import pandas as pd

from queueiq.eda import figures as F
from queueiq.eda.survival import aalen_johansen
from queueiq.style import OUTCOME_COLORS

CURVE = aalen_johansen(
    pd.Series(range(1, 101), dtype=float),
    pd.Series(["withdrawn", "operational", "censored", "withdrawn"] * 25),
)


def test_cif_overall_draws_both_outcomes_in_fixed_colors():
    fig = F.cif_overall(CURVE, "t")
    colors = {line.get_color() for line in fig.axes[0].get_lines()}
    assert {OUTCOME_COLORS["operational"], OUTCOME_COLORS["withdrawn"]} <= colors
    plt.close(fig)


def test_small_multiples_hides_unused_panels():
    fig = F.cif_small_multiples({"A": CURVE, "B": CURVE, "C": CURVE}, "t", ncols=2)
    assert sum(ax.get_visible() for ax in fig.axes) == 3
    plt.close(fig)


def test_bar_figures_build():
    cohorts = pd.DataFrame(
        {"operational": [10.0], "withdrawn": [60.0], "in_queue": [30.0]},
        index=pd.Index([2015], name="entry_year"),
    )
    for fig in (F.cohort_outcomes(cohorts, "t"), F.requests_per_year(pd.Series({2015: 5}), "t")):
        assert fig.axes
        plt.close(fig)
