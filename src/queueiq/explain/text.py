"""Plain-language explanation of one project's prediction, for the dashboard."""

import pandas as pd

from queueiq.explain.groups import TARGETS, describe

MIN_POINTS = 0.5  # contributions smaller than half a percentage point are not mentioned


def narrative(table: pd.DataFrame, row: pd.Series, reference: dict, top: int = 3) -> dict[str, str]:
    """One sentence block per target, e.g.
    'Chance of withdrawing within 3 years: 81.1%. Model's average prediction for recent entrants:
     50.8%; observed for 2022 entrants (latest with full follow-up): 37.2%. Pushing it up: ...'

    reference[target] = {"cohort": year, "observed": share} (see explain/reference.py).
    """
    texts = {}
    for target, (_, h, label) in TARGETS.items():
        p, base = table.attrs["prediction"][target], table.attrs["baseline"][target]
        ref = reference[target]
        contrib = (table[target] * 100).sort_values(key=abs, ascending=False)
        contrib = contrib[contrib.abs() >= MIN_POINTS].head(top)
        up = [f"{g} ({v:+.1f} points: {describe(g, row)})" for g, v in contrib.items() if v > 0]
        down = [f"{g} ({v:+.1f} points: {describe(g, row)})" for g, v in contrib.items() if v < 0]
        parts = [
            f"{label}: {100 * p:.1f}%.",
            f"Model's average prediction for recent entrants: {100 * base:.1f}%; observed for "
            f"{ref['cohort']} entrants (latest with full {h // 12}-year follow-up): "
            f"{100 * ref['observed']:.1f}%.",
        ]
        if up:
            parts.append("Pushing it up: " + "; ".join(up) + ".")
        if down:
            parts.append("Pulling it down: " + "; ".join(down) + ".")
        if not up and not down:
            parts.append("No feature group moves it by half a point or more.")
        texts[target] = " ".join(parts)
    return texts
