"""Plain-language explanation of one project's prediction, for the dashboard."""

import pandas as pd

from queueiq.explain.groups import TARGETS, describe

MIN_POINTS = 0.5  # contributions smaller than half a percentage point are not mentioned


def narrative(table: pd.DataFrame, row: pd.Series, top: int = 3) -> dict[str, str]:
    """One sentence block per target, e.g.
    'Chance of withdrawing within 3 years: 48% (typical recent project: 42%).
     Pushing it up: county congestion (+5.1 points: 23 earlier requests ...).'"""
    texts = {}
    for target, (_, _, label) in TARGETS.items():
        p, base = table.attrs["prediction"][target], table.attrs["baseline"][target]
        contrib = (table[target] * 100).sort_values(key=abs, ascending=False)
        contrib = contrib[contrib.abs() >= MIN_POINTS].head(top)
        up = [f"{g} ({v:+.1f} points: {describe(g, row)})" for g, v in contrib.items() if v > 0]
        down = [f"{g} ({v:+.1f} points: {describe(g, row)})" for g, v in contrib.items() if v < 0]
        parts = [f"{label}: {100 * p:.1f}% (typical recent project: {100 * base:.1f}%)."]
        if up:
            parts.append("Pushing it up: " + "; ".join(up) + ".")
        if down:
            parts.append("Pulling it down: " + "; ".join(down) + ".")
        if not up and not down:
            parts.append("No feature group moves it by half a point or more.")
        texts[target] = " ".join(parts)
    return texts
