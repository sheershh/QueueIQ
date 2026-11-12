"""What actually happened to comparable past projects (decisions D-057).

The dashboard leads with real outcomes, not model output: for the project's grid operator and
technology, the share of past entrants that had withdrawn, were operating, or were still waiting
`horizon` months after entering. Only entry years whose projects all have that much follow-up
before the data cutoff are used (the same rule as explain/reference.observed_reference), so
nothing is censored. Groups with fewer than MIN_N projects fall back to the grid operator alone.
"""

import pandas as pd

from queueiq.features.tabular import fuel_group

MIN_N = 30
YEARS = 4  # the latest four fully followed entry years


def _followed(df: pd.DataFrame, horizon: int) -> pd.Series:
    """True if the outcome at `horizon` months is known (event before it, or followed past it)."""
    event_before = (df["outcome"] != "censored") & (df["time_months"] < horizon)
    return event_before | (df["time_months"] >= horizon)


def comparable_table(entries: pd.DataFrame, cutoff: pd.Timestamp, horizon: int = 36) -> dict:
    """Outcome shares by (iso, fuel) and by iso alone, for the latest fully followed entry years."""
    scoped = entries[entries["in_model_scope"]].copy()
    last = int(
        scoped.loc[
            scoped["queue_date"] <= cutoff - pd.DateOffset(months=horizon), "queue_year"
        ].max()
    )
    # The last eligible year must be complete: every entrant of that year reaches the horizon.
    while pd.Timestamp(f"{last}-12-31") > cutoff - pd.DateOffset(months=horizon):
        last -= 1
    years = (last - YEARS + 1, last)
    df = scoped[scoped["queue_year"].between(*years)]
    df = df[_followed(df, horizon)].assign(fuel=fuel_group(df["fuel_type"]))

    def shares(g: pd.DataFrame) -> dict:
        wd = ((g["outcome"] == "withdrawn") & (g["time_months"] < horizon)).mean()
        op = ((g["outcome"] == "operational") & (g["time_months"] < horizon)).mean()
        return {
            "n": len(g),
            "withdrawn": float(wd),
            "operational": float(op),
            "waiting": float(1 - wd - op),
        }

    return {
        "years": list(years),
        "horizon_months": horizon,
        "min_n": MIN_N,
        "by_iso_fuel": {f"{i}|{f}": shares(g) for (i, f), g in df.groupby(["iso", "fuel"])},
        "by_iso": {i: shares(g) for i, g in df.groupby("iso")},
    }


def comparable_outcomes(table: dict, iso: str, fuel: str) -> dict:
    """The group used for one project: ISO × technology, or the ISO alone if that group is small."""
    span = {"years": table["years"], "horizon_months": table["horizon_months"]}
    group = table["by_iso_fuel"].get(f"{iso}|{fuel}")
    if group and group["n"] >= table["min_n"]:
        return {**group, "iso": iso, "fuel": fuel, "fallback": False, **span}
    return {**table["by_iso"][iso], "iso": iso, "fuel": None, "fallback": True, **span}
