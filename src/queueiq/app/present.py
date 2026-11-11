"""Formatting and tables for the dashboard. Pure functions: no Streamlit, easy to test."""

import pandas as pd

from queueiq.style import OUTCOME_LABELS

STAT_LABELS = {
    "withdrawn_3y": "Withdraws within 3 years",
    "operational_3y": "Operating within 3 years",
    "operational_5y": "Operating within 5 years",
    "months_in_queue_5y": "Expected months in queue",
}
FUEL_LABELS = {
    "solar": "Solar",
    "wind": "Wind",
    "battery": "Battery storage",
    "gas": "Gas",
    "other": "Other (hydro, nuclear, offshore wind, ...)",
}


def pct(x: float) -> str:
    """Percent with one decimal; values under 1% keep one significant decimal too."""
    return f"{100 * x:.1f}%"


def span(low: float, high: float, months: bool = False) -> str:
    if months:
        return f"{low:.0f}–{high:.0f} months"
    return f"{100 * low:.1f}–{100 * high:.1f}%"


STAT_HELP = {
    "withdrawn_3y": "Chance the project has left the queue within 36 months of entry.",
    "operational_3y": "Chance the project has reached commercial operation within 36 months.",
    "operational_5y": "Chance the project has reached commercial operation within 60 months.",
    "months_in_queue_5y": "Expected months still waiting in the queue during the first 60 months "
    "after entry (60 would mean it never leaves in that time).",
}


def stat_value(h: dict) -> str:
    return f"{h['model']:.0f} of {h['months']}" if h["outcome"] is None else pct(h["model"])


def stat_range(h: dict) -> str:
    if h["outcome"] is None:
        return f"Past-error range {h['low']:.0f}–{h['high']:.0f}"
    return "Past-error range " + span(h["low"], h["high"])


def range_numbers(h: dict) -> str:
    return stat_range(h).removeprefix("Past-error range ")


def summary_html(pred: dict) -> str:
    """The summary row: one block per headline number, ruled in its outcome's color."""
    blocks = []
    for h in pred["headline"]:
        kind = h["outcome"] or "queue"
        blocks.append(
            f'<div class="qi-stat qi-{kind}">'
            f'<div class="qi-stat-label" title="{STAT_HELP[h["key"]]}">'
            f"{STAT_LABELS[h['key']]}</div>"
            f'<div class="qi-stat-value">{stat_value(h)}</div>'
            f'<div class="qi-stat-range">Past-error range '
            f'<span class="qi-nowrap">{range_numbers(h)}</span></div></div>'
        )
    return f'<div class="qi-stats">{"".join(blocks)}</div>'


def curve_table(pred: dict, step: int = 6) -> pd.DataFrame:
    """The chart as a table (every `step` months): the accessible view of the curves."""
    rows = []
    for m in range(step, len(pred["months"]) + 1, step):
        row = {"Months since entry": m}
        for k in ("withdrawn", "operational"):
            c = pred["curves"][k]
            row[OUTCOME_LABELS[k]] = pct(c["model"][m - 1])
            row[f"{OUTCOME_LABELS[k]}: past-error range"] = span(c["low"][m - 1], c["high"][m - 1])
        rows.append(row)
    return pd.DataFrame(rows)


def compare_table(preds: list[dict]) -> pd.DataFrame:
    rows = []
    for p in preds:
        row = {"Project": p["project"]["name"]}
        for h in p["headline"]:
            row[STAT_LABELS[h["key"]]] = f"{stat_value(h)} ({range_numbers(h)})"
        rows.append(row)
    return pd.DataFrame(rows)


def describe_project(proj: dict) -> str:
    fuel = FUEL_LABELS[proj["fuel"]].split(" (")[0]
    storage = " + storage" if proj["storage"] and proj["fuel"] != "battery" else ""
    return f"{proj['capacity_mw']:,.0f} MW {fuel.lower()}{storage}, {proj['state']} ({proj['iso']})"


def long_date(iso_date: str) -> str:
    d = pd.Timestamp(iso_date)
    return f"{d.day} {d.month_name()} {d.year}"


def unique_name(name: str, taken: list[str]) -> str:
    if name not in taken:
        return name
    i = 2
    while f"{name} ({i})" in taken:
        i += 1
    return f"{name} ({i})"
