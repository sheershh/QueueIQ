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


def describe_project(proj: dict, county: dict | None = None) -> str:
    fuel = FUEL_LABELS[proj["fuel"]].split(" (")[0]
    storage = " + storage" if proj["storage"] and proj["fuel"] != "battery" else ""
    where = proj["state"]
    if county:
        name = (
            f"county {county['name']}" if county["key"] == "unknown" else f"{county['name']} County"
        )
        where = f"{name}{' (assumed)' if county['assumed'] else ''}, {proj['state']}"
    return f"{proj['capacity_mw']:,.0f} MW {fuel.lower()}{storage}, {where} ({proj['iso']})"


def assumed_county_text(pred: dict, project: dict | object) -> str:
    state = project["state"] if isinstance(project, dict) else project.state
    return (
        f"County assumed: {pred['county']['name']}, the county in {state} with the most past "
        "requests. Choose a county on the left to change it."
    )


OUTCOME_PHRASES = {"withdrawn": "withdrawing", "operational": "reaching operation"}


def band_check_text(past: dict) -> str:
    """How the most recent checked cohort compares with the band (report-only check, D-051)."""
    check = past["final_test_check"]
    years = "–".join(str(y) for y in check["test_years"])
    parts = []
    for k in ("withdrawn", "operational"):
        v = check["outcomes"][k]
        band = span(v["band_low"], v["band_high"])
        what = f"{OUTCOME_PHRASES[k]} within 3 years"
        if v["inside_band"]:
            parts.append(
                f"For {what}, the observed {pct(v['observed'])} fell inside the band ({band})."
            )
        else:
            cases = f"{v['cases']} project" + ("" if v["cases"] == 1 else "s")
            parts.append(
                f"For {what}, the most recent checked cohort was off by more than the band: "
                f"the band for {years} entrants was {band}, but {pct2(v['observed'])} "
                f"happened ({cases})."
            )
    return " ".join(parts)


def pct2(x: float) -> str:
    """Percent with two decimals for values under 1% (so 0.03% doesn't print as 0.0%)."""
    return f"{100 * x:.2f}%" if x < 0.01 else pct(x)


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


# -- history-first answer (D-057, D-058) ------------------------------------------------------
TECH_LABELS = {
    "solar": "Solar",
    "wind": "Wind",
    "battery": "Battery",
    "gas": "Gas",
    "other": "Other",
}
TECH_PHRASES = {
    "solar": "solar",
    "wind": "wind",
    "battery": "battery",
    "gas": "gas",
    "other": "other-technology",
}
GRID_ORDER = ("withdrawn", "operational", "waiting")
GRID_LABELS = {"withdrawn": "Withdrawn", "operational": "Operating", "waiting": "Still waiting"}


def plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def share_text(x: float) -> str:
    if x == 0:
        return "none"
    if x < 0.005:
        return "under 1%"
    return f"{round(100 * x)}%"


def history_sentence(c: dict, iso: str, fuel: str) -> str:
    """'Of 2,163 PJM solar projects that entered in 2019–2022, 32% had withdrawn within 3 years,
    under 1% were operating and 68% were still waiting.'"""
    years = f"{c['years'][0]}–{c['years'][1]}"
    horizon = c["horizon_months"] // 12
    tail = (
        f"{share_text(c['withdrawn'])} had withdrawn within {horizon} years, "
        f"{share_text(c['operational'])} were operating and {share_text(c['waiting'])} were "
        "still waiting."
    )
    if c["fallback"]:
        return (
            f"Too few {iso} {TECH_PHRASES[fuel]} projects to report on their own, so this uses "
            f"all {c['n']:,} {iso} projects that entered in {years}: {tail}"
        )
    return f"Of {c['n']:,} {iso} {TECH_PHRASES[fuel]} projects that entered in {years}, {tail}"


def grid_counts(c: dict) -> dict[str, int]:
    """Shares as whole squares out of 100 (largest remainder, so they always add up to 100)."""
    raw = {k: 100 * c[k] for k in GRID_ORDER}
    counts = {k: int(v) for k, v in raw.items()}
    for k in sorted(GRID_ORDER, key=lambda k: raw[k] - counts[k], reverse=True)[
        : 100 - sum(counts.values())
    ]:
        counts[k] += 1
    return counts


def outcome_grid_html(c: dict) -> str:
    """100 squares, one per 1% of comparable projects, filled once in outcome colors."""
    counts = grid_counts(c)
    cells, i = [], 0
    for k in GRID_ORDER:
        for _ in range(counts[k]):
            cells.append(f'<i class="qi-cell qi-{k}" style="--i:{i}"></i>')
            i += 1
    summary = ", ".join(f"{counts[k]} {GRID_LABELS[k].lower()}" for k in GRID_ORDER)
    legend = "".join(
        f'<span><i class="qi-cell qi-{k}"></i>{GRID_LABELS[k]} {share_text(c[k])}</span>'
        for k in GRID_ORDER
    )
    return (
        f'<div class="qi-grid" role="img" aria-label="Out of 100 comparable projects: {summary}">'
        f'{"".join(cells)}</div><div class="qi-grid-legend">{legend}</div>'
    )


def evidence_text(e: dict) -> str:
    """Plain-language reading of the model's 3-year AUC for one outcome."""
    return (
        f"{e['label'].capitalize()}: in back-tests, given one project that had this outcome and "
        f"one that did not, the model ranked them the right way round {round(100 * e['auc'])}% "
        "of the time (a coin flip would be 50%)."
    )


def percentile_html(label: str, t: dict, years: list[int]) -> str:
    p = round(t["percentile"])
    return (
        f'<div class="qi-pctl"><div class="qi-pctl-label">{label}</div>'
        f'<div class="qi-pctl-value">Higher than {p}% of {years[0]}–{years[1]} entrants</div>'
        f'<div class="qi-track" role="img" aria-label="{p}th percentile">'
        f'<span class="qi-marker" style="left:{p}%"></span></div>'
        f'<div class="qi-muted">{evidence_text(t["evidence"])}</div></div>'
    )


def short_warning(cal: dict) -> str:
    wd = cal["withdrawn"]
    return (
        "The model's own percentages run high on recent projects: for 2021–22 entrants it "
        f"predicted {round(100 * wd['predicted'])}% would withdraw within 3 years, and "
        f"{round(100 * wd['observed'])}% did. That is why the main answer above uses real "
        "outcomes, and the model only ranks this project against others."
    )
