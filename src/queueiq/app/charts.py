"""Altair charts for the dashboard. Every color comes from queueiq.style.

Outlook: cumulative chance of each outcome by month, the past-error range as a shaded band,
the region past the longest checked horizon greyed, and the observed rate of the latest fully
followed cohort as a ring. Hovering anywhere shows exact values for that month.
"""

import altair as alt
import pandas as pd

from queueiq.style import NEUTRAL_SERIES, OUTCOME_COLORS, OUTCOME_LABELS, UI, UI_FONT, UI_FONT_SIZES

ORDER = ["withdrawn", "operational"]
BODY = UI_FONT_SIZES["body"]
FONT = f"{UI_FONT}, sans-serif"
SHORT = {"withdrawn": "Withdrawn", "operational": "Operating"}  # tooltip keys
COLOR = alt.Scale(
    domain=[OUTCOME_LABELS[k] for k in ORDER], range=[OUTCOME_COLORS[k] for k in ORDER]
)


def theme(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(font=FONT, background=UI["paper"])
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=UI["muted"],
            titleColor=UI["muted"],
            labelFontSize=BODY,
            titleFontSize=BODY,
            titleFontWeight="normal",
            gridColor=UI["rule"],
            domainColor=UI["rule"],
            tickColor=UI["rule"],
        )
        .configure_text(font=FONT, fontSize=BODY, color=UI["ink"])
    )


def curve_frame(pred: dict) -> pd.DataFrame:
    """Long table: one row per month and outcome, month 0 included (everyone still waiting)."""
    rows = []
    for k in ORDER:
        c = pred["curves"][k]
        rows.append(
            {"month": 0, "outcome": OUTCOME_LABELS[k], "model": 0.0, "low": 0.0, "high": 0.0}
        )
        for i, m in enumerate(pred["months"]):
            rows.append(
                {
                    "month": m,
                    "outcome": OUTCOME_LABELS[k],
                    "model": c["model"][i],
                    "low": c["low"][i],
                    "high": c["high"][i],
                }
            )
    return pd.DataFrame(rows)


def hover_frame(pred: dict) -> pd.DataFrame:
    """Wide table for the tooltip: both outcomes for one month in one place."""
    out = pd.DataFrame({"month": pred["months"]})
    for k in ORDER:
        c, name = pred["curves"][k], SHORT[k]
        out[name] = [f"{100 * v:.1f}%" for v in c["model"]]
        out[f"{name}, range"] = [
            f"{100 * lo:.1f}–{100 * hi:.1f}%" for lo, hi in zip(c["low"], c["high"], strict=True)
        ]
    return out


def outlook(pred: dict, reference: dict, evaluated_months: int, height: int = 340) -> alt.Chart:
    data = curve_frame(pred)
    last = int(max(pred["months"]))
    x = alt.X(
        "month:Q",
        title="Months since entering the queue",
        scale=alt.Scale(domain=[0, last], nice=False),
        axis=alt.Axis(values=list(range(0, last + 1, 12)), grid=False),
    )
    y_scale = alt.Scale(domain=[0, 1])
    y_axis = alt.Axis(format="%", tickCount=5, title="Cumulative chance")

    unchecked = (
        alt.Chart(pd.DataFrame({"start": [evaluated_months], "end": [last]}))
        .mark_rect(color=UI["panel"])
        .encode(x="start:Q", x2="end:Q")
    )
    unchecked_label = (
        alt.Chart(pd.DataFrame({"month": [evaluated_months], "y": [0.98]}))
        .mark_text(
            align="left", baseline="top", dx=6, lineBreak="\n", color=UI["muted"], fontSize=BODY
        )
        .encode(
            x="month:Q",
            y=alt.Y("y:Q", scale=y_scale),
            text=alt.value(f"Not checked\nbeyond {evaluated_months // 12} years"),
        )
    )
    band = (
        alt.Chart(data)
        .mark_area(opacity=0.22)
        .encode(
            x=x,
            y=alt.Y("low:Q", scale=y_scale, axis=y_axis),
            y2="high:Q",
            color=alt.Color("outcome:N", scale=COLOR, legend=None),
        )
    )
    lines = (
        alt.Chart(data)
        .mark_line(strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y("model:Q", scale=y_scale),
            color=alt.Color("outcome:N", scale=COLOR, legend=None),
        )
    )
    ends = data[data["month"] == last]
    labels = (
        alt.Chart(ends)
        .mark_text(align="right", dx=-4, dy=-9)
        .encode(x="month:Q", y=alt.Y("model:Q", scale=y_scale), text="outcome:N")
    )
    ref = pd.DataFrame(
        [
            {
                "month": h,
                "observed": r["observed"],
                "what": f"Observed for all {r['cohort']} entrants (n = {r['n']:,})",
                "value": f"{100 * r['observed']:.1f}% {label} within {h // 12} years",
            }
            for key, label, h in (
                ("withdrawn_3y", "withdrew", 36),
                ("operational_5y", "reached operation", 60),
            )
            for r in [reference[key]]
        ]
    )
    rings = (
        alt.Chart(ref)
        .mark_point(shape="circle", size=70, filled=False, strokeWidth=1.5, color=UI["ink"])
        .encode(
            x="month:Q",
            y=alt.Y("observed:Q", scale=y_scale),
            tooltip=[alt.Tooltip("what:N", title="Ring"), alt.Tooltip("value:N", title="Rate")],
        )
    )
    hover = hover_frame(pred)
    nearest = alt.selection_point(
        nearest=True, on="pointerover", fields=["month"], empty=False, clear="pointerout"
    )
    tooltip = [alt.Tooltip("month:Q", title="Month")] + [
        alt.Tooltip(f"{c}:N") for c in hover.columns[1:]
    ]
    catcher = (
        alt.Chart(hover)
        .mark_rule(opacity=0, strokeWidth=8)
        .encode(x="month:Q", tooltip=tooltip)
        .add_params(nearest)
    )
    rule = (
        alt.Chart(hover)
        .mark_rule(color=UI["muted"], strokeWidth=1)
        .encode(x="month:Q")
        .transform_filter(nearest)
    )
    chart = alt.layer(
        unchecked, unchecked_label, band, lines, labels, rule, catcher, rings
    ).properties(height=height, width="container")
    return theme(chart)


def compare(preds: list[dict], key: str, outcome: str, title: str) -> alt.Chart:
    """Dot (model) and line (past-error range) per project for one headline number."""
    rows = []
    for p in preds:
        h = next(h for h in p["headline"] if h["key"] == key)
        rows.append(
            {
                "project": p["project"]["name"],
                "model": h["model"],
                "low": h["low"],
                "high": h["high"],
                "Model": f"{100 * h['model']:.1f}%",
                "Past-error range": f"{100 * h['low']:.1f}–{100 * h['high']:.1f}%",
            }
        )
    df = pd.DataFrame(rows)
    color = OUTCOME_COLORS[outcome]
    y = alt.Y("project:N", title=None, sort=None, axis=alt.Axis(labelLimit=160, grid=False))
    hi = max(0.05, float(df[["model", "high"]].max().max()) * 1.15)
    x = alt.X(
        "low:Q",
        title=title,
        scale=alt.Scale(domain=[0, min(1, hi)]),
        axis=alt.Axis(format="%", tickCount=4),
    )
    tip = [alt.Tooltip("project:N", title="Project"), "Model:N", "Past-error range:N"]
    rng = (
        alt.Chart(df)
        .mark_rule(color=color, strokeWidth=2, opacity=0.45)
        .encode(y=y, x=x, x2="high:Q", tooltip=tip)
    )
    dot = (
        alt.Chart(df)
        .mark_point(filled=True, size=90, color=color, opacity=1)
        .encode(y=y, x="model:Q", tooltip=tip)
    )
    return theme(alt.layer(rng, dot).properties(height=44 * len(df) + 30, width="container"))


def nice_step(extent: float) -> float:
    """Tick step for a symmetric axis [-2*step, 2*step] that covers +/- extent."""
    for s in (0.5, 1, 2.5, 5, 10, 25, 50):
        if 2 * s >= extent:
            return s
    return 50.0


def contributions(target: dict) -> alt.Chart:
    """Horizontal bars: percentage points each feature group adds to or takes from the
    prediction, relative to the model's average for recent entrants."""
    df = pd.DataFrame(target["contributions"])
    df["group"] = df["group"].str.capitalize()
    df["direction"] = df["points"].map(lambda v: "up" if v > 0 else "down")
    df["label"] = df["points"].map(lambda v: "0.0" if abs(v) < 0.05 else f"{v:+.1f}")
    df["Points"] = df["label"] + " points"
    step = nice_step(float(df["points"].abs().max()) * 1.3)
    lim = 2 * step
    y = alt.Y("group:N", sort=None, title=None, axis=alt.Axis(grid=False, labelLimit=140))
    x = alt.X(
        "points:Q",
        title="Percentage points",
        scale=alt.Scale(domain=[-lim, lim], nice=False),
        axis=alt.Axis(values=[-lim, -step, 0, step, lim]),
    )
    color = alt.Color(
        "direction:N",
        scale=alt.Scale(
            domain=["up", "down"], range=[OUTCOME_COLORS[target["outcome"]], NEUTRAL_SERIES]
        ),
        legend=None,
    )
    tip = [
        alt.Tooltip("group:N", title="Group"),
        alt.Tooltip("value:N", title="This project"),
        "Points:N",
    ]
    bars = (
        alt.Chart(df).mark_bar(height=14, cornerRadius=2).encode(y=y, x=x, color=color, tooltip=tip)
    )
    text = (
        alt.Chart(df)
        .mark_text(
            dx=alt.ExprRef("datum.points >= 0 ? 6 : -6"),
            align=alt.ExprRef("datum.points >= 0 ? 'left' : 'right'"),
        )
        .encode(y=y, x="points:Q", text="label:N")
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=UI["muted"]).encode(x="x:Q")
    return theme(alt.layer(bars, zero, text).properties(height=34 * len(df), width="container"))
