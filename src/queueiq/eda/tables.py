"""Summary tables for the EDA report. All inputs come from queue_entries.parquet."""

import numpy as np
import pandas as pd

from queueiq.eda.survival import aalen_johansen, value_at

HORIZONS_MONTHS = (36, 60)  # 3 and 5 years
# Curves stop once fewer than this many projects are still observed (D-019); horizon
# estimates need at least this many projects with complete follow-up (D-022).
MIN_AT_RISK = 30
ERAS = {"2010-2014": (2010, 2014), "2015-2019": (2015, 2019), "2020-2025": (2020, 2025)}
MAIN_FUELS = ("solar", "wind", "battery", "gas")


def in_scope(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["in_model_scope"]].copy()


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Entry era and a coarse fuel group (main fuels, everything else as 'other')."""
    out = df.copy()
    era = pd.Series(pd.NA, index=out.index, dtype="string")
    for name, (lo, hi) in ERAS.items():
        era[out["queue_year"].between(lo, hi).fillna(False)] = name
    out["era"] = era
    out["fuel_group"] = out["fuel_type"].where(out["fuel_type"].isin(MAIN_FUELS), "other")
    return out


def outcome_shares(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Raw status as of the cutoff, as % of requests (NOT censoring-aware)."""
    known = df[df["outcome"] != "unknown"]
    label = known["outcome"].replace({"censored": "in_queue"})
    table = pd.crosstab(known[by], label, normalize="index").mul(100).round(1)
    table = table.reindex(columns=["operational", "withdrawn", "in_queue"], fill_value=0.0)
    table.insert(0, "requests", known.groupby(by).size())
    return table


def cohort_table(scoped: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Per entry year: size, raw outcome shares, and how much follow-up the cohort can have."""
    table = outcome_shares(scoped, "queue_year")
    start = pd.to_datetime(table.index.astype(str) + "-12-31")
    table["max_followup_years"] = ((cutoff - start).days / 365.25 + 1).round(1)
    table["min_followup_years"] = ((cutoff - start).days / 365.25).round(1)
    table.index.name = "entry_year"
    return table


def curves_by(scoped: pd.DataFrame, by: str | None) -> dict[str, pd.DataFrame]:
    """Aalen-Johansen curves for the whole scope (by=None) or per group."""
    if by is None:
        return {"All": aalen_johansen(scoped["time_months"], scoped["outcome"])}
    return {
        str(name): aalen_johansen(g["time_months"], g["outcome"])
        for name, g in scoped.groupby(by, sort=True)
    }


def cif_summary(scoped: pd.DataFrame, by: str | None, cutoff: pd.Timestamp) -> pd.DataFrame:
    """% reaching operation / withdrawn within 3 and 5 years, using complete follow-up only.

    For horizon h, only projects that entered at least h months before the cutoff are used,
    so every one of them was observed for the full h months (D-022). Pooling cohorts with
    different follow-up under a strong time trend mixes them: the early part of a pooled
    curve comes from recent cohorts and its tail only from old ones. With complete
    follow-up there is no censoring before h, so the Aalen-Johansen value equals the plain
    share with that event by h.

    Blank where fewer than MIN_AT_RISK projects in the group qualify.
    """
    rows: dict[str, dict] = {}
    groups = scoped.groupby(by, sort=True) if by else [("All", scoped)]
    for name, g in groups:
        row: dict = {"requests": len(g)}
        for h in HORIZONS_MONTHS:
            yrs = h // 12
            eligible = g[g["queue_date"] <= cutoff - pd.DateOffset(months=h)]
            row[f"full_{yrs}y_followup"] = len(eligible)
            enough = len(eligible) >= MIN_AT_RISK
            curve = aalen_johansen(eligible["time_months"], eligible["outcome"]) if enough else None
            for k in ("operational", "withdrawn"):
                row[f"{k}_by_{yrs}y_%"] = (
                    round(100 * value_at(curve, f"cif_{k}", h), 1) if enough else np.nan
                )
        rows[str(name)] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = by or "scope"
    return table


def observed_durations(scoped: pd.DataFrame, by: str) -> pd.DataFrame:
    """Median months to the event AMONG projects that had it (conditional, biased short)."""
    events = scoped[scoped["outcome"].isin(["operational", "withdrawn"])]
    g = events.groupby([by, "outcome"])["time_months"]
    table = pd.DataFrame({"n": g.size(), "median_months": g.median().round(1)}).unstack("outcome")
    table.columns = [f"{outcome}_{stat}" for stat, outcome in table.columns]
    order = ["operational_n", "operational_median_months", "withdrawn_n", "withdrawn_median_months"]
    return table.reindex(columns=order)


def capacity_by_fuel(scoped: pd.DataFrame) -> pd.DataFrame:
    g = scoped.groupby("fuel_group")["capacity_mw"]
    return pd.DataFrame(
        {
            "requests": g.size(),
            "median_mw": g.median().round(1),
            "p25_mw": g.quantile(0.25).round(1),
            "p75_mw": g.quantile(0.75).round(1),
            "share_with_storage_%": scoped.groupby("fuel_group")["has_storage"]
            .mean()
            .mul(100)
            .round(1),
        }
    )


def lbnl_crosscheck(df: pd.DataFrame, raw: pd.DataFrame) -> dict[str, float]:
    """Recompute two LBNL headline figures on ALL regions, to check our reading of the file.

    LBNL (emp.lbl.gov/queues, 2026 edition): of capacity requesting interconnection 2000-2020,
    13% operational, 75% withdrawn, 10% active by end of 2025; and median request-to-operation
    time over 5 years for projects that came online in 2025 (regions with data).
    """
    total_mw = raw[["mw_1", "mw_2", "mw_3"]].fillna(0).sum(axis=1).to_numpy()
    cohort = df["queue_year"].between(2000, 2020).fillna(False).to_numpy()
    status = df["status"].to_numpy()
    mw = total_mw[cohort]
    st = status[cohort]
    share = {s: 100 * mw[st == s].sum() / mw.sum() for s in ["operational", "withdrawn", "active"]}

    online_2025 = (df["operation_date"].dt.year == 2025) & df["queue_date"].notna()
    months = (df.loc[online_2025, "operation_date"] - df.loc[online_2025, "queue_date"]).dt.days
    return {
        "mw_share_operational_%": round(share["operational"], 1),
        "mw_share_withdrawn_%": round(share["withdrawn"], 1),
        "mw_share_active_%": round(share["active"], 1),
        "online_2025_n": int(online_2025.sum()),
        "online_2025_median_years": round(float(months.median()) / 365.25, 2),
    }
