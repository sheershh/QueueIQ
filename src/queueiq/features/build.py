"""`make features`: data/processed/features.parquet + reports/features.md.

Features are computed for every row (so earlier entries from any region count as
congestion), and `in_model_scope` is carried over so models can filter.
"""

import pandas as pd

from queueiq import config
from queueiq.cleaning.report import to_markdown
from queueiq.features.congestion import county_key, prior_counts
from queueiq.features.tabular import project_features

PROCESSED_FILE = config.PROCESSED_DIR / "queue_entries.parquet"
FEATURES_FILE = config.PROCESSED_DIR / "features.parquet"
REPORT_FILE = config.REPORTS_DIR / "features.md"

LOOKBACK = pd.DateOffset(years=5)
MOMENTUM = pd.DateOffset(years=1)

# Columns feature code may read. Everything here is known at entry, except withdrawal_date,
# which congestion uses only as "withdrawal_date < this project's entry date".
ENTRY_SAFE_COLUMNS = [
    "project_id",
    "iso",
    "state",
    "county",
    "fips_code",
    "fuel_type",
    "is_hybrid",
    "has_storage",
    "capacity_mw",
    "queue_date",
    "queue_year",
]
CONGESTION_ONLY_COLUMNS = ["withdrawal_date"]

FEATURES = {
    "capacity_mw": "Capacity of the primary resource (MW)",
    "log_capacity_mw": "log(1 + capacity_mw)",
    "fuel_group": "Primary fuel: solar, wind, battery, gas, or other",
    "is_hybrid": "More than one resource type listed",
    "has_storage": "Any listed resource is storage",
    "iso": "Grid operator (PJM, MISO, CAISO, SPP in model scope)",
    "state": "State (raw; any grouping of rare states is fitted on training data in Phase 4)",
    "queue_year": "Entry year (numeric)",
    "county_known": "County could be identified (else county features are missing)",
    "county_prior_requests_5y": "Requests in the same county in the 5 years before entry",
    "county_prior_mw_5y": "Their total MW",
    "county_prior_withdrawn_5y": "Of those, withdrawn with a known date before entry",
    "county_prior_withdrawn_share_5y": (
        "county_prior_withdrawn_5y / county_prior_requests_5y (missing if no earlier requests)"
    ),
    "state_prior_requests_5y": "Requests in the same state in the 5 years before entry",
    "state_prior_mw_5y": "Their total MW",
    "state_prior_withdrawn_5y": "Of those, withdrawn with a known date before entry",
    "state_prior_withdrawn_share_5y": (
        "state_prior_withdrawn_5y / state_prior_requests_5y (missing if no earlier requests)"
    ),
    "state_prior_requests_1y": "Requests in the same state in the 12 months before entry",
}


def _congestion(df: pd.DataFrame, key: pd.Series, prefix: str, lookback, tag: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "key": key,
            "queue_date": df["queue_date"],
            "withdrawal_date": df["withdrawal_date"],
            "capacity_mw": df["capacity_mw"],
        }
    )
    c = prior_counts(frame, lookback)
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_prior_requests_{tag}"] = c["requests"]
    if tag == "5y":
        out[f"{prefix}_prior_mw_{tag}"] = c["mw"]
        out[f"{prefix}_prior_withdrawn_{tag}"] = c["withdrawn"]
        share = c["withdrawn"] / c["requests"]
        out[f"{prefix}_prior_withdrawn_share_{tag}"] = share.where(c["requests"] > 0)
    return out


def build_features(entries: pd.DataFrame) -> pd.DataFrame:
    """Feature table for every row of `entries`. Reads only entry-safe columns."""
    df = entries[ENTRY_SAFE_COLUMNS + CONGESTION_ONLY_COLUMNS]
    ckey = county_key(df)
    feats = pd.concat(
        [
            df[["project_id"]],
            project_features(df),
            pd.DataFrame({"county_known": ckey.notna()}, index=df.index),
            _congestion(df, ckey, "county", LOOKBACK, "5y"),
            _congestion(df, df["state"], "state", LOOKBACK, "5y"),
            _congestion(df, df["state"], "state", MOMENTUM, "1y"),
        ],
        axis=1,
    )
    assert list(feats.columns[1:]) == list(FEATURES), "FEATURES registry out of sync"
    return feats


def undated_withdrawal_share(entries: pd.DataFrame, scope: pd.Series) -> float:
    """DIAGNOSTIC ONLY (reads status): % of in-scope projects' 5-year county priors that are
    withdrawn projects with no withdrawal date, i.e. withdrawals we can never count."""
    ckey = county_key(entries)
    focal = entries.loc[scope & ckey.notna(), ["queue_date"]].assign(key=ckey)
    pool = entries.loc[ckey.notna() & entries["queue_date"].notna()].assign(key=ckey)
    pool["undated"] = (pool["status"] == "withdrawn") & pool["withdrawal_date"].isna()
    m = focal.reset_index().merge(
        pool[["key", "queue_date", "undated"]], on="key", suffixes=("", "_p")
    )
    m = m[(m["queue_date_p"] < m["queue_date"]) & (m["queue_date_p"] >= m["queue_date"] - LOOKBACK)]
    return round(100 * m["undated"].mean(), 2)


def summary_table(feats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, desc in FEATURES.items():
        col = feats[name]
        row = {"feature": name, "meaning": desc, "missing_%": round(100 * col.isna().mean(), 1)}
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            q = col.astype(float).quantile([0.1, 0.5, 0.9])
            row.update(
                {"p10": round(q[0.1], 2), "median": round(q[0.5], 2), "p90": round(q[0.9], 2)}
            )
        else:
            top = col.astype("string").value_counts(normalize=True).head(3)
            row["top values"] = ", ".join(f"{k} {100 * v:.0f}%" for k, v in top.items())
        rows.append(row)
    return pd.DataFrame(rows)


def range_coverage(scoped: pd.DataFrame, split_year: int = 2020) -> pd.DataFrame:
    """% of later-cohort rows whose value lies inside the range seen in earlier cohorts.

    A model trained on earlier cohorts has never seen values outside that range: trees
    treat them like the edge value, linear models extrapolate the trend.
    """
    early = scoped[scoped["queue_year"] < split_year]
    late = scoped[scoped["queue_year"] >= split_year]
    rows = []
    for name in FEATURES:
        col = scoped[name]
        if not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            continue
        lo, hi = early[name].min(), early[name].max()
        known = late[name].dropna()
        inside = known.between(lo, hi).mean() if len(known) else float("nan")
        rows.append(
            {
                "feature": name,
                "earlier_min": round(lo, 2),
                "earlier_max": round(hi, 2),
                "later_rows_inside_range_%": round(100 * inside, 1),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    entries = pd.read_parquet(PROCESSED_FILE)
    feats = build_features(entries)
    feats["in_model_scope"] = entries["in_model_scope"]
    assert len(feats) == len(entries) and feats["project_id"].is_unique
    feats.to_parquet(FEATURES_FILE, index=False)

    scoped = feats[feats["in_model_scope"]]
    by_year = scoped.groupby("queue_year")[
        [
            "state_prior_requests_1y",
            "state_prior_requests_5y",
            "county_prior_requests_5y",
            "county_prior_withdrawn_share_5y",
        ]
    ].median()
    county_missing = scoped.groupby("iso")["county_known"].apply(
        lambda s: round(100 * (~s).mean(), 1)
    )

    md = f"""# Features (generated by `make features`, do not edit by hand)

Feature table: `data/processed/features.parquet`, {len(feats):,} rows (one per queue entry).
Statistics below cover the {len(scoped):,} rows in model scope.

All features use only information available on the project's entry date (decisions D-023 to
D-025). Congestion features count requests from **any** region in the same county/state that
entered **strictly before** this project within the look-back window. Withdrawals count only if
their date is known and strictly before entry.

## Feature summary (model scope)
{to_markdown(summary_table(scoped))}

## Known gaps
- **Undated withdrawals:** {undated_withdrawal_share(entries, entries["in_model_scope"])}% of the
  5-year county look-back entries for in-scope projects are withdrawn projects with no
  withdrawal date. They count as requests but can never count as withdrawn, so
  `*_prior_withdrawn_*` slightly understates true withdrawals. (This diagnostic reads `status`;
  the features do not.)
- **County not identified (% of in-scope rows):**

{to_markdown(county_missing.rename("county_missing_%").to_frame(), index=True)}

## Can later cohorts be described with values seen earlier? (model scope)
With a time-based split, a model trained on earlier entry years must score later ones. For each
numeric feature, this is the % of 2020–2025 entries whose value lies inside the range seen in
2010–2019 entries. `queue_year` is 0% by construction: every test year is new. Congestion
measures mostly stay inside the earlier range, which is why they are kept alongside `queue_year`
as at-entry proxies for the queue boom (D-025).

{to_markdown(range_coverage(scoped))}

## Congestion by entry year (medians, model scope)

{to_markdown(by_year.round(2), index=True)}
"""
    REPORT_FILE.write_text(md)
    print(
        f"Wrote {FEATURES_FILE.relative_to(config.PROJECT_ROOT)} ({len(feats.columns)} columns)"
        f" and {REPORT_FILE.relative_to(config.PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
