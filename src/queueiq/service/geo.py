"""Queue crowding per county on the entry date, for the map (decisions D-059).

For each county key: requests that entered in the 5 years before the entry date, and how many of
them had withdrawn (with a known date) before it. These are exactly the county congestion features
a new project entering that county would get (features/congestion.prior_counts), computed for
every county at once.
"""

import json
from pathlib import Path

import pandas as pd

from queueiq.features.build import LOOKBACK


def county_crowding(hist: pd.DataFrame, entry_date: pd.Timestamp) -> pd.DataFrame:
    """One row per county key: requests_5y, withdrawn_5y, withdrawn_share (NaN if no requests)."""
    window = hist[
        hist["county_key"].notna()
        & (hist["queue_date"] >= entry_date - LOOKBACK)
        & (hist["queue_date"] < entry_date)
    ]
    withdrawn = window["withdrawal_date"].notna() & (window["withdrawal_date"] < entry_date)
    out = (
        window.assign(withdrawn=withdrawn)
        .groupby("county_key")
        .agg(requests_5y=("project_id", "size"), withdrawn_5y=("withdrawn", "sum"))
    )
    out["withdrawn_share"] = out["withdrawn_5y"] / out["requests_5y"]
    return out


ASSETS = Path(__file__).resolve().parents[1] / "assets"


def map_counties() -> list[dict]:
    """[{key, name, state}] for every county on the map (covered states), if the shapes exist."""
    path = ASSETS / "counties.geojson"
    if not path.exists():
        return []
    return [f["properties"] for f in json.loads(path.read_text())["features"]]
