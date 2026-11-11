"""Queue congestion at the moment a project enters: what was already in line nearby.

For a project entering on date d, among all LBNL rows (any region) in the same county or
state, look at requests that entered in the look-back window [d - lookback, d):
    requests   - how many there were
    mw         - their total capacity (zero/negative capacity adds 0 MW)
    withdrawn  - how many of them had a withdrawal date strictly before d

Leakage rules (docs/decisions.md D-023, D-024):
- "Earlier" means strictly earlier. Same-day entries (cluster windows) are not counted:
  nobody knows on the day who else files that day.
- A withdrawal counts only if its date is known and strictly before d. A withdrawn project
  with no withdrawal date counts as a request but never as withdrawn.
- Rows without a queue date can't be placed in time and are never counted.
- `status` and outcome columns are never read: they describe the end of 2025.
"""

import numpy as np
import pandas as pd

# The only columns congestion features may read. withdrawal_date is used solely through
# the comparison withdrawal_date < entry date.
INPUT_COLUMNS = ["key", "queue_date", "withdrawal_date", "capacity_mw"]


def _as_days(s: pd.Series, missing: float) -> np.ndarray:
    """Dates as float days since epoch; missing dates become `missing` (+inf / -inf / nan)."""
    out = s.to_numpy(dtype="datetime64[D]").astype("int64").astype(float)
    out[s.isna().to_numpy()] = missing
    return out


def prior_counts(frame: pd.DataFrame, lookback: pd.DateOffset) -> pd.DataFrame:
    """Requests / MW / withdrawn among earlier entries with the same key, per row of `frame`.

    frame needs exactly INPUT_COLUMNS. Rows with a missing key or queue_date get NaN.
    """
    frame = frame[INPUT_COLUMNS]
    out = pd.DataFrame(np.nan, index=frame.index, columns=["requests", "mw", "withdrawn"])
    usable = frame[frame["key"].notna() & frame["queue_date"].notna()]

    for _, g in usable.groupby("key", sort=False):
        entry = _as_days(g["queue_date"], np.nan)
        window_start = _as_days(g["queue_date"] - lookback, np.nan)
        withdrawn_on = _as_days(g["withdrawal_date"], np.inf)  # unknown = never "before"
        mw = g["capacity_mw"].clip(lower=0).fillna(0).to_numpy()

        # Row i = the project being described, column j = a candidate earlier request.
        earlier = (entry[None, :] < entry[:, None]) & (entry[None, :] >= window_start[:, None])
        known_withdrawn = earlier & (withdrawn_on[None, :] < entry[:, None])

        out.loc[g.index, "requests"] = earlier.sum(axis=1)
        out.loc[g.index, "mw"] = earlier @ mw
        out.loc[g.index, "withdrawn"] = known_withdrawn.sum(axis=1)
    return out


def county_key(df: pd.DataFrame) -> pd.Series:
    """5-digit FIPS code; if that is missing, '<STATE>|<county name>'; else missing."""
    by_name = df["state"] + "|" + df["county"].str.lower().str.strip()
    return df["fips_code"].fillna(by_name)
