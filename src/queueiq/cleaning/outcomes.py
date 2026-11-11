"""Turn each queue entry's status and dates into a survival outcome.

Rules (docs/decisions.md D-010):
- operational -> event 'operational' at operation_date
- withdrawn   -> event 'withdrawn' at withdrawal_date
- active, suspended -> 'censored' at the data cutoff (still in the queue then)
- anything else (e.g. 'unknown') -> 'unknown', no time
- an event dated after the cutoff had not happened yet as of the cutoff, so it is
  censored at the cutoff (administrative censoring) and flagged.
- an event with no date keeps its outcome but gets no time (time_months = NaN).
"""

from datetime import date

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 365.25 / 12  # average month length, 30.4375 days

EVENT_DATE_COLUMN = {"operational": "operation_date", "withdrawn": "withdrawal_date"}
CENSORED_STATUSES = {"active", "suspended"}


def assign_outcomes(df: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    cutoff_ts = pd.Timestamp(cutoff)
    out = df.copy()

    outcome = pd.Series("unknown", index=out.index, dtype="string")
    end_date = pd.Series(pd.NaT, index=out.index, dtype="datetime64[us]")

    for status, col in EVENT_DATE_COLUMN.items():
        is_status = out["status"] == status
        outcome[is_status] = status
        end_date[is_status] = out.loc[is_status, col]

    is_censored = out["status"].isin(CENSORED_STATUSES)
    outcome[is_censored] = "censored"
    end_date[is_censored] = cutoff_ts

    after_cutoff = outcome.isin(EVENT_DATE_COLUMN.keys()) & (end_date > cutoff_ts)
    outcome[after_cutoff] = "censored"
    end_date[after_cutoff] = cutoff_ts

    out["outcome"] = outcome
    out["end_date"] = end_date
    out["censored_at_cutoff_after_event"] = after_cutoff
    days = (out["end_date"] - out["queue_date"]).dt.days
    out["time_months"] = (days / DAYS_PER_MONTH).astype("float64").where(days.notna(), np.nan)
    return out
