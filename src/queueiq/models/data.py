"""Time-based train/test split for survival models (decisions D-027).

Training uses projects that entered before the origin date, with outcomes cut off at the
origin: only what was known on that date. A 2014 project that withdrew in 2021 is, for
training purposes, still in the queue on 2018-01-01. The test set is a later block of entry
years whose outcomes are observed up to the data cutoff.

Event codes: 0 = censored, 1 = operational, 2 = withdrawn.
Time is a month index: month 1 covers [0, 1) months after entry, month 2 covers [1, 2), etc.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from queueiq.eda.survival import EVENTS
from queueiq.features.build import FEATURES

EVENT_CODE = {"censored": 0, "operational": 1, "withdrawn": 2}
CAUSES = {1: "operational", 2: "withdrawn"}
assert tuple(CAUSES.values()) == EVENTS
DAYS_PER_MONTH = 365.25 / 12

# capacity_mw is dropped in favour of its log (same information, better scaled).
MODEL_FEATURES = [f for f in FEATURES if f != "capacity_mw"]


@dataclass
class Split:
    X_train: pd.DataFrame
    time_train: np.ndarray
    event_train: np.ndarray
    X_test: pd.DataFrame
    time_test: np.ndarray  # observed to the data cutoff
    event_test: np.ndarray
    origin: pd.Timestamp
    test_years: tuple[int, int]
    test_dropped_short_followup: int = 0


def month_index(months: pd.Series) -> np.ndarray:
    """Continuous months -> integer month index (an event at 0.4 months is in month 1)."""
    return (np.floor(months.to_numpy(dtype=float)) + 1).astype(int)


def outcome_as_of(entries: pd.DataFrame, as_of: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """(month index, event code) using only what was known on `as_of`.

    Events dated on or after `as_of` had not happened yet: censored at `as_of`.
    """
    is_event = entries["outcome"].isin(CAUSES.values()) & (entries["end_date"] < as_of)
    end = entries["end_date"].where(is_event, as_of)
    months = (end - entries["queue_date"]).dt.days / DAYS_PER_MONTH
    event = entries["outcome"].map(EVENT_CODE).where(is_event, 0).astype(int).to_numpy()
    return month_index(months), event


def make_split(
    entries: pd.DataFrame,
    features: pd.DataFrame,
    origin: pd.Timestamp,
    test_years: tuple[int, int],
    test_as_of: pd.Timestamp | None = None,
    min_test_followup_months: int | None = None,
) -> Split:
    """Rows must be in model scope; entries and features are aligned on project_id.

    test_as_of: if given, test outcomes are also cut off at that date (used for the inner
    validation split, so it never sees outcomes after the outer origin).
    min_test_followup_months: drop test rows censored before this many months, so every
    remaining test project has a known outcome at the horizon. The count is recorded.
    """
    outcome_cols = [
        "project_id",
        "in_model_scope",
        "queue_date",
        "outcome",
        "end_date",
        "time_months",
    ]
    df = entries[outcome_cols].merge(
        features[["project_id", *MODEL_FEATURES]], on="project_id", how="inner"
    )
    df = df[df["in_model_scope"]]
    train = df[df["queue_date"] < origin]
    test = df[df["queue_year"].between(*test_years)]
    if not test.empty and test["queue_date"].min() < origin:
        raise ValueError("test cohorts must start on or after the origin date")

    t_train, e_train = outcome_as_of(train, origin)
    if test_as_of is None:
        t_test = month_index(test["time_months"])
        e_test = test["outcome"].map(EVENT_CODE).astype(int).to_numpy()
    else:
        t_test, e_test = outcome_as_of(test, test_as_of)
    dropped = 0
    if min_test_followup_months is not None:
        keep = (e_test != 0) | (t_test > min_test_followup_months)
        dropped = int((~keep).sum())
        test, t_test, e_test = test[keep], t_test[keep], e_test[keep]
    return Split(
        X_train=train[MODEL_FEATURES].reset_index(drop=True),
        time_train=t_train,
        event_train=e_train,
        X_test=test[MODEL_FEATURES].reset_index(drop=True),
        time_test=t_test,
        event_test=e_test,
        origin=origin,
        test_years=test_years,
        test_dropped_short_followup=dropped,
    )
