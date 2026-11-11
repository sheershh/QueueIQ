"""Non-parametric survival estimates with two competing events.

aalen_johansen() answers "what share of projects had reached operation (or withdrawn)
within t months of entering the queue?" while handling projects that are still waiting
(censored). At each time an event happens:

    S(t)     = S(t-) * (1 - d / n)            Kaplan-Meier: share still in the queue
    CIF_k(t) = CIF_k(t-) + S(t-) * d_k / n    Aalen-Johansen: cumulative share ending in k

n = projects still at risk (time >= t), d_k = events of type k at t, d = all events at t.
A project censored at t still counts as at risk at t (it was observed up to t).
At every t: S(t) + CIF_operational(t) + CIF_withdrawn(t) = 1.

Why not just "share of projects that withdrew"? Recent projects have not had time to
finish, so a raw share understates outcomes; censoring-aware estimates use each project
only for as long as it was actually observed.
"""

import numpy as np
import pandas as pd

EVENTS = ("operational", "withdrawn")


def aalen_johansen(time: pd.Series, outcome: pd.Series) -> pd.DataFrame:
    """Return one row per distinct time: n_at_risk, events, S, and one CIF per event type.

    time: months from entry to event or censoring (no missing values).
    outcome: 'operational', 'withdrawn', or anything else (= censored).
    """
    t = np.asarray(time, dtype=float)
    o = np.asarray(outcome, dtype=object)
    if np.isnan(t).any():
        raise ValueError("time contains missing values; filter them out first")
    if (t < 0).any():
        raise ValueError("time must be non-negative")

    times = np.unique(t)
    n_at_risk = np.array([(t >= u).sum() for u in times])
    d = {k: np.array([((t == u) & (o == k)).sum() for u in times]) for k in EVENTS}
    d_all = sum(d.values())

    surv_after = np.cumprod(1 - d_all / n_at_risk)
    surv_before = np.concatenate([[1.0], surv_after[:-1]])
    cif = {k: np.cumsum(surv_before * d[k] / n_at_risk) for k in EVENTS}

    return pd.DataFrame(
        {
            "time": times,
            "n_at_risk": n_at_risk,
            "events": d_all,
            "still_in_queue": surv_after,
            **{f"cif_{k}": cif[k] for k in EVENTS},
        }
    )


def value_at(curve: pd.DataFrame, column: str, t: float) -> float:
    """Step-function lookup: the estimate at time t (last change at or before t)."""
    before = curve[curve["time"] <= t]
    if before.empty:
        return 0.0 if column.startswith("cif_") else 1.0
    return float(before[column].iloc[-1])


def at_risk_at(curve: pd.DataFrame, t: float) -> int:
    """Projects still under observation (not yet ended or censored) just before time t."""
    later = curve[curve["time"] >= t]
    return int(later["n_at_risk"].iloc[0]) if not later.empty else 0
