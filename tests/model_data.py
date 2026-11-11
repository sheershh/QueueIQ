"""Synthetic entries + features tables shaped like the real processed data, for model tests."""

import numpy as np
import pandas as pd

from queueiq.models.data import MODEL_FEATURES

CUTOFF = pd.Timestamp("2025-12-31")


def synthetic(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    q = pd.Timestamp("2010-01-01") + pd.to_timedelta(rng.integers(0, 365 * 11 - 40, n), unit="D")
    fuel = rng.choice(["solar", "wind", "battery", "gas", "other"], n)
    p_op = np.where(fuel == "gas", 0.4, 0.1)  # signal: gas is built more often
    u = rng.random(n)
    outcome = np.where(u < p_op, "operational", np.where(u < 0.85, "withdrawn", "censored"))
    months = rng.uniform(1, 90, n)
    end = q + pd.to_timedelta(months * 30.4375, unit="D")
    end = pd.Series(end).where(outcome != "censored", CUTOFF)
    outcome = np.where((outcome != "censored") & (end > CUTOFF), "censored", outcome)
    end = end.where(outcome != "censored", CUTOFF)
    entries = pd.DataFrame(
        {
            "project_id": [f"P{i}" for i in range(n)],
            "in_model_scope": True,
            "queue_date": q,
            "outcome": pd.array(outcome, dtype="string"),
            "end_date": end.to_numpy(),
            "time_months": (end.to_numpy() - q).days / 30.4375,
            "state": pd.array(rng.choice(["VA", "TX"], n), dtype="string"),
            "county": pd.array([None] * n, dtype="string"),
            "fips_code": pd.array([f"0000{i}" for i in rng.integers(1, 6, n)], dtype="string"),
        }
    )
    entries["queue_year"] = entries["queue_date"].dt.year
    entries["status"] = entries["outcome"].replace({"censored": "active"})
    entries["withdrawal_date"] = entries["end_date"].where(entries["outcome"] == "withdrawn")
    entries["scope_exclusion_reason"] = pd.array([None] * n, dtype="string")
    features = pd.DataFrame({"project_id": entries["project_id"]})
    for col in MODEL_FEATURES:
        features[col] = rng.uniform(0, 50, n)
    features["log_capacity_mw"] = rng.uniform(1, 6, n)
    features["queue_year"] = pd.array(q.year, dtype="Int64")
    features["fuel_group"] = pd.array(fuel, dtype="string")
    features["iso"] = pd.array(rng.choice(["PJM", "MISO", "CAISO", "SPP"], n), dtype="string")
    features["state"] = pd.array(
        rng.choice(["VA", "TX", "CA", "ZZ"], n, p=[0.4, 0.3, 0.29, 0.01]), dtype="string"
    )
    for col in ("is_hybrid", "has_storage"):
        features[col] = rng.random(n) < 0.2
    features["county_known"] = rng.random(n) < 0.95
    for col in [c for c in MODEL_FEATURES if c.startswith("county_prior")]:
        features.loc[~features["county_known"], col] = np.nan
    features["county_prior_withdrawn_share_5y"] = features["county_prior_withdrawn_share_5y"] / 50
    features["state_prior_withdrawn_share_5y"] = features["state_prior_withdrawn_share_5y"] / 50
    return entries, features
