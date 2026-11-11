"""A small, real Service built from synthetic queue entries (no real data or saved model needed).

Entries go through the real feature pipeline and a small forest is fitted on them, so the
service, API and dashboard helpers are tested end to end.
"""

import numpy as np
import pandas as pd

from queueiq.explain import deploy
from queueiq.explain.reference import observed_reference
from queueiq.features.build import build_features
from queueiq.models.rsf import CauseSpecificRSF
from queueiq.service.features import history
from queueiq.service.inputs import Options
from queueiq.service.predict import Service

CUTOFF = pd.Timestamp("2025-12-31")
STATES = {"PJM": ("VA", ["51001", "51003", "51005"]), "MISO": ("IL", ["17001", "17003"])}
MULTIPLIERS = [
    {"operational": 0.5, "withdrawn": 0.8},
    {"operational": 1.2, "withdrawn": 0.6},
    {"operational": 0.3, "withdrawn": 0.9},
]


def synthetic_entries(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    iso = rng.choice(list(STATES), n)
    state = np.array([STATES[i][0] for i in iso])
    fips = np.array([rng.choice(STATES[i][1]) for i in iso], dtype=object)
    fips[rng.random(n) < 0.05] = None
    fuel = rng.choice(["solar", "wind", "battery", "gas", "hydro"], n)
    storage = (fuel == "battery") | ((fuel == "solar") & (rng.random(n) < 0.3))
    q = pd.Timestamp("2012-01-01") + pd.to_timedelta(rng.integers(0, 365 * 14, n), unit="D")
    u = rng.random(n)
    outcome = np.where(u < np.where(fuel == "gas", 0.4, 0.1), "operational", "withdrawn")
    outcome = np.where(u > 0.85, "censored", outcome)
    end = pd.Series(q + pd.to_timedelta(rng.uniform(1, 90, n) * 30.4375, unit="D"))
    outcome = np.where(end > CUTOFF, "censored", outcome)
    end = end.where(outcome != "censored", CUTOFF)
    df = pd.DataFrame(
        {
            "project_id": [f"P{i}" for i in range(n)],
            "iso": pd.array(iso, dtype="string"),
            "state": pd.array(state, dtype="string"),
            "county": pd.array(
                [None if f is None else f"County {f}" for f in fips], dtype="string"
            ),
            "fips_code": pd.array(fips, dtype="string"),
            "fuel_type": pd.array(fuel, dtype="string"),
            "is_hybrid": (fuel != "battery") & storage,
            "has_storage": storage,
            "capacity_mw": rng.uniform(1, 500, n).round(1),
            "queue_date": q,
            "outcome": pd.array(outcome, dtype="string"),
            "end_date": end.to_numpy(),
            "in_model_scope": True,
        }
    )
    df["queue_year"] = df["queue_date"].dt.year
    df["time_months"] = (df["end_date"] - df["queue_date"]).dt.days / 30.4375
    df["withdrawal_date"] = df["end_date"].where(df["outcome"] == "withdrawn")
    # One row that shares a VA county FIPS but lists another state (as a few LBNL rows do).
    df.loc[0, ["state", "fips_code", "in_model_scope"]] = ["NC", "51001", False]
    return df


def synthetic_service(seed: int = 0) -> Service:
    entries = synthetic_entries(seed=seed)
    feats = build_features(entries)
    feats["in_model_scope"] = entries["in_model_scope"]
    X, t, ev, _ = deploy.deployment_data(entries, feats)
    model = CauseSpecificRSF(n_estimators=10, min_samples_leaf=10, exclude=("queue_year",))
    model.fit(X, t, ev)
    meta = {
        "primary_configuration": "rsf_cause_specific|qy=0|recal=0",
        "trained_on_rows": len(X),
        "outcomes_as_of": str(CUTOFF.date()),
        "entry_date": "2026-01-01",
        "background_years": [2021, 2025],
        "background_size": 8,
        "observed_reference": observed_reference(entries, CUTOFF),
        "calibration_warning": "Synthetic warning.",
        "past_error_range": {
            "origins": [
                {"block": "A", "origin": 2014 + i, "multiplier": m}
                for i, m in enumerate(MULTIPLIERS)
            ]
        },
    }
    background = X.sample(8, random_state=0).reset_index(drop=True)
    return Service(
        deploy.Deployment(model, background, meta), history(entries), Options.from_entries(entries)
    )
