"""Features for a hypothetical project, computed by the same code as the training features.

The project is added as one extra row to the real queue entries of its state, and
`build_features` runs on that small table. So "what was already in line" is counted exactly as
it was for every training row: earlier entries only, 5-year look-back, withdrawals only if dated
before entry. No separate re-implementation can drift from the training pipeline.
"""

import pandas as pd

from queueiq.features.build import (
    CONGESTION_ONLY_COLUMNS,
    ENTRY_SAFE_COLUMNS,
    LOOKBACK,
    build_features,
)
from queueiq.features.congestion import county_key
from queueiq.models.data import MODEL_FEATURES
from queueiq.service.inputs import Options, ProjectInput

NEW_ID = "__new_project__"
HISTORY_COLUMNS = ENTRY_SAFE_COLUMNS + CONGESTION_ONLY_COLUMNS


def history(entries: pd.DataFrame) -> pd.DataFrame:
    """The only columns the service keeps from the queue data (+ the county key, for lookup)."""
    hist = entries[HISTORY_COLUMNS + ["in_model_scope"]].copy()
    hist["county_key"] = county_key(hist)
    return hist


def project_row(p: ProjectInput, options: Options, entry_date: pd.Timestamp) -> pd.DataFrame:
    has_storage, is_hybrid = p.flags()
    county, _ = options.resolve_county(p)
    return pd.DataFrame(
        {
            "project_id": [NEW_ID],
            "iso": pd.array([p.iso], dtype="string"),
            "state": pd.array([p.state], dtype="string"),
            "county": pd.array([county.county if county else None], dtype="string"),
            "fips_code": pd.array([county.fips_code if county else None], dtype="string"),
            "fuel_type": pd.array([p.fuel], dtype="string"),
            "is_hybrid": [is_hybrid],
            "has_storage": [has_storage],
            "capacity_mw": [float(p.capacity_mw)],
            "queue_date": [entry_date],
            "queue_year": [entry_date.year],
            "withdrawal_date": [pd.NaT],
        }
    )


def project_features(
    p: ProjectInput, hist: pd.DataFrame, options: Options, entry_date: pd.Timestamp
) -> pd.DataFrame:
    """One row of MODEL_FEATURES for project p entering on entry_date."""
    # Same state, or same county key: a few LBNL rows share a FIPS code but list another state,
    # and the training features count them as county neighbours.
    county, _ = options.resolve_county(p)
    nearby = (hist["state"] == p.state) | (county is not None and hist["county_key"] == county.key)
    window = hist[
        nearby & (hist["queue_date"] >= entry_date - LOOKBACK) & (hist["queue_date"] < entry_date)
    ]
    row = project_row(p, options, entry_date)
    table = pd.concat([window[HISTORY_COLUMNS], row], ignore_index=True)
    table["queue_year"] = table["queue_year"].astype("Int64")
    feats = build_features(table)
    return feats[feats["project_id"] == NEW_ID][MODEL_FEATURES].reset_index(drop=True)
