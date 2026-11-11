"""Map the raw LBNL sheet onto QueueIQ's unified schema.

This step renames and normalizes only. It removes no rows and makes no judgment
about data quality (that is flags.py and scope.py).
"""

import pandas as pd

STORAGE_TYPES = {"Battery", "Other Storage"}


def _snake(s: pd.Series) -> pd.Series:
    """'Offshore Wind' -> 'offshore_wind'; missing stays missing."""
    return s.str.strip().str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")


def make_project_ids(entity: pd.Series, q_id: pd.Series) -> pd.Series:
    """Unique ID per row: '<entity>::<q_id>', with '#2', '#3', ... added to reused IDs.

    LBNL says q_id is only unique within a transmission provider ("entity"). Some
    (entity, q_id) pairs still repeat; those rows keep their order and get a suffix
    so every row has a unique key. They are flagged separately (flag_duplicate_queue_id).
    """
    base = entity.astype("string").fillna("NA") + "::" + q_id.astype("string").fillna("NA")
    occurrence = base.groupby(base).cumcount() + 1
    suffix = occurrence.map(lambda k: "" if k == 1 else f"#{k}")
    return base + suffix


def to_unified(raw: pd.DataFrame) -> pd.DataFrame:
    types = raw[["type_1", "type_2", "type_3"]]
    fips = pd.to_numeric(raw["fips_code"], errors="coerce").astype("Int64")

    out = pd.DataFrame(
        {
            "project_id": make_project_ids(raw["entity"], raw["q_id"]),
            "queue_id": raw["q_id"].astype("string"),
            "entity": raw["entity"].astype("string"),
            "iso": raw["region"].astype("string"),
            "state": raw["state"].astype("string").str.strip().str.upper(),
            "county": raw["county"].astype("string").str.strip(),
            "fips_code": fips.astype("string").str.zfill(5),
            "poi_name": raw["poi_name"].astype("string"),
            "utility": raw["utility"].astype("string"),
            "developer": raw["developer"].astype("string"),
            "project_type": _snake(raw["project_type"].astype("string")),
            "service": raw["service"].astype("string"),
            "fuel_type": _snake(raw["type_1"].astype("string")),
            "fuel_type_detail": raw["type_clean"].astype("string"),
            "is_hybrid": raw["type_2"].notna(),
            "has_storage": types.isin(STORAGE_TYPES).any(axis=1),
            "capacity_mw": pd.to_numeric(raw["mw_1"], errors="coerce"),
            "queue_date": pd.to_datetime(raw["q_date"]),
            "status": _snake(raw["q_status"].astype("string")),
            "operation_date": pd.to_datetime(raw["on_date"]),
            "withdrawal_date": pd.to_datetime(raw["wd_date"]),
            # Post-entry information: kept for EDA only, never usable as model features.
            "proposed_online_date": pd.to_datetime(raw["prop_date"]),
            "ia_date": pd.to_datetime(raw["ia_date"]),
            "ia_phase": raw["IA_phase_clean"].astype("string"),
        }
    )
    out["queue_year"] = out["queue_date"].dt.year.astype("Int64")
    return out
