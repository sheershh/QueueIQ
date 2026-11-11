"""Project characteristics known on the day of entry."""

import numpy as np
import pandas as pd

MAIN_FUELS = ("solar", "wind", "battery", "gas")


def fuel_group(fuel_type: pd.Series) -> pd.Series:
    """Main fuels kept as-is; everything else (hydro, nuclear, offshore wind, ...) -> 'other'."""
    return fuel_type.where(fuel_type.isin(MAIN_FUELS), "other").astype("string")


def project_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "capacity_mw": df["capacity_mw"],
            "log_capacity_mw": np.log1p(df["capacity_mw"].clip(lower=0)),
            "fuel_group": fuel_group(df["fuel_type"]),
            "is_hybrid": df["is_hybrid"],
            "has_storage": df["has_storage"],
            "iso": df["iso"],
            "state": df["state"],
            "queue_year": df["queue_year"],
        },
        index=df.index,
    )
