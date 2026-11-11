import numpy as np
import pandas as pd

from queueiq.features.build import range_coverage
from queueiq.features.congestion import county_key
from queueiq.features.tabular import fuel_group, project_features


def test_county_key_prefers_fips_then_state_and_name():
    df = pd.DataFrame(
        {
            "fips_code": pd.array(["01001", None, None], dtype="string"),
            "state": pd.array(["AL", "TX", None], dtype="string"),
            "county": pd.array(["Autauga", " Harris ", "Somewhere"], dtype="string"),
        }
    )
    assert county_key(df).tolist()[:2] == ["01001", "TX|harris"]
    assert pd.isna(county_key(df).iloc[2])  # no FIPS and no state: unknown


def test_fuel_group_folds_rare_fuels():
    s = pd.Series(["solar", "hydro", "gas", "offshore_wind"], dtype="string")
    assert fuel_group(s).tolist() == ["solar", "other", "gas", "other"]


def test_log_capacity_handles_zero():
    df = pd.DataFrame(
        {
            "capacity_mw": [0.0, np.e - 1],
            "fuel_type": "solar",
            "is_hybrid": False,
            "has_storage": False,
            "iso": "PJM",
            "state": "VA",
            "queue_year": 2015,
        }
    )
    out = project_features(df)
    assert out["log_capacity_mw"].tolist() == [0.0, 1.0]


def test_range_coverage_flags_values_outside_earlier_range():
    from queueiq.features.build import FEATURES

    scoped = pd.DataFrame({name: 1.0 for name in FEATURES}, index=range(4))
    for name in ("fuel_group", "iso", "state"):
        scoped[name] = "x"
    for name in ("is_hybrid", "has_storage", "county_known"):
        scoped[name] = True
    scoped["queue_year"] = [2015, 2016, 2021, 2022]
    scoped["capacity_mw"] = [10.0, 20.0, 15.0, 99.0]  # 15 inside [10, 20], 99 outside
    table = range_coverage(scoped).set_index("feature")
    assert table.loc["capacity_mw", "later_rows_inside_range_%"] == 50.0
    assert table.loc["queue_year", "later_rows_inside_range_%"] == 0.0
    assert "fuel_group" not in table.index
