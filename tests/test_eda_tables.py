import numpy as np
import pandas as pd
import pytest

from queueiq.eda import tables as T


def scoped_frame(n_op=40, n_wd=60, n_cens=20, iso="PJM", year=2015, fuel="solar"):
    """Synthetic in-scope rows: events spread over 1..80 months, censored at 100 months."""
    rows = (
        [("operational", 10 + i) for i in range(n_op)]
        + [("withdrawn", 1 + i) for i in range(n_wd)]
        + [("censored", 100.0)] * n_cens
    )
    return pd.DataFrame(
        {
            "outcome": [r[0] for r in rows],
            "time_months": [float(r[1]) for r in rows],
            "iso": iso,
            "queue_year": pd.array([year] * len(rows), dtype="Int64"),
            "fuel_type": fuel,
            "capacity_mw": 100.0,
            "has_storage": False,
            "in_model_scope": True,
        }
    )


def test_outcome_shares_rows_sum_to_100():
    shares = T.outcome_shares(scoped_frame(), "iso")
    row = shares.loc["PJM"]
    assert row["requests"] == 120
    assert row[["operational", "withdrawn", "in_queue"]].sum() == pytest.approx(100.0)


def test_add_groups_assigns_era_and_folds_rare_fuels():
    df = pd.concat([scoped_frame(year=2012, fuel="gas"), scoped_frame(year=2021, fuel="hydro")])
    g = T.add_groups(df)
    assert set(g["era"]) == {"2010-2014", "2020-2025"}
    assert set(g["fuel_group"]) == {"gas", "other"}


def test_cif_summary_blanks_horizons_with_too_few_projects_at_risk():
    small = scoped_frame(n_op=5, n_wd=10, n_cens=5)  # 20 projects in total: below MIN_AT_RISK
    table = T.cif_summary(small, "iso")
    assert np.isnan(table.loc["PJM", "operational_by_3y_%"])
    big = scoped_frame()
    table = T.cif_summary(big, None)
    assert table.loc["All", "at_risk_3y"] >= T.MIN_AT_RISK
    assert not np.isnan(table.loc["All", "withdrawn_by_3y_%"])


def test_observed_durations_are_medians_among_events_only():
    table = T.observed_durations(scoped_frame(n_op=3, n_wd=3, n_cens=10), "iso")
    assert table.loc["PJM", "operational_median_months"] == 11.0  # 10, 11, 12
    assert table.loc["PJM", "withdrawn_n"] == 3


def test_lbnl_crosscheck_is_capacity_weighted():
    df = pd.DataFrame(
        {
            "queue_year": pd.array([2005, 2010, 2015, 2022], dtype="Int64"),
            "status": ["operational", "withdrawn", "active", "operational"],
            "queue_date": pd.to_datetime(["2005-01-01", "2010-01-01", "2015-01-01", "2022-01-01"]),
            "operation_date": pd.to_datetime(["2025-01-01", None, None, "2025-01-01"]),
        }
    )
    raw = pd.DataFrame(
        {"mw_1": [100.0, 300.0, 100.0, 50.0], "mw_2": [None, 100.0, None, None], "mw_3": [None] * 4}
    )
    check = T.lbnl_crosscheck(df, raw)
    # 2000-2020 cohort: 100 MW operational, 400 MW withdrawn, 100 MW active (2022 row excluded)
    assert check["mw_share_operational_%"] == pytest.approx(100 / 6, abs=0.05)
    assert check["mw_share_withdrawn_%"] == pytest.approx(400 / 6, abs=0.05)
    assert check["online_2025_n"] == 2
