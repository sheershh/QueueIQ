import math

import pandas as pd

from tests.conftest import pipeline


def row(df, i):
    return df.iloc[i]


def test_each_status_maps_to_the_documented_outcome():
    df = pipeline(
        [
            {"q_status": "operational", "on_date": pd.Timestamp("2018-06-01")},
            {"q_status": "withdrawn", "wd_date": pd.Timestamp("2017-01-01")},
            {"q_status": "active"},
            {"q_status": "suspended"},
            {"q_status": "unknown"},
        ]
    )
    assert df["outcome"].tolist() == ["operational", "withdrawn", "censored", "censored", "unknown"]
    assert row(df, 0)["end_date"] == pd.Timestamp("2018-06-01")
    assert row(df, 2)["end_date"] == pd.Timestamp("2025-12-31")  # censored at the cutoff
    assert pd.isna(row(df, 4)["end_date"])


def test_time_is_measured_in_average_months_from_entry():
    df = pipeline(
        [
            {
                "q_date": pd.Timestamp("2020-01-01"),
                "q_status": "withdrawn",
                "wd_date": pd.Timestamp("2021-01-01"),
            }
        ]
    )
    assert math.isclose(row(df, 0)["time_months"], 366 / 30.4375)


def test_event_after_cutoff_is_censored_at_cutoff():
    df = pipeline([{"q_status": "withdrawn", "wd_date": pd.Timestamp("2026-01-16")}])
    r = row(df, 0)
    assert r["outcome"] == "censored"
    assert r["end_date"] == pd.Timestamp("2025-12-31")
    assert r["censored_at_cutoff_after_event"]


def test_missing_event_date_keeps_outcome_but_has_no_time():
    df = pipeline([{"q_status": "operational", "on_date": None}])
    assert row(df, 0)["outcome"] == "operational"
    assert math.isnan(row(df, 0)["time_months"])


def test_outcome_uses_the_date_matching_the_status():
    """A withdrawn row that also has an operation date is timed by its withdrawal date."""
    df = pipeline(
        [
            {
                "q_status": "withdrawn",
                "wd_date": pd.Timestamp("2016-01-01"),
                "on_date": pd.Timestamp("2019-01-01"),
            }
        ]
    )
    assert row(df, 0)["end_date"] == pd.Timestamp("2016-01-01")
