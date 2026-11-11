"""Leakage tests for the feature pipeline.

1. A hand-built example whose counts were worked out by hand (one county, every edge case).
2. "Hide the future": for any project, deleting everything that happened after its entry
   date must not change its features.
3. Features never depend on status or outcome columns.
"""

import numpy as np
import pandas as pd
import pytest

from queueiq.features.build import CONGESTION_ONLY_COLUMNS, ENTRY_SAFE_COLUMNS, build_features

T = pd.Timestamp
NaT = pd.NaT


def entries(rows: list[dict]) -> pd.DataFrame:
    """Minimal entries table. status is included only to prove it is ignored."""
    base = {
        "iso": "PJM",
        "state": "XX",
        "county": "c1",
        "fips_code": "00001",
        "fuel_type": "solar",
        "is_hybrid": False,
        "has_storage": False,
        "capacity_mw": 10.0,
        "withdrawal_date": NaT,
        "status": "active",
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["queue_date"] = pd.to_datetime(df["queue_date"])
    df["withdrawal_date"] = pd.to_datetime(df["withdrawal_date"])
    df["queue_year"] = df["queue_date"].dt.year.astype("Int64")
    for c in ["iso", "state", "county", "fips_code", "fuel_type", "status", "project_id"]:
        df[c] = df[c].astype("string")
    return df


# Focal project F enters county c1 on 2020-06-01. Its 5-year window is [2015-06-01, 2020-06-01).
HAND = [
    {
        "project_id": "A",
        "queue_date": T("2014-01-01"),
        "withdrawal_date": T("2016-01-01"),
        "capacity_mw": 100.0,
    },  # entered before the window: not counted
    {
        "project_id": "B",
        "queue_date": T("2016-03-01"),
        "withdrawal_date": T("2018-05-01"),
        "capacity_mw": 50.0,
    },  # in window, withdrew before F entered: request + withdrawn
    {
        "project_id": "C",
        "queue_date": T("2018-07-01"),
        "status": "withdrawn",
        "capacity_mw": 200.0,
    },  # withdrawn but undated: request, never withdrawn
    {
        "project_id": "D",
        "queue_date": T("2019-09-01"),
        "withdrawal_date": T("2021-01-01"),
        "capacity_mw": 80.0,
    },  # withdrew AFTER F entered: request, not withdrawn
    {
        "project_id": "G",
        "queue_date": T("2019-01-01"),
        "withdrawal_date": T("2020-06-01"),
        "capacity_mw": 0.0,
    },  # withdrew ON F's entry day: request, not withdrawn; 0 MW
    {"project_id": "E", "queue_date": T("2020-06-01"), "capacity_mw": 30.0},  # same day as F
    {"project_id": "H", "queue_date": NaT, "withdrawal_date": T("2017-01-01")},  # no entry date
    {
        "project_id": "K",
        "queue_date": T("2017-01-01"),
        "withdrawal_date": T("2018-01-01"),
        "capacity_mw": 40.0,
        "county": "c2",
        "fips_code": "00002",
    },  # other county, same state
    {"project_id": "F", "queue_date": T("2020-06-01"), "capacity_mw": 120.0},
]


@pytest.fixture
def hand():
    return build_features(entries(HAND)).set_index("project_id")


def test_hand_example_county_counts(hand):
    f = hand.loc["F"]
    assert f["county_prior_requests_5y"] == 4  # B, C, D, G
    assert f["county_prior_mw_5y"] == 330.0  # 50 + 200 + 80 + 0
    assert f["county_prior_withdrawn_5y"] == 1  # B only
    assert f["county_prior_withdrawn_share_5y"] == pytest.approx(0.25)


def test_hand_example_state_counts(hand):
    f = hand.loc["F"]
    assert f["state_prior_requests_5y"] == 5  # B, C, D, G + K from the other county
    assert f["state_prior_mw_5y"] == 370.0
    assert f["state_prior_withdrawn_5y"] == 2  # B, K
    assert f["state_prior_requests_1y"] == 1  # window [2019-06-01, 2020-06-01): only D


def test_same_day_entries_see_the_same_past(hand):
    cols = [c for c in hand.columns if "prior" in c]
    pd.testing.assert_series_equal(hand.loc["E", cols], hand.loc["F", cols], check_names=False)


def test_edge_rows(hand):
    assert hand.loc["A", "county_prior_requests_5y"] == 0  # nothing entered before A
    assert np.isnan(hand.loc["A", "county_prior_withdrawn_share_5y"])  # 0/0 is undefined
    # H has no entry date: it can't be placed in time, so it gets no features ...
    assert hand.loc["H", ["county_prior_requests_5y", "state_prior_requests_5y"]].isna().all()


def test_window_is_relative_to_each_project(hand):
    # B entered 2016-03-01; its window is [2011-03-01, 2016-03-01).
    # A entered 2014-01-01 (inside) and withdrew 2016-01-01 (before B entered).
    assert hand.loc["B", "county_prior_requests_5y"] == 1
    assert hand.loc["B", "county_prior_withdrawn_5y"] == 1
    # ... and H (no entry date) is never counted for anyone, even with a withdrawal date.
    assert hand.loc["F", "state_prior_requests_5y"] == 5


def random_entries(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    q = T("2005-01-01") + pd.to_timedelta(rng.integers(0, 365 * 21, n), unit="D")
    q = q.floor("D")
    wd = q + pd.to_timedelta(rng.integers(0, 365 * 6, n), unit="D")
    wd = pd.Series(wd).where(rng.random(n) < 0.6)  # 40% never withdraw
    rows = [
        {
            "project_id": f"P{i}",
            "queue_date": q[i],
            "withdrawal_date": wd[i],
            "capacity_mw": float(rng.integers(0, 500)),
            "fips_code": f"0000{rng.integers(1, 4)}",
            "state": "XX" if rng.random() < 0.7 else "YY",
            "status": rng.choice(["active", "withdrawn", "operational"]),
        }
        for i in range(n)
    ]
    df = entries(rows)
    df.loc[rng.choice(n, 10, replace=False), "queue_date"] = NaT  # some undated entries
    df["queue_year"] = df["queue_date"].dt.year.astype("Int64")
    return df


def known_at(df: pd.DataFrame, d: pd.Timestamp, rng) -> pd.DataFrame:
    """The dataset as it could have looked on date d: later entries gone, later withdrawals
    unknown, and status/outcome scrambled (it describes the future)."""
    past = df[~(df["queue_date"] > d)].copy()
    past.loc[past["withdrawal_date"] >= d, "withdrawal_date"] = NaT
    past["status"] = rng.permutation(past["status"].to_numpy())
    return past


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_features_are_unchanged_when_the_future_is_hidden(seed):
    rng = np.random.default_rng(seed)
    df = random_entries(seed=seed)
    full = build_features(df).set_index("project_id")
    focal_ids = rng.choice(df.loc[df["queue_date"].notna(), "project_id"], 25, replace=False)
    for pid in focal_ids:
        d = df.loc[df["project_id"] == pid, "queue_date"].iloc[0]
        truncated = build_features(known_at(df, d, rng)).set_index("project_id")
        pd.testing.assert_series_equal(full.loc[pid], truncated.loc[pid], check_names=False)


def test_features_ignore_status_and_outcome_columns():
    df = random_entries(seed=3)
    noisy = df.assign(
        status="operational",
        outcome="withdrawn",
        operation_date=T("1990-01-01"),
        ia_date=T("1990-01-01"),
        proposed_online_date=T("1990-01-01"),
    )
    pd.testing.assert_frame_equal(build_features(df), build_features(noisy))


def test_feature_inputs_contain_no_outcome_information():
    forbidden = {
        "status",
        "outcome",
        "end_date",
        "time_months",
        "operation_date",
        "ia_date",
        "ia_phase",
        "proposed_online_date",
        "censored_at_cutoff_after_event",
    }
    assert not forbidden & set(ENTRY_SAFE_COLUMNS + CONGESTION_ONLY_COLUMNS)
    out = build_features(random_entries(seed=4))
    assert not forbidden & set(out.columns)
