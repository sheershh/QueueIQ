"""Phase 9: comparable history, percentiles, evidence labels, county crowding and the map."""

import json

import numpy as np
import pandas as pd
import pytest

from queueiq.app import geomap
from queueiq.app.present import (
    grid_counts,
    history_sentence,
    outcome_grid_html,
    percentile_html,
    share_text,
)
from queueiq.explain.reference import signal_strength
from queueiq.service.features import project_features
from queueiq.service.geo import county_crowding
from queueiq.service.history import MIN_N, comparable_outcomes, comparable_table
from queueiq.service.inputs import ProjectInput
from tests.service_data import synthetic_service

CUTOFF = pd.Timestamp("2025-12-31")


@pytest.fixture(scope="module")
def svc():
    return synthetic_service()


def entries(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["queue_date"] = pd.to_datetime(df["queue_date"])
    df["queue_year"] = df["queue_date"].dt.year
    df["in_model_scope"] = True
    return df


def test_comparable_table_uses_complete_cohorts_and_known_outcomes():
    base = {"iso": "PJM", "fuel_type": "solar"}
    rows = [
        # 2022 entrants all reach 36 months by the cutoff; 2023 entrants do not and are excluded.
        {**base, "queue_date": "2022-03-01", "outcome": "withdrawn", "time_months": 10.0},
        {**base, "queue_date": "2022-05-01", "outcome": "operational", "time_months": 30.0},
        {**base, "queue_date": "2021-05-01", "outcome": "censored", "time_months": 55.0},
        {
            **base,
            "queue_date": "2020-05-01",
            "outcome": "withdrawn",
            "time_months": 40.0,
        },  # after 36
        {**base, "queue_date": "2023-01-15", "outcome": "withdrawn", "time_months": 5.0},
    ]
    t = comparable_table(entries(rows), CUTOFF, horizon=36)
    assert t["years"] == [2019, 2022]
    g = t["by_iso_fuel"]["PJM|solar"]
    assert g["n"] == 4
    assert g["withdrawn"] == pytest.approx(0.25)  # only the 10-month withdrawal counts
    assert g["operational"] == pytest.approx(0.25)
    assert g["waiting"] == pytest.approx(0.5)


def test_small_groups_fall_back_to_the_grid_operator():
    t = {
        "years": [2019, 2022],
        "horizon_months": 36,
        "min_n": MIN_N,
        "by_iso_fuel": {"PJM|gas": {"n": 5, "withdrawn": 0.2, "operational": 0.4, "waiting": 0.4}},
        "by_iso": {"PJM": {"n": 500, "withdrawn": 0.3, "operational": 0.1, "waiting": 0.6}},
    }
    c = comparable_outcomes(t, "PJM", "gas")
    assert c["fallback"] and c["n"] == 500 and c["horizon_months"] == 36
    assert "Too few PJM gas projects" in history_sentence(c, "PJM", "gas")


def test_signal_strength_reads_the_metrics_file():
    m = pd.DataFrame(
        [
            {
                "block": "C",
                "model": "m",
                "outcome": "withdrawn",
                "tau": 36,
                "auc": 0.559,
                "n": 9,
                "cases": 3,
            },
            {
                "block": "A",
                "model": "m",
                "outcome": "withdrawn",
                "tau": 36,
                "auc": 0.6,
                "n": 9,
                "cases": 3,
            },
            {
                "block": "C",
                "model": "m",
                "outcome": "operational",
                "tau": 36,
                "auc": np.nan,
                "n": 9,
                "cases": 1,
            },
            {
                "block": "A-pooled",
                "model": "m",
                "outcome": "operational",
                "tau": 36,
                "auc": 0.723,
                "n": 9,
                "cases": 5,
            },
            {
                "block": "B",
                "model": "m",
                "outcome": "operational",
                "tau": 36,
                "auc": 0.88,
                "n": 9,
                "cases": 5,
            },
        ]
    )
    s = signal_strength(m, "m")
    assert s["withdrawn"]["label"] == "weak evidence" and s["withdrawn"]["block"] == "C"
    assert (
        s["operational"]["block"] == "A-pooled" and s["operational"]["label"] == "moderate evidence"
    )
    assert s["operational"]["auc_max"] == pytest.approx(0.88)


def test_percentiles_are_in_range_and_follow_the_prediction(svc):
    p = ProjectInput(iso="PJM", state="VA", fuel="gas", capacity_mw=300)
    out = svc.percentiles(p)
    for t in out["targets"].values():
        assert 0 <= t["percentile"] <= 100 and "label" in t["evidence"]
    q = np.asarray(svc.meta["percentile_reference"]["quantiles"]["withdrawn_3y"])
    assert (np.diff(q) >= 0).all()


def test_county_crowding_equals_the_model_features(svc):
    crowd = county_crowding(svc.hist, svc.entry_date)
    for key in crowd.index:
        state = next(s for s, cs in svc.options.counties.items() if any(c.key == key for c in cs))
        iso = next((i for i, ss in svc.options.states.items() if state in ss), None)
        if iso is None:
            continue
        f = project_features(
            ProjectInput(iso=iso, state=state, county=key, fuel="solar", capacity_mw=50),
            svc.hist,
            svc.options,
            svc.entry_date,
        )
        assert f["county_prior_requests_5y"][0] == crowd.loc[key, "requests_5y"]
        assert f["county_prior_withdrawn_5y"][0] == crowd.loc[key, "withdrawn_5y"]


def test_grid_and_text_helpers():
    c = {"withdrawn": 0.319, "operational": 0.004, "waiting": 0.677}
    counts = grid_counts(c)
    assert sum(counts.values()) == 100 and counts["withdrawn"] == 32
    assert outcome_grid_html(c).count('class="qi-cell qi-') == 103  # 100 cells + 3 legend keys
    assert share_text(0.004) == "under 1%" and share_text(0) == "none"
    html = percentile_html(
        "x", {"percentile": 72.4, "evidence": {"label": "weak evidence", "auc": 0.56}}, [2021, 2025]
    )
    assert "Higher than 72% of 2021–2025 entrants" in html and "56% of the time" in html


def test_map_assets_and_deck(svc):
    counties = geomap.shapes("counties")["features"]
    states = geomap.shapes("states")["features"]
    covered = {f["properties"]["state"] for f in states if f["properties"]["covered"]}
    assert {f["properties"]["state"] for f in counties} == covered
    assert all(len(f["properties"]["key"]) == 5 for f in counties)
    deck = json.loads(geomap.deck(svc.crowding(), "51001").to_json())
    assert {layer["id"] for layer in deck["layers"]} == {"states", "counties"}
    assert geomap.centroid("51001") is not None


def test_map_selection_parsing():
    event = {"selection": {"objects": {"counties": [{"properties": {"key": "17031"}}]}}}
    assert geomap.selected_key(event) == "17031"
    assert geomap.selected_key({"selection": {"objects": {}}}) is None
    assert geomap.selected_key(None) is None
