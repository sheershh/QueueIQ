import numpy as np
import pandas as pd
import pytest

from queueiq.explain import deploy
from queueiq.explain.groups import GROUPS, TARGETS, describe
from queueiq.explain.permutation import grouped_permutation_importance
from queueiq.explain.text import narrative
from queueiq.models.data import MODEL_FEATURES
from tests.model_data import synthetic


def test_groups_cover_every_model_feature_except_queue_year_once():
    grouped = [c for cols in GROUPS.values() for c in cols]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(MODEL_FEATURES) - {"queue_year"}


class OnlyCapacity:
    """Toy model: predictions depend on capacity only (both outcomes)."""

    def predict_cif(self, X):
        p = (X["log_capacity_mw"].to_numpy() / 10)[:, None] * np.linspace(0.1, 1, 96)[None, :]
        return {"operational": p * 0.3, "withdrawn": p * 0.6}


def test_permutation_importance_is_exactly_zero_for_unused_groups():
    _, f = synthetic(n=300, seed=2)
    X = f.drop(columns="project_id")
    rng = np.random.default_rng(0)
    time, event = rng.integers(1, 30, len(X)), rng.choice([1, 2], len(X))
    null = {k: np.full((len(X), 96), 0.3) for k in ("operational", "withdrawn")}
    imp = grouped_permutation_importance(OnlyCapacity(), X, time, event, null, GROUPS, repeats=2)
    unused = imp.drop(index="size")
    assert (unused.abs().to_numpy() == 0).all()
    assert imp.loc["size"].abs().sum() > 0


def row(**kw):
    base = {
        "log_capacity_mw": np.log1p(150),
        "fuel_group": "solar",
        "is_hybrid": True,
        "has_storage": True,
        "iso": "PJM",
        "state": "VA",
        "county_known": True,
        "county_prior_requests_5y": 12.0,
        "county_prior_withdrawn_5y": 4.0,
        "state_prior_requests_5y": 300.0,
        "state_prior_withdrawn_share_5y": 0.25,
        "state_prior_requests_1y": 60.0,
    }
    return pd.Series({**base, **kw})


def test_describe_each_group():
    r = row()
    assert describe("size", r) == "150 MW"
    assert describe("technology", r) == "solar, with storage, hybrid"
    assert "12 earlier requests in the county" in describe("county congestion", r)
    assert describe("county congestion", row(county_known=False)) == "county unknown"
    assert "25% already withdrawn" in describe("state congestion", r)


REF = {
    "withdrawn_3y": {"cohort": 2022, "observed": 0.372, "n": 1579},
    "operational_5y": {"cohort": 2020, "observed": 0.022, "n": 1435},
}


def test_narrative_labels_the_baseline_and_shows_the_observed_reference():
    table = pd.DataFrame(
        {"withdrawn_3y": [0.10, -0.03, 0.001], "operational_5y": [0.0, 0.0, 0.0]},
        index=["state congestion", "size", "technology"],
    )
    table = table.reindex(list(GROUPS)).fillna(0.0)
    table.attrs["prediction"] = {"withdrawn_3y": 0.62, "operational_5y": 0.03}
    table.attrs["baseline"] = {"withdrawn_3y": 0.55, "operational_5y": 0.03}
    text = narrative(table, row(), REF)
    w = text["withdrawn_3y"]
    assert w.startswith(TARGETS["withdrawn_3y"][2] + ": 62.0%.")
    assert "Model's average prediction for recent entrants: 55.0%" in w
    assert "observed for 2022 entrants (latest with full 3-year follow-up): 37.2%" in w
    assert "typical" not in w
    assert "Pushing it up: state congestion (+10.0 points" in w
    assert "Pulling it down: size (-3.0 points: 150 MW)" in w
    assert "technology" not in w  # 0.1 points is below the reporting threshold
    assert "No feature group moves it" in text["operational_5y"]


def test_calibration_warning_states_both_gaps():
    from queueiq.explain.reference import calibration_warning

    cal = {
        "withdrawn": {"predicted": 0.516, "observed": 0.343, "cases": 1131, "n": 3299},
        "operational": {"predicted": 0.046, "observed": 0.0003, "cases": 1, "n": 3299},
    }
    w = calibration_warning(cal)
    assert "51.6%" in w and "34.3%" in w and "4.6%" in w and "0.03%" in w and "n = 3,299" in w


def test_observed_reference_uses_latest_fully_followed_cohort():
    from queueiq.explain.reference import observed_reference

    entries, _ = synthetic(n=600, seed=9)
    entries["queue_year"] = entries["queue_date"].dt.year
    ref = observed_reference(entries, pd.Timestamp("2025-12-31"))
    assert ref["withdrawn_3y"]["cohort"] == entries["queue_year"].max()  # synthetic ends 2020
    assert 0 <= ref["operational_5y"]["observed"] <= 1


def test_primary_model_mapping():
    m = deploy.primary_model({"primary_configuration": "rsf_cause_specific|qy=0|recal=0"})
    assert m.exclude == ("queue_year",)
    with pytest.raises(NotImplementedError):
        deploy.primary_model({"primary_configuration": "rsf_cause_specific|qy=0|recal=1"})
    with pytest.raises(NotImplementedError):
        deploy.primary_model({"primary_configuration": "recent_cohort"})


def test_deployment_data_uses_every_in_scope_row_and_background_is_recent():
    entries, features = synthetic(n=400, seed=4)
    X, t, e, ids = deploy.deployment_data(entries, features)
    assert len(X) == len(ids) == entries["in_model_scope"].sum()
    assert set(e) <= {0, 1, 2}
    old = deploy.BACKGROUND_YEARS
    try:
        deploy.BACKGROUND_YEARS = (2016, 2020)
        bg = deploy.background_sample(X)
        assert len(bg) == deploy.BACKGROUND_SIZE and bg["queue_year"].between(2016, 2020).all()
    finally:
        deploy.BACKGROUND_YEARS = old
