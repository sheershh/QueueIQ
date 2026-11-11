"""Every model fits on synthetic data and returns valid cumulative incidence curves."""

import numpy as np
import pandas as pd
import pytest

from queueiq.models.base import GRID
from queueiq.models.baseline import GroupAalenJohansen
from queueiq.models.cox import CauseSpecificCox
from queueiq.models.data import make_split
from queueiq.models.lgbm import DiscreteTimeLGBM
from queueiq.models.rsf import CauseSpecificRSF
from tests.model_data import synthetic


@pytest.fixture(scope="module")
def split():
    entries, features = synthetic(n=800)
    return make_split(
        entries, features, pd.Timestamp("2018-01-01"), (2018, 2020), min_test_followup_months=60
    )


@pytest.mark.parametrize(
    "model",
    [
        GroupAalenJohansen(min_group=30),
        CauseSpecificCox(),
        CauseSpecificRSF(n_estimators=20),
        DiscreteTimeLGBM(num_rounds=20),
    ],
    ids=lambda m: m.name,
)
def test_model_returns_valid_cif(model, split):
    cif = model.fit(split.X_train, split.time_train, split.event_train).predict_cif(split.X_test)
    for k in ("operational", "withdrawn"):
        assert cif[k].shape == (len(split.X_test), len(GRID))
        assert (cif[k] >= -1e-12).all()
        assert (np.diff(cif[k], axis=1) >= -1e-9).all()  # cumulative: never decreases
    assert (cif["operational"] + cif["withdrawn"] <= 1 + 1e-9).all()


def test_gas_signal_is_learned(split):
    """Synthetic data builds gas projects more often; a fitted model should rank them higher."""
    model = CauseSpecificCox().fit(split.X_train, split.time_train, split.event_train)
    cif = model.predict_cif(split.X_test)["operational"][:, 59]
    gas = (split.X_test["fuel_group"] == "gas").to_numpy()
    assert cif[gas].mean() > cif[~gas].mean()


def test_lgbm_round_selection_uses_only_given_data(split):
    rounds = DiscreteTimeLGBM.select_rounds(
        split.X_train,
        split.time_train,
        split.event_train,
        split.X_train,
        split.time_train,
        split.event_train,
        max_rounds=50,
    )
    assert 1 <= rounds <= 50


def test_cox_reports_ph_test_and_hazard_ratios(split):
    model = CauseSpecificCox().fit(split.X_train, split.time_train, split.event_train)
    ph = model.ph_test()
    assert set(ph["cause"]) == {"operational", "withdrawn"}
    assert (model.hazard_ratios() > 0).all().all()
