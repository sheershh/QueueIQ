"""Aalen-Johansen checked against a small example worked out by hand.

Five projects (months, outcome):
  1 operational | 2 withdrawn | 2 censored | 3 withdrawn | 4 censored
t=1: at risk 5, 1 operational -> CIF_op = 1 * 1/5 = 0.2,   S = 0.8
t=2: at risk 4 (the project censored at 2 still counts), 1 withdrawn
     -> CIF_wd = 0.8 * 1/4 = 0.2,                            S = 0.8 * 3/4 = 0.6
t=3: at risk 2, 1 withdrawn -> CIF_wd = 0.2 + 0.6 * 1/2 = 0.5, S = 0.3
t=4: at risk 1, no event                                       S = 0.3
"""

import numpy as np
import pandas as pd
import pytest

from queueiq.eda.survival import aalen_johansen, at_risk_at, value_at

TIME = [1, 2, 2, 3, 4]
OUTCOME = ["operational", "withdrawn", "censored", "withdrawn", "censored"]


@pytest.fixture
def curve():
    return aalen_johansen(pd.Series(TIME), pd.Series(OUTCOME))


def test_hand_worked_example(curve):
    assert curve["time"].tolist() == [1, 2, 3, 4]
    assert curve["n_at_risk"].tolist() == [5, 4, 2, 1]
    np.testing.assert_allclose(curve["cif_operational"], [0.2, 0.2, 0.2, 0.2])
    np.testing.assert_allclose(curve["cif_withdrawn"], [0.0, 0.2, 0.5, 0.5])
    np.testing.assert_allclose(curve["still_in_queue"], [0.8, 0.6, 0.3, 0.3])


def test_shares_always_sum_to_one(curve):
    total = curve["still_in_queue"] + curve["cif_operational"] + curve["cif_withdrawn"]
    np.testing.assert_allclose(total, 1.0)


def test_without_censoring_it_equals_simple_proportions():
    rng = np.random.default_rng(0)
    t = rng.integers(1, 50, size=200)
    o = rng.choice(["operational", "withdrawn"], size=200)
    curve = aalen_johansen(pd.Series(t), pd.Series(o))
    for k in ("operational", "withdrawn"):
        expected = np.mean(o == k)
        assert value_at(curve, f"cif_{k}", 1000) == pytest.approx(expected)


def test_ignoring_censoring_would_understate_outcomes(curve):
    """Raw share withdrawn is 2/5 = 0.4; the censoring-aware estimate at t=3 is 0.5."""
    assert value_at(curve, "cif_withdrawn", 3) > OUTCOME.count("withdrawn") / len(OUTCOME)


def test_step_lookup_and_at_risk(curve):
    assert value_at(curve, "cif_withdrawn", 0.5) == 0.0  # before any event
    assert value_at(curve, "still_in_queue", 0.5) == 1.0
    assert value_at(curve, "cif_withdrawn", 2.9) == pytest.approx(0.2)  # holds until next step
    assert at_risk_at(curve, 2.5) == 2
    assert at_risk_at(curve, 99) == 0


def test_rejects_missing_or_negative_time():
    with pytest.raises(ValueError, match="missing"):
        aalen_johansen(pd.Series([1.0, np.nan]), pd.Series(["withdrawn", "withdrawn"]))
    with pytest.raises(ValueError, match="non-negative"):
        aalen_johansen(pd.Series([-1.0]), pd.Series(["withdrawn"]))
