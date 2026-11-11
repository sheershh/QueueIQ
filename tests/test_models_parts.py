import numpy as np
import pandas as pd
import pytest

from queueiq.evaluation.horizon import horizon_labels, horizon_metrics
from queueiq.models.combine import cif_from_cumulative_hazards, step_on_grid
from queueiq.models.lgbm import expand_person_period
from queueiq.models.preprocess import Preprocessor, redundant_columns
from tests.model_data import synthetic


def test_combine_hand_example():
    """Month 1: dH_op .1, dH_wd .2 -> CIF_op .1, CIF_wd .2, S .7.
    Month 2: dH_wd .3 -> CIF_wd .2 + .7 * .3 = .41, S .49. Shares sum to 1."""
    cif = cif_from_cumulative_hazards(np.array([[0.1, 0.1]]), np.array([[0.2, 0.5]]))
    np.testing.assert_allclose(cif["operational"], [[0.1, 0.1]])
    np.testing.assert_allclose(cif["withdrawn"], [[0.2, 0.41]])
    np.testing.assert_allclose(1 - cif["operational"] - cif["withdrawn"], [[0.7, 0.49]])


def test_combine_caps_a_month_whose_hazards_exceed_one():
    cif = cif_from_cumulative_hazards(np.array([[0.9]]), np.array([[0.6]]))
    assert cif["operational"][0, 0] + cif["withdrawn"][0, 0] == pytest.approx(1.0)


def test_step_on_grid():
    out = step_on_grid(np.array([2.0, 5.0]), np.array([[0.1, 0.3]]), np.array([1, 2, 4, 5, 9]))
    np.testing.assert_allclose(out, [[0.0, 0.1, 0.1, 0.3, 0.3]])


def test_person_period_expansion_hand_example():
    # rows: event op in month 4 | censored month 4 | censored month 3 | withdrawn month 1
    pp = expand_person_period(np.array([4, 4, 3, 1]), np.array([1, 0, 0, 2]))
    assert pp.to_dict("list") == {
        "row": [0, 0, 1, 3],
        "interval": [1, 2, 1, 1],
        "label": [0, 1, 0, 2],
    }


def test_preprocessing_learns_only_from_training_rows():
    _, f = synthetic()
    train, test = f.iloc[:400].copy(), f.iloc[400:].copy()
    test.loc[test.index[:5], "state"] = "NEW"  # a state never seen in training
    prep = Preprocessor("linear", min_state_rows=50).fit(train)
    assert "ZZ" not in prep.states_  # rare in training -> grouped as other
    Z = prep.transform(test)
    assert list(Z.columns) == prep.columns_
    assert not Z.isna().any().any()
    lin = prep.transform(train)
    assert lin["queue_year"].mean() == pytest.approx(0, abs=1e-9)  # scaled with train stats


def test_lgbm_mode_keeps_missing_values_and_categories():
    _, f = synthetic()
    Z = Preprocessor("lgbm").fit(f).transform(f)
    assert Z["county_prior_requests_5y"].isna().any()
    assert isinstance(Z["iso"].dtype, pd.CategoricalDtype)


def test_redundant_columns_finds_constants_duplicates_and_mirrors():
    df = pd.DataFrame(
        {
            "a": [0.0, 1, 0],
            "b": [1.0, 0, 1],
            "c": [5.0, 5, 5],
            "d": [0.0, 1, 0],
            "x": [0.3, 0.1, 0.2],
        }
    )
    assert redundant_columns(df) == ["b", "c", "d"]


def test_horizon_labels_and_metrics():
    time, event = np.array([10, 40, 70, 61]), np.array([2, 1, 0, 2])
    assert horizon_labels(time, event, 2, 36).tolist() == [1, 0, 0, 0]
    with pytest.raises(ValueError, match="censored before month 72"):
        horizon_labels(time, event, 2, 72)
    m = horizon_metrics(np.array([0.9, 0.1, 0.2, 0.3]), np.array([1, 0, 0, 0]))
    assert m["auc"] == 1.0
    assert m["brier"] == pytest.approx((0.01 + 0.01 + 0.04 + 0.09) / 4, abs=1e-4)
