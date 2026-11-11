"""Exact grouped Shapley values checked on toy models with known answers."""

import numpy as np
import pandas as pd
import pytest

from queueiq.explain.shapley import explain, shapley_from_values

BG = pd.DataFrame({"a": [0.0, 2.0, 4.0], "b": [1.0, 1.0, 4.0], "c": [5.0, 6.0, 7.0]})
X = pd.DataFrame({"a": [10.0], "b": [3.0], "c": [0.0]})
GROUPS = {"A": ["a"], "B": ["b"], "C": ["c"]}


def predict_additive(df):
    return {"y": 2 * df["a"].to_numpy() + 3 * df["b"].to_numpy()}


def test_additive_model_gets_coefficient_times_distance_from_background_mean():
    phi = explain(predict_additive, X, BG, GROUPS)["y"]
    # a: 2 * (10 - mean(0, 2, 4) = 2) = 16;  b: 3 * (3 - mean(1, 1, 4) = 2) = 3;  c unused: 0
    assert phi.tolist() == pytest.approx([16.0, 3.0, 0.0])


def test_values_add_up_to_prediction_minus_baseline():
    def predict(df):
        a, b, c = (df[k].to_numpy() for k in "abc")
        return {"y": a * b + np.sin(c) + a}

    table = explain(predict, X, BG, GROUPS)
    total = table["y"].sum()
    assert total == pytest.approx(table.attrs["prediction"]["y"] - table.attrs["baseline"]["y"])
    assert table.attrs["prediction"]["y"] == pytest.approx(10 * 3 + np.sin(0) + 10)


def test_pure_interaction_is_split_equally():
    """f = a * b with a single background row (0, 0): v(empty) = v(a) = v(b) = 0, v(ab) = 30,
    so a and b each get half of 30."""
    phi = explain(
        lambda df: {"y": (df["a"] * df["b"]).to_numpy()}, X, BG.iloc[[0]].assign(b=0.0), GROUPS
    )["y"]
    assert phi.tolist() == pytest.approx([15.0, 15.0, 0.0])


def test_grouping_columns_together():
    phi = explain(predict_additive, X, BG, {"AB": ["a", "b"], "C": ["c"]})["y"]
    assert phi.tolist() == pytest.approx([19.0, 0.0])


def test_shapley_from_values_two_player_hand_example():
    # v(empty)=0, v({0})=4, v({1})=2, v({0,1})=10 -> phi0 = (4 + (10-2))/2 = 6, phi1 = 4
    assert shapley_from_values(np.array([0.0, 4.0, 2.0, 10.0]), 2).tolist() == [6.0, 4.0]
