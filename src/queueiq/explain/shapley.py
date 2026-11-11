"""Exact Shapley values over feature groups, for any model (decisions D-040).

Question answered: "how much did each group of features move this project's prediction away
from a typical project?" For a set S of groups, v(S) is the average prediction when the project
keeps its own values for the groups in S and takes the values of each background project for the
rest (interventional / marginal Shapley). Each group's Shapley value is its average marginal
contribution over all orders in which groups could be added:

    phi_g = sum over S not containing g of  |S|! (n - |S| - 1)! / n!  *  (v(S + g) - v(S))

With n = 7 groups there are only 2^7 = 128 sets S, so this is computed exactly (no sampling).
Guarantees: the values add up exactly, sum_g phi_g = f(project) - mean f(background), and a group
the model ignores gets exactly 0.
"""

from collections.abc import Callable
from math import factorial

import numpy as np
import pandas as pd

Predict = Callable[[pd.DataFrame], dict[str, np.ndarray]]


def coalition_frame(x: pd.DataFrame, background: pd.DataFrame, groups: list[list[str]]):
    """Rows for every (coalition mask, background row): background values, overwritten with the
    project's values for the groups in the mask. Returns (frame, masks)."""
    n, B = len(groups), len(background)
    masks = np.arange(2**n)
    frame = pd.concat([background] * len(masks), ignore_index=True)
    mask_of_row = np.repeat(masks, B)
    for g, cols in enumerate(groups):
        use_x = (mask_of_row >> g) & 1 == 1
        for c in cols:
            frame.loc[use_x, c] = x[c].iloc[0]
    return frame, masks


def shapley_from_values(v: np.ndarray, n: int) -> np.ndarray:
    """v: value of each coalition, indexed by bit mask -> phi (n,)."""
    weights = [factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)]
    phi = np.zeros(n)
    for mask in range(2**n):
        size = bin(mask).count("1")
        for g in range(n):
            if not mask >> g & 1:
                phi[g] += weights[size] * (v[mask | (1 << g)] - v[mask])
    return phi


def explain(
    predict: Predict, x: pd.DataFrame, background: pd.DataFrame, groups: dict[str, list[str]]
) -> pd.DataFrame:
    """Shapley values per group and target for one project (x: one-row frame).

    Returns a frame indexed by group with one column per target, plus attrs
    'prediction' and 'baseline' (dicts by target) so the parts can be checked to add up.
    """
    names, cols = list(groups), list(groups.values())
    frame, masks = coalition_frame(x, background, cols)
    out = predict(frame)
    result = {}
    prediction, baseline = {}, {}
    for target, values in out.items():
        v = np.asarray(values, dtype=float).reshape(len(masks), len(background)).mean(axis=1)
        result[target] = shapley_from_values(v, len(cols))
        prediction[target], baseline[target] = float(v[-1]), float(v[0])
    table = pd.DataFrame(result, index=pd.Index(names, name="group"))
    table.attrs["prediction"], table.attrs["baseline"] = prediction, baseline
    return table
