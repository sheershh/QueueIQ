"""Metric checks against hand-worked examples and brute-force pair counting."""

import itertools

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from queueiq.evaluation import bootstrap as bs
from queueiq.evaluation.metrics import (
    assert_full_followup,
    auc_pairs,
    calibration_bins,
    cindex_pairs,
    ece,
    ibs_per_project,
)
from queueiq.evaluation.recalibrate import apply_scaling, hazards_from_cif, solve_scaling

# Five projects, horizon tau = 3 months. event 1 = operational, 2 = withdrawn, 0 = still waiting.
TIME = np.array([1, 2, 2, 3, 9])
EVENT = np.array([1, 2, 1, 2, 0])
RISK = np.array([0.9, 0.2, 0.5, 0.4, 0.1])  # predicted F_operational(3)


def test_cindex_hand_example():
    """Cases (operational by month 3): A (t=1) and C (t=2).
    A is comparable with everyone free of events at t=1 (B, C, D, E) and with the withdrawn
    projects (B, D): {B, C, D, E}. A's risk .9 beats all four -> 4 concordant.
    C (t=2) is comparable with D, E (t > 2) and with withdrawn B, D: {B, D, E}.
    C's .5 beats .2, .4, .1 -> 3 concordant. C = 7/7 = 1.0."""
    p = cindex_pairs(TIME, EVENT, 1, RISK, tau=3)
    assert p.n_cases == 2 and p.n_pairs == 7
    assert p.value() == 1.0
    worse = RISK.copy()
    worse[2] = 0.3  # C now loses to D (0.4): 6 of 7
    assert cindex_pairs(TIME, EVENT, 1, worse, 3).value() == pytest.approx(6 / 7)


def brute_cindex(time, event, cause, risk, tau):
    num = den = 0.0
    for i, j in itertools.permutations(range(len(time)), 2):
        is_case = event[i] == cause and time[i] <= tau
        if is_case and (time[j] > time[i] or event[j] not in (0, cause)):
            den += 1
            num += 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
    return num / den


def test_cindex_matches_brute_force_on_random_data():
    rng = np.random.default_rng(0)
    time = rng.integers(1, 60, 120)
    event = rng.choice([0, 1, 2], 120)
    time[event == 0] = 100  # censored rows are followed past tau
    risk = np.round(rng.random(120), 1)  # many ties
    for cause in (1, 2):
        got = cindex_pairs(time, event, cause, risk, tau=36).value()
        assert got == pytest.approx(brute_cindex(time, event, cause, risk, 36))


def test_auc_equals_sklearn_when_unweighted():
    rng = np.random.default_rng(1)
    time, event = rng.integers(1, 60, 200), rng.choice([0, 1, 2], 200)
    time[event == 0] = 100
    risk = rng.random(200)
    y = ((event == 2) & (time <= 36)).astype(int)
    assert auc_pairs(time, event, 2, risk, 36).value() == pytest.approx(roc_auc_score(y, risk))


def test_weights_act_like_duplicated_rows():
    w = np.array([[2, 1, 1, 1, 1]], dtype=np.float32)  # project A drawn twice
    dup = np.r_[0, np.arange(5)]
    p_w = cindex_pairs(TIME, EVENT, 1, np.array([0.9, 0.2, 0.3, 0.4, 0.1]), 3).value(w)[0]
    p_dup = cindex_pairs(TIME[dup], EVENT[dup], 1, np.array([0.9, 0.2, 0.3, 0.4, 0.1])[dup], 3)
    assert p_w == pytest.approx(p_dup.value())


def test_ibs_hand_example():
    """One project, withdrawn in month 2, constant prediction 0.5 over tau=3:
    month 1: (0 - .5)^2 = .25, month 2: (1 - .5)^2 = .25, month 3: .25 -> IBS .25."""
    cif = np.full((1, 96), 0.5)
    assert ibs_per_project(np.array([2]), np.array([2]), 2, cif, 3)[0] == pytest.approx(0.25)
    perfect = np.zeros((1, 96))
    perfect[0, 1:] = 1.0
    assert ibs_per_project(np.array([2]), np.array([2]), 2, perfect, 3)[0] == 0.0


def test_full_followup_guard():
    with pytest.raises(ValueError, match="follow-up"):
        assert_full_followup(np.array([5, 20]), np.array([0, 1]), tau=36)


def test_calibration_bins_and_ece():
    pred = np.array([0.1, 0.1, 0.9, 0.9])
    obs = np.array([0, 0, 1, 0])
    bins = calibration_bins(pred, obs, n_bins=2)
    assert bins == [(0.1, 0.0, 2), (0.9, 0.5, 2)]
    assert ece(bins) == pytest.approx((2 * 0.1 + 2 * 0.4) / 4)


def test_project_bootstrap_counts_sum_to_n():
    W = bs.project_weights(50, B=200, seed=0)
    assert W.shape == (200, 50) and (W.sum(axis=1) == 50).all()


def test_group_bootstrap_moves_whole_groups_together():
    groups = np.array(["a", "a", "b", "b", "b", "c"])
    W = bs.group_weights(groups, B=300, seed=0)
    assert (W[:, 0] == W[:, 1]).all() and (W[:, 2] == W[:, 4]).all()
    assert set(np.unique(W[:, 5])) <= {0, 1, 2, 3}  # c drawn 0..3 times out of 3 group draws


def test_interval_ignores_nan():
    assert bs.interval(np.array([np.nan, *range(101)])) == (2.5, 97.5)


def curves(n=4, seed=0):
    rng = np.random.default_rng(seed)
    h_op, h_wd = rng.uniform(0, 0.01, (n, 96)), rng.uniform(0, 0.03, (n, 96))
    from queueiq.models.combine import cif_from_cumulative_hazards

    return cif_from_cumulative_hazards(np.cumsum(h_op, 1), np.cumsum(h_wd, 1)), h_op, h_wd


def test_hazards_round_trip():
    cif, h_op, _ = curves()
    np.testing.assert_allclose(hazards_from_cif(cif)["operational"], h_op, atol=1e-12)


def test_scaling_by_one_changes_nothing_and_keeps_shares_valid():
    cif, _, _ = curves()
    same = apply_scaling(cif, {"operational": 1.0, "withdrawn": 1.0})
    np.testing.assert_allclose(same["withdrawn"], cif["withdrawn"], atol=1e-12)
    big = apply_scaling(cif, {"operational": 50.0, "withdrawn": 50.0})
    assert (big["operational"] + big["withdrawn"] <= 1 + 1e-9).all()


def test_solver_hits_the_target_rates():
    cif, _, _ = curves(n=40, seed=3)
    target = {"operational": 0.05, "withdrawn": 0.40}
    c = solve_scaling(cif, target, tau=36)
    fitted = apply_scaling(cif, c)
    assert fitted["operational"][:, 35].mean() == pytest.approx(0.05, abs=1e-4)
    assert fitted["withdrawn"][:, 35].mean() == pytest.approx(0.40, abs=1e-4)
