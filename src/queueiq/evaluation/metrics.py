"""Competing-risks metrics exactly as defined in docs/evaluation_protocol.md, section 4.

Every evaluation cohort has full follow-up to the horizon tau, so the inverse-probability-of-
censoring weights these metrics normally need are all 1. `assert_full_followup` enforces it.

Inputs: `time` is the month index (D-027), `event` is 0/1/2, `cif` is an (n, 96) curve where
column t-1 is the cumulative incidence at the end of month t. Optional `w` are bootstrap
weights (resample counts): a weight of 2 means the project was drawn twice.
"""

import numpy as np

MIN_CASES = 20  # below this, AUC and C-index are "not estimable" (protocol section 4)


def assert_full_followup(time: np.ndarray, event: np.ndarray, tau: int) -> None:
    followed = ((event != 0) & (time <= tau)) | (time > tau)
    if not followed.all():
        raise ValueError(f"{(~followed).sum()} rows lack follow-up to month {tau}")


def event_by(time: np.ndarray, event: np.ndarray, cause: int, t: int) -> np.ndarray:
    return ((event == cause) & (time <= t)).astype(float)


def brier_by_month(time, event, cause, cif_k: np.ndarray, tau: int) -> np.ndarray:
    """(n, tau): squared error at the end of each month 1..tau."""
    months = np.arange(1, tau + 1)
    observed = ((event[:, None] == cause) & (time[:, None] <= months[None, :])).astype(float)
    return (observed - cif_k[:, :tau]) ** 2


def ibs_per_project(time, event, cause, cif_k, tau) -> np.ndarray:
    """Each project's contribution to the integrated Brier score (mean over months 1..tau)."""
    return brier_by_month(time, event, cause, cif_k, tau).mean(axis=1)


def weighted_mean(x: np.ndarray, w: np.ndarray | None) -> float:
    return float(x.mean()) if w is None else float((x * w).sum() / w.sum())


class Pairwise:
    """Weighted pair-counting for AUC and C-index.

    For each case i and each project j: comparable[i, j] says whether (i, j) is a valid pair;
    the pair scores 1 if case i's risk is higher, 1/2 if tied, 0 otherwise. With weights w,
        value = sum_i sum_j w_i w_j comparable_ij score_ij / sum_i sum_j w_i w_j comparable_ij
    Precomputing the two matrices makes each bootstrap resample a matrix-vector product.
    """

    def __init__(self, risk: np.ndarray, cases: np.ndarray, comparable: np.ndarray):
        self.case_idx = np.flatnonzero(cases)
        rc = risk[self.case_idx][:, None]
        score = (rc > risk[None, :]) + 0.5 * (rc == risk[None, :])
        self.den = comparable.astype(np.float32)
        self.num = (comparable * score).astype(np.float32)

    @property
    def n_cases(self) -> int:
        return len(self.case_idx)

    @property
    def n_pairs(self) -> int:
        return int(self.den.sum())

    def value(self, W: np.ndarray | None = None) -> np.ndarray | float:
        """W: None (no resampling) or (B, n) weights -> float or (B,) values."""
        if W is None:
            return float(self.num.sum() / self.den.sum()) if self.den.sum() else np.nan
        W = W.astype(np.float32)
        wc = W[:, self.case_idx]  # (B, cases)
        num = np.einsum("bc,cb->b", wc, self.num @ W.T)
        den = np.einsum("bc,cb->b", wc, self.den @ W.T)
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def auc_pairs(time, event, cause, risk, tau) -> Pairwise:
    """Cumulative/dynamic AUC(tau): cases = event `cause` by tau; controls = everyone else."""
    cases = event_by(time, event, cause, tau).astype(bool)
    comparable = np.broadcast_to(~cases[None, :], (cases.sum(), len(cases)))
    return Pairwise(risk, cases, comparable)


def cindex_pairs(time, event, cause, risk, tau) -> Pairwise:
    """Truncated competing-risks C-index (Wolbers): case i has `cause` at T_i <= tau; j is
    comparable if T_j > T_i (still free of any event at T_i) or j had the other cause."""
    cases = event_by(time, event, cause, tau).astype(bool)
    other = (event != 0) & (event != cause)
    ti = time[cases][:, None]
    comparable = (time[None, :] > ti) | other[None, :]
    return Pairwise(risk, cases, comparable)


def calibration_bins(pred: np.ndarray, observed: np.ndarray, n_bins: int = 10):
    """Equal-count bins of the prediction: (mean predicted, observed share, size) per bin."""
    order = np.argsort(pred, kind="stable")
    bins = [b for b in np.array_split(order, n_bins) if len(b)]
    return [(float(pred[b].mean()), float(observed[b].mean()), len(b)) for b in bins]


def ece(bins) -> float:
    total = sum(n for _, _, n in bins)
    return float(sum(n * abs(p - o) for p, o, n in bins) / total)
