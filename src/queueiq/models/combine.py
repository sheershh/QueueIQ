"""Turn two cause-specific cumulative hazards into cumulative incidence (CIF) curves.

Same product-limit logic as the Aalen-Johansen estimator (eda/survival.py), but per project:
at each month t with hazard increments dH_op(t), dH_wd(t):

    CIF_k(t) = CIF_k(t-1) + S(t-1) * dH_k(t)
    S(t)     = S(t-1) * (1 - dH_op(t) - dH_wd(t))

So S + CIF_op + CIF_wd = 1 at every t. Increments are capped so a month's total hazard
never exceeds 1.
"""

import numpy as np


def cif_from_cumulative_hazards(H_op: np.ndarray, H_wd: np.ndarray) -> dict[str, np.ndarray]:
    """H_*: (n_projects, n_months) cumulative hazards on a common monthly grid 1..T."""
    dH_op = np.diff(H_op, axis=1, prepend=0.0).clip(min=0)
    dH_wd = np.diff(H_wd, axis=1, prepend=0.0).clip(min=0)
    total = dH_op + dH_wd
    scale = np.where(total > 1, 1 / np.where(total > 0, total, 1), 1.0)
    dH_op, dH_wd = dH_op * scale, dH_wd * scale

    surv_after = np.cumprod(1 - dH_op - dH_wd, axis=1)
    surv_before = np.concatenate([np.ones((len(H_op), 1)), surv_after[:, :-1]], axis=1)
    return {
        "operational": np.cumsum(surv_before * dH_op, axis=1),
        "withdrawn": np.cumsum(surv_before * dH_wd, axis=1),
    }


def step_on_grid(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Right-continuous step function evaluated on `grid`; 0 before the first time.

    values: (n, len(times)) -> returns (n, len(grid)).
    """
    idx = np.searchsorted(times, grid, side="right") - 1
    out = np.zeros((values.shape[0], len(grid)))
    ok = idx >= 0
    out[:, ok] = values[:, idx[ok]]
    return out
