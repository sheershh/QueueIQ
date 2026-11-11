"""Hazard-scaling recalibration (protocol section 7).

Any model's curves are turned back into monthly cause-specific hazards,
    h_k(t) = dF_k(t) / S(t-1),
each hazard is multiplied by a constant c_k, and the curves are rebuilt with the same
product-limit formula as everywhere else (so S + F_op + F_wd = 1 still holds). c_op and c_wd
are chosen so the model reproduces the observed 36-month rates of the most recent fully
observed cohort, which is known at the origin. The test period is never used.
"""

import numpy as np

from queueiq.models.combine import cif_from_cumulative_hazards

CAUSE_KEYS = ("operational", "withdrawn")


def hazards_from_cif(cif: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    F_op, F_wd = cif["operational"], cif["withdrawn"]
    surv = 1 - F_op - F_wd
    surv_prev = np.concatenate([np.ones((len(surv), 1)), surv[:, :-1]], axis=1)
    out = {}
    for k, F in (("operational", F_op), ("withdrawn", F_wd)):
        dF = np.diff(F, axis=1, prepend=0.0).clip(min=0)
        out[k] = np.where(surv_prev > 1e-12, dF / np.where(surv_prev > 1e-12, surv_prev, 1), 0.0)
    return out


def apply_scaling(cif: dict[str, np.ndarray], c: dict[str, float]) -> dict[str, np.ndarray]:
    h = hazards_from_cif(cif)
    H_op = np.cumsum(h["operational"] * c["operational"], axis=1)
    H_wd = np.cumsum(h["withdrawn"] * c["withdrawn"], axis=1)
    return cif_from_cumulative_hazards(H_op, H_wd)


def solve_scaling(
    cif_ref: dict[str, np.ndarray],
    observed: dict[str, float],
    tau: int = 36,
    rounds: int = 20,
    iters: int = 50,
    bounds: tuple[float, float] = (0.01, 100.0),
) -> dict[str, float]:
    """Constants c_k so that mean F_k(tau) over the reference cohort equals observed[k]."""
    lo_b, hi_b = np.log(bounds[0]), np.log(bounds[1])
    c = {k: 1.0 for k in CAUSE_KEYS}
    for _ in range(rounds):
        for k in CAUSE_KEYS:
            if observed[k] <= 0:
                c[k] = bounds[0]
                continue
            lo, hi = lo_b, hi_b
            for _ in range(iters):
                mid = (lo + hi) / 2
                trial = {**c, k: float(np.exp(mid))}
                mean_pred = apply_scaling(cif_ref, trial)[k][:, tau - 1].mean()
                lo, hi = (mid, hi) if mean_pred < observed[k] else (lo, mid)
            c[k] = float(np.exp((lo + hi) / 2))
    return c


def past_error_band(
    cif: dict[str, np.ndarray], multipliers: list[dict[str, float]]
) -> dict[str, np.ndarray]:
    """Pointwise min and max over curves redrawn with each past year's multipliers
    (decisions D-046; the multipliers come from evaluation/error_range.py).

    cif: {outcome: (n, months)}. Returns {f"{outcome}_low"/"_high": (n, months)}.
    """
    redrawn = [apply_scaling(cif, c) for c in multipliers]
    out = {}
    for k in CAUSE_KEYS:
        stack = np.stack([r[k] for r in redrawn])
        out[f"{k}_low"], out[f"{k}_high"] = stack.min(axis=0), stack.max(axis=0)
    return out
