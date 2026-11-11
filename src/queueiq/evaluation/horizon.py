"""Metrics at a fixed horizon h, for test cohorts observed for at least h (decisions D-031).

With full follow-up there is no censoring before h, so "event k by h" is a plain yes/no
label for every test project, and standard binary metrics apply directly:
- AUC: chance a project that had event k by h is ranked above one that did not
- Brier score: mean squared error of the predicted probability
- mean predicted vs observed rate: calibration-in-the-large
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def horizon_labels(time: np.ndarray, event: np.ndarray, cause: int, h: int) -> np.ndarray:
    """1 if event `cause` happened in month <= h. Raises if any row lacks follow-up to h."""
    followed = ((event != 0) & (time <= h)) | (time > h)
    if not followed.all():
        raise ValueError(f"{(~followed).sum()} rows are censored before month {h}")
    return ((event == cause) & (time <= h)).astype(int)


def horizon_metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "observed_%": round(100 * y.mean(), 2),
        "mean_pred_%": round(100 * float(pred.mean()), 2),
        "auc": round(float(roc_auc_score(y, pred)), 3) if 0 < y.sum() < len(y) else np.nan,
        "brier": round(float(np.mean((pred - y) ** 2)), 4),
    }
