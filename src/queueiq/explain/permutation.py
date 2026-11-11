"""Grouped permutation importance on a held-out test cohort (decisions D-041).

For each feature group, shuffle that group's columns together across test projects (breaking
their link to the outcome but keeping their own joint pattern), re-predict, and measure how
much the metric drops. A big drop means the model relies on that group out of sample.
Report only: nothing here changes any model.
"""

import numpy as np
import pandas as pd

from queueiq.evaluation.metrics import MIN_CASES, auc_pairs, event_by, ibs_per_project

CAUSES = {"operational": 1, "withdrawn": 2}
TAU = 36


def scores(cif, null_cif, time, event) -> dict[str, float]:
    out = {}
    for k, code in CAUSES.items():
        ibs = ibs_per_project(time, event, code, cif[k], TAU).mean()
        null = ibs_per_project(time, event, code, null_cif[k], TAU).mean()
        out[f"{k}_skill"] = 1 - ibs / null
        cases = event_by(time, event, code, TAU).sum()
        out[f"{k}_auc"] = (
            auc_pairs(time, event, code, cif[k][:, TAU - 1], TAU).value()
            if cases >= MIN_CASES
            else np.nan
        )
    return out


def grouped_permutation_importance(
    model,
    X: pd.DataFrame,
    time,
    event,
    null_cif,
    groups: dict[str, list[str]],
    repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Mean drop (base - permuted) per group and metric over `repeats` shuffles."""
    rng = np.random.default_rng(seed)
    base = scores(model.predict_cif(X), null_cif, time, event)
    rows = []
    for group, cols in groups.items():
        drops = []
        for _ in range(repeats):
            perm = rng.permutation(len(X))
            Xp = X.copy()
            Xp[cols] = X[cols].iloc[perm].to_numpy()
            s = scores(model.predict_cif(Xp), null_cif, time, event)
            drops.append({m: base[m] - s[m] for m in base})
        d = pd.DataFrame(drops)
        rows.append({"group": group, **d.mean().to_dict()})
    return pd.DataFrame(rows).set_index("group")
