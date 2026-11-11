"""Bootstrap resampling weights (protocol section 5).

Instead of copying data, each resample is a vector of counts: how many times each project was
drawn. Resampling *groups* draws whole groups (e.g. all projects of one county and entry year)
and gives every project the count of its group. Projects that share a place and year also
share shocks, so group resampling gives wider, more honest intervals when outcomes cluster.
"""

import numpy as np
import pandas as pd

B_DEFAULT = 1000


def project_weights(n: int, B: int = B_DEFAULT, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1 / n), size=B).astype(np.float32)


def group_weights(groups, B: int = B_DEFAULT, seed: int = 42) -> np.ndarray:
    codes, uniques = pd.factorize(pd.Series(groups), sort=True)
    G = len(uniques)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, G, size=(B, G))
    counts = np.stack([np.bincount(row, minlength=G) for row in draws])
    return counts[:, codes].astype(np.float32)


def interval(values: np.ndarray) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return (float(lo), float(hi))
