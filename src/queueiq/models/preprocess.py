"""Feature preprocessing, fitted on training rows only (decisions D-026, D-028).

Three output modes:
- "linear" (Cox): log1p on counts/MW, missing values -> training median + a 0/1 indicator,
  standardize with training mean/std, one-hot categories with the first level as reference.
- "tree" (random survival forest): missing -> median + indicator, all one-hot levels,
  no scaling (trees don't care about scale).
- "lgbm" (LightGBM): categories as pandas categoricals; missing values left as NaN
  (LightGBM handles them natively).
In every mode, states with fewer than `min_state_rows` training rows become "other", and
categories never seen in training become "other" (or NaN for LightGBM).
"""

import numpy as np
import pandas as pd

CATEGORICAL = ["fuel_group", "iso", "state"]
BINARY = ["is_hybrid", "has_storage", "county_known"]
SKEWED = [
    "county_prior_requests_5y",
    "county_prior_mw_5y",
    "county_prior_withdrawn_5y",
    "state_prior_requests_5y",
    "state_prior_mw_5y",
    "state_prior_withdrawn_5y",
    "state_prior_requests_1y",
]
NUMERIC = [
    "log_capacity_mw",
    "queue_year",
    *SKEWED,
    "county_prior_withdrawn_share_5y",
    "state_prior_withdrawn_share_5y",
]
OTHER = "other"


def redundant_columns(df: pd.DataFrame) -> list[str]:
    """Columns a linear model can't separate from others: constants, and 0/1 columns that
    duplicate or mirror (1 - x) an earlier 0/1 column. E.g. every county 'missing' indicator
    equals 1 - county_known. Left in, they make the Cox fit singular."""
    drop, seen = [], []
    for c in df.columns:
        col = df[c].to_numpy()
        if np.all(col == col[0]):
            drop.append(c)
            continue
        if set(np.unique(col)) <= {0.0, 1.0}:
            if any(np.array_equal(col, s) or np.array_equal(col, 1 - s) for s in seen):
                drop.append(c)
                continue
            seen.append(col)
    return drop


class Preprocessor:
    def __init__(self, mode: str, min_state_rows: int = 50):
        if mode not in ("linear", "tree", "lgbm"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.min_state_rows = min_state_rows

    def _categories(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[CATEGORICAL].astype("string").fillna(OTHER)
        out["state"] = out["state"].where(out["state"].isin(self.states_), OTHER)
        for c in CATEGORICAL:
            out[c] = out[c].where(out[c].isin(self.levels_[c]), OTHER)
        return out

    def _numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        num = X[NUMERIC].astype(float).copy()
        if self.mode == "linear":
            num[SKEWED] = np.log1p(num[SKEWED].clip(lower=0))
        return num

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        counts = X["state"].astype("string").value_counts()
        self.states_ = set(counts[counts >= self.min_state_rows].index)
        cats = X[CATEGORICAL].astype("string").fillna(OTHER)
        cats["state"] = cats["state"].where(cats["state"].isin(self.states_), OTHER)
        self.levels_ = {c: sorted(cats[c].unique()) for c in CATEGORICAL}
        for c in CATEGORICAL:
            if OTHER not in self.levels_[c]:
                self.levels_[c].append(OTHER)

        num = self._numeric(X)
        self.medians_ = num.median()
        self.with_missing_ = [c for c in NUMERIC if num[c].isna().any()]
        if self.mode == "linear":
            filled = num.fillna(self.medians_)
            self.means_ = filled.mean()
            self.stds_ = filled.std().replace(0, 1.0)
        self.columns_ = None
        full = self.transform(X)
        self.dropped_ = redundant_columns(full) if self.mode == "linear" else []
        self.columns_ = [c for c in full.columns if c not in self.dropped_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cats = self._categories(X)
        num = self._numeric(X)
        binary = X[BINARY].astype(float)

        if self.mode == "lgbm":
            out = pd.concat([num, binary], axis=1)
            for c in CATEGORICAL:
                out[c] = pd.Categorical(cats[c], categories=self.levels_[c])
            return out

        indicators = pd.DataFrame(
            {f"{c}_missing": num[c].isna().astype(float) for c in self.with_missing_},
            index=X.index,
        )
        num = num.fillna(self.medians_)
        if self.mode == "linear":
            num = (num - self.means_) / self.stds_
        dummies = []
        for c in CATEGORICAL:
            levels = self.levels_[c][1:] if self.mode == "linear" else self.levels_[c]
            for lv in levels:
                dummies.append((cats[c] == lv).astype(float).rename(f"{c}={lv}"))
        out = pd.concat([num, binary, indicators, *dummies], axis=1)
        if self.columns_ is not None:
            out = out.reindex(columns=self.columns_, fill_value=0.0)
        return out
