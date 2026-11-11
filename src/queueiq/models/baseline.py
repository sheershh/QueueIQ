"""Baseline: Aalen-Johansen curves by ISO x fuel, with fallbacks for small groups.

No features beyond region and fuel; every project in a group gets the same curve. Any model
worth using has to beat this.
"""

import numpy as np
import pandas as pd

from queueiq.eda.survival import aalen_johansen
from queueiq.models.base import GRID, CompetingRisksModel
from queueiq.models.combine import step_on_grid
from queueiq.models.data import CAUSES


class GroupAalenJohansen(CompetingRisksModel):
    name = "baseline_aj_iso_fuel"

    def __init__(self, min_group: int = 50):
        self.min_group = min_group

    def _curve(self, time, event) -> dict[str, np.ndarray]:
        outcome = pd.Series(event).map(CAUSES).fillna("censored")
        c = aalen_johansen(pd.Series(time, dtype=float), outcome)
        return {
            k: step_on_grid(c["time"].to_numpy(), c[f"cif_{k}"].to_numpy()[None, :], GRID)[0]
            for k in CAUSES.values()
        }

    def fit(self, X, time, event):
        keys = X["iso"].astype(str) + "|" + X["fuel_group"].astype(str)
        self.curves_: dict[str, dict] = {"all": self._curve(time, event)}
        for level, labels in (("pair", keys), ("fuel", X["fuel_group"].astype(str))):
            for value in labels.unique():
                mask = (labels == value).to_numpy()
                if mask.sum() >= self.min_group:
                    self.curves_[f"{level}:{value}"] = self._curve(time[mask], event[mask])
        return self

    def group_of(self, row) -> str:
        for key in (f"pair:{row['iso']}|{row['fuel_group']}", f"fuel:{row['fuel_group']}"):
            if key in self.curves_:
                return key
        return "all"

    def predict_cif(self, X):
        groups = [self.group_of(r) for _, r in X.iterrows()]
        return {k: np.vstack([self.curves_[g][k] for g in groups]) for k in CAUSES.values()}

    def params(self):
        return {"min_group": self.min_group, "n_curves": len(self.curves_)}


class PooledAalenJohansen(CompetingRisksModel):
    """Null model: one Aalen-Johansen curve for everyone, no features (protocol section 2)."""

    name = "null_aj"

    def fit(self, X, time, event):
        self.curve_ = GroupAalenJohansen()._curve(time, event)
        return self

    def predict_cif(self, X):
        return {k: np.tile(v, (len(X), 1)) for k, v in self.curve_.items()}


class RecentCohortAJ(CompetingRisksModel):
    """'Most recent fully observed cohort' baseline (protocol section 6).

    Uses only the training rows that entered in `ref_year` (the latest entry year whose
    `horizon`-month outcomes are fully known at the origin) and predicts that cohort's
    Aalen-Johansen curve by ISO, with an all-ISO fallback for ISOs below `min_group`.
    Values after `horizon` are held flat; only months <= horizon are scored.
    """

    name = "recent_cohort"

    def __init__(self, ref_year: int, horizon: int = 36, min_group: int = 30):
        self.ref_year = ref_year
        self.horizon = horizon
        self.min_group = min_group

    def fit(self, X, time, event):
        ref = (X["queue_year"] == self.ref_year).to_numpy()
        t, e = time[ref], event[ref]
        followed = ((e != 0) & (t <= self.horizon)) | (t > self.horizon)
        if ref.sum() == 0 or not followed.all():
            raise ValueError(
                f"reference cohort {self.ref_year} is not fully observed to {self.horizon} months"
            )
        aj = GroupAalenJohansen()
        self.curves_ = {"all": self._flat(aj._curve(t, e))}
        iso = X.loc[ref, "iso"].astype(str).to_numpy()
        for value in np.unique(iso):
            mask = iso == value
            if mask.sum() >= self.min_group:
                self.curves_[value] = self._flat(aj._curve(t[mask], e[mask]))
        self.n_ref_ = int(ref.sum())
        return self

    def _flat(self, curves: dict) -> dict:
        out = {}
        for k, v in curves.items():
            v = v.copy()
            v[self.horizon :] = v[self.horizon - 1]
            out[k] = v
        return out

    def predict_cif(self, X):
        keys = [i if i in self.curves_ else "all" for i in X["iso"].astype(str)]
        return {k: np.vstack([self.curves_[g][k] for g in keys]) for k in CAUSES.values()}

    def params(self):
        return {"ref_year": self.ref_year, "horizon": self.horizon, "min_group": self.min_group}
