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
