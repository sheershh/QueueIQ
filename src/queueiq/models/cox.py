"""Cause-specific Cox models: one per outcome, the other outcome treated as censored.

Each model gives a cumulative hazard per project; combine.py turns the pair into CIFs.
A ridge penalty keeps the many correlated congestion features stable.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from queueiq.models.base import GRID, CompetingRisksModel
from queueiq.models.combine import cif_from_cumulative_hazards
from queueiq.models.data import CAUSES
from queueiq.models.preprocess import Preprocessor


class CauseSpecificCox(CompetingRisksModel):
    name = "cox_cause_specific"

    def __init__(self, penalizer: float = 0.1):
        self.penalizer = penalizer

    def fit(self, X, time, event):
        self.prep_ = Preprocessor("linear").fit(X)
        Z = self.prep_.transform(X)
        self.models_: dict[int, CoxPHFitter] = {}
        self._train = {}
        for code in CAUSES:
            df = Z.assign(T=time, E=(event == code).astype(int))
            cph = CoxPHFitter(penalizer=self.penalizer).fit(df, duration_col="T", event_col="E")
            self.models_[code] = cph
            self._train[code] = df
        return self

    def predict_cif(self, X):
        Z = self.prep_.transform(X)
        H = {
            code: cph.predict_cumulative_hazard(Z, times=GRID).to_numpy().T
            for code, cph in self.models_.items()
        }
        return cif_from_cumulative_hazards(H[1], H[2])

    def ph_test(self) -> pd.DataFrame:
        """Proportional-hazards test (rank time transform) for every covariate and cause."""
        rows = []
        for code, cph in self.models_.items():
            res = proportional_hazard_test(cph, self._train[code], time_transform="rank")
            s = res.summary.reset_index().rename(columns={"index": "covariate"})
            s["cause"] = CAUSES[code]
            rows.append(s[["cause", "covariate", "test_statistic", "p"]])
        return pd.concat(rows, ignore_index=True)

    def hazard_ratios(self) -> pd.DataFrame:
        return pd.concat(
            {CAUSES[c]: np.exp(m.params_) for c, m in self.models_.items()}, axis=1
        ).rename_axis("covariate")

    def params(self):
        return {"penalizer": self.penalizer}
