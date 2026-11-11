"""Cause-specific random survival forests (scikit-survival), one per outcome.

scikit-survival's forest handles a single event type, so each outcome gets its own forest
with the other outcome treated as censored (the same idea as cause-specific Cox). Their
cumulative hazards are combined into CIFs with combine.py.
"""

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from queueiq.config import SEED
from queueiq.models.base import GRID, CompetingRisksModel
from queueiq.models.combine import cif_from_cumulative_hazards, step_on_grid
from queueiq.models.data import CAUSES
from queueiq.models.preprocess import Preprocessor


class CauseSpecificRSF(CompetingRisksModel):
    name = "rsf_cause_specific"

    def __init__(
        self,
        n_estimators: int = 300,
        min_samples_leaf: int = 20,
        max_features="sqrt",
        exclude: tuple[str, ...] = (),
    ):
        self.exclude = tuple(exclude)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def fit(self, X, time, event):
        self.prep_ = Preprocessor("tree", exclude=self.exclude).fit(X)
        Z = self.prep_.transform(X).to_numpy()
        self.forests_ = {}
        for code in CAUSES:
            y = Surv.from_arrays(event=event == code, time=time.astype(float))
            forest = RandomSurvivalForest(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                low_memory=False,
                n_jobs=-1,
                random_state=SEED,
            )
            self.forests_[code] = forest.fit(Z, y)
        return self

    def predict_cif(self, X):
        Z = self.prep_.transform(X).to_numpy()
        H = {}
        for code, forest in self.forests_.items():
            chf = forest.predict_cumulative_hazard_function(Z, return_array=True)
            H[code] = step_on_grid(forest.unique_times_, chf, GRID.astype(float))
        return cif_from_cumulative_hazards(H[1], H[2])

    def params(self):
        return {
            "n_estimators": self.n_estimators,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
        }
