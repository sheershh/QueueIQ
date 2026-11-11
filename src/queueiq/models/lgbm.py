"""Discrete-time competing-risks model with LightGBM (decisions D-030).

Each project becomes one row per 3-month interval it was observed in ("person-period" data).
The label for each interval is 0 = nothing happened, 1 = became operational, 2 = withdrew. A
multiclass model then predicts, for any interval j, the probability of each label given the
project features and j. Those probabilities are discrete hazards h_k(j), which are combined
into cumulative incidence just like Aalen-Johansen:

    CIF_k(j) = CIF_k(j-1) + S(j-1) * h_k(j),   S(j) = S(j-1) * (1 - h_1(j) - h_2(j))

Censoring convention: a project censored part-way through an interval contributes only the
intervals it was observed for in full.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from queueiq.config import SEED
from queueiq.models.base import GRID, CompetingRisksModel
from queueiq.models.combine import step_on_grid
from queueiq.models.preprocess import CATEGORICAL, Preprocessor

INTERVAL_MONTHS = 3
N_INTERVALS = int(GRID.max() // INTERVAL_MONTHS)  # 32 intervals = 96 months
PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def expand_person_period(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """One row per (project, interval). Returns columns: row (project index), interval, label.

    time is the month index (month m covers [m-1, m) months after entry).
    Event in month m -> intervals 1..ceil(m/3), label = cause in the last one.
    Censored in month m -> only intervals fully observed: 1..floor((m-1)/3), all label 0.
    """
    last = np.where(
        event > 0, np.ceil(time / INTERVAL_MONTHS), np.floor((time - 1) / INTERVAL_MONTHS)
    )
    last = np.minimum(last.astype(int), N_INTERVALS)
    rows = np.repeat(np.arange(len(time)), last)
    interval = np.concatenate([np.arange(1, n + 1) for n in last]) if len(time) else np.array([])
    label = np.zeros(len(rows), dtype=int)
    is_last = interval == last[rows]
    label[is_last] = event[rows][is_last]
    return pd.DataFrame({"row": rows, "interval": interval.astype(int), "label": label})


class DiscreteTimeLGBM(CompetingRisksModel):
    name = "lgbm_discrete_time"

    def __init__(self, num_rounds: int = 300, exclude: tuple[str, ...] = ()):
        self.num_rounds = num_rounds
        self.exclude = tuple(exclude)

    def _design(self, Z: pd.DataFrame, pp: pd.DataFrame) -> pd.DataFrame:
        X = Z.iloc[pp["row"].to_numpy()].reset_index(drop=True)
        X["interval"] = pp["interval"].to_numpy()
        return X

    @staticmethod
    def select_rounds(X_tr, t_tr, e_tr, X_va, t_va, e_va, max_rounds=2000, exclude=()) -> int:
        """Pick the number of boosting rounds by early stopping on a later validation cohort."""
        prep = Preprocessor("lgbm", exclude=exclude).fit(X_tr)
        m = DiscreteTimeLGBM()
        pp_tr, pp_va = expand_person_period(t_tr, e_tr), expand_person_period(t_va, e_va)
        train = lgb.Dataset(
            m._design(prep.transform(X_tr), pp_tr), pp_tr["label"], categorical_feature=CATEGORICAL
        )
        valid = lgb.Dataset(m._design(prep.transform(X_va), pp_va), pp_va["label"], reference=train)
        booster = lgb.train(
            PARAMS,
            train,
            num_boost_round=max_rounds,
            valid_sets=[valid],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return int(booster.best_iteration)

    def fit(self, X, time, event):
        self.prep_ = Preprocessor("lgbm", exclude=self.exclude).fit(X)
        pp = expand_person_period(time, event)
        data = lgb.Dataset(
            self._design(self.prep_.transform(X), pp), pp["label"], categorical_feature=CATEGORICAL
        )
        self.booster_ = lgb.train(PARAMS, data, num_boost_round=self.num_rounds)
        return self

    def predict_hazards(self, X) -> np.ndarray:
        """(n, N_INTERVALS, 3) class probabilities per interval."""
        Z = self.prep_.transform(X)
        n = len(Z)
        pp = pd.DataFrame(
            {
                "row": np.repeat(np.arange(n), N_INTERVALS),
                "interval": np.tile(np.arange(1, N_INTERVALS + 1), n),
            }
        )
        probs = self.booster_.predict(self._design(Z, pp))
        return probs.reshape(n, N_INTERVALS, 3)

    def predict_cif(self, X):
        p = self.predict_hazards(X)
        h_op, h_wd = p[:, :, 1], p[:, :, 2]
        surv_after = np.cumprod(1 - h_op - h_wd, axis=1)
        surv_before = np.concatenate([np.ones((len(p), 1)), surv_after[:, :-1]], axis=1)
        ends = np.arange(1, N_INTERVALS + 1) * INTERVAL_MONTHS
        return {
            "operational": step_on_grid(ends, np.cumsum(surv_before * h_op, axis=1), GRID),
            "withdrawn": step_on_grid(ends, np.cumsum(surv_before * h_wd, axis=1), GRID),
        }

    def params(self):
        return {
            "num_rounds": self.num_rounds,
            "interval_months": INTERVAL_MONTHS,
            "exclude": ",".join(self.exclude),
            **PARAMS,
        }
