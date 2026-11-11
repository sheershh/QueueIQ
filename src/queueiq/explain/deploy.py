"""The deployment model: the primary configuration refit on all in-scope data (decisions D-039).

Phase 5 evaluated the *procedure* (fit on the past, predict the future). The deployed model uses
the same procedure with everything known at the data cutoff: all in-scope entrants, outcomes as
of 2025-12-31. Its own accuracy can't be measured until new cohorts mature; the Phase 5 results
are the evidence for it.
"""

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from queueiq import config
from queueiq.explain.groups import TARGETS
from queueiq.models.cox import CauseSpecificCox
from queueiq.models.data import MODEL_FEATURES, make_split
from queueiq.models.rsf import CauseSpecificRSF

DEPLOY_ORIGIN = pd.Timestamp("2026-01-01")  # day after the cutoff: every dated event counts
DEPLOY_DIR = config.PROJECT_ROOT / "models"
MODEL_FILE = DEPLOY_DIR / "deploy_primary.joblib"
COX_FILE = DEPLOY_DIR / "deploy_cox_reference.joblib"
BACKGROUND_FILE = DEPLOY_DIR / "deploy_background.parquet"
META_FILE = DEPLOY_DIR / "deploy_meta.json"
BACKGROUND_YEARS = (2021, 2025)  # "typical recent project" for explanations
BACKGROUND_SIZE = 64


def primary_model(selection: dict):
    """Instantiate the frozen primary configuration (only the RSF / Cox families are mapped)."""
    name = selection["primary_configuration"]
    fam, *parts = name.split("|")
    opts = dict(p.split("=") for p in parts)
    exclude = ("queue_year",) if opts.get("qy") == "0" else ()
    if opts.get("recal") == "1":
        raise NotImplementedError("deploying a recalibrated configuration is not implemented")
    if fam == "rsf_cause_specific":
        return CauseSpecificRSF(exclude=exclude)
    if fam == "cox_cause_specific":
        return CauseSpecificCox(exclude=exclude)
    raise NotImplementedError(f"no deployment mapping for {fam}")


def deployment_data(entries: pd.DataFrame, features: pd.DataFrame):
    """(X, time, event, project_ids) for every in-scope project, outcomes as of the cutoff."""
    split = make_split(entries, features, DEPLOY_ORIGIN, (DEPLOY_ORIGIN.year, DEPLOY_ORIGIN.year))
    return split.X_train, split.time_train, split.event_train, split.train_ids


def background_sample(X: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    recent = X[X["queue_year"].between(*BACKGROUND_YEARS)]
    return recent.sample(BACKGROUND_SIZE, random_state=seed).reset_index(drop=True)


def predictor(model):
    """Model -> function returning the dashboard's headline probabilities."""

    def predict(X: pd.DataFrame) -> dict[str, np.ndarray]:
        cif = model.predict_cif(X)
        return {t: cif[k][:, h - 1] for t, (k, h, _) in TARGETS.items()}

    return predict


@dataclass
class Deployment:
    model: object
    background: pd.DataFrame
    meta: dict


def load() -> Deployment:
    return Deployment(
        model=joblib.load(MODEL_FILE),
        background=pd.read_parquet(BACKGROUND_FILE),
        meta=json.loads(META_FILE.read_text()),
    )


__all__ = ["MODEL_FEATURES", "Deployment", "load", "predictor", "primary_model"]
