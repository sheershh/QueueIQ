"""The deployment model: the primary configuration refit on all in-scope data (decisions D-039).

Phase 5 evaluated the *procedure* (fit on the past, predict the future). The deployed model uses
the same procedure with everything known at the data cutoff: all in-scope entrants, outcomes as
of 2025-12-31. Its own accuracy can't be measured until new cohorts mature; the Phase 5 results
are the evidence for it.
"""

import json
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from queueiq import config
from queueiq.explain.groups import TARGETS
from queueiq.explain.reference import (
    calibration_warning,
    calibration_warning_values,
    observed_reference,
    signal_strength,
)
from queueiq.models.cox import CauseSpecificCox
from queueiq.models.data import MODEL_FEATURES, make_split
from queueiq.models.rsf import CauseSpecificRSF
from queueiq.service.history import comparable_table

DEPLOY_ORIGIN = pd.Timestamp("2026-01-01")  # day after the cutoff: every dated event counts
DEPLOY_DIR = config.PROJECT_ROOT / "models"
MODEL_FILE = DEPLOY_DIR / "deploy_primary.joblib"
COX_FILE = DEPLOY_DIR / "deploy_cox_reference.joblib"
BACKGROUND_FILE = DEPLOY_DIR / "deploy_background.parquet"
META_FILE = DEPLOY_DIR / "deploy_meta.json"
RANGE_FILE = config.REPORTS_DIR / "past_error_range.json"  # from `make error-range` (committed)
METRICS_FILE = config.REPORTS_DIR / "evaluation_metrics.csv"  # Phase 5 results (committed)
BACKGROUND_YEARS = (2021, 2025)
HISTORY_HORIZONS = (36, 60)  # months; the dashboard shows "within 3 years" and "within 5 years"
PERCENTILE_YEARS = (2021, 2025)  # "recent entrants" that percentiles compare against
PERCENTILE_TARGETS = {
    "withdrawn_3y": ("withdrawn", 36),
    "operational_3y": ("operational", 36),
}  # "typical recent project" for explanations
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


def build(entries: pd.DataFrame, features: pd.DataFrame, selection: dict) -> dict:
    """Fit and save the deployment model, the Cox reference, the explanation background and the
    metadata the dashboard shows. Needs only the processed data and committed reports."""
    primary = selection["primary_configuration"]
    X, t, ev, ids = deployment_data(entries, features)
    start = time.perf_counter()
    model = primary_model(selection).fit(X, t, ev)
    fit_s = round(time.perf_counter() - start, 1)
    cox = CauseSpecificCox(exclude=model.exclude).fit(X, t, ev)
    background = background_sample(X)
    DEPLOY_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE, compress=3)
    joblib.dump(cox, COX_FILE, compress=3)
    background.to_parquet(BACKGROUND_FILE, index=False)
    metrics = pd.read_csv(METRICS_FILE)
    cal = calibration_warning_values(metrics, primary)
    past = json.loads(RANGE_FILE.read_text())
    if past["primary_configuration"] != primary:
        raise ValueError("past_error_range.json was built for a different primary configuration")
    meta = {
        "primary_configuration": primary,
        "trained_on_rows": len(X),
        "events": {
            "censored": int((ev == 0).sum()),
            "operational": int((ev == 1).sum()),
            "withdrawn": int((ev == 2).sum()),
        },
        "outcomes_as_of": str(config.DATA_CUTOFF),
        "entry_date": str(DEPLOY_ORIGIN.date()),
        "background_years": list(BACKGROUND_YEARS),
        "background_size": len(background),
        "observed_reference": observed_reference(entries, pd.Timestamp(config.DATA_CUTOFF)),
        "phase5_final_test_calibration": cal,
        "calibration_warning": calibration_warning(cal),
        "past_error_range": past,
        "percentile_reference": percentile_reference(model, X),
        "signal_strength": signal_strength(metrics, primary),
        "comparable_outcomes": {
            str(h): comparable_table(entries, pd.Timestamp(config.DATA_CUTOFF), horizon=h)
            for h in HISTORY_HORIZONS
        },
    }
    META_FILE.write_text(json.dumps(meta, indent=2) + "\n")
    return {
        "model": model,
        "cox": cox,
        "background": background,
        "meta": meta,
        "X": X,
        "ids": ids,
        "fit_seconds": fit_s,
    }


def percentile_reference(model, X: pd.DataFrame) -> dict:
    """101 quantiles (0th to 100th percentile) of the model's 3-year predictions over recent
    in-scope entrants. A new project is placed within them (Service.percentiles)."""
    recent = X[X["queue_year"].between(*PERCENTILE_YEARS)]
    cif = model.predict_cif(recent)
    qs = np.linspace(0, 1, 101)
    return {
        "years": list(PERCENTILE_YEARS),
        "n": len(recent),
        "quantiles": {
            t: [float(v) for v in np.quantile(cif[k][:, h - 1], qs)]
            for t, (k, h) in PERCENTILE_TARGETS.items()
        },
    }


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
