"""`make train`: fit all models on one time-based split and compare them on the test cohorts.

Split (decisions D-027): train on 2010-2017 entrants with outcomes cut off at 2018-01-01;
test on 2018-2020 entrants, each observed for at least 5 years. No model setting is chosen
using the test cohorts. Every run is logged to a local MLflow store (mlruns/).
"""

import logging
import time as clock

import joblib
import mlflow
import numpy as np
import pandas as pd

from queueiq import config
from queueiq.cleaning.report import to_markdown
from queueiq.evaluation.horizon import horizon_labels, horizon_metrics
from queueiq.models.base import GRID
from queueiq.models.baseline import GroupAalenJohansen
from queueiq.models.cox import CauseSpecificCox
from queueiq.models.data import CAUSES, Split, make_split
from queueiq.models.lgbm import DiscreteTimeLGBM
from queueiq.models.rsf import CauseSpecificRSF

ORIGIN = pd.Timestamp("2018-01-01")
TEST_YEARS = (2018, 2020)
INNER_ORIGIN = pd.Timestamp("2016-01-01")  # inner split inside training, for LightGBM rounds
INNER_VALID_YEARS = (2016, 2017)
HORIZONS = (36, 60)

ENTRIES_FILE = config.PROCESSED_DIR / "queue_entries.parquet"
FEATURES_FILE = config.PROCESSED_DIR / "features.parquet"
MODELS_DIR = config.PROJECT_ROOT / "models"
MLRUNS_DIR = config.PROJECT_ROOT / "mlruns"
REPORT_FILE = config.REPORTS_DIR / "model_comparison.md"
TABLE_FILE = config.REPORTS_DIR / "model_comparison.csv"
EXPERIMENT = "queueiq-phase4"


def evaluate(model, split: Split) -> list[dict]:
    cif = model.predict_cif(split.X_test)
    rows = []
    for code, cause in CAUSES.items():
        for h in HORIZONS:
            y = horizon_labels(split.time_test, split.event_test, code, h)
            pred = cif[cause][:, np.searchsorted(GRID, h)]
            rows.append(
                {
                    "model": model.name,
                    "event": cause,
                    "horizon_years": h // 12,
                    **horizon_metrics(pred, y),
                }
            )
    return rows


def transfer_diagnostics(models, split: Split, h: int = 36) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Does what a model learned on the training years carry over to the test years?

    - AUC on the training cohorts themselves (in-sample, so optimistic) vs on the test cohorts.
      A big drop means the pattern changed over time, not just that the model overfits.
    - Observed 3-year rates by ISO in the training cohorts vs the test cohorts.
    Diagnostics only: nothing here is used to change any model.
    """
    from sklearn.metrics import roc_auc_score

    # Whole entry years with >= h months before the origin. Filtering row by row ("followed to
    # h") would keep recent projects only if they had an event early: selection on the outcome.
    ok = (split.X_train["queue_year"] + h // 12 < split.origin.year).to_numpy()
    X_tr, t_tr, e_tr = split.X_train[ok], split.time_train[ok], split.event_train[ok]
    rows = []
    for model in models:
        cif_tr, cif_te = model.predict_cif(X_tr), model.predict_cif(split.X_test)
        for code, cause in CAUSES.items():
            y_tr = horizon_labels(t_tr, e_tr, code, h)
            y_te = horizon_labels(split.time_test, split.event_test, code, h)
            col = np.searchsorted(GRID, h)
            rows.append(
                {
                    "model": model.name,
                    "event": cause,
                    "auc_train_cohorts_in_sample": round(
                        roc_auc_score(y_tr, cif_tr[cause][:, col]), 3
                    ),
                    "auc_test_cohorts": round(roc_auc_score(y_te, cif_te[cause][:, col]), 3),
                }
            )
    auc = pd.DataFrame(rows).sort_values(["event", "model"])

    rates = {}
    for label, X, t, e in (
        ("train cohorts", X_tr, t_tr, e_tr),
        ("test cohorts", split.X_test, split.time_test, split.event_test),
    ):
        for code, cause in CAUSES.items():
            y = horizon_labels(t, e, code, h)
            by_iso = pd.Series(y, index=X["iso"].to_numpy()).groupby(level=0).mean().mul(100)
            rates[f"{cause}_by_{h // 12}y_% ({label})"] = by_iso.round(1)
    years = sorted(X_tr["queue_year"].unique())
    return auc, pd.DataFrame(rates).rename_axis(f"iso (train cohorts {years[0]}-{years[-1]})")


def split_summary(split: Split) -> pd.DataFrame:
    def counts(event):
        return {
            name: int((event == code).sum()) for code, name in {0: "censored", **CAUSES}.items()
        }

    return pd.DataFrame(
        [
            {
                "set": f"train (entered before {split.origin.date()}, outcomes as of that date)",
                "projects": len(split.X_train),
                **counts(split.event_train),
            },
            {
                "set": f"test (entered {split.test_years[0]}-{split.test_years[1]}, "
                "outcomes to cutoff)",
                "projects": len(split.X_test),
                **counts(split.event_test),
            },
        ]
    )


def setup_mlflow() -> None:
    logging.getLogger("alembic").setLevel(logging.WARNING)
    MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(MLRUNS_DIR / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def main() -> None:
    entries = pd.read_parquet(ENTRIES_FILE)
    features = pd.read_parquet(FEATURES_FILE)
    split = make_split(
        entries, features, ORIGIN, TEST_YEARS, min_test_followup_months=max(HORIZONS)
    )
    inner = make_split(entries, features, INNER_ORIGIN, INNER_VALID_YEARS, test_as_of=ORIGIN)

    print("Choosing LightGBM rounds on the inner split (training cohorts only) ...")
    rounds = DiscreteTimeLGBM.select_rounds(
        inner.X_train,
        inner.time_train,
        inner.event_train,
        inner.X_test,
        inner.time_test,
        inner.event_test,
    )

    models = [
        GroupAalenJohansen(),
        CauseSpecificCox(),
        CauseSpecificRSF(),
        DiscreteTimeLGBM(rounds),
    ]
    setup_mlflow()
    MODELS_DIR.mkdir(exist_ok=True)
    results, ph = [], None
    for model in models:
        print(f"Fitting {model.name} ...")
        start = clock.perf_counter()
        model.fit(split.X_train, split.time_train, split.event_train)
        fit_seconds = round(clock.perf_counter() - start, 1)
        rows = evaluate(model, split)
        results += rows  # fit time goes to MLflow only, so reports are identical across runs
        joblib.dump(model, MODELS_DIR / f"{model.name}.joblib")
        with mlflow.start_run(run_name=model.name):
            mlflow.log_params(
                {"origin": str(ORIGIN.date()), "test_years": str(TEST_YEARS), **model.params()}
            )
            for r in rows:
                tag = f"{r['event']}_{r['horizon_years']}y"
                mlflow.log_metrics({f"auc_{tag}": r["auc"], f"brier_{tag}": r["brier"]})
            mlflow.log_metric("fit_seconds", fit_seconds)
        if isinstance(model, CauseSpecificCox):
            ph = model.ph_test()

    table = pd.DataFrame(results).sort_values(["event", "horizon_years", "model"])
    table.to_csv(TABLE_FILE, index=False)
    diagnostics = transfer_diagnostics(models, split)
    REPORT_FILE.write_text(build_report(split, table, ph, rounds, diagnostics))
    print(f"Wrote {REPORT_FILE.relative_to(config.PROJECT_ROOT)}; models saved to models/")


def build_report(
    split: Split,
    table: pd.DataFrame,
    ph: pd.DataFrame,
    rounds: int,
    diagnostics: tuple[pd.DataFrame, pd.DataFrame],
) -> str:
    auc_shift, rate_shift = diagnostics
    view = table.drop(columns=["n"]).set_index(["event", "horizon_years", "model"])
    ph_sig = ph.assign(violated=ph["p"] < 0.05).groupby("cause")["violated"].agg(["sum", "size"])
    ph_top = ph.sort_values("test_statistic", ascending=False).groupby("cause").head(5)
    return f"""# Model comparison (generated by `make train`, do not edit by hand)

One time-based split, identical for every model (decisions D-027 to D-031):

{to_markdown(split_summary(split))}

- Training outcomes are cut off at the origin date ({split.origin.date()}): only what was
  known then.
- Every test project was observed for at least 5 years, so "event by 3 / 5 years" is a plain
  yes/no label (no censoring before the horizon). {split.test_dropped_short_followup} test
  project(s) were dropped because they were still in the queue with slightly less than 60
  average-length months of follow-up (entered on the last days of {split.test_years[1]}).
- No setting was chosen using the test cohorts. LightGBM's number of rounds ({rounds}) was chosen
  by early stopping on an inner split inside the training cohorts (train before
  {INNER_ORIGIN.date()}, validate on {INNER_VALID_YEARS[0]}-{INNER_VALID_YEARS[1]} entrants).

## Test results
`auc`: chance that a project which had the event by the horizon is ranked above one that did not
(0.5 = random). `brier`: mean squared error of the predicted probability (lower is better).
`observed_%` vs `mean_pred_%`: calibration-in-the-large.

These are point estimates on one split with no confidence intervals yet. Differences of a few
hundredths may be noise, especially for `operational`, which has few events. Phase 5 adds
bootstrap intervals, a rolling-origin backtest, calibration curves, the C-index, and the
integrated Brier score.

{to_markdown(view, index=True)}

## Transfer diagnostics: do patterns from the training years hold later?
AUC at 3 years on the training cohorts (in-sample, optimistic) vs on the test cohorts. "Training
cohorts" here means the entry years with a full 3 years of follow-up before the origin.

{to_markdown(auc_shift)}

Observed 3-year rates by ISO, training cohorts vs test cohorts:

{to_markdown(rate_shift, index=True)}

If ranking quality drops sharply from the training to the test cohorts while the observed rates
also shift, the relationship changed over time (a regime shift), not only that the model
overfits. Phase 5 tests this across many origins with the rolling-origin backtest. These
diagnostics were computed after the models were fixed and were not used to change them.

## Cox proportional-hazards check
Covariates whose effect appears to change over time (rank-transformed Schoenfeld test, p < 0.05),
per cause: {", ".join(f"{c}: {int(r['sum'])} of {int(r['size'])}" for c, r in ph_sig.iterrows())}.
With thousands of projects this test flags even small departures. The Cox CIFs should be read
as averaged effects. Largest test statistics:

{to_markdown(ph_top.round(4))}
"""


if __name__ == "__main__":
    main()
