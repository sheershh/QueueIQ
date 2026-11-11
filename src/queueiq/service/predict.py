"""The prediction service shared by the API and the dashboard (decisions D-046 to D-048).

One hypothetical project entering the queue on the day after the data cutoff:
- curves: cumulative chance of reaching operation / withdrawing by each month (1-96), from the
  deployment model, with the past-error range around each curve;
- headline numbers: withdrawn within 3 years, operational within 3 and 5 years, and expected
  months in the queue during the first 5 years, each with its past-error range;
- explanation (optional, ~1 s): grouped Shapley contributions with plain-language text;
- context: the observed rates for the latest fully followed cohorts and the Phase 5 warning.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from queueiq import config
from queueiq.evaluation.recalibrate import CAUSE_KEYS, apply_scaling, past_error_band
from queueiq.explain import deploy
from queueiq.explain.groups import GROUPS, TARGETS, describe
from queueiq.explain.shapley import explain
from queueiq.explain.text import drivers, narrative
from queueiq.models.base import GRID
from queueiq.service.features import history, project_features
from queueiq.service.inputs import MAX_COMPARE, InputError, Options, ProjectInput

ENTRIES_FILE = config.PROCESSED_DIR / "queue_entries.parquet"
EVALUATED_MONTHS = 60  # longest horizon checked in Phase 5 (blocks A and B)
QUEUE_MONTHS = 60  # window for "expected months in the queue"
DATA_ATTRIBUTION = {
    "text": "Queued Up Data File. Copyright (c) 2026, The Regents of the University of California, "
    "through Lawrence Berkeley National Laboratory & GridTracker. CC BY 4.0.",
    "source": "https://emp.lbl.gov/queues",
    "license": "https://creativecommons.org/licenses/by/4.0/",
    "changes": "Filtered to PJM, MISO, CAISO and SPP entrants from 2010 on, cleaned, and used to "
    "derive features and fit models. LBNL and GridTracker do not endorse this project.",
}

HEADLINES = [
    ("withdrawn_3y", "withdrawn", 36, "Withdrawn within 3 years"),
    ("operational_3y", "operational", 36, "Reached operation within 3 years"),
    ("operational_5y", "operational", 60, "Reached operation within 5 years"),
]


def months_in_queue(cif: dict[str, np.ndarray], upto: int = QUEUE_MONTHS) -> np.ndarray:
    """Expected months still in the queue during months 0..upto (restricted mean, trapezoid on
    the monthly grid): the area under P(still in queue)."""
    surv = 1 - cif["operational"][:, :upto] - cif["withdrawn"][:, :upto]
    prev = np.concatenate([np.ones((len(surv), 1)), surv[:, :-1]], axis=1)
    return ((prev + surv) / 2).sum(axis=1)


@dataclass
class Service:
    deployment: deploy.Deployment
    hist: pd.DataFrame
    options: Options

    @classmethod
    def load(cls, entries_file: Path = ENTRIES_FILE) -> "Service":
        missing = [p for p in (deploy.MODEL_FILE, deploy.META_FILE, entries_file) if not p.exists()]
        if missing:
            names = ", ".join(str(p.relative_to(config.PROJECT_ROOT)) for p in missing)
            raise FileNotFoundError(f"Missing {names}. Run `make data && make train` first.")
        entries = pd.read_parquet(entries_file)
        return cls(deploy.load(), history(entries), Options.from_entries(entries))

    @property
    def meta(self) -> dict:
        return self.deployment.meta

    @property
    def entry_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.meta["entry_date"])

    @property
    def multipliers(self) -> list[dict[str, float]]:
        return [o["multiplier"] for o in self.meta["past_error_range"]["origins"]]

    # -- inputs ---------------------------------------------------------------------------
    def features(self, p: ProjectInput) -> pd.DataFrame:
        self.options.check(p)
        return project_features(p, self.hist, self.options, self.entry_date)

    # -- predictions ----------------------------------------------------------------------
    def _predict_rows(self, X: pd.DataFrame) -> list[dict]:
        cif = self.deployment.model.predict_cif(X)
        band = past_error_band(cif, self.multipliers)
        redrawn = [apply_scaling(cif, c) for c in self.multipliers]
        queue = months_in_queue(cif)
        queue_past = np.stack([months_in_queue(r) for r in redrawn])
        out = []
        for i in range(len(X)):
            curves = {
                k: {
                    "model": cif[k][i].tolist(),
                    "low": band[f"{k}_low"][i].tolist(),
                    "high": band[f"{k}_high"][i].tolist(),
                }
                for k in CAUSE_KEYS
            }
            headline = []
            for key, k, h, label in HEADLINES:
                past = [r[k][i, h - 1] for r in redrawn]
                headline.append(
                    {
                        "key": key,
                        "label": label,
                        "outcome": k,
                        "months": h,
                        "model": float(cif[k][i, h - 1]),
                        "low": float(min(past)),
                        "high": float(max(past)),
                    }
                )
            headline.append(
                {
                    "key": "months_in_queue_5y",
                    "label": f"Expected months in the queue, of the first {QUEUE_MONTHS}",
                    "outcome": None,
                    "months": QUEUE_MONTHS,
                    "model": float(queue[i]),
                    "low": float(queue_past[:, i].min()),
                    "high": float(queue_past[:, i].max()),
                }
            )
            out.append({"months": GRID.tolist(), "curves": curves, "headline": headline})
        return out

    def predict(self, p: ProjectInput) -> dict:
        X = self.features(p)
        return {
            "project": p.model_dump(),
            "county": self.options.county_info(p),
            **self._predict_rows(X)[0],
        }

    def compare(self, projects: list[ProjectInput]) -> list[dict]:
        if not 1 <= len(projects) <= MAX_COMPARE:
            raise InputError([f"Compare between 1 and {MAX_COMPARE} projects."])
        problems, rows = [], []
        for p in projects:
            try:
                rows.append(self.features(p))
            except InputError as e:
                problems += [f"{p.name}: {msg}" for msg in e.problems]
        if problems:
            raise InputError(problems)
        preds = self._predict_rows(pd.concat(rows, ignore_index=True))
        return [
            {"project": p.model_dump(), "county": self.options.county_info(p), **r}
            for p, r in zip(projects, preds, strict=True)
        ]

    # -- explanation ----------------------------------------------------------------------
    def explain(self, p: ProjectInput) -> dict:
        X = self.features(p)
        table = explain(
            deploy.predictor(self.deployment.model), X, self.deployment.background, GROUPS
        )
        row = X.iloc[0]
        targets = {}
        for target, (outcome, h, label) in TARGETS.items():
            targets[target] = {
                "label": label,
                "outcome": outcome,
                "months": h,
                "prediction": float(table.attrs["prediction"][target]),
                "baseline": float(table.attrs["baseline"][target]),
                "contributions": [
                    {
                        "group": g,
                        "points": float(100 * table.loc[g, target]),
                        "value": describe(g, row),
                    }
                    for g in GROUPS
                ],
            }
        return {
            "project": p.model_dump(),
            "county": self.options.county_info(p),
            "targets": targets,
            "text": narrative(table, row, self.meta["observed_reference"]),
            "drivers": drivers(table, row),
        }

    # -- context shown with every prediction ----------------------------------------------
    def context(self) -> dict:
        m = self.meta
        return {
            "entry_date": m["entry_date"],
            "outcomes_as_of": m["outcomes_as_of"],
            "trained_on_rows": m["trained_on_rows"],
            "primary_configuration": m["primary_configuration"],
            "evaluated_months": EVALUATED_MONTHS,
            "past_error_range": m["past_error_range"],
            "observed_reference": m["observed_reference"],
            "calibration_warning": m["calibration_warning"],
            "background_years": m["background_years"],
            "data_attribution": DATA_ATTRIBUTION,
        }
