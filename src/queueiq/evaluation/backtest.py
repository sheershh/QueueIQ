"""Fit every protocol candidate at one origin and score the test cohort (protocol sections 1-7).

One call = one origin: train on entrants before the origin (outcomes as of the origin), predict
the test cohort, add recalibrated variants using the reference cohort Y-4, and keep everything
needed for metrics and bootstraps. No decision is made here.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from queueiq.evaluation.metrics import (
    MIN_CASES,
    assert_full_followup,
    auc_pairs,
    calibration_bins,
    cindex_pairs,
    ece,
    event_by,
    ibs_per_project,
)
from queueiq.evaluation.recalibrate import apply_scaling, solve_scaling
from queueiq.features.congestion import county_key
from queueiq.models.baseline import GroupAalenJohansen, PooledAalenJohansen, RecentCohortAJ
from queueiq.models.cox import CauseSpecificCox
from queueiq.models.data import CONGESTION_FEATURES, make_split
from queueiq.models.lgbm import DiscreteTimeLGBM
from queueiq.models.rsf import CauseSpecificRSF

CAUSE_CODE = {"operational": 1, "withdrawn": 2}
FAMILIES = ("cox_cause_specific", "lgbm_discrete_time", "rsf_cause_specific")
FAMILY_RANK = {
    "recent_cohort": 0,
    "baseline_aj_iso_fuel": 1,
    "cox_cause_specific": 2,
    "lgbm_discrete_time": 3,
    "rsf_cause_specific": 4,
}
REF_LAG_36 = 4  # reference cohort for 36 months: entry year Y - 4
REF_LAG_60 = 6
MIN_IN_SCOPE_YEAR = 2010


def candidate_names() -> list[str]:
    """The 15 selection candidates of protocol section 2."""
    names = ["recent_cohort", "baseline_aj_iso_fuel|recal=0", "baseline_aj_iso_fuel|recal=1"]
    for fam in FAMILIES:
        for qy in (1, 0):
            for recal in (0, 1):
                names.append(f"{fam}|qy={qy}|recal={recal}")
    return names


def ablation_names() -> list[str]:
    return [f"{fam}|no_congestion" for fam in FAMILIES]


def complexity_key(name: str) -> tuple[int, int, int]:
    """Tie-rule order (protocol section 3): family, then queue_year, then recalibration."""
    parts = dict(p.split("=") for p in name.split("|")[1:] if "=" in p)
    return (FAMILY_RANK[name.split("|")[0]], int(parts.get("qy", 0)), int(parts.get("recal", 0)))


def _family_model(fam: str, exclude: tuple[str, ...], inner) -> object:
    if fam == "cox_cause_specific":
        return CauseSpecificCox(exclude=exclude)
    if fam == "rsf_cause_specific":
        return CauseSpecificRSF(exclude=exclude)
    rounds = DiscreteTimeLGBM.select_rounds(
        inner.X_train,
        inner.time_train,
        inner.event_train,
        inner.X_test,
        inner.time_test,
        inner.event_test,
        exclude=exclude,
    )
    return DiscreteTimeLGBM(num_rounds=rounds, exclude=exclude)


@dataclass
class OriginResult:
    block: str
    origin_year: int
    test_years: tuple[int, int]
    horizons: tuple[int, ...]
    ids: np.ndarray
    time: np.ndarray
    event: np.ndarray
    iso: np.ndarray
    entry_year: np.ndarray
    county_year: np.ndarray
    preds: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    recal_constants: dict[str, dict[str, float]] = field(default_factory=dict)
    lgbm_rounds: dict[str, int] = field(default_factory=dict)
    dropped_short_followup: int = 0
    n_train: int = 0


def run_origin(
    entries, features, block: str, origin_year: int, test_years, horizons
) -> OriginResult:
    origin = pd.Timestamp(f"{origin_year}-01-01")
    split = make_split(
        entries, features, origin, test_years, min_test_followup_months=max(horizons)
    )
    inner = make_split(
        entries,
        features,
        pd.Timestamp(f"{origin_year - 2}-01-01"),
        (origin_year - 2, origin_year - 1),
        test_as_of=origin,
    )
    for h in horizons:
        assert_full_followup(split.time_test, split.event_test, h)

    fit = lambda m: m.fit(split.X_train, split.time_train, split.event_train)  # noqa: E731
    models = {
        "null_aj": fit(PooledAalenJohansen()),
        "baseline_aj_iso_fuel|recal=0": fit(GroupAalenJohansen()),
        "recent_cohort": fit(RecentCohortAJ(origin_year - REF_LAG_36, horizon=36)),
    }
    if 60 in horizons and origin_year - REF_LAG_60 >= MIN_IN_SCOPE_YEAR:
        models["recent_cohort_60"] = fit(RecentCohortAJ(origin_year - REF_LAG_60, horizon=60))
    lgbm_rounds = {}
    for fam in FAMILIES:
        for variant, exclude in (
            ("qy=1|recal=0", ()),
            ("qy=0|recal=0", ("queue_year",)),
            ("no_congestion", CONGESTION_FEATURES),
        ):
            model = fit(_family_model(fam, exclude, inner))
            models[f"{fam}|{variant}"] = model
            if fam == "lgbm_discrete_time":
                lgbm_rounds[variant] = model.num_rounds

    preds = {name: m.predict_cif(split.X_test) for name, m in models.items()}

    # Recalibration (protocol section 7): reference cohort Y-4, outcomes known at the origin.
    ref = (split.X_train["queue_year"] == origin_year - REF_LAG_36).to_numpy()
    t_ref, e_ref = split.time_train[ref], split.event_train[ref]
    assert_full_followup(t_ref, e_ref, 36)
    observed = {k: float(event_by(t_ref, e_ref, c, 36).mean()) for k, c in CAUSE_CODE.items()}
    constants = {}
    for name in [n for n in models if n.endswith("recal=0")]:
        c = solve_scaling(models[name].predict_cif(split.X_train[ref]), observed, tau=36)
        recal_name = name.replace("recal=0", "recal=1")
        preds[recal_name] = apply_scaling(preds[name], c)
        constants[recal_name] = c

    ids = split.test_ids
    info = entries.set_index("project_id").loc[ids]
    ckey = county_key(info.reset_index()).to_numpy()
    year = split.X_test["queue_year"].to_numpy().astype(int)
    county_year = np.array(
        [
            f"{k}|{y}" if isinstance(k, str) else f"solo:{i}"
            for k, y, i in zip(ckey, year, ids, strict=True)
        ]
    )
    return OriginResult(
        block=block,
        origin_year=origin_year,
        test_years=tuple(test_years),
        horizons=tuple(horizons),
        ids=ids,
        time=split.time_test,
        event=split.event_test,
        iso=split.X_test["iso"].astype(str).to_numpy(),
        entry_year=year,
        county_year=county_year,
        preds={n: {k: v.astype(np.float32) for k, v in p.items()} for n, p in preds.items()},
        recal_constants=constants,
        lgbm_rounds=lgbm_rounds,
        dropped_short_followup=split.test_dropped_short_followup,
        n_train=len(split.X_train),
    )


def scored_name(name: str, tau: int) -> str:
    """The recent-cohort baseline is horizon-specific: its 60-month version is scored at 60."""
    return "recent_cohort_60" if name == "recent_cohort" and tau == 60 else name


def point_metrics(res: OriginResult, names: list[str], tau: int) -> list[dict]:
    """Point estimates for each name and outcome at horizon tau, with event counts."""
    rows = []
    for k, code in CAUSE_CODE.items():
        null_ibs = ibs_per_project(res.time, res.event, code, res.preds["null_aj"][k], tau).mean()
        y = event_by(res.time, res.event, code, tau)
        for name in names:
            key = scored_name(name, tau)
            if key not in res.preds:
                continue
            cif = res.preds[key][k]
            risk = cif[:, tau - 1]
            ibs = ibs_per_project(res.time, res.event, code, cif, tau).mean()
            auc, cidx = (
                auc_pairs(res.time, res.event, code, risk, tau),
                cindex_pairs(res.time, res.event, code, risk, tau),
            )
            estimable = auc.n_cases >= MIN_CASES
            bins = calibration_bins(risk, y)
            rows.append(
                {
                    "block": res.block,
                    "origin": res.origin_year,
                    "model": name,
                    "outcome": k,
                    "tau": tau,
                    "n": len(y),
                    "cases": int(y.sum()),
                    "auc": auc.value() if estimable else np.nan,
                    "cindex": cidx.value() if estimable else np.nan,
                    "c_pairs": cidx.n_pairs,
                    "brier_tau": float(((y - risk) ** 2).mean()),
                    "ibs": float(ibs),
                    "skill": float(1 - ibs / null_ibs),
                    "mean_pred": float(risk.mean()),
                    "observed": float(y.mean()),
                    "ece": ece(bins),
                }
            )
    return rows
