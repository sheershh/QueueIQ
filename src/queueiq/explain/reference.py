"""Reference numbers shown next to every explanation (decisions D-043).

The Shapley baseline is the model's *average prediction* for recent entrants, not an observed
rate. So each explanation also shows (a) the observed rate for the most recent cohort with full
follow-up to that horizon, and (b) the Phase 5 final-test calibration warning (the model
predicted far more withdrawals and completions than happened).
"""

import pandas as pd

from queueiq.explain.groups import TARGETS


def observed_reference(entries: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, dict]:
    """Per target: the latest entry year whose projects all have follow-up to the horizon,
    its size, and the observed share with the event by the horizon (model scope)."""
    scoped = entries[entries["in_model_scope"]]
    out = {}
    for target, (outcome, h, _) in TARGETS.items():
        eligible = scoped[scoped["queue_date"] <= cutoff - pd.DateOffset(months=h)]
        year = int(eligible["queue_year"].max())
        cohort = scoped[scoped["queue_year"] == year]
        followed = (cohort["outcome"] != "censored") & (cohort["time_months"] < h) | (
            cohort["time_months"] >= h
        )
        cohort = cohort[followed]
        hit = (cohort["outcome"] == outcome) & (cohort["time_months"] < h)
        out[target] = {"cohort": year, "n": len(cohort), "observed": float(hit.mean())}
    return out


def calibration_warning_values(metrics: pd.DataFrame, primary: str) -> dict[str, dict]:
    """Phase 5 final test (block C), primary configuration, 3-year horizon."""
    m = metrics[(metrics["block"] == "C") & (metrics["model"] == primary) & (metrics["tau"] == 36)]
    return {
        r["outcome"]: {
            "predicted": float(r["mean_pred"]),
            "observed": float(r["observed"]),
            "cases": int(r["cases"]),
            "n": int(r["n"]),
        }
        for _, r in m.iterrows()
    }


def calibration_warning(cal: dict[str, dict]) -> str:
    def pct(x):
        return f"{100 * x:.2f}%" if x < 0.01 else f"{100 * x:.1f}%"

    wd, op = cal["withdrawn"], cal["operational"]
    return (
        "Caution: these probabilities run high. On the Phase 5 final test (2021-22 entrants, "
        f"n = {wd['n']:,}) the model predicted {pct(wd['predicted'])} withdrawing within 3 years "
        f"against {pct(wd['observed'])} observed, and {pct(op['predicted'])} reaching operation "
        f"within 3 years against {pct(op['observed'])} observed ({op['cases']} project). Use the "
        "numbers to compare projects with each other, not as literal odds."
    )
