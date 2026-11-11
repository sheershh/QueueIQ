"""Past-error range for the dashboard (decisions D-046).

For every past test year in blocks A and B (2014-2020 entrants), find the hazard multipliers
(c_op, c_wd) that would have made the primary configuration's average 3-year prediction match
what actually happened to that cohort (same solver as protocol section 7). A new project's
curves are then redrawn with each past year's multipliers. The envelope of those curves is the
"past-error range" (recalibrate.past_error_band): where the outcome would land if the model is as
wrong as it was in one of those years.

- Only blocks A and B are used. Block C (the final test) stays report-only, as the protocol says.
- This is not a confidence interval. It is a spread of past calibration errors, with 7 years
  behind it. It says nothing about errors larger than the worst past year.
- `make error-range` writes reports/past_error_range.json from the saved Phase 5 results
  (data/interim/eval). The file is committed so the deployment build works from a clean clone.
"""

import json

import joblib

from queueiq import config
from queueiq.evaluation.backtest import CAUSE_CODE, OriginResult
from queueiq.evaluation.dev import DEV_ORIGINS, EVAL_DIR, SELECTION_FILE
from queueiq.evaluation.metrics import event_by
from queueiq.evaluation.recalibrate import CAUSE_KEYS, solve_scaling

B_ORIGINS = (2018, 2019, 2020)  # same as evaluation.final.B_ORIGINS (not imported: heavy module)
ORIGINS = [("A", y) for y in DEV_ORIGINS] + [("B", y) for y in B_ORIGINS]
TAU = 36
RANGE_FILE = config.REPORTS_DIR / "past_error_range.json"


def origin_multipliers(res: OriginResult, primary: str, tau: int = TAU) -> dict:
    """Multipliers that map this origin's mean predicted tau-month rates onto observed ones."""
    cif = res.preds[primary]
    observed = {
        k: float(event_by(res.time, res.event, CAUSE_CODE[k], tau).mean()) for k in CAUSE_KEYS
    }
    c = solve_scaling(cif, observed, tau=tau)
    return {
        "block": res.block,
        "origin": res.origin_year,
        "n": len(res.time),
        "predicted": {k: float(cif[k][:, tau - 1].mean()) for k in CAUSE_KEYS},
        "observed": observed,
        "multiplier": c,
    }


def main() -> None:
    primary = json.loads(SELECTION_FILE.read_text())["primary_configuration"]
    rows = [
        origin_multipliers(joblib.load(EVAL_DIR / f"{b}_{y}.joblib"), primary) for b, y in ORIGINS
    ]
    out = {
        "primary_configuration": primary,
        "horizon_months": TAU,
        "blocks_used": "A (2014-2017) and B (2018-2020); block C (final test) not used",
        "origins": rows,
    }
    RANGE_FILE.write_text(json.dumps(out, indent=2) + "\n")
    for r in rows:
        parts = [
            f"{k} {100 * r['predicted'][k]:.1f}% -> {100 * r['observed'][k]:.1f}% "
            f"(x{r['multiplier'][k]:.2f})"
            for k in ("withdrawn", "operational")
        ]
        print(f"{r['block']} {r['origin']}: n={r['n']:,}  " + "  ".join(parts))
    print(f"Wrote {RANGE_FILE.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
