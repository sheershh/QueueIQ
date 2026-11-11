"""Past-error range for the dashboard (decisions D-046).

For every past test year in blocks A and B (2014-2020 entrants), find the hazard multipliers
(c_op, c_wd) that would have made the primary configuration's average 3-year prediction match
what actually happened to that cohort (same solver as protocol section 7). A new project's
curves are then redrawn with each past year's multipliers. The envelope of those curves is the
"past-error range" (recalibrate.past_error_band): where the outcome would land if the model is as
wrong as it was in one of those years.

- Only blocks A and B build the band. Block C (the final test) is used afterwards, report-only, to
  check whether its miss falls inside the band (`final_test_check`); nothing is fitted on it.
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
from queueiq.evaluation.recalibrate import CAUSE_KEYS, apply_scaling, solve_scaling

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


def final_test_check(res: OriginResult, primary: str, multipliers: list[dict], tau: int = TAU):
    """Report only: the final-test cohort's observed rate against the band for that cohort (the
    cohort's mean prediction redrawn with each past year's multipliers)."""
    row = origin_multipliers(res, primary, tau)
    out = {"block": res.block, "test_years": list(res.test_years), "n": row["n"], "outcomes": {}}
    for k in CAUSE_KEYS:
        means = [
            float(apply_scaling(res.preds[primary], c)[k][:, tau - 1].mean()) for c in multipliers
        ]
        lo, hi, obs = min(means), max(means), row["observed"][k]
        out["outcomes"][k] = {
            "predicted": row["predicted"][k],
            "observed": obs,
            "cases": int(((res.event == CAUSE_CODE[k]) & (res.time <= tau)).sum()),
            "band_low": lo,
            "band_high": hi,
            "inside_band": bool(lo <= obs <= hi),
            "multiplier_needed": row["multiplier"][k],
        }
    return out


def main() -> None:
    primary = json.loads(SELECTION_FILE.read_text())["primary_configuration"]
    rows = [
        origin_multipliers(joblib.load(EVAL_DIR / f"{b}_{y}.joblib"), primary) for b, y in ORIGINS
    ]
    out = {
        "primary_configuration": primary,
        "horizon_months": TAU,
        "blocks_used": "A (2014-2017) and B (2018-2020); block C (final test) only checks the band",
        "origins": rows,
        "final_test_check": final_test_check(
            joblib.load(EVAL_DIR / "C_2021.joblib"), primary, [r["multiplier"] for r in rows]
        ),
    }
    RANGE_FILE.write_text(json.dumps(out, indent=2) + "\n")
    for r in rows:
        parts = [
            f"{k} {100 * r['predicted'][k]:.1f}% -> {100 * r['observed'][k]:.1f}% "
            f"(x{r['multiplier'][k]:.2f})"
            for k in ("withdrawn", "operational")
        ]
        print(f"{r['block']} {r['origin']}: n={r['n']:,}  " + "  ".join(parts))
    for k, v in out["final_test_check"]["outcomes"].items():
        print(
            f"Final test {k}: observed {100 * v['observed']:.2f}% vs band "
            f"{100 * v['band_low']:.1f}-{100 * v['band_high']:.1f}% -> inside={v['inside_band']}"
        )
    print(f"Wrote {RANGE_FILE.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
