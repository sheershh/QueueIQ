"""`make evaluate-dev`: Block A (development) and the frozen model selection.

Implements docs/evaluation_protocol.md sections 1-3. Output:
- data/interim/eval/A_<origin>.joblib  (predictions, reused by the final run)
- reports/evaluation_dev_metrics.csv  (per-origin point metrics)
- reports/evaluation_selection.json   (scores + the selection; committed BEFORE blocks B and C)
"""

import hashlib
import json
import subprocess

import joblib
import pandas as pd

from queueiq import config
from queueiq.evaluation.backtest import (
    candidate_names,
    complexity_key,
    point_metrics,
    run_origin,
)

PROTOCOL_FILE = config.PROJECT_ROOT / "docs" / "evaluation_protocol.md"
PROTOCOL_TAG = "eval-protocol-v1"
EVAL_DIR = config.INTERIM_DIR / "eval"
ENTRIES_FILE = config.PROCESSED_DIR / "queue_entries.parquet"
FEATURES_FILE = config.PROCESSED_DIR / "features.parquet"
SELECTION_FILE = config.REPORTS_DIR / "evaluation_selection.json"
DEV_METRICS_FILE = config.REPORTS_DIR / "evaluation_dev_metrics.csv"

DEV_ORIGINS = (2014, 2015, 2016, 2017)
SELECTION_TAU = 36
TIE_MARGIN = 0.005


def dev_horizons(origin: int) -> tuple[int, ...]:
    return (36, 60) if origin <= 2015 else (36,)


def protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL_FILE.read_bytes()).hexdigest()


def protocol_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-list", "-n", "1", PROTOCOL_TAG],
            capture_output=True,
            text=True,
            check=True,
            cwd=config.PROJECT_ROOT,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def selection_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    """Score_c = mean over the 4 origins and 2 outcomes of Skill at 36 months (section 3)."""
    m = metrics[(metrics["tau"] == SELECTION_TAU) & metrics["model"].isin(candidate_names())]
    per = m.pivot_table(index="model", columns=["origin", "outcome"], values="skill")
    expected = len(DEV_ORIGINS) * 2
    if per.shape[1] != expected or per.isna().any().any():
        raise ValueError("every candidate needs a skill for every origin and outcome")
    scores = per.mean(axis=1).rename("score").to_frame()
    scores["complexity"] = [complexity_key(n) for n in scores.index]
    return scores.sort_values("score", ascending=False)


def select(scores: pd.DataFrame) -> tuple[str, str]:
    """Highest score; but the least complex candidate within TIE_MARGIN of the best wins."""
    best_name, best = scores.index[0], scores["score"].iloc[0]
    near = scores[scores["score"] >= best - TIE_MARGIN]
    chosen = min(near.index, key=complexity_key)
    reason = (
        "highest score"
        if chosen == best_name
        else f"tie rule: within {TIE_MARGIN} of the best ({best_name}, {best:.4f}) and less complex"
    )
    return chosen, reason


def main() -> None:
    entries = pd.read_parquet(ENTRIES_FILE)
    features = pd.read_parquet(FEATURES_FILE)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows, origins = [], {}
    for y in DEV_ORIGINS:
        print(f"Block A, origin {y} ...")
        res = run_origin(entries, features, "A", y, (y, y), dev_horizons(y))
        joblib.dump(res, EVAL_DIR / f"A_{y}.joblib")
        names = [*candidate_names(), "null_aj", *[n for n in res.preds if "no_congestion" in n]]
        for tau in res.horizons:
            rows += point_metrics(res, names, tau)
        origins[y] = {
            "n_train": res.n_train,
            "n_test": len(res.ids),
            "lgbm_rounds": res.lgbm_rounds,
            "recal_constants": res.recal_constants,
        }

    metrics = pd.DataFrame(rows)
    metrics.to_csv(DEV_METRICS_FILE, index=False, float_format="%.6g")
    scores = selection_scores(metrics)
    primary, reason = select(scores)
    skills = metrics[(metrics["tau"] == SELECTION_TAU) & (metrics["model"] == primary)].set_index(
        ["origin", "outcome"]
    )["skill"]
    payload = {
        "protocol_file": "docs/evaluation_protocol.md",
        "protocol_sha256": protocol_sha256(),
        "protocol_tag": PROTOCOL_TAG,
        "protocol_commit": protocol_commit(),
        "primary_configuration": primary,
        "selection_reason": reason,
        "scores": {n: round(float(s), 6) for n, s in scores["score"].items()},
        "primary_skill_by_origin": {f"{o}|{k}": round(float(v), 6) for (o, k), v in skills.items()},
        "origins": {str(k): v for k, v in origins.items()},
    }
    SELECTION_FILE.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"Primary configuration: {primary} ({reason})")
    print(scores.head(8).to_string())


if __name__ == "__main__":
    main()
