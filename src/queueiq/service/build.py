"""Fit and save the deployment model the API and dashboard serve (part of `make train`).

Needs data/processed/{queue_entries,features}.parquet and the committed Phase 5 outputs
(reports/evaluation_selection.json, evaluation_metrics.csv, past_error_range.json). Writes models/.
"""

import json

import pandas as pd

from queueiq import config
from queueiq.explain import deploy

SELECTION_FILE = config.REPORTS_DIR / "evaluation_selection.json"


def main() -> None:
    selection = json.loads(SELECTION_FILE.read_text())
    entries = pd.read_parquet(config.PROCESSED_DIR / "queue_entries.parquet")
    features = pd.read_parquet(config.PROCESSED_DIR / "features.parquet")
    print(f"Fitting the deployment model ({selection['primary_configuration']}) on all data ...")
    built = deploy.build(entries, features, selection)
    print(
        f"Saved models/ ({built['meta']['trained_on_rows']:,} projects, "
        f"fit in {built['fit_seconds']} s)"
    )


if __name__ == "__main__":
    main()
