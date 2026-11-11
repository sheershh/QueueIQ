"""End-to-end `make train` on synthetic processed data in a temp directory."""

import pandas as pd

from queueiq import config
from queueiq.models import train
from tests.model_data import synthetic


def test_make_train_writes_comparison_and_models(tmp_path, monkeypatch):
    entries, features = synthetic(n=900, seed=1)
    entries.to_parquet(tmp_path / "entries.parquet")
    features.to_parquet(tmp_path / "features.parquet")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(train, "ENTRIES_FILE", tmp_path / "entries.parquet")
    monkeypatch.setattr(train, "FEATURES_FILE", tmp_path / "features.parquet")
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train, "MLRUNS_DIR", tmp_path / "mlruns")
    monkeypatch.setattr(train, "REPORT_FILE", tmp_path / "model_comparison.md")
    monkeypatch.setattr(train, "TABLE_FILE", tmp_path / "model_comparison.csv")

    train.main()

    table = pd.read_csv(tmp_path / "model_comparison.csv")
    assert set(table["model"]) == {
        "baseline_aj_iso_fuel",
        "cox_cause_specific",
        "rsf_cause_specific",
        "lgbm_discrete_time",
    }
    assert len(table) == 4 * 2 * 2  # models x events x horizons
    assert len(list((tmp_path / "models").glob("*.joblib"))) == 4
    report = (tmp_path / "model_comparison.md").read_text()
    assert "## Transfer diagnostics" in report and "TBD" not in report
