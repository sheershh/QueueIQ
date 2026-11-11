"""End-to-end Phase 5 on synthetic data: dev (selection freeze) -> final (B, C, report)."""

import json

import pytest

from queueiq import config
from queueiq.evaluation import dev, final
from tests.model_data import synthetic


@pytest.fixture
def env(tmp_path, monkeypatch):
    entries, features = synthetic(n=1200, seed=7)
    entries.to_parquet(tmp_path / "entries.parquet")
    features.to_parquet(tmp_path / "features.parquet")
    for mod in (dev, final):
        monkeypatch.setattr(mod, "ENTRIES_FILE", tmp_path / "entries.parquet")
        monkeypatch.setattr(mod, "FEATURES_FILE", tmp_path / "features.parquet")
        monkeypatch.setattr(mod, "EVAL_DIR", tmp_path / "eval")
        monkeypatch.setattr(mod, "SELECTION_FILE", tmp_path / "selection.json")
        monkeypatch.setattr(mod, "DEV_ORIGINS", (2014,))
    monkeypatch.setattr(dev, "DEV_METRICS_FILE", tmp_path / "dev_metrics.csv")
    monkeypatch.setattr(final, "B_ORIGINS", (2016,))
    monkeypatch.setattr(final, "C_ORIGIN", 2018)
    monkeypatch.setattr(final, "C_TEST_YEARS", (2018, 2019))
    monkeypatch.setattr(final, "REPORT_FILE", tmp_path / "evaluation.md")
    monkeypatch.setattr(final, "METRICS_FILE", tmp_path / "metrics.csv")
    monkeypatch.setattr(final, "BOOT_FILE", tmp_path / "boot.csv")
    monkeypatch.setattr(final, "MODEL_CARD", tmp_path / "model_card.md")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    return tmp_path


def test_final_refuses_without_a_frozen_selection(env):
    with pytest.raises(RuntimeError, match="No frozen selection"):
        final.main(B=20)


def test_dev_then_final_end_to_end(env):
    dev.main()
    sel = json.loads((env / "selection.json").read_text())
    assert sel["primary_configuration"] in sel["scores"]
    assert len(sel["scores"]) == 15

    final.main(B=30)
    report = (env / "evaluation.md").read_text()
    assert "## Final test (block C, run once)" in report
    assert "PRIMARY" in report and "TBD" not in report
    card = (env / "model_card.md").read_text()
    assert "## Limitations and failure cases" in card
    assert len(list((env / "figures").glob("*.png"))) == 2


def test_final_refuses_if_protocol_changed(env):
    dev.main()
    sel = json.loads((env / "selection.json").read_text())
    sel["protocol_sha256"] = "0" * 64
    (env / "selection.json").write_text(json.dumps(sel))
    with pytest.raises(RuntimeError, match="changed after the selection"):
        final.main(B=20)
