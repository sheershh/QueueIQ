"""End-to-end: run `make data`'s main() on a tiny fake LBNL file in a temp directory."""

import hashlib

import pandas as pd

from queueiq import config
from queueiq.ingestion import build, lbnl
from tests.conftest import make_raw_frame

T = pd.Timestamp


def write_fake_lbnl_file(path, rows):
    path.parent.mkdir(parents=True)
    with pd.ExcelWriter(path) as xl:
        banner = pd.DataFrame([["RETURN TO CONTENTS"]])
        banner.to_excel(xl, sheet_name=lbnl.SHEET_NAME, index=False, header=False)
        make_raw_frame(rows).to_excel(xl, sheet_name=lbnl.SHEET_NAME, index=False, startrow=1)


def test_build_writes_all_outputs_and_keeps_every_row(tmp_path, monkeypatch):
    rows = [
        {"q_status": "withdrawn", "wd_date": T("2018-01-01")},  # in scope
        {"q_status": "operational", "on_date": T("2019-05-01")},  # in scope
        {"region": "West", "entity": "APS"},  # out: region
        {"q_status": "withdrawn", "wd_date": None},  # out: missing event date
        {"mw_1": 20000.0},  # in scope, but flagged as huge capacity
    ]
    raw_file = tmp_path / "data" / "raw" / lbnl.RAW_FILENAME
    write_fake_lbnl_file(raw_file, rows)

    # Point every path at the temp directory and accept the fake file's checksum.
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "RAW_DIR", raw_file.parent)
    monkeypatch.setattr(config, "INTERIM_DIR", tmp_path / "data" / "interim")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(lbnl, "EXPECTED_SHA256", hashlib.sha256(raw_file.read_bytes()).hexdigest())
    monkeypatch.setattr(build, "INTERIM_FILE", tmp_path / "data" / "interim" / "sheet.parquet")
    monkeypatch.setattr(build, "PROCESSED_FILE", tmp_path / "data" / "processed" / "q.parquet")
    monkeypatch.setattr(build, "WATERFALL_FILE", tmp_path / "reports" / "waterfall.csv")
    monkeypatch.setattr(build, "REPORT_FILE", tmp_path / "reports" / "data_quality.md")

    build.main()

    out = pd.read_parquet(build.PROCESSED_FILE)
    assert len(out) == len(rows)
    assert out["in_model_scope"].tolist() == [True, True, False, False, True]
    assert pd.read_csv(build.WATERFALL_FILE)["rows_remaining"].iloc[-1] == 3
    report = build.REPORT_FILE.read_text()
    assert "## Cleaning waterfall" in report
    assert "### Examples: huge_capacity" in report
