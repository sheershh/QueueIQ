"""End-to-end: fake LBNL file -> `make data` -> `make eda`, all in a temp directory."""

import hashlib
import itertools

import pandas as pd

from queueiq import config
from queueiq.eda import build as eda_build
from queueiq.ingestion import build as data_build
from queueiq.ingestion import lbnl
from tests.test_build import write_fake_lbnl_file

T = pd.Timestamp
FUELS = ["Solar", "Wind", "Battery", "Gas", "Hydro"]
YEARS = [2012, 2017, 2022]
STATUSES = ["withdrawn", "operational", "active"]


def fake_rows():
    rows = []
    combos = itertools.product(FUELS, YEARS, STATUSES, range(3))  # 45 requests per entry year
    for i, (fuel, year, status, _) in enumerate(combos):
        row = {
            "type_1": fuel,
            "type_clean": fuel,
            "q_status": status,
            "q_date": T(f"{year}-03-01"),
            "region": ["PJM", "MISO"][i % 2],
        }
        if status == "withdrawn":
            row["wd_date"] = T(f"{year + 1}-06-01")
        if status == "operational":
            row["on_date"] = T(f"{year + 3}-01-01")
        rows.append(row)
    return rows


def test_make_eda_writes_report_and_figures(tmp_path, monkeypatch):
    raw_file = tmp_path / "data" / "raw" / lbnl.RAW_FILENAME
    write_fake_lbnl_file(raw_file, fake_rows())

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "RAW_DIR", raw_file.parent)
    monkeypatch.setattr(config, "INTERIM_DIR", tmp_path / "data" / "interim")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(lbnl, "EXPECTED_SHA256", hashlib.sha256(raw_file.read_bytes()).hexdigest())
    processed = tmp_path / "data" / "processed" / "queue_entries.parquet"
    interim = tmp_path / "data" / "interim" / "sheet.parquet"
    monkeypatch.setattr(data_build, "INTERIM_FILE", interim)
    monkeypatch.setattr(data_build, "PROCESSED_FILE", processed)
    monkeypatch.setattr(data_build, "WATERFALL_FILE", tmp_path / "reports" / "waterfall.csv")
    monkeypatch.setattr(data_build, "REPORT_FILE", tmp_path / "reports" / "data_quality.md")
    monkeypatch.setattr(eda_build, "PROCESSED_FILE", processed)
    monkeypatch.setattr(eda_build, "INTERIM_FILE", interim)
    monkeypatch.setattr(eda_build, "REPORT_FILE", tmp_path / "reports" / "eda.md")

    data_build.main()
    eda_build.main()

    report = (tmp_path / "reports" / "eda.md").read_text()
    assert "## Key findings" in report
    assert "## Cross-check against LBNL" in report
    assert "TBD" not in report
    pngs = sorted(p.name for p in (tmp_path / "reports" / "figures").glob("*.png"))
    assert len(pngs) == 6
