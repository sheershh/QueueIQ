"""Helpers that build tiny, hand-made LBNL-shaped tables for tests (no real data needed)."""

import pandas as pd
import pytest

from queueiq.cleaning.flags import add_quality_flags
from queueiq.cleaning.outcomes import assign_outcomes
from queueiq.cleaning.unify import to_unified
from queueiq.ingestion.lbnl import EXPECTED_COLUMNS

CUTOFF = pd.Timestamp("2025-12-31").date()

DEFAULT_ROW = {
    "q_id": "Q1",
    "q_status": "active",
    "q_date": pd.Timestamp("2015-06-01"),
    "region": "PJM",
    "entity": "PJM",
    "state": "VA",
    "county": "Fairfax",
    "fips_code": 51059,
    "project_type": "Generation",
    "type_1": "Solar",
    "type_clean": "Solar",
    "mw_1": 100.0,
}


def make_raw_frame(rows: list[dict]) -> pd.DataFrame:
    """Raw-sheet-shaped DataFrame: every expected column, defaults filled in, overrides applied."""
    records = []
    for i, overrides in enumerate(rows):
        rec = dict.fromkeys(EXPECTED_COLUMNS)
        rec.update(DEFAULT_ROW)
        rec["q_id"] = f"Q{i + 1}"
        rec.update(overrides)
        records.append(rec)
    df = pd.DataFrame.from_records(records, columns=EXPECTED_COLUMNS)
    for col in ["q_date", "prop_date", "on_date", "wd_date", "ia_date"]:
        df[col] = pd.to_datetime(df[col])
    return df


def pipeline(rows: list[dict]) -> pd.DataFrame:
    """unify -> outcomes -> flags, the same order as `make data`."""
    df = to_unified(make_raw_frame(rows))
    df = assign_outcomes(df, CUTOFF)
    return add_quality_flags(df, CUTOFF)


@pytest.fixture
def raw_rows():
    return make_raw_frame
