import pandas as pd
import pytest

from queueiq.cleaning.flags import FLAGS
from tests.conftest import pipeline

T = pd.Timestamp

CASES = {
    "unknown_status": {"q_status": "unknown"},
    "missing_queue_date": {"q_date": None},
    "missing_event_date": {"q_status": "withdrawn", "wd_date": None},
    "event_before_entry": {"q_status": "operational", "on_date": T("2014-01-01")},
    "event_on_entry_day": {"q_status": "withdrawn", "wd_date": T("2015-06-01")},
    "event_after_cutoff": {"q_status": "withdrawn", "wd_date": T("2026-01-10")},
    "nonpositive_capacity": {"mw_1": 0.0},
    "huge_capacity": {"mw_1": 16875.0},
    "status_date_conflict": {"q_status": "active", "on_date": T("2020-01-01")},
}


@pytest.mark.parametrize("flag,overrides", CASES.items())
def test_each_flag_fires_on_its_case_and_not_on_a_clean_row(flag, overrides):
    clean = {"q_status": "withdrawn", "wd_date": T("2018-01-01")}
    df = pipeline([overrides, clean])
    assert df[f"flag_{flag}"].tolist() == [True, False]


def test_duplicate_queue_id_flags_every_copy():
    df = pipeline(
        [
            {"q_id": "A", "entity": "PJM"},
            {"q_id": "A", "entity": "PJM"},
            {"q_id": "A", "entity": "MISO"},
        ]
    )
    assert df["flag_duplicate_queue_id"].tolist() == [True, True, False]


def test_every_flag_is_tested_and_documented():
    assert set(CASES) | {"duplicate_queue_id"} == set(FLAGS)


def test_flags_never_change_or_drop_rows():
    df = pipeline([{"mw_1": -5.0}, {"q_date": None}])
    assert len(df) == 2
    assert df["capacity_mw"].tolist() == [-5.0, 100.0]
