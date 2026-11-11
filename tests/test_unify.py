import pandas as pd

from queueiq.cleaning.unify import make_project_ids, to_unified
from tests.conftest import make_raw_frame


def test_project_ids_are_unique_and_suffix_reused_ids():
    ids = make_project_ids(
        pd.Series(["PJM", "PJM", "MISO", "PJM"]), pd.Series(["A1", "A1", "A1", "B2"])
    )
    assert ids.tolist() == ["PJM::A1", "PJM::A1#2", "MISO::A1", "PJM::B2"]


def test_unify_normalizes_fields_and_keeps_every_row():
    raw = make_raw_frame(
        [
            {
                "type_1": "Offshore Wind",
                "type_clean": "Offshore Wind",
                "fips_code": 1001,
                "state": " va ",
            },
            {"type_1": "Solar", "type_2": "Battery", "type_clean": "Solar+Battery"},
            {"type_1": "Wind", "q_status": "Withdrawn", "project_type": None},
        ]
    )
    df = to_unified(raw)

    assert len(df) == len(raw)
    assert df["fuel_type"].tolist() == ["offshore_wind", "solar", "wind"]
    assert df["fips_code"].iloc[0] == "01001"  # leading zero restored
    assert df["state"].iloc[0] == "VA"
    assert df["is_hybrid"].tolist() == [False, True, False]
    assert df["has_storage"].tolist() == [False, True, False]
    assert df["status"].iloc[2] == "withdrawn"
    assert pd.isna(df["project_type"].iloc[2])
    assert df["queue_year"].iloc[0] == 2015
