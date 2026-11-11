import pandas as pd

from queueiq.cleaning.scope import SCOPE_RULES, apply_model_scope
from tests.conftest import pipeline

T = pd.Timestamp


def test_reason_is_the_first_rule_failed_and_no_rows_are_deleted():
    df = pipeline(
        [
            {},  # in scope (active PJM 2015)
            {"region": "West"},  # out: region
            {"region": "West", "q_date": None},  # out: missing date comes first
            {"q_date": T("2008-03-01")},  # out: before 2010
            {"q_status": "withdrawn", "wd_date": None},  # out: missing event date
            {"project_type": "Upgrade"},  # out: not new generation
            {"mw_1": 0.0},  # out: capacity
        ]
    )
    out, _ = apply_model_scope(df)

    assert len(out) == len(df)
    assert out["in_model_scope"].tolist() == [True] + [False] * 6
    assert out["scope_exclusion_reason"].tolist()[1:] == [
        "region_out_of_scope",
        "missing_queue_date",
        "entered_before_2010",
        "missing_event_date",
        "not_new_generation",
        "nonpositive_capacity",
    ]
    assert pd.isna(out["scope_exclusion_reason"].iloc[0])


def test_waterfall_adds_up():
    df = pipeline([{}, {"region": "ERCOT"}, {"q_status": "unknown"}, {"mw_1": -1.0}, {}])
    out, wf = apply_model_scope(df)

    assert list(wf["step"]) == ["start"] + [name for name, _, _ in SCOPE_RULES]
    assert wf["rows_remaining"].iloc[0] == len(df)
    assert wf["rows_removed"].sum() + wf["rows_remaining"].iloc[-1] == len(df)
    assert (wf["rows_remaining"].diff().dropna() == -wf["rows_removed"].iloc[1:]).all()
    assert wf["rows_remaining"].iloc[-1] == out["in_model_scope"].sum() == 2
