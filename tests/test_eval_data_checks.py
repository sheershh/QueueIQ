import pandas as pd
import pytest

from queueiq.evaluation.data_checks import gap_bound, missing_withdrawal_dates_by_year


def test_missing_dates_counted_only_for_otherwise_eligible_withdrawn_projects():
    T = pd.Timestamp
    entries = pd.DataFrame(
        {
            "queue_year": [2021, 2021, 2021, 2021, 2022],
            "status": ["withdrawn", "withdrawn", "withdrawn", "active", "withdrawn"],
            "withdrawal_date": [T("2022-01-01"), pd.NaT, pd.NaT, pd.NaT, T("2023-01-01")],
            "in_model_scope": [True, False, False, True, True],
            "scope_exclusion_reason": [
                None,
                "missing_event_date",
                "region_out_of_scope",
                None,
                None,
            ],
        }
    )
    t = missing_withdrawal_dates_by_year(entries)
    assert t.loc[2021, "withdrawn_projects"] == 2  # the out-of-region row doesn't count
    assert t.loc[2021, "missing_withdrawal_date"] == 1
    assert t.loc[2022, "missing_%"] == 0.0


def test_gap_bound_hand_example():
    """100 test projects, 30 withdrawn, 10 undated withdrawals excluded, 50% predicted:
    bound = 40/110 = 36.4%; gap 20 points; closes (0.3636 - 0.30) / 0.20 = 31.8%."""
    t = pd.DataFrame({"missing_withdrawal_date": [6, 4]}, index=[2021, 2022])
    b = gap_bound(t, (2021, 2022), n_test=100, cases=30, predicted=0.5)
    assert b["excluded_undated_withdrawals"] == 10
    assert b["observed_upper_bound"] == pytest.approx(40 / 110)
    assert b["share_of_gap_explainable"] == pytest.approx((40 / 110 - 0.3) / 0.2)
