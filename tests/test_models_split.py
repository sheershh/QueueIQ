import numpy as np
import pandas as pd
import pytest

from queueiq.models.data import make_split, month_index, outcome_as_of
from tests.model_data import synthetic

T = pd.Timestamp


def test_month_index_puts_early_events_in_month_one():
    assert month_index(pd.Series([0.0, 0.4, 1.0, 35.99, 36.0])).tolist() == [1, 1, 2, 36, 37]


def test_outcomes_are_cut_off_at_the_as_of_date():
    e = pd.DataFrame(
        {
            "queue_date": [T("2015-01-01")] * 3,
            "outcome": ["withdrawn", "withdrawn", "censored"],
            "end_date": [T("2016-01-01"), T("2019-06-01"), T("2025-12-31")],
        }
    )
    time, event = outcome_as_of(e, T("2018-01-01"))
    assert event.tolist() == [2, 0, 0]  # the 2019 withdrawal was not known on 2018-01-01
    # 2015-01-01 -> 2018-01-01 is 1096 days = 36.01 months -> month index 37
    assert time.tolist()[1:] == [37, 37]


def test_split_keeps_training_before_origin_and_never_shows_later_outcomes():
    entries, features = synthetic()
    origin = T("2018-01-01")
    split = make_split(entries, features, origin, (2018, 2020), min_test_followup_months=60)
    assert split.X_train["queue_year"].max() <= 2017
    assert split.X_test["queue_year"].between(2018, 2020).all()
    # Hide-the-future: changing any outcome after the origin must not change training data.
    future = entries["end_date"] >= origin
    altered = entries.assign(
        outcome=entries["outcome"].where(~future, "operational"),
        end_date=entries["end_date"].where(~future, T("2019-01-01")),
    )
    split2 = make_split(altered, features, origin, (2018, 2020))
    np.testing.assert_array_equal(split.time_train, split2.time_train)
    np.testing.assert_array_equal(split.event_train, split2.event_train)


def test_test_rows_short_of_followup_are_dropped_and_counted():
    entries, features = synthetic()
    late = entries["queue_date"] == entries["queue_date"].max()
    entries.loc[late, ["outcome", "end_date"]] = ["censored", T("2025-12-31")]
    entries.loc[late, "queue_date"] = T("2020-12-31")
    entries["time_months"] = (entries["end_date"] - entries["queue_date"]).dt.days / 30.4375
    features.loc[late, "queue_year"] = 2020
    split = make_split(
        entries, features, T("2018-01-01"), (2018, 2020), min_test_followup_months=60
    )
    assert split.test_dropped_short_followup == int(late.sum())


def test_test_years_must_not_overlap_training():
    entries, features = synthetic()
    with pytest.raises(ValueError, match="origin"):
        make_split(entries, features, T("2018-01-01"), (2017, 2019))
