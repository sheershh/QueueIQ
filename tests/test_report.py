import pandas as pd

from queueiq.cleaning.report import missingness_table, to_markdown
from tests.conftest import pipeline


def test_event_date_missingness_counts_only_rows_with_that_status():
    df = pipeline(
        [
            {"q_status": "withdrawn", "wd_date": pd.Timestamp("2018-01-01")},
            {"q_status": "withdrawn", "wd_date": None},
            {"q_status": "active"},  # no withdrawal date, but it is not supposed to have one
        ]
    )
    table = missingness_table(df, "iso")
    assert table.loc["PJM", "withdrawal_date (withdrawn rows)"] == 50.0
    assert table.loc["All", "rows"] == 3


def test_markdown_escapes_pipes_and_blanks_missing_values():
    md = to_markdown(pd.DataFrame({"id": ["a|b"], "x": [None]}))
    assert "a\\|b" in md
    assert md.splitlines()[-1] == "| a\\|b |  |"
