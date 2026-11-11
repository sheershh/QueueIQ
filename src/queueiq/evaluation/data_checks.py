"""Could data gaps explain part of the final-test calibration gap? (decisions D-044)

Withdrawn projects with no withdrawal date were excluded from the model scope (D-011/D-012,
rule `missing_event_date`). Each exclusion removes a withdrawal from its cohort, which lowers the
observed withdrawal rate. This module measures how many there are and bounds the effect.
"""

import pandas as pd


def missing_withdrawal_dates_by_year(entries: pd.DataFrame, first_year: int = 2010) -> pd.DataFrame:
    """Withdrawn projects that pass every other scope rule, by entry year: how many lack a date."""
    eligible = entries["in_model_scope"] | (
        entries["scope_exclusion_reason"] == "missing_event_date"
    )
    wd = entries[
        eligible & (entries["status"] == "withdrawn") & (entries["queue_year"] >= first_year)
    ]
    g = wd.groupby("queue_year")
    table = pd.DataFrame(
        {
            "withdrawn_projects": g.size(),
            "missing_withdrawal_date": g["withdrawal_date"].apply(lambda s: int(s.isna().sum())),
        }
    )
    table["missing_%"] = (
        100 * table["missing_withdrawal_date"] / table["withdrawn_projects"]
    ).round(1)
    table.index.name = "entry_year"
    return table


def gap_bound(
    table: pd.DataFrame, years: tuple[int, int], n_test: int, cases: int, predicted: float
) -> dict:
    """Upper bound: put every excluded undated withdrawal back into the test cohort and assume each
    happened within the horizon. The observed rate can't be higher than this from this cause."""
    k = int(table.loc[years[0] : years[1], "missing_withdrawal_date"].sum())
    observed = cases / n_test
    bound = (cases + k) / (n_test + k)
    gap = predicted - observed
    return {
        "excluded_undated_withdrawals": k,
        "observed": observed,
        "observed_upper_bound": bound,
        "predicted": predicted,
        "gap_points": 100 * gap,
        "share_of_gap_explainable": (bound - observed) / gap if gap else float("nan"),
    }
