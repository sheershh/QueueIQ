"""Decide which rows the survival models may use, and record why each other row is out.

Rows are never deleted from the processed dataset. Each row gets in_model_scope
(True/False) and, if False, the FIRST rule it failed (scope_exclusion_reason).
Applying the rules in order gives the cleaning waterfall: rows remaining after each step.
"""

from collections.abc import Callable

import pandas as pd

# Regions where both event dates are ~90%+ complete (docs/decisions.md D-011).
MODEL_ISOS = ("PJM", "MISO", "CAISO", "SPP")
# Before 2010, 15-60% of withdrawal dates are missing even in these regions (D-012).
MIN_ENTRY_YEAR = 2010
NON_GENERATION_TYPES = ("surplus", "upgrade", "replacement")

# (step name, reason shown in reports, rule that marks rows to EXCLUDE)
ScopeRule = tuple[str, str, Callable[[pd.DataFrame], pd.Series]]

SCOPE_RULES: list[ScopeRule] = [
    ("unknown_status", "Status is 'unknown'", lambda d: d["flag_unknown_status"]),
    ("missing_queue_date", "No queue entry date", lambda d: d["flag_missing_queue_date"]),
    (
        "not_new_generation",
        "Surplus/upgrade/replacement request, not a new plant",
        lambda d: d["project_type"].isin(NON_GENERATION_TYPES).fillna(False),
    ),
    (
        "region_out_of_scope",
        f"Region outside {', '.join(MODEL_ISOS)} (event dates too incomplete)",
        lambda d: ~d["iso"].isin(MODEL_ISOS).fillna(False),
    ),
    (
        "entered_before_2010",
        f"Entered before {MIN_ENTRY_YEAR} (withdrawal dates often missing)",
        lambda d: (d["queue_year"] < MIN_ENTRY_YEAR).fillna(False),
    ),
    (
        "duplicate_queue_id",
        "Reused queue ID; cannot tell which row is correct",
        lambda d: d["flag_duplicate_queue_id"],
    ),
    ("missing_event_date", "Event date missing", lambda d: d["flag_missing_event_date"]),
    ("event_before_entry", "Event date before entry date", lambda d: d["flag_event_before_entry"]),
    (
        "nonpositive_capacity",
        "Capacity zero, negative, or missing",
        lambda d: d["flag_nonpositive_capacity"],
    ),
]


def apply_model_scope(
    df: pd.DataFrame, rules: list[ScopeRule] = SCOPE_RULES
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df with in_model_scope + scope_exclusion_reason, waterfall table)."""
    out = df.copy()
    reason = pd.Series(pd.NA, index=out.index, dtype="string")
    remaining = pd.Series(True, index=out.index)
    steps = [
        {
            "step": "start",
            "reason": "All rows in the LBNL sheet",
            "rows_removed": 0,
            "rows_remaining": int(remaining.sum()),
        }
    ]

    for name, description, rule in rules:
        hit = remaining & rule(out).astype(bool)
        reason[hit] = name
        remaining &= ~hit
        steps.append(
            {
                "step": name,
                "reason": description,
                "rows_removed": int(hit.sum()),
                "rows_remaining": int(remaining.sum()),
            }
        )

    out["in_model_scope"] = remaining
    out["scope_exclusion_reason"] = reason
    return out, pd.DataFrame(steps)
