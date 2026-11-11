"""Data-quality flags. Each flag marks suspicious rows; nothing is changed or removed here.

Every flag is a boolean column named flag_<name>. FLAGS documents what each one means
and is used for the data-quality report.
"""

from datetime import date

import pandas as pd

HUGE_CAPACITY_MW = 3000  # larger than almost any single US plant; see decisions D-013

FLAGS = {
    "unknown_status": "Status is not one of active/withdrawn/suspended/operational.",
    "missing_queue_date": "No queue entry date, so no time origin.",
    "missing_event_date": "Operational or withdrawn, but the matching event date is empty.",
    "event_before_entry": "Event date is earlier than the queue entry date (impossible).",
    "event_on_entry_day": "Event happened on the queue entry date (time = 0).",
    "event_after_cutoff": "Event date is after the data cutoff; censored at the cutoff.",
    "nonpositive_capacity": "Capacity is zero, negative, or missing.",
    "huge_capacity": f"Capacity above {HUGE_CAPACITY_MW} MW (possible unit or entry error).",
    "duplicate_queue_id": "Same (entity, queue ID) appears on more than one row.",
    "status_date_conflict": (
        "Dates contradict status: active with an operation/withdrawal date, withdrawn with "
        "an operation date, or operational with a withdrawal date."
    ),
}


def add_quality_flags(df: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Expects columns from unify.to_unified and outcomes.assign_outcomes."""
    out = df.copy()
    status = out["status"]
    is_event = out["outcome"].isin(["operational", "withdrawn"])
    base_id = out["entity"].fillna("NA") + "::" + out["queue_id"].fillna("NA")

    flags = {
        "unknown_status": ~status.isin(["active", "withdrawn", "suspended", "operational"]),
        "missing_queue_date": out["queue_date"].isna(),
        "missing_event_date": is_event & out["end_date"].isna(),
        "event_before_entry": is_event & (out["end_date"] < out["queue_date"]),
        "event_on_entry_day": is_event & (out["end_date"] == out["queue_date"]),
        "event_after_cutoff": out["censored_at_cutoff_after_event"],
        "nonpositive_capacity": ~(out["capacity_mw"] > 0),
        "huge_capacity": out["capacity_mw"] > HUGE_CAPACITY_MW,
        "duplicate_queue_id": base_id.duplicated(keep=False),
        "status_date_conflict": (
            (
                (status == "active")
                & (out["operation_date"].notna() | out["withdrawal_date"].notna())
            )
            | ((status == "withdrawn") & out["operation_date"].notna())
            | ((status == "operational") & out["withdrawal_date"].notna())
        ),
    }
    assert set(flags) == set(FLAGS), "every flag needs a description in FLAGS"
    for name, mask in flags.items():
        out[f"flag_{name}"] = mask.fillna(False).astype(bool)
    return out
