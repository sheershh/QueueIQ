"""Feature groups used for explanations (decisions D-039).

Explaining 16 correlated features one by one gives noisy, hard-to-read answers (the five county
congestion columns move together). Grouping them into seven ideas a developer recognises keeps
explanations readable, and makes exact Shapley values cheap (2^7 = 128 combinations).
"""

GROUPS: dict[str, list[str]] = {
    "size": ["log_capacity_mw"],
    "technology": ["fuel_group", "is_hybrid", "has_storage"],
    "grid operator": ["iso"],
    "state": ["state"],
    "county congestion": [
        "county_known",
        "county_prior_requests_5y",
        "county_prior_mw_5y",
        "county_prior_withdrawn_5y",
        "county_prior_withdrawn_share_5y",
    ],
    "state congestion": [
        "state_prior_requests_5y",
        "state_prior_mw_5y",
        "state_prior_withdrawn_5y",
        "state_prior_withdrawn_share_5y",
    ],
    "state momentum": ["state_prior_requests_1y"],
}

# What the explanation describes: the two numbers the dashboard headlines.
TARGETS = {
    "withdrawn_3y": ("withdrawn", 36, "Chance of withdrawing within 3 years"),
    "operational_5y": ("operational", 60, "Chance of reaching operation within 5 years"),
}


def describe(group: str, row) -> str:
    """Plain-language value of a group for one project (row: a Series of raw features)."""
    if group == "size":
        import math

        return f"{math.expm1(row['log_capacity_mw']):,.0f} MW"
    if group == "technology":
        parts = [str(row["fuel_group"])]
        if row["has_storage"] and row["fuel_group"] != "battery":
            parts.append("with storage")
        if row["is_hybrid"]:
            parts.append("hybrid")
        return ", ".join(parts)
    if group == "grid operator":
        return str(row["iso"])
    if group == "state":
        return str(row["state"])
    if group == "county congestion":
        if not row["county_known"]:
            return "county unknown"
        n, wd = row["county_prior_requests_5y"], row["county_prior_withdrawn_5y"]
        return f"{n:.0f} earlier requests in the county in 5 years, {wd:.0f} already withdrawn"
    if group == "state congestion":
        n, share = row["state_prior_requests_5y"], row["state_prior_withdrawn_share_5y"]
        share_txt = "" if share != share else f", {100 * share:.0f}% already withdrawn"
        return f"{n:.0f} earlier requests in the state in 5 years{share_txt}"
    if group == "state momentum":
        return f"{row['state_prior_requests_1y']:.0f} requests in the state in the past year"
    raise KeyError(group)
