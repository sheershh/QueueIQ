"""What a user can ask about: one hypothetical project entering the queue (decisions D-047).

Two layers of validation:
1. `ProjectInput` (pydantic) checks types and simple ranges (capacity > 0, known fuel and ISO).
2. `Options.check` checks against the data: the state must have enough in-scope history in that
   grid operator, the county must be in that state, and capacity must lie inside the range the
   model was trained on (a forest cannot extrapolate beyond it).
"""

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from queueiq.features.congestion import county_key
from queueiq.features.tabular import MAIN_FUELS

ISOS = ("PJM", "MISO", "CAISO", "SPP")  # model scope (D-019)
FUELS = (*MAIN_FUELS, "other")
MIN_STATE_ROWS = 20  # states with fewer in-scope projects in that ISO are not offered
MAX_COMPARE = 5


class ProjectInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field("Project", min_length=1, max_length=40, description="Label for comparisons")
    iso: Literal["PJM", "MISO", "CAISO", "SPP"]
    state: str = Field(min_length=2, max_length=2, description="Two-letter state code")
    county: str | None = Field(
        None, description="County key from /options (5-digit FIPS); empty = county not specified"
    )
    fuel: Literal["solar", "wind", "battery", "gas", "other"]
    capacity_mw: float = Field(gt=0, description="Capacity of the primary resource, MW")
    storage: bool = Field(False, description="Paired with battery storage (ignored for battery)")

    def flags(self) -> tuple[bool, bool]:
        """(has_storage, is_hybrid), following how LBNL rows combine them (D-047)."""
        if self.fuel == "battery":
            return True, False
        return self.storage, self.storage


class InputError(ValueError):
    """Input is well-formed but outside what the model can answer."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass
class County:
    key: str
    name: str
    fips_code: str | None
    county: str | None
    requests: int  # LBNL rows with this county key (any year, any region)


@dataclass
class Options:
    states: dict[str, list[str]]  # iso -> states offered
    counties: dict[str, list[County]]  # state -> counties seen in LBNL data (any region)
    capacity_max: float
    largest_state: dict[str, str]  # iso -> state with the most in-scope projects (form default)

    @classmethod
    def from_entries(cls, entries: pd.DataFrame) -> "Options":
        scoped = entries[entries["in_model_scope"]]
        counts = scoped.groupby(["iso", "state"]).size()
        states = {
            iso: sorted(counts[iso][counts[iso] >= MIN_STATE_ROWS].index.tolist())
            if iso in counts.index.get_level_values(0)
            else []
            for iso in ISOS
        }
        offered = {s for v in states.values() for s in v}
        df = entries[entries["state"].isin(offered)].assign(key=county_key(entries))
        df = df[df["key"].notna()]
        counties: dict[str, list[County]] = {}
        for (state, key), g in df.groupby(["state", "key"], sort=True):
            names = g["county"].dropna()
            name = names.mode().iloc[0] if len(names) else key
            fips = g["fips_code"].dropna()
            counties.setdefault(state, []).append(
                County(
                    key=key,
                    name=str(name),
                    fips_code=str(fips.iloc[0]) if len(fips) else None,
                    county=str(name) if len(names) else None,
                    requests=len(g),
                )
            )
        for state in counties:
            counties[state].sort(key=lambda c: c.name)
        largest = {iso: counts[iso].idxmax() for iso in ISOS if states[iso]}
        return cls(states, counties, float(scoped["capacity_mw"].max()), largest)

    def busiest_county(self, state: str) -> County | None:
        """Form default: the county with the most past requests in the state."""
        return max(self.counties.get(state, []), key=lambda c: c.requests, default=None)

    def county(self, state: str, key: str) -> County | None:
        return next((c for c in self.counties.get(state, []) if c.key == key), None)

    def check(self, p: ProjectInput) -> None:
        problems = []
        if p.state not in self.states[p.iso]:
            problems.append(
                f"{p.state} is not offered for {p.iso}: the model has fewer than "
                f"{MIN_STATE_ROWS} past projects there. Choose one of "
                f"{', '.join(self.states[p.iso])}."
            )
        elif p.county is not None and self.county(p.state, p.county) is None:
            problems.append(f"County {p.county!r} is not a {p.state} county in the LBNL data.")
        if p.capacity_mw > self.capacity_max:
            problems.append(
                f"Capacity must be at most {self.capacity_max:,.0f} MW, the largest project "
                "the model was trained on."
            )
        if problems:
            raise InputError(problems)

    def as_dict(self) -> dict:
        return {
            "isos": list(ISOS),
            "fuels": list(FUELS),
            "states": self.states,
            "counties": {
                s: [{"key": c.key, "name": c.name} for c in v] for s, v in self.counties.items()
            },
            "capacity_mw": {"min_exclusive": 0, "max": self.capacity_max},
            "max_compare": MAX_COMPARE,
        }
