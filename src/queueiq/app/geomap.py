"""The 3D county map (decisions D-059).

Counties in the covered states are extruded: height = requests filed there in the 5 years before
the entry date (square-root scale, so a few very busy counties don't flatten the rest), color =
share of those requests already withdrawn. States the model does not cover are flat grey.
Clicking a county selects it. Shapes: US Census cartographic boundaries (scripts/build_geo.py).
"""

import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pydeck as pdk

from queueiq.style import CROWDING_RAMP, MAP_NO_DATA, UI, UI_FONT

ASSETS = Path(__file__).resolve().parents[1] / "assets"  # shared with the service
METERS_PER_SQRT_REQUEST = 12_000
HOME = {"latitude": 38.6, "longitude": -95.5, "zoom": 3.25, "pitch": 45, "bearing": -8}


def rgb(hexc: str, alpha: int = 255) -> list[int]:
    return [int(hexc[i : i + 2], 16) for i in (1, 3, 5)] + [alpha]


@lru_cache(maxsize=2)
def shapes(name: str) -> dict:
    return json.loads((ASSETS / f"{name}.geojson").read_text())


def ramp_color(share: float) -> list[int]:
    if share != share:  # NaN: no requests in the window
        return rgb(MAP_NO_DATA)
    return rgb(CROWDING_RAMP[min(int(share * len(CROWDING_RAMP)), len(CROWDING_RAMP) - 1)])


def county_features(crowding: pd.DataFrame, selected: str | None) -> list[dict]:
    feats = []
    for f in shapes("counties")["features"]:
        p = f["properties"]
        row = crowding.loc[p["key"]] if p["key"] in crowding.index else None
        n = int(row["requests_5y"]) if row is not None else 0
        share = float(row["withdrawn_share"]) if row is not None else float("nan")
        withdrawn = "no requests" if n == 0 else f"{100 * share:.0f}% already withdrawn"
        feats.append(
            {
                **f,
                "properties": {
                    **p,
                    "elevation": METERS_PER_SQRT_REQUEST * math.sqrt(n),
                    "color": rgb(UI["ink"]) if p["key"] == selected else ramp_color(share),
                    "tooltip": f"{p['name']}, {p['state']}\n{n} request{'' if n == 1 else 's'} "
                    "in the last 5 years, "
                    f"{withdrawn}\nClick to choose this county",
                },
            }
        )
    return feats


def _points(coords) -> list:
    if isinstance(coords[0], (int, float)):
        return [coords]
    return [pt for c in coords for pt in _points(c)]


def centroid(key: str) -> tuple[float, float] | None:
    """(latitude, longitude): mean of the county outline's points; enough to aim the map."""
    for f in shapes("counties")["features"]:
        if f["properties"]["key"] == key:
            pts = _points(f["geometry"]["coordinates"])
            return (sum(p[1] for p in pts) / len(pts), sum(p[0] for p in pts) / len(pts))
    return None


def deck(crowding: pd.DataFrame, selected: str | None) -> pdk.Deck:
    states = [
        {
            **f,
            "properties": {
                **f["properties"],
                "fill": [0, 0, 0, 0] if f["properties"]["covered"] else rgb(MAP_NO_DATA),
                "tooltip": f"{f['properties']['name']}\n"
                + ("Covered" if f["properties"]["covered"] else "No reliable queue history here"),
            },
        }
        for f in shapes("states")["features"]
    ]
    view = dict(HOME)
    c = centroid(selected) if selected else None
    if c:
        view.update(latitude=c[0], longitude=c[1], zoom=5.0, pitch=45, bearing=-12)
    layers = [
        pdk.Layer(
            "GeoJsonLayer",
            {"type": "FeatureCollection", "features": states},
            id="states",
            pickable=True,
            stroked=True,
            filled=True,
            get_fill_color="properties.fill",
            get_line_color=rgb("#FFFFFF"),
            line_width_min_pixels=1,
        ),
        pdk.Layer(
            "GeoJsonLayer",
            {"type": "FeatureCollection", "features": county_features(crowding, selected)},
            id="counties",
            pickable=True,
            auto_highlight=True,
            highlight_color=rgb(UI["accent"], 220),
            extruded=True,
            wireframe=False,
            get_elevation="properties.elevation",
            get_fill_color="properties.color",
            get_line_color=rgb("#FFFFFF", 160),
            material={"ambient": 0.55, "diffuse": 0.6, "shininess": 20},
        ),
    ]
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(**view, transitionDuration=900),
        map_provider="carto",
        map_style=pdk.map_styles.CARTO_LIGHT,
        tooltip={
            "text": "{tooltip}",
            "style": {
                "backgroundColor": UI["ink"],
                "color": UI["paper"],
                "fontFamily": UI_FONT,
                "fontSize": "14px",
            },
        },
    )


def selected_key(event) -> str | None:
    """County key from st.pydeck_chart's selection state, or None."""
    try:
        objs = event["selection"]["objects"].get("counties") or []
    except (KeyError, TypeError, AttributeError):
        return None
    if not objs:
        return None
    obj = objs[0]
    props = obj.get("properties", obj) if isinstance(obj, dict) else {}
    return props.get("key")
