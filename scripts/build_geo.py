"""County and state shapes for the dashboard map (`make geo`, decisions D-059).

Source: US Census Bureau cartographic boundary files, 2024, 1:20,000,000 (public domain).
The zip files are downloaded to data/raw/geo/ (never modified) and checked against the SHA-256
values below before use. Output (committed, small):
    src/queueiq/assets/counties.geojson  counties in the states the model covers
    src/queueiq/assets/states.geojson    all states, flagged covered or not
Coordinates are rounded to 3 decimals (about 100 m), plenty for a national map.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd
import shapefile  # pyshp

from queueiq import config
from queueiq.service.inputs import Options

BASE = "https://www2.census.gov/geo/tiger/GENZ2024/shp"
FILES = {
    "cb_2024_us_county_20m.zip": "c9a54fd95422436b8822e0b070a45a9bc3de08eb13f943cdba66e7a9c2a135ff",
    "cb_2024_us_state_20m.zip": "7bc773d83c01b6df69b8aada9c2b5983d97f22f4551947cc5c00b87c663223ef",
}
RAW = config.RAW_DIR / "geo"
OUT = config.PROJECT_ROOT / "src" / "queueiq" / "assets"
DIGITS = 3


def fetch(name: str) -> Path:
    path = RAW / name
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{BASE}/{name}", path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FILES[name]:
        raise SystemExit(f"{name}: SHA-256 {digest} does not match the pinned value")
    return path


def rounded(geometry: dict) -> dict:
    def walk(x):
        return round(x, DIGITS) if isinstance(x, float) else [walk(v) for v in x]

    return {"type": geometry["type"], "coordinates": walk(geometry["coordinates"])}


def features(path: Path, keep, props) -> list[dict]:
    reader = shapefile.Reader(str(path))
    out = []
    for sr in reader.iterShapeRecords():
        rec = sr.record.as_dict()
        if keep(rec):
            out.append(
                {
                    "type": "Feature",
                    "properties": props(rec),
                    "geometry": rounded(sr.shape.__geo_interface__),
                }
            )
    return out


def main() -> None:
    options = Options.from_entries(pd.read_parquet(config.PROCESSED_DIR / "queue_entries.parquet"))
    covered = {s for states in options.states.values() for s in states}
    counties = features(
        fetch("cb_2024_us_county_20m.zip"),
        keep=lambda r: r["STUSPS"] in covered,
        props=lambda r: {"key": r["GEOID"], "name": r["NAME"], "state": r["STUSPS"]},
    )
    states = features(
        fetch("cb_2024_us_state_20m.zip"),
        keep=lambda r: r["STUSPS"] != "PR",
        props=lambda r: {
            "state": r["STUSPS"],
            "name": r["NAME"],
            "covered": r["STUSPS"] in covered,
        },
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for name, feats in (("counties", counties), ("states", states)):
        path = OUT / f"{name}.geojson"
        path.write_text(
            json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":"))
        )
        size = path.stat().st_size / 1e6
        print(
            f"Wrote {path.relative_to(config.PROJECT_ROOT)}: {len(feats)} features, {size:.1f} MB"
        )


if __name__ == "__main__":
    main()
