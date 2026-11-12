"""Streamlit Community Cloud entry point for the QueueIQ dashboard (decisions D-056).

The model and the trimmed queue history are not in git (D-045). On first start this script
downloads them from the repository's GitHub Release named in assets.json, checks each file's
SHA-256, and then runs the normal dashboard (src/queueiq/app/main.py). Locally, where `make train`
has already written the same files, nothing is downloaded.
"""

import hashlib
import json
import runpy
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
MANIFEST = json.loads(Path(__file__).with_name("assets.json").read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_assets() -> None:
    base = f"https://github.com/{MANIFEST['repo']}/releases/download/{MANIFEST['tag']}"
    for name, spec in MANIFEST["files"].items():
        dest = ROOT / spec["path"]
        if dest.exists():
            # Present already: built locally by `make train` (never overwritten), or downloaded
            # by an earlier session. Downloads land in a .part file first, so a partial download
            # never takes this path.
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        urllib.request.urlretrieve(f"{base}/{name}", part)
        if sha256(part) != spec["sha256"]:
            part.unlink()
            raise RuntimeError(f"Checksum mismatch for {name}; refusing to load it.")
        part.replace(dest)


fetch_assets()
runpy.run_path(str(ROOT / "src" / "queueiq" / "app" / "main.py"), run_name="__main__")
