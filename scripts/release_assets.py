"""Build the files the hosted dashboard downloads, and their manifest (`make assets`).

Writes build/assets/ (the three deployed model files and the queue history trimmed to the columns
the service reads) and deploy/streamlit_cloud/assets.json (name, destination, size, SHA-256).
Publishing them is a separate, explicit step:
    gh release create <tag> build/assets/* --title ... --notes-file ...
"""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from queueiq.explain import deploy
from queueiq.service.features import HISTORY_COLUMNS
from queueiq.service.predict import ENTRIES_FILE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "assets"
MANIFEST = ROOT / "deploy" / "streamlit_cloud" / "assets.json"
REPO = "sheershh/QueueIQ"
TAG = "deploy-assets-v1"
HISTORY_NAME = "queue_history.parquet"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    files = {}
    for src in (deploy.MODEL_FILE, deploy.META_FILE, deploy.BACKGROUND_FILE):
        shutil.copy(src, OUT / src.name)
        files[src.name] = src.relative_to(ROOT)
    pd.read_parquet(ENTRIES_FILE)[[*HISTORY_COLUMNS, "in_model_scope"]].to_parquet(
        OUT / HISTORY_NAME, index=False
    )
    files[HISTORY_NAME] = ENTRIES_FILE.relative_to(ROOT)
    manifest = {
        "repo": REPO,
        "tag": TAG,
        "files": {
            name: {
                "path": str(dest),
                "bytes": (OUT / name).stat().st_size,
                "sha256": sha256(OUT / name),
            }
            for name, dest in files.items()
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(v["bytes"] for v in manifest["files"].values()) / 1e6
    print(f"Wrote {OUT.relative_to(ROOT)} ({total:.0f} MB) and {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
