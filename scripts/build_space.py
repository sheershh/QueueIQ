"""Assemble the Hugging Face Space in build/hf_space/ (`make space`). Uploads nothing.

Contents: the Space README and entry point (deploy/huggingface/), the queueiq package, the
Streamlit theme, requirements pinned from uv.lock (runtime group only), the three deployed model
files, and the queue history trimmed to the columns the service reads.
"""

import shutil
import subprocess
from pathlib import Path

import pandas as pd

from queueiq.explain import deploy
from queueiq.service.features import HISTORY_COLUMNS
from queueiq.service.predict import ENTRIES_FILE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "hf_space"
MODEL_FILES = (deploy.MODEL_FILE, deploy.META_FILE, deploy.BACKGROUND_FILE)
KEEP_COLUMNS = [*HISTORY_COLUMNS, "in_model_scope"]  # everything Service.load reads


def main() -> None:
    missing = [p for p in (*MODEL_FILES, ENTRIES_FILE) if not p.exists()]
    if missing:
        raise SystemExit(
            f"Missing {[str(p.relative_to(ROOT)) for p in missing]}: run make data && make train"
        )
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(ROOT / "deploy" / "huggingface", OUT)
    shutil.copytree(
        ROOT / "src" / "queueiq",
        OUT / "src" / "queueiq",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(ROOT / ".streamlit", OUT / ".streamlit")
    shutil.copy(ROOT / "LICENSE", OUT / "LICENSE")
    (OUT / "models").mkdir()
    for f in MODEL_FILES:
        shutil.copy(f, OUT / "models" / f.name)
    (OUT / "data" / "processed").mkdir(parents=True)
    pd.read_parquet(ENTRIES_FILE)[KEEP_COLUMNS].to_parquet(
        OUT / "data" / "processed" / ENTRIES_FILE.name, index=False
    )
    req = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-default-groups",
            "--no-hashes",
            "--no-emit-project",
            "--format",
            "requirements-txt",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (OUT / "requirements.txt").write_text(req)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"Built {OUT.relative_to(ROOT)} ({size:.0f} MB). Nothing was uploaded.")


if __name__ == "__main__":
    main()
