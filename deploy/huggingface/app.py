"""Hugging Face Spaces entry point: runs the QueueIQ dashboard from this folder.

The Space holds a copy of src/queueiq, the deployed model (models/) and a trimmed queue history
(data/processed/queue_entries.parquet), assembled by `make space` in the QueueIQ repository.
"""

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("QUEUEIQ_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

runpy.run_path(str(ROOT / "src" / "queueiq" / "app" / "main.py"), run_name="__main__")
