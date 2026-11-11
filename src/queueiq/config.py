"""Project-wide paths and constants.

Every module should import paths and the random seed from here instead of
hard-coding them, so the whole pipeline can be moved or re-seeded in one place.
"""

from pathlib import Path

# src/queueiq/config.py -> parents[2] is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Single seed used by every stochastic step (splits, bootstraps, model training).
SEED = 42


def ensure_dirs() -> None:
    """Create the data and report directories if they do not exist yet."""
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
