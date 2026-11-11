# Decisions log

Each entry: what was decided, why, and the alternatives that were considered.

## Phase 0: Setup (2026-11-11)

### D-001: uv for environments and pinning
- **Decision:** Use `uv`. Ranges go in `pyproject.toml` and exact versions (with hashes) in `uv.lock`.
  `make setup` and CI run `uv sync --frozen`, which fails instead of silently upgrading if the lock is stale.
- **Why:** One tool handles the Python version, the virtualenv, and the lockfile. The lock makes installs
  reproducible across machines and CI.
- **Alternatives:** pip-tools (requirements.txt) works too, but needs a separate step to manage Python
  versions. Poetry is heavier and slower.

### D-002: Python 3.12
- **Decision:** Pin Python to 3.12 (`.python-version`, `requires-python = ">=3.12,<3.13"`).
- **Why:** The scientific packages planned for later (scikit-survival, lifelines, LightGBM, SHAP) reliably have
  prebuilt wheels for 3.12. Newer versions (3.13/3.14) are more likely to be missing wheels for some of them.
- **Alternatives:** 3.11 (older than it needs to be), 3.13+ (packaging risk).

### D-003: Add runtime dependencies phase by phase
- **Decision:** Phase 0 has no runtime dependencies, only dev tools (pytest, pytest-cov, ruff, pre-commit).
  Each later phase adds what it needs with `uv add`, in the same commit as the code that uses it.
- **Why:** Every dependency is then tied to a reason and a commit, which keeps reviews small.
- **Alternatives:** Install the whole stack up front. This is faster at first but makes the environment bigger
  and harder to explain.

### D-004: src layout
- **Decision:** Code lives in `src/queueiq/` and is installed as a package. Tests import it the way a user would.
- **Why:** Tests can't accidentally import from the working directory, so packaging mistakes show up early.

### D-005: Ruff for linting and formatting, run through uv in pre-commit
- **Decision:** Ruff does both linting and formatting (rule sets E, W, F, I, B, UP, SIM, NPY; line length 100).
  The pre-commit hooks call `uv run ruff`, so they use the version in `uv.lock`.
- **Why:** One fast tool replaces flake8, isort, and black. Running it through uv means the commit hook, CI,
  and local runs can't disagree about the ruff version.
- **Alternatives:** The official `ruff-pre-commit` repo. It pins its own ruff version, which can drift from the lock.

### D-006: Guard against committing data
- **Decision:** `data/` and `mlruns/` are gitignored, and pre-commit blocks files over 1 MB.
- **Why:** Raw data is re-downloaded by scripts (reproducibility). Committing it would bloat the repo and could
  raise licensing problems.

### D-007: Central config for paths and seed
- **Decision:** `src/queueiq/config.py` holds all paths and one global `SEED = 42`.
- **Why:** There is one place to change paths, and every random step is seeded from the same constant.
