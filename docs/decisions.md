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

## Phase 1: Ingestion (2026-11-11)

### D-008: Manual download, verified by checksum
- **Decision:** `make data` does not download anything. The user downloads the file by hand, and the pipeline
  checks its SHA-256 against the value recorded in `docs/data_sources.md`. It stops if the file is missing or
  different.
- **Why:** LBNL requires a registration form and blocks automated requests (HTTP 403). A checksum still makes
  the run reproducible: anyone can confirm they have the exact bytes this project used.
- **Alternatives:** Scraping past the registration (against the site's terms) or committing the file to git
  (redistribution adds licensing obligations and the file is 15 MB).

### D-009: Read one sheet, never change raw files
- **Decision:** Read only `03. Complete Queue Data`. Fail if its 30 column names differ from what we expect.
  Save the sheet unchanged to `data/interim/` and do all cleaning in code.
- **Why:** If LBNL changes the layout in a future edition, the pipeline fails loudly instead of quietly
  misreading columns.

### D-010: Outcome mapping and censoring
- **Data cutoff:** 2025-12-31. The 2026 edition contains data "through the end of 2025".
- **Mapping (status, then outcome, then end date):**
  - `operational`: event `operational` at `operation_date`.
  - `withdrawn`: event `withdrawn` at `withdrawal_date`.
  - `active`: `censored` at the cutoff.
  - `suspended`: `censored` at the cutoff. A suspended project has not reached either final outcome and can
    resume. 124 suspended rows are in model scope.
  - `unknown`: outcome `unknown`, excluded from model scope. There are 10 such rows, and the status is not in
    the codebook.
- **Events dated after the cutoff** (46 in scope, all withdrawals dated in January 2026) are censored at the
  cutoff. As of 2025-12-31 they had not happened yet, so treating them as known events would use information
  from after the cutoff. They are flagged `flag_event_after_cutoff`.
- **Time scale:** months = days / 30.4375 (the average month length).
- **Which date is used:** only the date that matches the status. A withdrawn row that also has an
  operation date is timed by its withdrawal date (see D-013).
- **Alternatives:** Treat `suspended` as a third competing event. It is too rare (124 rows in scope) to model
  well. Exclude `suspended` rows. That would remove projects that were still in the queue, which biases the
  sample toward projects that had already finished.

### D-011: Model regions are PJM, MISO, CAISO, and SPP (chosen by the author)
- **Decision:** Only these four regions are in model scope. All 9 regions stay in the processed dataset for
  descriptive EDA.
- **Why:** Survival models need event dates. Share of withdrawn rows that have a withdrawal date: CAISO 98%,
  PJM 98%, MISO 94%, SPP 73% overall (95%+ from 2010). By contrast: West 12%, Southeast 35%, NYISO 49%,
  ERCOT 49%. ISO-NE has no operation dates at all (0%). Full table in `reports/data_quality.md`.
- **Alternatives rejected:** Impute missing dates (this invents event timing and breaks rule 1). Drop undated
  rows in every region (this mostly removes withdrawn West and Southeast projects, so withdrawals would look
  rarer than they are). Add ERCOT, where about a quarter of withdrawals have no date.
- **Consequence:** Results describe these four ISOs only. The README and model card must say so.

### D-012: Model scope starts at entry year 2010 (chosen by the author)
- **Decision:** Requests that entered before 2010 are out of model scope in all four regions. This replaced an
  earlier rule that applied only to SPP.
- **Why:** In the model regions, withdrawal dates are 15–60% missing for each entry year before 2010 (2001:
  60%, 2008: 21%), and under ~5% from 2010 on. Dropping undated rows from the old cohorts would remove
  withdrawals unevenly, making those cohorts look more successful than they were.
- **Result:** Model scope covers entry years 2010–2025. The missing-event-date rule then removes 170 rows,
  about 1.1% of the 14,989 rows still in scope at that step.

### D-013: Flag suspicious rows instead of fixing them
- **Decision:** `src/queueiq/cleaning/flags.py` adds one boolean column per check. Counts and examples
  appear in `reports/data_quality.md`. Only impossible or unusable rows are excluded from model scope:
  missing entry date, missing event date, event before entry, capacity of 0 or less, unknown status, and
  duplicate IDs.
- **Kept in scope but flagged (for review):**
  - Capacity over 3,000 MW: 3 rows in scope. 3,000 MW is above almost any single US plant, but large wind,
    hydro, or hybrid requests can be real.
  - Event on the entry day (time = 0): 7 rows in scope. To be handled when choosing models in Phase 4,
    since some survival libraries reject time = 0.
  - Status and dates that contradict each other: 340 rows in scope (213 withdrawn rows that also have an
    operation date, 123 active rows that have an operation or withdrawal date, and 4 operational rows that
    have a withdrawal date). LBNL's status field is treated as the source of truth, and the event is timed
    by the date that matches it. A likely explanation is that the extra dates are planned or partial dates,
    but this has not been verified.

### D-014: Project ID and duplicate queue IDs
- **Decision:** `project_id = <entity>::<q_id>`, with `#2`, `#3`, ... added when the same pair repeats.
  Every copy of a repeated pair is excluded from model scope.
- **Why:** LBNL documents `entity + q_id` as the unique key, but 135 pairs (270 rows) repeat. The copies are
  not identical, so there is no way to tell which one is correct. None of them are in the model regions,
  so this rule removes 0 rows from scope.

### D-015: Capacity is the primary resource's MW; hybrids and storage are flags
- **Decision:** `capacity_mw = mw_1`. Separate `is_hybrid` and `has_storage` booleans. `fuel_type` is the
  primary type (`type_1`).
- **Why:** `mw_1` is never missing. LBNL notes that `mw_2` and `mw_3` are often missing for hybrids (their
  imputed values are excluded from this file), so adding the three columns together would give inconsistent
  totals.

### D-016: Only new generation requests
- **Decision:** `surplus`, `upgrade`, and `replacement` requests (200 rows) are excluded from model scope.
  Rows with a missing `project_type` are kept (305 in scope).
- **Why:** The project models whether a new plant gets built. Uprates and replacements behave differently.
  LBNL also notes that not every upgrade is labelled as one, so some rows with a missing type may be
  upgrades. This is a known limitation.

### D-017: Rows are never deleted, and every exclusion has a reason
- **Decision:** `queue_entries.parquet` keeps all 38,201 rows. `in_model_scope` and `scope_exclusion_reason`
  (the first rule a row fails) record the scope. `reports/cleaning_waterfall.csv` lists the row count after
  each rule. The build asserts that the processed dataset has as many rows as the raw sheet.
- **Why:** Every exclusion can be audited and reversed, and Phase 2 can still describe all regions.
