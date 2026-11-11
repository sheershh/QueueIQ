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

## Phase 2: Exploratory analysis (2026-11-11)

### D-018: Our own Aalen-Johansen estimator
- **Decision:** `src/queueiq/eda/survival.py` implements Kaplan-Meier (share still in the queue) and
  Aalen-Johansen (cumulative share reaching each outcome) directly in about 30 lines. It is tested against a
  hand-worked example and against simple proportions when there is no censoring.
- **Why:** The estimator is the backbone of the EDA and of the Phase 4 baseline, so it has to be fully
  understood and explainable. Tied event times are handled exactly, with no random jitter. lifelines'
  `AalenJohansenFitter` adds random jitter to ties by default.
- **Alternatives:** lifelines or scikit-survival. These come in Phase 4 for Cox and random survival
  forests, where reimplementing would add nothing.
- **Limit:** No confidence bands in the EDA. Bootstrap intervals come in Phase 5 alongside the model metrics.

### D-019: Report an estimate only when at least 30 projects are still observed
- **Decision:** Curves stop, and 3- and 5-year estimates are left blank, once fewer than 30 projects in
  that group are still being observed (`MIN_AT_RISK = 30`).
- **Why:** The tail of a survival curve rests on the few projects observed that long, so it swings wildly.
  This is the EDA version of the evaluation horizon cap used in Phase 5.
- **Alternatives:** A fixed horizon cap per cohort (used for evaluation in Phase 5), or showing the full
  tail with confidence bands (planned for Phase 5).

### D-020: Raw shares and censoring-aware estimates are always labelled
- **Decision:** Raw status shares ("% withdrawn as of 2025") are always labelled "raw" and are never used
  as estimates of outcome probability. Median time to an event is shown only "among projects that had that
  event", with a warning that it is biased short.
- **Why:** These are the two classic mistakes with censored data. Keeping them visibly separate in the
  report is the point of this phase.

### D-021: One visual system, an `eda` subpackage, and committed figures
- **Decision:** `src/queueiq/style.py` fixes the palette for every chart:
  - operational = `#2a78d6` (blue) and withdrawn = `#eb6834` (orange). This pair passes a palette check for
    color-blind separation and contrast.
  - still in queue = neutral gray, and non-outcome series = a darker neutral, so blue and orange keep
    their meaning.
  - groups are shown as small multiples (one small panel per group) rather than extra colors.
  The EDA logic lives in a new `src/queueiq/eda/` subpackage. The figures (about 350 KB in total) are
  committed so the report renders on GitHub.
- **Why:** The same outcome is the same color everywhere, including the Phase 7 dashboard. Code in `src/`
  is testable, whereas notebooks are for exploring only.

### Cross-check result
Recomputed on all regions, our numbers match LBNL's published headline figures: capacity requesting
interconnection 2000–2020 is 12.8% operational, 75.2% withdrawn, and 10.6% active, against LBNL's
13% / 75% / 10%. The median time from request to operation for projects online in 2025 is 5.07 years,
against LBNL's "over 5 years". The full table is in `reports/eda.md`.

### D-022: Horizon estimates use only cohorts with complete follow-up (correction to Phase 2)
- **Problem found:** The first Phase 2 report gave a 5-year estimate for "2020–2025 entrants" of 0.7%
  operational. Only 2020 entrants have had 5 years: all 419 projects still observed at 5 years entered in
  2020. The pooled curve's early part came from 2021–2025 entrants and its 5-year value from 2020 entrants
  alone, so it described no real group. Because outcomes trend strongly with entry year, pooling cohorts
  that have different follow-up mixes them. In technical terms, censoring (set by entry year) is not
  independent of the outcome (which also depends on entry year).
- **Decision:** An estimate at horizon h uses only requests that entered at least h before the cutoff,
  so every one of them was observed for the full horizon. With complete follow-up there is no censoring
  before h, and the Aalen-Johansen value equals the plain share with the event by h. Cells are blank when
  fewer than 30 requests qualify. The report also gains a per-entry-year table, and its curves carry a note
  that their tails come from the oldest cohorts only.
- **Effect:**
  - Model scope within 5 years: 9.8% operational and 65.2% withdrawn (2010–2020 entrants). The pooled
    version had said 6.9% and 70.3%.
  - 2020 entrants: 2.2% operational within 5 years, against 15.7% for 2010–2014 entrants.
- **Lesson carried into modelling:** Under a time trend, a survival model needs entry year (or a proxy for
  it) as a feature, for the same reason. Evaluation must also respect the horizon cap.

## Phase 3: Features (2026-11-11)

### D-023: How congestion is defined
- **Decision:** For a project entering on date d, among **all** LBNL rows in the same county (and,
  separately, the same state), consider requests with `queue_date` in **[d − 5 years, d)**. Count them,
  sum their MW, and count how many have a `withdrawal_date` **strictly before d**. Also compute the
  withdrawn share and the number of state requests in the previous 12 months.
- **Strictly earlier, not "on or before":** 79% of in-scope projects share their entry date with another
  request in the same state. These are cluster submission windows. Nobody knows on the day who else files
  that day, and a withdrawal stamped on the entry day can't be put in order relative to the entry. Strict
  inequality is the leakage-safe choice. The cost is that same-cluster peers are not counted.
- **All regions, not just the model ISOs:** Congestion is about physical location, so a non-ISO utility's
  request in the same county also competes for the grid. Out-of-scope rows (pre-2010, upgrades, other
  regions) still count as earlier requests.
- **5-year look-back instead of all history:** With no look-back limit, 6.9% of a project's earlier county
  entries are undated withdrawals, mostly old ones. These are "zombie" requests that will never count as
  withdrawn. A 5-year window cuts that to 3.7% (see `reports/features.md`) and stops the counts growing just
  because the dataset starts in 1995.
- **County key:** 5-digit FIPS code. If that is missing, state + county name. Otherwise the county is
  unknown, and its features are missing with `county_known = False`.
- **Alternatives:** Count only the same ISO. Count only requests still active at entry; this needs exit
  dates, which are missing for some older rows, so undated withdrawals would wrongly count as active.
  Point of interconnection (POI) level, which is a stretch goal.

### D-024: Missing dates in the congestion counts
| Case | Handling | Why |
|---|---|---|
| Earlier request with no `queue_date` | Not counted | Can't be placed in time |
| Earlier request withdrawn, no `withdrawal_date` | Counted as a request, never as withdrawn | We can't know it had withdrawn by d; its 2025 status is future information |
| Withdrawal dated on or after d | Not counted as withdrawn | Not yet known at entry |
| Project with no county | County features missing, `county_known = False` | 13.6% of in-scope CAISO projects; Phase 4 must handle it (LightGBM: native; Cox: impute within fold + indicator) |
| No earlier requests | `*_withdrawn_share_5y` missing (0/0) | Undefined, not zero |
| Capacity ≤ 0 or missing | Counted as a request, adds 0 MW | The request existed; its size is unknown |
- **Known bias:** `*_prior_withdrawn_*` slightly understates true withdrawals (by about 3.7% of earlier
  county requests). This bias is the same at training and prediction time, so the model learns from the
  same imperfect signal it will see in use.

### D-025: Entry year under a time trend and a time-based split
- **Problem:** Test cohorts are always later than training cohorts, so `queue_year` always has to be
  extrapolated. Trees (LightGBM, random survival forests) treat any later year like the last training year.
  Cox extends the trend in a straight line.
- **Decision:**
  - Keep `queue_year` as a numeric feature (one-hot categories would be useless for unseen years).
  - Also provide at-entry congestion measures as proxies for the queue boom. For these, 85–100% of
    2020–2025 entries fall inside the range seen in 2010–2019 entries; for `queue_year` it is 0%
    (table in `reports/features.md`).
- **Why keep entry year at all:** Censoring is set by entry year, and outcome risk also depends on entry
  year. Survival models assume censoring is independent of the outcome *given the features*, which is only
  plausible if entry year (or a proxy) is a feature. D-022 shows the bias when it is left out.
- **To be decided by evidence in Phase 4/5:** an ablation with and without `queue_year` in the
  rolling-origin backtest.

### D-026: The feature code can't read outcomes
- `build_features` first narrows its input to `ENTRY_SAFE_COLUMNS` + `withdrawal_date`. `status`, outcome
  dates, interconnection-agreement data and the proposed online date never reach feature code. The
  proposed online date is excluded because it can be revised after entry.
- `state` is stored raw. Grouping rare states, encoding, and imputation happen in Phase 4 inside each
  training fold, because fitting them on all rows would let test-period data shape the features.
- **Leakage tests** (`tests/test_features_leakage.py`):
  1. A hand-built county example worked out by hand, covering every edge case.
  2. A "hide the future" test: deleting everything after a project's entry date and scrambling `status`
     must leave its features unchanged.
  3. A test that outcome columns are neither read nor produced.

  Planting three deliberate leaks (counting future entries, counting future withdrawals, counting same-day
  withdrawals) made the tests fail each time, which shows they can catch leaks.
- **No look at outcomes during feature design:** feature-versus-outcome relationships were not examined
  on the full dataset. That analysis happens in Phase 4 on training folds only, so test-period outcomes
  can't influence which features are chosen.

## Phase 4: Modeling (2026-11-11)

### D-027: One time-based split for every model
- **Training:** in-scope projects that entered before **2018-01-01**, with outcomes **cut off at that date**.
  An event dated on or after 2018-01-01 counts as censored at 2018-01-01. This is exactly what was knowable
  on the origin date. Without the cutoff, shocks after 2018 (such as later mass withdrawals) would reach the
  training labels.
- **Test:** projects that entered in **2018–2020**, with outcomes observed to the data cutoff. Every one of
  them has at least 5 years of follow-up, so 3- and 5-year metrics need no censoring correction. Two test
  projects that entered on 2020-12-31 are dropped and the count is reported. They were still active after
  5 calendar years, but that is 59.99 average-length months, and the horizon check stays strict.
- **Not tested here:** 2021+ entrants (too little follow-up for 5 years). Phase 5's rolling-origin
  backtest covers them at 3 years.
- **Time unit:** month index, where month m covers [m−1, m) months after entry. Predictions are on months
  1–96.

### D-028: Preprocessing is fitted on training rows only
- States with fewer than 50 training rows, and any state not seen in training, become `other`.
- **Cox ("linear" mode):**
  - log1p on counts and MW, then standardize with the training mean and standard deviation;
  - missing values become the training median plus a 0/1 indicator;
  - one-hot categories with the first level as the reference;
  - redundant columns are dropped: constants, and 0/1 columns that duplicate or mirror an earlier one. In
    the real data this drops 6 missing-indicators that equal `1 − county_known` (or each other) and
    `iso=other`. Without this step the Cox fit was singular.
- **Random survival forest ("tree" mode):** the same filling and indicators, all levels one-hot, no scaling.
- **LightGBM:** native categories and native handling of missing values.

### D-029: Cause-specific Cox and cause-specific random survival forest
- **Decision:** One model per outcome, with the other outcome treated as censored. Each model gives a
  per-project cumulative hazard, and `combine.py` turns the pair into cumulative incidence with the same
  product-limit formula as Aalen-Johansen, so S + CIF_op + CIF_wd = 1.
- **Why:** Cause-specific hazards are the standard way to build competing-risks curves from single-event
  models. lifelines and scikit-survival both fit single-event models only.
- **Alternative:** Fine-Gray subdistribution hazards. It models the CIF directly, but neither library
  implements it, and coefficients are harder to interpret.
- **Settings (fixed, not tuned on test):**
  - Cox: ridge penalty 0.1, because the congestion features are correlated.
  - Forest: 300 trees, `min_samples_leaf = 20`, `max_features = "sqrt"`, seed 42.
- **Proportional-hazards check:** 4 of 49 covariates for operation and 3 of 49 for withdrawal have
  p < 0.05. The largest are log capacity (operation), `iso=MISO` and `queue_year` (withdrawal). With
  thousands of rows this test is very sensitive, so the Cox effects are read as averages over time.

### D-030: Discrete-time LightGBM
- **Person-period data:** one row per project per 3-month interval, labelled 0 (nothing), 1 (operational)
  or 2 (withdrawn). The interval number is a feature. Predicted class probabilities are discrete hazards,
  combined into CIFs with the product-limit formula.
- **Censoring convention:** a project censored part-way through an interval contributes only the intervals
  it was observed for in full.
- **Rounds:** chosen by early stopping on an inner split inside the training data (train on entrants before
  2016-01-01, validate on 2016–2017 entrants with outcomes cut off at 2018-01-01). The result was 31 rounds.
  The test cohorts were never used.
- **Fixed settings:** learning rate 0.05, 15 leaves, 200 minimum rows per leaf, feature and bagging fraction
  0.8, L2 penalty 1.0, deterministic, seed 42.

### D-031: Phase 4 metrics, and the transfer diagnostics
- **Metrics:** for each outcome at 3 and 5 years, on test cohorts with complete follow-up: AUC (ranking),
  Brier score (squared error), and mean predicted vs observed rate (calibration-in-the-large). The
  C-index, integrated Brier score, calibration curves and bootstrap intervals come in Phase 5.
- **Transfer diagnostic:** AUC on the training cohorts (in-sample) against AUC on the test cohorts, plus
  the observed 3-year rates by ISO in each. This separates a regime shift from ordinary overfitting.
- **Pitfall caught while building it:** a first version kept training rows "followed to 3 years" one row at
  a time. That keeps recent projects only if they had an event early, which is selection on the outcome.
  The fix uses whole entry years with full follow-up before the origin (2010–2014).
- **Commitment:** the diagnostics were computed after all model settings were fixed. Nothing was changed
  to improve test results.

### D-032: Tooling
- **MLflow:** version 3.18 refuses the plain-folder tracking store, so runs go to a local SQLite file
  (`mlruns/mlflow.db`) with artifacts in `mlruns/artifacts`. It is still local with no server, and it is
  gitignored. View it with `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
- **Fitted models:** saved to `models/*.joblib` (gitignored) for Phases 5–7.
- **macOS:** LightGBM needs the OpenMP runtime: `brew install libomp`. Linux CI already has it.

### Phase 4 results in brief (from `reports/model_comparison.md`)
- **Operational:** all three models rank projects well above the baseline. 3-year AUC is 0.81–0.86 against
  0.72 for the baseline, and 5-year AUC is 0.70–0.74 against 0.63. But every model overpredicts the level
  (about 7–8% predicted against 2.0% observed at 3 years), because completion fell after the training years.
- **Withdrawn:** test AUC is about 0.49–0.54, close to chance, even though the models rank withdrawals in
  the training cohorts (in-sample 0.64–0.80). At 3 years the ISO × fuel baseline (0.575) ranks
  withdrawals better than any model. Observed 3-year withdrawal fell between 2010–14 and 2018–20
  entrants in every ISO (for example SPP 59% to 22%, PJM 58% to 38%). Withdrawal timing changed across
  eras and the models can't see that from the features.
- **Next:** Phase 5 measures how stable this is across many origins (rolling-origin backtest), with
  confidence intervals and the planned ablations, including with and without `queue_year`.

## Phase 5: Evaluation and rigor (2026-11-11)

### D-033: A pre-registered protocol and an audit trail in git
- **Decision:** All Phase 5 choices are defined in `docs/evaluation_protocol.md`: the blocks, the exact
  selection score, the tie rule, the metrics, the bootstrap units, the recalibration method, and what the
  final test must report. It was committed and tagged **before any result was computed**.
- **Audit trail (git order):**
  1. protocol `9e141b7` (tag `eval-protocol-v1`);
  2. block A plus the frozen selection, `7cca0db`;
  3. final-run code before it was run, `ce12185`;
  4. block B and C results (the next commit).
- **Why:** The 2018–2020 cohorts had already been seen in Phase 4, so they could no longer be an untouched
  test. Decisions use only cohorts 2014–2017 (information known by 2021-01-01). The final test (2021–22
  entrants) was scored once.
- **Re-runs:** none. Presentation fixes after the final run were made with `make evaluate-render`, which
  rebuilds the report from saved predictions and metric files without refitting or re-scoring (D-038).

### D-034: Metric implementation
- AUC (cumulative/dynamic) and the truncated competing-risks C-index (Wolbers) use one pair-counting class
  with optional bootstrap weights. They are tested against a hand-worked example, a brute-force pair count
  with ties, and scikit-learn's AUC. IBS and skill are computed per project, so bootstrap resamples cost a
  matrix product.
- All evaluation cohorts have full follow-up to the horizon (asserted in code). The
  inverse-probability-of-censoring weights the metrics normally need are therefore exactly 1, and are not
  applied.

### D-035: Bootstrap units: what we learned
- With 1,000 resamples of fixed predictions, county × entry-year intervals were a median **1.23× (block A)
  and 1.27× (block C)** wider than project intervals. ISO × entry-year intervals were 3.0× wider, but rest
  on only 16 groups (unreliable).
- **Reading:** outcomes cluster in place and time, so project-level intervals are somewhat optimistic. The
  report shows both. The spread of per-origin metrics is larger again (withdrawal AUC 0.47–0.60 across seven
  origins), which says era-to-era variation is the dominant source of uncertainty.

### D-036: Recalibration, and what it did
- The method is in protocol section 7: hazards are scaled so the model reproduces the 36-month rates of the
  Y−4 cohort.
- **Development (A):** recalibration helped only the forest a little. The tie rule then preferred the
  version without it.
- **Final test (C):** recalibration improved every family's skill (for example forest withdrawal skill
  0.041 → 0.127, Cox 0.173 → 0.216), because outcome rates kept falling and the Y−4 cohort (2017) was closer
  to 2021–22 than pooled history. The fitted constants for the forest fell from 0.85–1.08 in development to
  0.56 (operational) and 0.85 (withdrawn) at the 2021 origin.
- **Not acted on:** the primary configuration is unchanged (protocol section 8).

### D-037: Phase 5 results in brief (from `reports/evaluation.md`)
- **Primary configuration:** cause-specific random survival forest, without `queue_year`, without
  recalibration.
- **Final test (2021–22 entrants, n = 3,299, 3 years):**
  - **Withdrawal:** AUC 0.559 [0.539, 0.579] and Brier skill 0.041 [0.030, 0.051] (project intervals;
    county-year intervals are wider). It overpredicts (51.6% predicted against 34.3% observed).
  - **Operational:** only **1** case, so ranking is not estimable. Predicted 4.6% against 0.03% observed.
- **Against the baselines:** the primary configuration beats both the "most recent fully observed cohort"
  baseline and the ISO × fuel baseline on withdrawal skill, in development and on the final test (paired
  intervals exclude 0).
- **The selection did not transfer.** Cox *with* `queue_year` *and* recalibration was clearly worse than
  the primary configuration in development (paired withdrawal skill difference −0.050 to −0.025) and
  clearly better on the final test (+0.16 to +0.19). The best choice changed between eras. That is
  consistent with the regime shift found in Phases 2 and 4, and it is the main limitation.
- **Congestion ablation:** no consistent benefit. Without congestion features, withdrawal AUC is slightly
  higher for every family. With them, operational AUC is higher for Cox and the forest in development
  (0.693 vs 0.649 and 0.686 vs 0.626).
- **What a version-2 protocol would do:** pre-register the recalibrated Cox with `queue_year` as a candidate
  primary configuration, and evaluate it on cohorts that are still unseen (2023+ entrants) once their
  3-year outcomes exist (end of 2026 data at the earliest). Switching now, after seeing block C, would be
  tuning on the test set.

### D-038: Re-rendering without re-scoring
- `make evaluate-render` rebuilds `reports/evaluation.md`, the figures and `docs/model_card.md` from the
  saved predictions (`data/interim/eval/*.joblib`) and the metric/bootstrap CSVs. It refits and re-scores
  nothing. A test checks that re-rendering leaves the metric files byte-identical. On the real run, checksums
  confirmed the metric, bootstrap and block C prediction files were unchanged by three presentation fixes:
  table headers, the note listing rows that beat the primary configuration, and small-percentage precision.
