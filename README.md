# QueueIQ

Survival and competing-risks models that estimate whether a proposed US energy project in a grid
interconnection queue reaches commercial operation or withdraws, and how long that takes.
Data: Lawrence Berkeley National Laboratory "Queued Up" dataset.

**Status:** Phase 7 (API and dashboard) complete. Write-up (Phase 8) next.

## Roadmap
0. Setup: environment, lint, tests, CI
1. Ingestion: LBNL data to a validated, unified dataset
2. Exploratory analysis
3. Leakage-safe features
4. Survival and competing-risks models
5. Evaluation: backtesting, calibration, confidence intervals
6. Explainability
7. API and dashboard
8. Write-up

## Run the dashboard and API

Needs the LBNL data file (see [Getting the data](#getting-the-data)), [uv](https://docs.astral.sh/uv/),
`make` and Docker with about 3 GB of memory for containers (4 GB is comfortable).

```bash
make setup                 # pinned dependencies
make data && make train    # processed data + the deployed model in models/ (under a minute)
docker compose up          # dashboard: http://localhost:8501   API: http://localhost:8000/docs
```

The first `docker compose up` builds the image (about 6 minutes on an Apple-silicon laptop, most of it
compiling one scientific dependency). The image holds code only. `models/` and `data/processed/` are
mounted read-only, so the model is never baked into the image or committed. Without Docker, run
`make app` (dashboard) and `make api` (API) instead.

| Outlook | Compare | Why this prediction |
|---|---|---|
| ![Outlook](docs/screenshots/outlook_laptop.png) | ![Compare](docs/screenshots/compare_laptop.png) | ![Why](docs/screenshots/why_laptop.png) |

Phone width: [outlook](docs/screenshots/outlook_phone.png), [compare](docs/screenshots/compare_phone.png),
[why](docs/screenshots/why_phone.png), [input error](docs/screenshots/error_phone.png).
Screenshots are regenerated with `make screenshots` while the app is running.

**How to read it.** For a project entering the queue on 1 January 2026, the lines are the deployed model's
chance of having withdrawn, or reached operation, by each month. The shaded band is the **past-error range**:
the same curves redrawn with each of the model's 2014–2020 calibration errors (decisions D-046). It is not a
confidence interval. In every past test year fewer projects withdrew than the model predicted, so the band
sits below the withdrawal line. Curves past 60 months were never checked against outcomes (greyed). Use the
numbers to compare setups; see the [model card](docs/model_card.md) for limits.

API example:

```bash
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"iso": "MISO", "state": "IL", "fuel": "wind", "capacity_mw": 200}'
```

`GET /options` lists valid states, counties and the capacity range. `POST /predict?explain=true` adds the
explanation, and `POST /compare` takes up to 5 projects.

## Quick start (full pipeline)

Requires [uv](https://docs.astral.sh/uv/) and `make`. uv installs the pinned Python version (3.12) itself.
On macOS, LightGBM also needs the OpenMP runtime: `brew install libomp`.

```bash
make setup   # install pinned dependencies + git hooks
make test    # run tests
make lint    # lint and format checks
make data    # verify the raw LBNL file and build the processed dataset
make eda     # exploratory analysis: reports/eda.md + figures
make features  # leakage-safe feature table + reports/features.md
make train   # fit all models on the time split + reports/model_comparison.md + deployed model
make evaluate-dev  # Phase 5 development backtest + frozen model selection
make evaluate      # Phase 5 extension + final test (run once) + evaluation report + model card
make explain       # deployment model + reports/explanations.md (grouped Shapley, permutation importance)
make error-range   # past-error range for the dashboard band (from saved Phase 5 results)
make app           # dashboard on :8501      make api   # API on :8000
make help    # list all targets
```

### Getting the data
The LBNL data file needs a free registration, so it can't be downloaded by a script.
1. Go to https://emp.lbl.gov/queues and download the **Queued Up 2026 Data File (XLSX)**.
2. Save it unchanged as `data/raw/LBNL_Ix_Queue_Data_File_thru2025.xlsx`.
3. Run `make data`. It checks the file's SHA-256 against [docs/data_sources.md](docs/data_sources.md),
   then writes `data/processed/queue_entries.parquet` and [reports/data_quality.md](reports/data_quality.md).

Data: *Queued Up Data File. Copyright (c) 2026, The Regents of the University of California, through
Lawrence Berkeley National Laboratory & GridTracker. CC BY 4.0.*

## Docs
- [docs/decisions.md](docs/decisions.md): design decisions and why they were made
- [docs/data_sources.md](docs/data_sources.md): data provenance, license, checksum
- [docs/data_dictionary.md](docs/data_dictionary.md): column definitions and which are safe as features
- [reports/data_quality.md](reports/data_quality.md): generated data-quality report and cleaning waterfall
- [reports/eda.md](reports/eda.md): exploratory analysis (censoring-aware outcome curves, cohorts, cross-check with LBNL)
- [reports/features.md](reports/features.md): feature definitions, coverage, and known gaps
- [reports/model_comparison.md](reports/model_comparison.md): all models on one time-based split, with transfer diagnostics
- [docs/evaluation_protocol.md](docs/evaluation_protocol.md): pre-registered evaluation protocol (tag `eval-protocol-v1`)
- [reports/evaluation.md](reports/evaluation.md): rolling-origin backtest, final test, bootstrap intervals, calibration
- [docs/model_card.md](docs/model_card.md): intended use, performance, limitations
- [reports/explanations.md](reports/explanations.md): what the model relies on, and why individual projects get their predictions
- [reports/past_error_range.json](reports/past_error_range.json): the past errors behind the dashboard's band
