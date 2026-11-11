# QueueIQ

Survival and competing-risks models that estimate whether a proposed US energy project in a grid
interconnection queue reaches commercial operation or withdraws, and how long that takes.
Data: Lawrence Berkeley National Laboratory "Queued Up" dataset.

**Status:** Phase 5 (pre-registered evaluation) complete. Explainability (Phase 6) next.

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

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and `make`. uv installs the pinned Python version (3.12) itself.
On macOS, LightGBM also needs the OpenMP runtime: `brew install libomp`.

```bash
make setup   # install pinned dependencies + git hooks
make test    # run tests
make lint    # lint and format checks
make data    # verify the raw LBNL file and build the processed dataset
make eda     # exploratory analysis: reports/eda.md + figures
make features  # leakage-safe feature table + reports/features.md
make train   # fit all models on the time split + reports/model_comparison.md
make evaluate-dev  # Phase 5 development backtest + frozen model selection
make evaluate      # Phase 5 extension + final test (run once) + evaluation report + model card
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
