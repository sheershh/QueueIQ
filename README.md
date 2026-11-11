# QueueIQ

Survival and competing-risks models that estimate whether a proposed US energy project in a grid
interconnection queue reaches commercial operation or withdraws, and how long that takes.
Data: Lawrence Berkeley National Laboratory "Queued Up" dataset.

**Status:** Phase 0 (project setup). No data, models, or results yet.

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

```bash
make setup   # install pinned dependencies + git hooks
make test    # run tests
make lint    # lint and format checks
make help    # list all targets
```

## Docs
- [docs/decisions.md](docs/decisions.md): design decisions and why they were made
- [docs/data_sources.md](docs/data_sources.md): data provenance (TBD, Phase 1)
- [docs/data_dictionary.md](docs/data_dictionary.md): column definitions (TBD, Phase 1)
- [docs/model_card.md](docs/model_card.md): model documentation (TBD, Phase 5)
