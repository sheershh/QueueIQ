# Every pipeline step is a make target. Later phases add: data, features, train, evaluate, app.
.PHONY: help setup data eda features train evaluate-dev evaluate evaluate-render error-range explain api app screenshots lint format test check clean

UV ?= uv

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

setup:  ## Install pinned dependencies from uv.lock and the git pre-commit hooks
	@command -v $(UV) >/dev/null || { echo "uv is not installed: see https://docs.astral.sh/uv/"; exit 1; }
	$(UV) sync --frozen
	@if [ -d .git ]; then $(UV) run pre-commit install; else echo "Not a git repo: skipping pre-commit install"; fi

data:  ## Verify the raw LBNL file and build data/processed/queue_entries.parquet + reports/data_quality.md
	$(UV) run python -m queueiq.ingestion.build

eda:  ## Exploratory analysis: reports/eda.md + reports/figures/ (needs make data first)
	$(UV) run python -m queueiq.eda.build

features:  ## Leakage-safe feature table: data/processed/features.parquet + reports/features.md
	$(UV) run python -m queueiq.features.build

FEATURES_FILE = data/processed/features.parquet

$(FEATURES_FILE): data/processed/queue_entries.parquet
	$(UV) run python -m queueiq.features.build

train: $(FEATURES_FILE)  ## Model comparison (reports/model_comparison.md, MLflow) + the deployment model the app serves (models/)
	$(UV) run python -m queueiq.models.train
	$(UV) run python -m queueiq.service.build

evaluate-dev:  ## Phase 5 block A (development) + frozen model selection (protocol section 3)
	$(UV) run python -m queueiq.evaluation.dev

evaluate:  ## Phase 5 blocks B + C (report only) using the frozen selection; evaluation report + model card
	$(UV) run python -m queueiq.evaluation.final

evaluate-render:  ## Rebuild evaluation report/figures/model card from saved results (no refit, no re-scoring)
	$(UV) run python -m queueiq.evaluation.final --render-only

error-range:  ## Past-error range for the dashboard from saved blocks A+B (reports/past_error_range.json)
	$(UV) run python -m queueiq.evaluation.error_range

explain:  ## Phase 6: deployment model (models/) + reports/explanations.md
	$(UV) run python -m queueiq.explain.build

api:  ## Run the FastAPI service locally on http://localhost:8000 (docs at /docs)
	$(UV) run uvicorn queueiq.api.main:app --port 8000

app:  ## Run the Streamlit dashboard locally on http://localhost:8501
	$(UV) run streamlit run src/queueiq/app/main.py

screenshots:  ## Dashboard screenshots at laptop and phone width (app must be running; needs Google Chrome)
	$(UV) run python scripts/screenshots.py

lint:  ## Check lint and formatting (no changes made)
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:  ## Auto-fix lint issues and format code
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

test:  ## Run the test suite with coverage
	$(UV) run pytest --cov=queueiq --cov-report=term-missing

check: lint test  ## Lint + tests (what CI runs)

clean:  ## Remove caches and build artifacts (never touches data/)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
