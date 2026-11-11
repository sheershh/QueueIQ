# Every pipeline step is a make target. Later phases add: data, features, train, evaluate, app.
.PHONY: help setup lint format test check clean

UV ?= uv

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

setup:  ## Install pinned dependencies from uv.lock and the git pre-commit hooks
	@command -v $(UV) >/dev/null || { echo "uv is not installed: see https://docs.astral.sh/uv/"; exit 1; }
	$(UV) sync --frozen
	@if [ -d .git ]; then $(UV) run pre-commit install; else echo "Not a git repo: skipping pre-commit install"; fi

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
