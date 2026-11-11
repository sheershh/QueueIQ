# One image for both services (API and dashboard). It holds code and runtime dependencies only:
# the trained model (models/) and the processed data (data/processed/) are mounted read-only at
# run time (docker-compose.yml), so the image never contains data or model files (D-045).

# --- build stage: compilers are needed only here (ecos, a scikit-survival dependency, has no
# --- prebuilt wheel for some platforms, e.g. Linux on Apple silicon)
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
# Runtime dependencies only (no training stack, no test tools), exactly as locked.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-default-groups --no-install-project
COPY src ./src
RUN uv sync --frozen --no-default-groups --no-editable

# --- runtime stage
FROM python:3.12-slim
# Fewer glibc malloc arenas: the forests predict on many threads, and per-thread arenas roughly
# doubled resident memory in testing.
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 QUEUEIQ_ROOT=/app MALLOC_ARENA_MAX=2
WORKDIR /app
RUN useradd --create-home --uid 1000 queueiq && mkdir -p models data/processed \
    && chown queueiq /app /app/models /app/data /app/data/processed
COPY --from=build --chown=queueiq /app/.venv /app/.venv
COPY --chown=queueiq src ./src
COPY --chown=queueiq .streamlit ./.streamlit
USER queueiq
EXPOSE 8000 8501
CMD ["uvicorn", "queueiq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
