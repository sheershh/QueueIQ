"""Smoke tests: the package installs, imports, and its paths point at this repo."""

import importlib

import pytest

import queueiq
from queueiq import config

SUBPACKAGES = [
    "ingestion",
    "cleaning",
    "features",
    "models",
    "evaluation",
    "explain",
    "api",
    "app",
]


def test_version_is_set():
    assert queueiq.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name):
    importlib.import_module(f"queueiq.{name}")


def test_project_root_is_repo_root():
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()


def test_data_dirs_live_under_project_root():
    for d in (config.RAW_DIR, config.INTERIM_DIR, config.PROCESSED_DIR):
        assert d.is_relative_to(config.DATA_DIR)
    assert config.DATA_DIR.parent == config.PROJECT_ROOT


def test_ensure_dirs_creates_missing_dirs(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    figs = tmp_path / "reports" / "figures"
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "INTERIM_DIR", tmp_path / "data" / "interim")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(config, "FIGURES_DIR", figs)

    config.ensure_dirs()
    config.ensure_dirs()  # second call must not fail

    assert raw.is_dir()
    assert figs.is_dir()


def test_seed_is_fixed_int():
    assert isinstance(config.SEED, int)
