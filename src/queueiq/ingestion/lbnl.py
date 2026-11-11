"""Read the LBNL "Queued Up" Excel file without changing it.

LBNL requires a (free) registration to download the file, so it cannot be fetched
by a script. The user downloads it by hand; this module checks that the file in
data/raw is byte-for-byte the release we documented (SHA-256), then reads the
project-level sheet and checks its columns.
"""

import hashlib
from pathlib import Path

import pandas as pd

from queueiq import config

SOURCE_PAGE = "https://emp.lbl.gov/queues"
SOURCE_URL = "https://emp.lbl.gov/sites/default/files/2026-05/LBNL_Ix_Queue_Data_File_thru2025.xlsx"
RAW_FILENAME = "LBNL_Ix_Queue_Data_File_thru2025.xlsx"
EXPECTED_SHA256 = "794582d3281c6a305e9615fcfec3fae9dc85be2165216d33760b677e976a08b6"

SHEET_NAME = "03. Complete Queue Data"
HEADER_ROW = 1  # row 0 is a "RETURN TO CONTENTS" link; column names are on row 1

EXPECTED_COLUMNS = [
    "q_id",
    "q_status",
    "q_date",
    "prop_date",
    "on_date",
    "wd_date",
    "ia_date",
    "IA_phase_raw",
    "IA_phase_clean",
    "county",
    "state",
    "fips_code",
    "poi_name",
    "region",
    "project_name",
    "utility",
    "entity",
    "developer",
    "cluster",
    "service",
    "project_type",
    "type_1",
    "type_2",
    "type_3",
    "type_clean",
    "mw_1",
    "mw_2",
    "mw_3",
    "q_year",
    "prop_year",
]

# Columns that mix numbers and text in the sheet; stored as text so parquet can hold them.
TEXT_COLUMNS = ["q_id", "IA_phase_raw", "poi_name", "project_name", "cluster"]

DOWNLOAD_HELP = (
    f"Download the 'Queued Up 2026 Data File' from {SOURCE_PAGE} (free registration), "
    f"and save it unchanged as data/raw/{RAW_FILENAME}."
)


class RawDataError(RuntimeError):
    """The raw file is missing, different from the documented release, or malformed."""


def default_raw_path() -> Path:
    return config.RAW_DIR / RAW_FILENAME


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, read in chunks so large files use little memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def verify_raw_file(path: Path, expected_sha256: str | None = None) -> None:
    """Fail loudly if the raw file is missing or is not the documented release."""
    expected_sha256 = expected_sha256 or EXPECTED_SHA256
    if not path.is_file():
        raise RawDataError(f"Raw file not found: {path}\n{DOWNLOAD_HELP}")
    actual = sha256_of(path)
    if actual != expected_sha256:
        raise RawDataError(
            f"Checksum mismatch for {path}\n  expected {expected_sha256}\n  actual   {actual}\n"
            "This is not the release documented in docs/data_sources.md. If LBNL published a "
            "new version, update the source metadata and decisions log before using it."
        )


def check_columns(df: pd.DataFrame) -> None:
    """The parser depends on these exact column names; stop if the sheet layout changed."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing or extra:
        raise RawDataError(f"Unexpected sheet layout. Missing columns: {missing}; extra: {extra}")


def read_queue_sheet(path: Path) -> pd.DataFrame:
    """Read the project-level sheet as-is (no cleaning), with mixed-type columns as text."""
    df = pd.read_excel(path, sheet_name=SHEET_NAME, header=HEADER_ROW)
    check_columns(df)
    for col in TEXT_COLUMNS:
        df[col] = df[col].map(lambda v: None if pd.isna(v) else str(v)).astype("string")
    return df
