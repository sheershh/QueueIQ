"""`make data`: raw LBNL file -> data/processed/queue_entries.parquet + data-quality report.

Steps: verify checksum -> read sheet (saved unchanged to data/interim) -> unified schema ->
outcomes -> quality flags -> model scope -> write outputs. The raw file is only ever read.
"""

from queueiq import config
from queueiq.cleaning.flags import add_quality_flags
from queueiq.cleaning.outcomes import assign_outcomes
from queueiq.cleaning.report import build_report
from queueiq.cleaning.scope import MODEL_ISOS, apply_model_scope
from queueiq.cleaning.unify import to_unified
from queueiq.ingestion import lbnl

INTERIM_FILE = config.INTERIM_DIR / "lbnl_queue_sheet.parquet"
PROCESSED_FILE = config.PROCESSED_DIR / "queue_entries.parquet"
WATERFALL_FILE = config.REPORTS_DIR / "cleaning_waterfall.csv"
REPORT_FILE = config.REPORTS_DIR / "data_quality.md"


def main() -> None:
    config.ensure_dirs()
    raw_path = lbnl.default_raw_path()

    print(f"Verifying {raw_path.name} ...")
    lbnl.verify_raw_file(raw_path)

    print(f"Reading sheet '{lbnl.SHEET_NAME}' ...")
    raw = lbnl.read_queue_sheet(raw_path)
    raw.to_parquet(INTERIM_FILE, index=False)

    df = to_unified(raw)
    df = assign_outcomes(df, config.DATA_CUTOFF)
    df = add_quality_flags(df, config.DATA_CUTOFF)
    df, waterfall = apply_model_scope(df)

    assert df["project_id"].is_unique
    assert len(df) == len(raw), "no row may be dropped from the processed dataset"

    df.to_parquet(PROCESSED_FILE, index=False)
    waterfall.to_csv(WATERFALL_FILE, index=False)
    meta = {
        "file": raw_path.name,
        "sha256": lbnl.EXPECTED_SHA256,
        "sheet": lbnl.SHEET_NAME,
        "cutoff": config.DATA_CUTOFF.isoformat(),
    }
    REPORT_FILE.write_text(build_report(df, waterfall, meta, MODEL_ISOS))

    print(
        f"Wrote {PROCESSED_FILE.relative_to(config.PROJECT_ROOT)}: {len(df)} rows, "
        f"{int(df['in_model_scope'].sum())} in model scope"
    )
    print(
        f"Wrote {REPORT_FILE.relative_to(config.PROJECT_ROOT)} and "
        f"{WATERFALL_FILE.relative_to(config.PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
