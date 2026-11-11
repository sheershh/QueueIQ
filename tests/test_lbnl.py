import hashlib

import pandas as pd
import pytest

from queueiq.ingestion import lbnl
from tests.conftest import make_raw_frame


def test_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"queue data" * 1000)
    assert lbnl.sha256_of(f, chunk_size=7) == hashlib.sha256(b"queue data" * 1000).hexdigest()


def test_verify_missing_file_explains_how_to_download(tmp_path):
    with pytest.raises(lbnl.RawDataError, match="emp.lbl.gov/queues"):
        lbnl.verify_raw_file(tmp_path / "missing.xlsx")


def test_verify_rejects_different_file(tmp_path):
    f = tmp_path / "file.xlsx"
    f.write_bytes(b"not the release")
    with pytest.raises(lbnl.RawDataError, match="Checksum mismatch"):
        lbnl.verify_raw_file(f)


def test_verify_accepts_matching_checksum(tmp_path):
    f = tmp_path / "file.xlsx"
    f.write_bytes(b"release bytes")
    lbnl.verify_raw_file(f, expected_sha256=hashlib.sha256(b"release bytes").hexdigest())


def test_check_columns_reports_missing_and_extra():
    df = make_raw_frame([{}]).drop(columns=["wd_date"]).assign(new_col=1)
    with pytest.raises(
        lbnl.RawDataError, match=r"Missing columns: \['wd_date'\].*extra: \['new_col'\]"
    ):
        lbnl.check_columns(df)


def test_read_queue_sheet_skips_banner_row_and_types_text_columns(tmp_path):
    """Mimic the real layout: a banner row, then the header, on the expected sheet name."""
    data = make_raw_frame([{"q_id": 123, "poi_name": "Sub A"}, {"q_id": "Q-7"}])
    path = tmp_path / "lbnl.xlsx"
    with pd.ExcelWriter(path) as xl:
        pd.DataFrame([["RETURN TO CONTENTS"]]).to_excel(
            xl, sheet_name=lbnl.SHEET_NAME, index=False, header=False
        )
        data.to_excel(xl, sheet_name=lbnl.SHEET_NAME, index=False, startrow=1)

    df = lbnl.read_queue_sheet(path)

    assert list(df.columns) == lbnl.EXPECTED_COLUMNS
    assert len(df) == 2
    assert df["q_id"].tolist() == ["123", "Q-7"]  # numbers and text both become text
    assert pd.api.types.is_datetime64_any_dtype(df["q_date"])
