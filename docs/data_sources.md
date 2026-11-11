# Data sources

## LBNL "Queued Up" interconnection queue data (only source for the MVP)

| Item | Value |
|---|---|
| Publisher | Lawrence Berkeley National Laboratory (Energy Markets & Planning), compiled with GridTracker |
| Release | Queued Up: 2026 Edition. The data file was last updated May 2026 and covers queues through the end of 2025 |
| Landing page | https://emp.lbl.gov/queues |
| File URL | https://emp.lbl.gov/sites/default/files/2026-05/LBNL_Ix_Queue_Data_File_thru2025.xlsx |
| File name | `LBNL_Ix_Queue_Data_File_thru2025.xlsx` (15,571,236 bytes) |
| SHA-256 | `794582d3281c6a305e9615fcfec3fae9dc85be2165216d33760b677e976a08b6` |
| Downloaded | 2026-11-11, by hand after free registration on the LBNL site |
| License | CC BY 4.0, as stated on the landing page |
| Coverage | All 7 ISOs/RTOs plus 50 non-ISO utilities (~98% of installed US capacity, per LBNL) |

**License and attribution.** The landing page says (checked 2026-11-11) that the data file is licensed CC BY 4.0
and may be used, shared, or adapted with attribution to Lawrence Berkeley National Laboratory and GridTracker.
Required attribution: *Queued Up Data File. Copyright (c) 2026, The Regents of the University of California,
through Lawrence Berkeley National Laboratory & GridTracker. CC BY 4.0.*

**Why the download is manual.** LBNL asks for a short registration before the file can be downloaded, and the
site refuses automated requests (HTTP 403). `make data` therefore does not download anything. It checks that
`data/raw/LBNL_Ix_Queue_Data_File_thru2025.xlsx` exists and that its SHA-256 matches the value above. If the
file is missing or different, it stops and explains how to get it. See decisions D-008.

### What is in the file
- 43 sheets. The pipeline reads only **`03. Complete Queue Data`**: one row per interconnection request
  (38,201 rows, 30 columns). Row 1 is a "RETURN TO CONTENTS" link and the column names are on row 2.
- **`04. Data Codebook`**: LBNL's definitions for every column. These are summarized in
  [data_dictionary.md](data_dictionary.md).
- The remaining sheets are LBNL's own summary tables (completion rates, durations, and so on). They are not
  used as inputs, but they are useful for sanity-checking the EDA in Phase 2.

### Findings from checking the source (2026-11-11)
- `q_id` is unique only within a transmission provider (`entity`), as LBNL notes. Even so, 135
  `(entity, q_id)` pairs appear on more than one row. All of them are outside the model regions.
- **Event dates are missing for whole regions.** This is the main data limitation. For example, only about
  12% of withdrawn projects in the non-ISO West have a withdrawal date, and ISO-NE has no operation dates at
  all. Before 2010, withdrawal dates are often missing even in the well-covered regions. This decided the
  model scope (D-011, D-012). Full tables are in [`reports/data_quality.md`](../reports/data_quality.md).
- Status labels found: `active`, `withdrawn`, `operational`, `suspended`, and 10 rows of `unknown`, which is
  not listed in the codebook.

### Policies
- Raw files are never modified. The copy in `data/raw/` is set to read-only. All cleaning happens in code
  that writes to `data/interim/` and `data/processed/`.
- Neither raw nor processed data is committed to git. The generated data-quality report and cleaning
  waterfall are committed, because they are small summaries.
