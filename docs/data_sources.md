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

### License re-check for a possible hosted demo (2026-11-11)
- **Landing page:** emp.lbl.gov/queues still states "The Queued Up data file is licensed CC BY 4.0. You may
  use, share, or adapt the dataset as long as you attribute it to Lawrence Berkeley National Laboratory and
  GridTracker". It links to the Creative Commons BY 4.0 deed.
- **Copyright line:** "Queued Up Data File. Copyright (c) 2026, The Regents of the University of California,
  through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S.
  Dept. of Energy) & GridTracker. CC BY 4.0." The same line also says "All rights reserved". We read the
  explicit CC BY 4.0 grant as governing, but the wording is mixed.
- **Download form:** the form shown before download is an email-list sign-up (name, email, company, job
  title, topic lists). Its only consent text is about receiving marketing emails (via Constant Contact).
  It shows no separate data-use terms.
- **What CC BY 4.0 allows for a hosted demo:** sharing and adapting, including derived statistics and
  trained models, provided we:
  1. credit LBNL and GridTracker;
  2. link to the license;
  3. say that changes were made (filtered, cleaned, modelled);
  4. don't imply endorsement.

  Our choice not to commit the raw file stays (reproducibility via checksum; D-008). This is a reading of
  the public license, not legal advice. Before any public deployment, the author should confirm with LBNL
  if in doubt, especially about the "subject to DOE approvals" and "All rights reserved" wording.

## Secondary source: US Census Bureau county and state shapes (Phase 9)
| Field | Value |
|---|---|
| Publisher | US Census Bureau, Geography Division |
| Product | Cartographic boundary files, 2024, 1:20,000,000 (`cb_2024_us_county_20m.zip`, `cb_2024_us_state_20m.zip`) |
| URL | https://www2.census.gov/geo/tiger/GENZ2024/shp/ |
| License | Public domain (US government work) |
| SHA-256 | county `c9a54fd95422436b8822e0b070a45a9bc3de08eb13f943cdba66e7a9c2a135ff`; state `7bc773d83c01b6df69b8aada9c2b5983d97f22f4551947cc5c00b87c663223ef` (checked by `make geo`) |
| Used for | Drawing the map only. County keys (5-digit FIPS) match LBNL's `fips_code`. No model input comes from these files. |

## Redistribution in the hosted app (Phases 8 and 9)
- The original XLSX file is never redistributed.
- The hosted dashboard (https://queueiq.streamlit.app) downloads files derived from it from this
  repository's GitHub Releases (`deploy-assets-v1`, then `deploy-assets-v2`):
  - the trained model and its metadata;
  - 64 recent rows of model features (the explanation baseline);
  - `queue_history.parquet`, the in-scope and out-of-scope queue rows trimmed to the 13 columns the
    service reads.
- Every release note, the README, the write-up and every app page carry the CC BY 4.0 attribution, a link
  to the license, a statement of the changes made, and a note that LBNL and GridTracker do not endorse the
  project (decisions D-045, D-053, D-056).
