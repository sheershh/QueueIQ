# Data dictionary: `data/processed/queue_entries.parquet`

One row per row of LBNL sheet `03. Complete Queue Data` (38,201 rows). No rows are dropped.
Which rows the models may use is recorded in `in_model_scope`.

**Leakage column:** "Feature-safe" means the value is known on the queue entry date. Only feature-safe columns
may be used as model inputs, to avoid data leakage.

## Identity and location

| Column | Type | Source column | Meaning / normalization | Feature-safe |
|---|---|---|---|---|
| `project_id` | string | `entity`, `q_id` | `<entity>::<q_id>`. Reused IDs get a `#2`, `#3`, ... suffix so every row is unique | n/a (key) |
| `queue_id` | string | `q_id` | Queue position as published. Numbers are stored as text | n/a |
| `entity` | string | `entity` | Transmission provider (ISO or utility) | yes |
| `iso` | string | `region` | One of the 7 ISOs, `West` (non-ISO), or `Southeast` (non-ISO) | yes |
| `state` | string | `state` | Two-letter code, trimmed and upper-cased | yes |
| `county` | string | `county` | First listed county if the project spans several | yes |
| `fips_code` | string | `fips_code` | 5-digit county FIPS code, zero-padded (`1001` becomes `01001`) | yes |
| `poi_name` | string | `poi_name` | Point of interconnection (substation or line tap), not standardized | yes |
| `utility` | string | `utility` | Utility name | yes |
| `developer` | string | `developer` | Developer name, not standardized. Missing for most requests | yes |

## Project characteristics

| Column | Type | Source column | Meaning / normalization | Feature-safe |
|---|---|---|---|---|
| `project_type` | string | `project_type` | `generation`, `surplus`, `upgrade`, `replacement`, or missing | yes |
| `service` | string | `service` | Interconnection service: NRIS, ERIS, NRIS/ERIS, Other | yes |
| `fuel_type` | string | `type_1` | Primary resource type, lower snake case (`offshore_wind`) | yes |
| `fuel_type_detail` | string | `type_clean` | LBNL's combined label for hybrids (`Solar+Battery`) | yes |
| `is_hybrid` | bool | `type_2` | True if a second resource type is listed | yes |
| `has_storage` | bool | `type_1..3` | True if any listed type is `Battery` or `Other Storage` | yes |
| `capacity_mw` | float | `mw_1` | Capacity of the primary resource, in MW (D-015) | yes |
| `queue_date` | datetime | `q_date` | Interconnection request date (time origin) | yes |
| `queue_year` | int | from `queue_date` | Entry year | yes |

## Outcome (targets, never features)

| Column | Type | Meaning |
|---|---|---|
| `status` | string | LBNL status as of the cutoff: `active`, `withdrawn`, `operational`, `suspended`, `unknown` |
| `operation_date` | datetime | Date the project became operational (`on_date`) |
| `withdrawal_date` | datetime | Date the project withdrew (`wd_date`) |
| `outcome` | string | `operational`, `withdrawn`, `censored`, or `unknown` (mapping in D-010) |
| `end_date` | datetime | Event date for events. The data cutoff (2025-12-31) for censored rows |
| `censored_at_cutoff_after_event` | bool | The event was dated after the cutoff, so the row was censored at the cutoff |
| `time_months` | float | `(end_date - queue_date)` in days / 30.4375. Missing if either date is missing |

## Information from after entry (EDA only, never features)

| Column | Source column | Why it is not feature-safe |
|---|---|---|
| `proposed_online_date` | `prop_date` | Can be revised during the process. The published value may be a later revision |
| `ia_date` | `ia_date` | Interconnection agreement signed after entry |
| `ia_phase` | `IA_phase_clean` | Study phase reached by the cutoff. LBNL imputes `IA Executed` for operational projects |

## Quality flags and scope

| Column | Meaning |
|---|---|
| `flag_*` | One boolean per data-quality check. The definitions are in `src/queueiq/cleaning/flags.py` (`FLAGS`), and counts and examples are in `reports/data_quality.md` |
| `in_model_scope` | True if the row passes every scope rule (D-011, D-012, D-017) |
| `scope_exclusion_reason` | Name of the first scope rule the row failed. Missing if the row is in scope |

LBNL columns not carried over: `project_name` (mostly missing), `cluster`, `IA_phase_raw`, `mw_2`, `mw_3`,
`prop_year`, and `q_year` (recomputed from `queue_date`). All of them remain in
`data/interim/lbnl_queue_sheet.parquet`.

## Feature table: `data/processed/features.parquet`

One row per queue entry, keyed by `project_id`, plus `in_model_scope`. Every feature is built only from
information available on the entry date. The full list, definitions and summary statistics are in
[`reports/features.md`](../reports/features.md), and the rules are in decisions D-023 to D-026.
