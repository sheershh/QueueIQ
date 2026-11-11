# Evaluation protocol (Phase 5)

Written and committed **before** any Phase 5 result is computed. It is tagged `eval-protocol-v1` in git, so
the git history shows that the rules came before the results. Any later change to this file is a new
version with its own tag and a stated reason. Results produced under an older version stay reported.

## 0. What has already been seen
- Phase 4 scored models trained before 2018-01-01 on the **2018–2020 entrants** (3- and 5-year horizons),
  and those results were looked at. The 2018–2020 cohorts are therefore **not** used for any decision below,
  and they are not the final test.
- Descriptive outcome counts per entry year are known from the EDA report (`reports/eda.md`). No model has
  been scored on any 2021 or later cohort.

## 1. Three blocks, and which ones drive decisions
All models are refit at every origin. Training uses in-scope projects that entered before the origin, with
outcomes cut off at the origin (decisions D-027). Each test cohort is the in-scope entrants of the stated
years, with outcomes observed to the data cutoff (2025-12-31). Test rows still in the queue with less than
the horizon's follow-up are dropped and counted.

| Block | Origins (start of year Y) | Test cohort | Horizons | Role |
|---|---|---|---|---|
| **A. Development** | 2014, 2015, 2016, 2017 | entrants in year Y | 36 months (selection); 60 months for Y = 2014, 2015 (report only) | **Only block used for decisions** |
| **B. Extension** | 2018, 2019, 2020 | entrants in year Y | 36 and 60 months | Report only, run after selection is frozen |
| **C. Final test** | 2021 | entrants in 2021–2022 | 36 months | Report only, **run once** with the frozen selection |

- **Block A uses only information known by 2021-01-01.** Its longest-dated label is a 2017 entrant at
  36 months, which ends in 2020.
- **If the final run fails for a technical reason** (a crash, not a disappointing result), it may be re-run
  after a fix that changes none of the choices in this protocol. Any re-run is recorded in
  `docs/decisions.md`.

## 2. Candidates for selection (15)
| Family | Variants |
|---|---|
| `recent_cohort` baseline (section 6) | 1 (no recalibration: it is already calibrated to the recent cohort) |
| `baseline_aj_iso_fuel` (Aalen-Johansen by ISO × fuel, all history) | recalibration {off, on} = 2 |
| `cox_cause_specific` | `queue_year` {with, without} × recalibration {off, on} = 4 |
| `lgbm_discrete_time` | same 4 |
| `rsf_cause_specific` | same 4 |

- **Settings:** all model settings are exactly as in Phase 4 (D-029, D-030). LightGBM's rounds are chosen
  separately at each origin Y by early stopping on an inner split: train on entrants before the start of
  Y−2 (outcomes as of Y−2), validate on Y−2…Y−1 entrants (outcomes as of Y).
- **Report-only ablation, not a candidate:** each of Cox, LightGBM and the forest is also fit **without the
  congestion features** (all `county_*`/`state_*` features and `county_known`), with `queue_year` in and
  recalibration off. It is reported in blocks A, B and C and is never used for selection.
- **Reference (not a candidate):** the null model is an Aalen-Johansen curve with no features, fitted on
  the same training data at each origin. It is used only to scale the selection score.

## 3. Selection score (exact)
For candidate c, origin Y ∈ {2014, 2015, 2016, 2017} and outcome k ∈ {operational, withdrawn}, over the
n_Y test projects i:

    BS_{c,Y,k}(t)  = (1 / n_Y) Σ_i ( 1[T_i ≤ t and cause_i = k] − F̂_{c,k}(t | x_i) )²     t = month 1…36
    IBS_{c,Y,k}    = (1 / 36) Σ_{t=1..36} BS_{c,Y,k}(t)
    Skill_{c,Y,k}  = 1 − IBS_{c,Y,k} / IBS_{null,Y,k}
    Score_c        = (1 / 8) Σ_Y Σ_k Skill_{c,Y,k}

- **Equal weights:** every origin counts equally, whatever its cohort size, and so does each outcome.
  Skill puts operational and withdrawal on the same scale; a raw IBS average would be dominated by
  withdrawal, whose IBS is about 10 times larger. Higher is better, and 0 means no better than the null.
- "T_i ≤ t" means the event fell in month index ≤ t (D-027 month convention). F̂ is read at the end of
  month t.
- **Choice:** the candidate with the highest Score_c is the **primary configuration**.
- **Tie rule:** if a less complex candidate scores within **0.005** of the best, the least complex such
  candidate is chosen. Complexity order, compared from left to right:
  - family: `recent_cohort` < `baseline_aj_iso_fuel` < `cox_cause_specific` < `lgbm_discrete_time` <
    `rsf_cause_specific`;
  - then without `queue_year` < with;
  - then recalibration off < on.
- **If a baseline wins, it becomes the primary configuration**, and the report says so.
- **What the primary configuration is used for:** the headline result, the model card, and the dashboard
  model (Phase 7).
- **Freezing:** the selection, per-candidate scores and per-origin skills are written to
  `reports/evaluation_selection.json` and committed **before** blocks B and C are run.

## 4. Metrics (defined for competing risks)
k is the outcome being scored, τ the horizon (36 or 60 months), and F̂_k(τ | x) the predicted cumulative
incidence.

- **Time-dependent AUC(τ)** (cumulative/dynamic, Blanche et al. 2013, "definition 2" controls). Cases are
  projects with event k by τ. Controls are everyone else: the competing event by τ, or no event by τ. The
  score is F̂_k(τ). It equals the probability that a random case scores above a random control (ties count
  ½).
- **Truncated C-index for competing risks** (Wolbers et al. 2009/2014). A pair (i, j) is comparable if i had
  event k at T_i ≤ τ, and j either had no event of any kind up to T_i (T_j > T_i) or had the competing event
  at any time. It is concordant if F̂_k(τ | x_i) > F̂_k(τ | x_j) (ties count ½). C = concordant ÷ comparable.
- **Brier(τ), IBS (months 1…τ) and skill:** as in section 3.
- **Calibration:**
  - calibration-in-the-large: mean F̂_k(τ) against the observed share with k by τ;
  - calibration curves: 10 equal-count bins of F̂_k(τ), mean predicted against observed share per bin;
  - ECE: the bin-size-weighted mean of |predicted − observed|.
- **Censoring:** every evaluation cohort has full follow-up to τ (rows that don't are dropped and counted),
  so the inverse-probability-of-censoring weights these metrics normally need all equal 1. The code asserts
  full follow-up. It does not apply the weights.
- **Event counts:** every metric row reports n, the number of cases, and (for the C-index) the number of
  comparable pairs.
- **Minimum events:** AUC and C-index are reported as **"not estimable"** when there are fewer than **20**
  cases, and no model comparison is drawn from them. IBS, Brier and calibration are still reported.
  Block C has 1 operational case at 36 months, so operational *ranking* is not evaluated there. Only
  operational *calibration* is.

## 5. Uncertainty
- **Resampling:** 1,000 bootstrap resamples (seed 42) of the **fixed test predictions**. No refitting, so
  the intervals cover test-sample uncertainty only. The interval is the 2.5th to 97.5th percentile.
- **Two resampling units, both reported:**
  - **projects;**
  - **county × entry-year groups.** Projects with no known county each form their own group.
- **Pooled block A** also reports **ISO × entry-year** groups (16), labelled unreliable because there are
  so few groups.
- **Paired differences:** each resample scores all candidates on the same projects, giving intervals for
  (a) every candidate minus the primary configuration, and (b) the primary configuration minus the
  `recent_cohort` baseline.
- **Across-origin spread:** the spread of per-origin metrics across blocks A and B is reported as the
  measure of uncertainty from one era to the next.

## 6. "Most recent fully observed cohort" baseline
- At origin Y and horizon τ, the reference cohort is the latest entry year whose τ-month outcomes are fully
  known at Y: **Y − 4 for 36 months**, and **Y − 6 for 60 months**.
- **Prediction:** the reference cohort's Aalen-Johansen cumulative incidence by **ISO**. Any ISO with fewer
  than 30 reference projects uses the all-ISO curve. With full follow-up this equals the observed share
  with each event by month t.
- **Scope:** the curve is defined up to τ. Only months ≤ τ are scored.

## 7. Recalibration for the calibration gap (no use of the test period)
- **Reference cohort:** R = entrants of year Y − 4. Their 36-month outcomes are fully known at origin Y.
  For each outcome k, o_k = the observed share of R with event k by month 36.
- **Recovering hazards:** the model's monthly curves are turned back into discrete cause-specific hazards,
  h_k(t) = ΔF̂_k(t) ÷ Ŝ(t−1).
- **Scaling:** each hazard is multiplied by a constant c_k. The curves are rebuilt with the product-limit
  formula (S + F_op + F_wd = 1 is preserved, and a month's total hazard is capped at 1).
- **Solving for c_op and c_wd:** they are chosen so that the mean rebuilt F̂_k(36) over R equals o_k for
  both outcomes. The solver alternates one-dimensional bisection on log c over [0.01, 100], 50 iterations
  each, 20 rounds. If o_k = 0, c_k = 0.01.
- **Scope:** c_k is fitted once per origin and model, and applied unchanged to every test prediction from
  that origin. It uses only data known at the origin. R is part of the training data, so the model's
  predictions for R are in-sample. This is accepted and stated.

## 8. What block C reports (required whatever block A selects)
- All 15 candidates, explicitly including Cox, LightGBM and the forest **with and without `queue_year`**
  and **with and without recalibration**.
- The congestion ablation (section 2).
- The **primary configuration** labelled as primary. All other rows are labelled **sensitivity analyses**.
  A sensitivity row that scores better on C **does not replace** the primary configuration, and the report
  says so whenever it happens.
- For every metric: the counts, the project bootstrap interval and the county × entry-year bootstrap
  interval.

## 9. Order of work
1. Commit and tag this protocol (`eval-protocol-v1`).
2. Implement and unit-test the metrics, bootstrap, recalibration and backtest code on synthetic data only.
3. Run block A. Write and commit `reports/evaluation_selection.json` (the selection freeze).
4. Run blocks B and C once. Write `reports/evaluation.md`, the figures and `docs/model_card.md`. Commit.
