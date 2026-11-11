"""`make explain`: deployment model + explanations report (Phase 6).

1. Fit the primary configuration on all in-scope data (deployment model) and a Cox reference.
2. Reproducibility check: refit the primary configuration at the Phase 5 origins and confirm the
   predictions equal the saved Phase 5 predictions.
3. Global: grouped permutation importance on the Phase 5 test cohorts (report only), and mean
   |Shapley| of the deployment model over recent projects.
4. Local: example explanations with plain-language text.
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from queueiq import config
from queueiq.cleaning.report import to_markdown
from queueiq.evaluation.dev import DEV_ORIGINS, EVAL_DIR, SELECTION_FILE
from queueiq.evaluation.final import C_ORIGIN, C_TEST_YEARS
from queueiq.explain import deploy
from queueiq.explain import figures as F
from queueiq.explain.groups import GROUPS, TARGETS
from queueiq.explain.permutation import grouped_permutation_importance
from queueiq.explain.shapley import explain
from queueiq.explain.text import narrative
from queueiq.models.data import make_split

ENTRIES_FILE = config.PROCESSED_DIR / "queue_entries.parquet"
FEATURES_FILE = config.PROCESSED_DIR / "features.parquet"
REPORT_FILE = config.REPORTS_DIR / "explanations.md"
N_GLOBAL = 100  # recent projects explained for the global mean |Shapley|
PERM_REPEATS = 5


def refit_origin(entries, features, selection, origin_year, test_years, horizon):
    """Refit the primary configuration exactly as Phase 5 did at one origin."""
    split = make_split(
        entries,
        features,
        pd.Timestamp(f"{origin_year}-01-01"),
        test_years,
        min_test_followup_months=horizon,
    )
    model = deploy.primary_model(selection).fit(split.X_train, split.time_train, split.event_train)
    return model, split


def main() -> None:
    selection = json.loads(SELECTION_FILE.read_text())
    primary = selection["primary_configuration"]
    entries = pd.read_parquet(ENTRIES_FILE)
    features = pd.read_parquet(FEATURES_FILE)

    # 1. Deployment model ------------------------------------------------------------------
    print("Fitting the deployment model on all in-scope data ...")
    built = deploy.build(entries, features, selection)
    model, cox, background, meta = built["model"], built["cox"], built["background"], built["meta"]
    X, ids, fit_s = built["X"], built["ids"], built["fit_seconds"]
    reference, warning = meta["observed_reference"], meta["calibration_warning"]

    # 2 + 3a. Reproducibility and permutation importance on Phase 5 test cohorts -------------
    perm, repro = {}, []
    for block, origin, years in [
        *[("A", y, (y, y)) for y in DEV_ORIGINS],
        ("C", C_ORIGIN, C_TEST_YEARS),
    ]:
        print(f"Permutation importance, block {block} origin {origin} ...")
        saved = joblib.load(EVAL_DIR / f"{block}_{origin}.joblib")
        m, split = refit_origin(entries, features, selection, origin, years, max(saved.horizons))
        cif = m.predict_cif(split.X_test)
        diff = max(float(np.abs(cif[k] - saved.preds[primary][k]).max()) for k in cif)
        repro.append(
            {
                "block": block,
                "origin": origin,
                "test_projects": len(split.X_test),
                "max_abs_difference_vs_phase5": f"{diff:.1e}",
            }
        )
        perm[(block, origin)] = grouped_permutation_importance(
            m,
            split.X_test,
            split.time_test,
            split.event_test,
            saved.preds["null_aj"],
            GROUPS,
            repeats=PERM_REPEATS,
        )
    perm_a = pd.concat([perm[("A", y)] for y in DEV_ORIGINS])
    perm_table = pd.concat(
        {
            "A mean (2014-17)": perm_a.groupby(level=0).mean(),
            "A min": perm_a.groupby(level=0).min(),
            "A max": perm_a.groupby(level=0).max(),
            "C (2021-22)": perm[("C", C_ORIGIN)],
        },
        axis=1,
    ).reindex(list(GROUPS))

    # 3b + 4. Shapley for the deployment model ---------------------------------------------
    print("Explaining recent projects with the deployment model ...")
    predict = deploy.predictor(model)
    recent = X[X["queue_year"] == deploy.BACKGROUND_YEARS[1]]
    sample = recent.sample(N_GLOBAL, random_state=7)
    tables = [explain(predict, sample.iloc[[i]], background, GROUPS) for i in range(len(sample))]
    mean_abs = (
        pd.concat([tb.abs() for tb in tables])
        .groupby(level=0)
        .mean()
        .mul(100)
        .reindex(list(GROUPS))
    )
    max_gap = max(
        abs(tb[c].sum() - (tb.attrs["prediction"][c] - tb.attrs["baseline"][c]))
        for tb in tables
        for c in TARGETS
    )

    preds = predict(recent)
    order = np.argsort(preds["withdrawn_3y"])
    picks = {
        "highest withdrawal risk": order[-1],
        "median": order[len(order) // 2],
        "lowest withdrawal risk": order[0],
    }
    examples = []
    for label, i in picks.items():
        x = recent.iloc[[i]]
        tb = explain(predict, x, background, GROUPS)
        pid = ids[recent.index[i]]
        examples.append((label, pid, x.iloc[0], tb, narrative(tb, x.iloc[0], reference)))

    figs = {
        "perm": save(F.permutation_importance(perm_table), "fig09_permutation_importance.png"),
        "local": save(
            F.local_explanation(
                examples[0][3], "Example: " + examples[0][0] + " (2025 entrant)", reference, warning
            ),
            "fig10_example_explanation.png",
        ),
    }
    REPORT_FILE.write_text(
        build_report(
            primary, meta, fit_s, repro, perm_table, mean_abs, max_gap, examples, cox, figs
        )
    )
    print(f"Wrote {REPORT_FILE.name}; deployment model in models/")


def reference_table(meta) -> pd.DataFrame:
    rows = []
    for target, (_, _, label) in TARGETS.items():
        ref = meta["observed_reference"][target]
        rows.append(
            {
                "headline number": label,
                "observed for latest fully followed cohort": f"{100 * ref['observed']:.1f}% "
                f"({ref['cohort']} entrants, n = {ref['n']:,})",
            }
        )
    return pd.DataFrame(rows)


def save(fig, name: str) -> str:
    fig.savefig(config.FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


def build_report(
    primary, meta, fit_s, repro, perm_table, mean_abs, max_gap, examples, cox, figs
) -> str:
    hr = cox.hazard_ratios()
    top_hr = (
        hr.assign(spread=np.log(hr).abs().max(axis=1))
        .sort_values("spread", ascending=False)
        .head(12)
        .drop(columns="spread")
        .round(3)
    )
    perm_views = {}
    for outcome in ("withdrawn", "operational"):
        rows = {}
        for metric in ("skill", "auc"):
            col = f"{outcome}_{metric}"
            mean, lo, hi = (perm_table[(k, col)] for k in ("A mean (2014-17)", "A min", "A max"))
            final = perm_table[("C (2021-22)", col)]
            rows[f"{metric} drop: development mean [range]"] = [
                f"{m:+.4f} [{a:+.4f}, {b:+.4f}]" for m, a, b in zip(mean, lo, hi, strict=True)
            ]
            rows[f"{metric} drop: final test"] = [
                "not estimable" if np.isnan(v) else f"{v:+.4f}" for v in final
            ]
        perm_views[outcome] = pd.DataFrame(rows, index=perm_table.index)
    example_md = []
    for label, pid, _row, tb, text in examples:
        view = (
            (tb * 100).round(2).rename(columns={k: v[2] + " (points)" for k, v in TARGETS.items()})
        )
        example_md += [
            f"### {label}: `{pid}`",
            "",
            *[f"- {s}" for s in text.values()],
            "",
            to_markdown(view, index=True),
            "",
        ]
    return f"""# Explanations (generated by `make explain`, do not edit by hand)

Model explained: the **primary configuration** `{primary}` frozen in Phase 5, refit on all
in-scope data as the **deployment model** ({meta["trained_on_rows"]:,} projects, outcomes as of
{meta["outcomes_as_of"]}; fit in {fit_s} s). Decisions D-039 to D-042.

Explanations describe **what the model relies on**, not what causes withdrawal or completion.

## 1. Reproducibility check
The primary configuration was refit at each Phase 5 origin. Its test predictions match the
saved Phase 5 predictions:

{to_markdown(pd.DataFrame(repro))}

## 2. What the model relies on out of sample (grouped permutation importance)
Each group's columns are shuffled together across the test cohort ({PERM_REPEATS} repeats), and
the drop in performance is measured: Brier skill vs the null model, and AUC (3 years). Higher means
the model relies more on that group. Values near 0 (or below 0) mean the group adds nothing out of
sample. Block A = the four development origins (mean, min, max). Block C = the final test (2021–22
entrants), where operational AUC is not estimable (1 case). Report only.

![Permutation importance]({figs["perm"]})

### Withdrawn
{to_markdown(perm_views["withdrawn"], index=True)}

### Operational
{to_markdown(perm_views["operational"], index=True)}

## 3. What moves individual predictions (deployment model, mean |Shapley|)
Exact grouped Shapley values (all 128 group combinations) for {N_GLOBAL} randomly chosen 2025
entrants. The comparison point is the average prediction over {meta["background_size"]} random
{meta["background_years"][0]}–{meta["background_years"][1]} entrants: the **model's average prediction for recent entrants**.
This is a model output, not an observed rate. See section 4 for observed rates.
Values are the mean absolute contribution in percentage points. For every explained project, the
contributions add up to the prediction minus the baseline, with a largest gap of {max_gap:.1e}.

{to_markdown(mean_abs.round(2).rename(columns={k: v[2] for k, v in TARGETS.items()}), index=True)}

Permutation importance (section 2) asks "does the model *need* this group to predict well on new
cohorts?" Shapley values (section 3) ask "how much does this group *move* the prediction?" A group
can move predictions a lot without improving accuracy. Where the two disagree, trust section 2 for
accuracy and section 3 for what the model is doing.

## 4. Example explanations (2025 entrants, still in the queue)
These are what the dashboard will show. Points are percentage points relative to the **model's
average prediction for recent ({meta["background_years"][0]}–{meta["background_years"][1]}) entrants**.
That baseline is shown next to the observed rate for the most recent cohort with full follow-up:

{to_markdown(reference_table(meta))}

> {meta["calibration_warning"]}

![Example explanation]({figs["local"]})

{chr(10).join(example_md)}

## 5. Cox hazard ratios (interpretable reference, not the deployed model)
A cause-specific Cox model with the same features, fit on the same data. Hazard ratios are per one
standard deviation of the (log-transformed for counts and MW) feature, or against the reference
category. These are the 12 covariates with the largest effect on either outcome. Caveats: the
features are correlated and ridge-penalised, proportional hazards is only approximate (D-029), and
none of this is causal.

{to_markdown(top_hr, index=True)}
"""


if __name__ == "__main__":
    main()
