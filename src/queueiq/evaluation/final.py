"""`make evaluate`: blocks B and C (report only) with the frozen selection; report + model card.

Implements docs/evaluation_protocol.md sections 4, 5 and 8. It refuses to run if the selection
has not been frozen, or if the protocol file changed after the selection was made.
"""

import json
import subprocess

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from queueiq import config
from queueiq.cleaning.report import to_markdown
from queueiq.evaluation import bootstrap as bs
from queueiq.evaluation import figures as F
from queueiq.evaluation.backtest import (
    CAUSE_CODE,
    OriginResult,
    ablation_names,
    candidate_names,
    point_metrics,
    run_origin,
)
from queueiq.evaluation.data_checks import gap_bound, missing_withdrawal_dates_by_year
from queueiq.evaluation.dev import (
    DEV_ORIGINS,
    ENTRIES_FILE,
    EVAL_DIR,
    FEATURES_FILE,
    SELECTION_FILE,
    protocol_sha256,
)
from queueiq.evaluation.metrics import (
    MIN_CASES,
    auc_pairs,
    calibration_bins,
    cindex_pairs,
    event_by,
    ibs_per_project,
)

B_ORIGINS = (2018, 2019, 2020)
C_ORIGIN = 2021
C_TEST_YEARS = (2021, 2022)
REPORT_FILE = config.REPORTS_DIR / "evaluation.md"
METRICS_FILE = config.REPORTS_DIR / "evaluation_metrics.csv"
BOOT_FILE = config.REPORTS_DIR / "evaluation_bootstrap.csv"
MODEL_CARD = config.PROJECT_ROOT / "docs" / "model_card.md"
TAU = 36


def pct(x: float) -> str:
    """Percent with enough precision that a tiny rate (1 in 3,299) doesn't print as 0.0%."""
    v = 100 * x
    return f"{v:.2f}%" if v < 1 else f"{v:.1f}%"


def load_selection() -> dict:
    if not SELECTION_FILE.exists():
        raise RuntimeError("No frozen selection: run `make evaluate-dev` and commit it first.")
    sel = json.loads(SELECTION_FILE.read_text())
    if sel["protocol_sha256"] != protocol_sha256():
        raise RuntimeError("docs/evaluation_protocol.md changed after the selection was frozen.")
    return sel


def git_commit_of(path: str) -> str:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", path],
            capture_output=True,
            text=True,
            check=True,
            cwd=config.PROJECT_ROOT,
        ).stdout.strip()
        return out or "uncommitted"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def pool(results: list[OriginResult], block: str) -> OriginResult:
    """Stack several origins' test cohorts (each scored by its own origin's models)."""
    names = set.intersection(*(set(r.preds) for r in results))
    cat = lambda attr: np.concatenate([getattr(r, attr) for r in results])  # noqa: E731
    return OriginResult(
        block=block,
        origin_year=0,
        test_years=(results[0].test_years[0], results[-1].test_years[1]),
        horizons=(TAU,),
        ids=cat("ids"),
        time=cat("time"),
        event=cat("event"),
        iso=cat("iso"),
        entry_year=cat("entry_year"),
        county_year=cat("county_year"),
        preds={
            n: {k: np.concatenate([r.preds[n][k] for r in results]) for k in CAUSE_CODE}
            for n in names
        },
    )


def role(name: str, primary: str) -> str:
    if name == primary:
        return "PRIMARY"
    if name in ablation_names():
        return "ablation (report only)"
    if name.startswith(("recent_cohort", "baseline_aj")):
        return "baseline"
    if name == "null_aj":
        return "null (reference)"
    return "sensitivity"


def bootstrap_block(res: OriginResult, names: list[str], units: dict, primary: str, B: int):
    """Intervals for every name/outcome/metric and paired differences, per resampling unit."""
    rows = []
    weights = {
        u: bs.project_weights(len(res.ids), B) if g is None else bs.group_weights(g, B)
        for u, g in units.items()
    }
    for k, code in CAUSE_CODE.items():
        y = event_by(res.time, res.event, code, TAU)
        null_ibs = ibs_per_project(res.time, res.event, code, res.preds["null_aj"][k], TAU)
        parts = {}
        for name in names:
            risk = res.preds[name][k][:, TAU - 1].astype(float)
            parts[name] = {
                "risk": risk,
                "ibs": ibs_per_project(res.time, res.event, code, res.preds[name][k], TAU),
                "auc": auc_pairs(res.time, res.event, code, risk, TAU)
                if y.sum() >= MIN_CASES
                else None,
                "cindex": cindex_pairs(res.time, res.event, code, risk, TAU)
                if y.sum() >= MIN_CASES
                else None,
            }
        for unit, W in weights.items():
            tot = W.sum(axis=1)
            null_tot = W @ null_ibs
            draws = {}
            for name, p in parts.items():
                d = {
                    "skill": 1 - (W @ p["ibs"]) / null_tot,
                    "ibs": (W @ p["ibs"]) / tot,
                    "brier_tau": (W @ (y - p["risk"]) ** 2) / tot,
                    "calibration_gap": (W @ (p["risk"] - y)) / tot,
                }
                for metric in ("auc", "cindex"):
                    if p[metric] is not None:
                        d[metric] = p[metric].value(W)
                draws[name] = d
                for metric, v in d.items():
                    lo, hi = bs.interval(v)
                    rows.append(
                        {
                            "block": res.block,
                            "unit": unit,
                            "model": name,
                            "outcome": k,
                            "metric": metric,
                            "lo": lo,
                            "hi": hi,
                        }
                    )
            for name, d in draws.items():
                if name == primary:
                    continue
                for metric in ("skill", "auc"):
                    if metric in d and metric in draws[primary]:
                        lo, hi = bs.interval(d[metric] - draws[primary][metric])
                        rows.append(
                            {
                                "block": res.block,
                                "unit": unit,
                                "model": name,
                                "outcome": k,
                                "metric": f"{metric}_minus_primary",
                                "lo": lo,
                                "hi": hi,
                            }
                        )
    return pd.DataFrame(rows)


def ci_text(boot: pd.DataFrame, block, model, outcome, metric, units) -> str:
    out = []
    for u in units:
        r = boot[
            (boot["block"] == block)
            & (boot["model"] == model)
            & (boot["outcome"] == outcome)
            & (boot["metric"] == metric)
            & (boot["unit"] == u)
        ]
        if len(r) and not np.isnan(r["lo"].iloc[0]):
            out.append(f"[{r['lo'].iloc[0]:.3f}, {r['hi'].iloc[0]:.3f}]")
        else:
            out.append("–")
    return " ".join(out)


def block_table(metrics, boot, block, outcome, names, primary, units) -> pd.DataFrame:
    m = metrics[
        (metrics["block"] == block) & (metrics["outcome"] == outcome) & (metrics["tau"] == TAU)
    ]
    rows = []
    for name in names:
        r = m[m["model"] == name]
        if r.empty:
            continue
        r = r.iloc[0]
        auc = (
            "not estimable"
            if np.isnan(r["auc"])
            else f"{r['auc']:.3f} {ci_text(boot, block, name, outcome, 'auc', units)}"
        )
        cidx = (
            "not estimable"
            if np.isnan(r["cindex"])
            else f"{r['cindex']:.3f} ({int(r['c_pairs']):,} pairs)"
        )
        rows.append(
            {
                "model": name,
                "role": role(name, primary),
                "n": int(r["n"]),
                "cases": int(r["cases"]),
                "AUC [CI by unit]": auc,
                "C-index": cidx,
                "skill [CI by unit]": f"{r['skill']:.3f} {ci_text(boot, block, name, outcome, 'skill', units)}",
                "Brier@3y": f"{r['brier_tau']:.4f}",
                "predicted vs observed": f"{pct(r['mean_pred'])} vs {pct(r['observed'])}",
                "ECE": f"{r['ece']:.3f}",
            }
        )
    return pd.DataFrame(rows)


def per_origin_table(metrics, names, tau, metric) -> pd.DataFrame:
    m = metrics[
        (metrics["tau"] == tau)
        & metrics["block"].isin(["A", "B", "C"])
        & metrics["model"].isin(names)
    ]
    t = m.pivot_table(index=["outcome", "model"], columns="origin", values=metric).round(3)
    cases = m[m["model"] == "null_aj"] if "null_aj" in names else m
    return t, cases.pivot_table(index="outcome", columns="origin", values="cases", aggfunc="first")


def main(B: int = bs.B_DEFAULT) -> None:
    sel = load_selection()
    primary = sel["primary_configuration"]
    entries = pd.read_parquet(ENTRIES_FILE)
    features = pd.read_parquet(FEATURES_FILE)
    results = {("A", y): joblib.load(EVAL_DIR / f"A_{y}.joblib") for y in DEV_ORIGINS}
    for y in B_ORIGINS:
        print(f"Block B (report only), origin {y} ...")
        results[("B", y)] = run_origin(entries, features, "B", y, (y, y), (36, 60))
        joblib.dump(results[("B", y)], EVAL_DIR / f"B_{y}.joblib")
    print("Block C (final test, run once) ...")
    results[("C", C_ORIGIN)] = run_origin(entries, features, "C", C_ORIGIN, C_TEST_YEARS, (36,))
    joblib.dump(results[("C", C_ORIGIN)], EVAL_DIR / f"C_{C_ORIGIN}.joblib")

    names = [*candidate_names(), *ablation_names()]
    rows = []
    for res in results.values():
        for tau in res.horizons:
            rows += point_metrics(res, [*names, "null_aj"], tau)
    pooled_a = pool([results[("A", y)] for y in DEV_ORIGINS], "A-pooled")
    rows += point_metrics(pooled_a, [*names, "null_aj"], TAU)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_FILE, index=False, float_format="%.6g")

    print("Bootstrapping (block A pooled, block C) ...")
    final = results[("C", C_ORIGIN)]
    boot = pd.concat(
        [
            bootstrap_block(
                pooled_a,
                names,
                {
                    "project": None,
                    "county-year": pooled_a.county_year,
                    "ISO-year": np.char.add(
                        np.char.add(pooled_a.iso.astype(str), "|"), pooled_a.entry_year.astype(str)
                    ),
                },
                primary,
                B,
            ),
            bootstrap_block(
                final, names, {"project": None, "county-year": final.county_year}, primary, B
            ),
        ]
    )
    boot.to_csv(BOOT_FILE, index=False, float_format="%.6g")

    n_iso_year = len(set(zip(pooled_a.iso, pooled_a.entry_year, strict=True)))
    write_outputs(sel, metrics, boot, results, primary, B, n_iso_year, entries)


def write_outputs(sel, metrics, boot, results, primary, B, n_iso_year, entries) -> None:
    figs = make_figures(metrics, results, primary)
    report = build_report(sel, metrics, boot, results, primary, figs, B, n_iso_year)
    report += data_gap_section(entries, metrics, primary)
    REPORT_FILE.write_text(report)
    MODEL_CARD.write_text(build_model_card(sel, metrics, boot, results, primary))
    print(f"Wrote {REPORT_FILE.name} and {MODEL_CARD.name}")


def render_only(B: int = bs.B_DEFAULT) -> None:
    """Rebuild the report, figures and model card from SAVED results. Nothing is refit or
    re-scored: it reads the stored predictions and the metric/bootstrap CSVs from the single
    final run, so presentation fixes never mean touching block C again."""
    sel = load_selection()
    primary = sel["primary_configuration"]
    results = {("A", y): joblib.load(EVAL_DIR / f"A_{y}.joblib") for y in DEV_ORIGINS}
    results |= {("B", y): joblib.load(EVAL_DIR / f"B_{y}.joblib") for y in B_ORIGINS}
    results[("C", C_ORIGIN)] = joblib.load(EVAL_DIR / f"C_{C_ORIGIN}.joblib")
    metrics, boot = pd.read_csv(METRICS_FILE), pd.read_csv(BOOT_FILE)
    pooled_a = pool([results[("A", y)] for y in DEV_ORIGINS], "A-pooled")
    n_iso_year = len(set(zip(pooled_a.iso, pooled_a.entry_year, strict=True)))
    write_outputs(
        sel, metrics, boot, results, primary, B, n_iso_year, pd.read_parquet(ENTRIES_FILE)
    )


def make_figures(metrics, results, primary) -> dict[str, str]:
    lines = {
        primary: ("Primary (selected in A)", "primary", "-"),
        "baseline_aj_iso_fuel|recal=0": ("ISO x fuel baseline", "baseline", "--"),
        "recent_cohort": ("Most recent cohort", "recent", ":"),
    }
    figs = {}
    fig = F.rolling_origin(metrics[metrics["block"].isin(["A", "B", "C"])], lines)
    figs["rolling"] = save(fig, "fig07_rolling_origin.png")
    final = results[("C", C_ORIGIN)]
    recal_name = primary.replace("recal=0", "recal=1") if "recal=0" in primary else None
    variants = {primary: "Primary"}
    if recal_name and recal_name in final.preds:
        variants[recal_name] = "Primary + recalibration"
    bins = {}
    for k, code in CAUSE_CODE.items():
        y = event_by(final.time, final.event, code, TAU)
        bins[k] = {
            label: calibration_bins(final.preds[n][k][:, TAU - 1], y)
            for n, label in variants.items()
        }
    figs["calibration"] = save(
        F.calibration(bins, "Calibration on the final test (2021-22 entrants, 3 years)"),
        "fig08_calibration_final.png",
    )
    return figs


def save(fig, name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


def ablation_table(metrics) -> pd.DataFrame:
    """Each family with all features vs without congestion features (queue_year in, no
    recalibration in both), at 3 years. Report only (protocol section 2)."""
    m = metrics[(metrics["tau"] == TAU) & metrics["block"].isin(["A-pooled", "C"])]
    rows = []
    for fam in ("cox_cause_specific", "lgbm_discrete_time", "rsf_cause_specific"):
        for block in ("A-pooled", "C"):
            for k in ("withdrawn", "operational"):
                g = m[(m["block"] == block) & (m["outcome"] == k)].set_index("model")
                full, noc = g.loc[f"{fam}|qy=1|recal=0"], g.loc[f"{fam}|no_congestion"]
                rows.append(
                    {
                        "family": fam,
                        "block": block,
                        "outcome": k,
                        "cases": int(full["cases"]),
                        "skill with": round(full["skill"], 3),
                        "skill without": round(noc["skill"], 3),
                        "AUC with": round(full["auc"], 3),
                        "AUC without": round(noc["auc"], 3),
                    }
                )
    return pd.DataFrame(rows)


def ordered_names(primary: str) -> list[str]:
    rest = [n for n in candidate_names() if n != primary]
    return [primary, *rest, *ablation_names()]


def spread(metrics, primary) -> pd.DataFrame:
    m = metrics[
        (metrics["model"] == primary) & (metrics["tau"] == TAU) & metrics["block"].isin(["A", "B"])
    ]
    t = m.groupby("outcome")[["auc", "skill"]].agg(["mean", "std", "min", "max"]).round(3)
    t.columns = [f"{metric} {stat}" for metric, stat in t.columns]
    return t


def unit_width_ratio(boot) -> pd.DataFrame:
    b = boot.assign(width=boot["hi"] - boot["lo"])
    b = b[~b["metric"].str.contains("minus")]
    w = b.pivot_table(index=["block", "model", "outcome", "metric"], columns="unit", values="width")
    out = {}
    for unit in [u for u in w.columns if u != "project"]:
        ratio = (w[unit] / w["project"]).dropna()
        out[unit] = ratio.groupby(level="block").median().round(2)
    return pd.DataFrame(out).rename_axis("block")


def better_than_primary(metrics, primary, block) -> list[str]:
    """Rows whose skill beats the primary configuration on `block`, per outcome."""
    m = metrics[(metrics["block"] == block) & (metrics["tau"] == TAU)]
    notes = []
    for outcome in ("withdrawn", "operational"):
        mo = m[m["outcome"] == outcome].set_index("model")["skill"]
        p = mo.get(primary)
        winners = [
            f"`{n}` ({mo[n]:.3f})" for n in ordered_names(primary)[1:] if n in mo and mo[n] > p
        ]
        if winners:
            notes.append(f"{outcome} (primary {p:.3f}): " + ", ".join(winners))
    return notes


def build_report(sel, metrics, boot, results, primary, figs, B, n_iso_year) -> str:
    names = ordered_names(primary)
    final = results[("C", C_ORIGIN)]
    c_units, a_units = ["project", "county-year"], ["project", "county-year", "ISO-year"]
    b_rows = metrics[metrics["block"] == "B"]
    show = [primary, "baseline_aj_iso_fuel|recal=0", "recent_cohort", "null_aj"]
    skill_36, cases_36 = per_origin_table(metrics, show, 36, "skill")
    auc_36, _ = per_origin_table(metrics, show, 36, "auc")
    skill_60, cases_60 = per_origin_table(metrics, show, 60, "skill")
    better = better_than_primary(metrics, primary, "C")
    recal_rows = [
        {
            "origin": f"{b}-{y}",
            "c_operational": round(
                r.recal_constants[primary.replace("recal=0", "recal=1")]["operational"], 3
            )
            if "recal=0" in primary
            else None,
            "c_withdrawn": round(
                r.recal_constants[primary.replace("recal=0", "recal=1")]["withdrawn"], 3
            )
            if "recal=0" in primary
            else None,
        }
        for (b, y), r in results.items()
    ]
    c_desc = (
        f"entrants {C_TEST_YEARS[0]}-{C_TEST_YEARS[1]}, trained on entrants before "
        f"{C_ORIGIN}-01-01 (outcomes as of that date); {len(final.ids):,} projects, "
        f"{final.dropped_short_followup} dropped for follow-up just short of 36 months"
    )
    return f"""# Evaluation (generated by `make evaluate`, do not edit by hand)

Protocol: [`docs/evaluation_protocol.md`](../docs/evaluation_protocol.md), tag `{sel["protocol_tag"]}`
(commit `{sel["protocol_commit"][:7]}`). The selection was frozen in
[`reports/evaluation_selection.json`](evaluation_selection.json) (commit `{git_commit_of("reports/evaluation_selection.json")}`)
before blocks B and C were run.

**Primary configuration:** `{primary}` ({sel["selection_reason"]}).
Every other row is a sensitivity analysis, an ablation, or a baseline. None of them can replace
the primary configuration after the fact.

Reading the tables:
- `cases` is the number of projects with that event by 3 years. AUC and C-index are "not
  estimable" below {MIN_CASES} cases.
- `skill` = 1 − IBS / IBS(null), where the null model is one curve with no features. 0 means no
  better than the null.
- Intervals are 95% bootstrap percentile intervals over {B:,} resamples, shown in this order:
  {", ".join(f"`{u}`" for u in a_units)}.

## Final test (block C, run once): {c_desc}
### Withdrawn by 3 years
{to_markdown(block_table(metrics, boot, "C", "withdrawn", names, primary, c_units))}

### Operational by 3 years
Only {int(event_by(final.time, final.event, 1, TAU).sum())} test project(s) reached operation within
3 years, so ranking (AUC, C-index) can't be evaluated. Skill and calibration still can.

{to_markdown(block_table(metrics, boot, "C", "operational", names, primary, c_units))}

{"**Rows whose block C skill beats the primary configuration** (they do not replace it, protocol section 8):" + "".join(chr(10) + "- " + b for b in better) if better else "No row beat the primary configuration on block C."}

![Calibration on the final test]({figs["calibration"]})

## Development (block A pooled: entrants 2014–2017, each scored by its own origin's models)
### Withdrawn by 3 years
{to_markdown(block_table(metrics, boot, "A-pooled", "withdrawn", names, primary, a_units))}

### Operational by 3 years
{to_markdown(block_table(metrics, boot, "A-pooled", "operational", names, primary, a_units))}

The ISO-year intervals rest on only {n_iso_year} groups and are labelled unreliable (protocol section 5).

## Rolling-origin view (blocks A, B, C at 3 years)
![Rolling-origin performance]({figs["rolling"]})

Skill at 3 years, by origin:

{to_markdown(skill_36, index=True)}

AUC at 3 years, by origin:

{to_markdown(auc_36, index=True)}

Cases at 3 years, by origin:

{to_markdown(cases_36, index=True)}

Skill at 5 years (report only; origins with 5 years of follow-up), with case counts:

{to_markdown(skill_60, index=True)}

{to_markdown(cases_60, index=True)}

Spread of the primary configuration across the seven origins in blocks A and B (the measure of
era-to-era uncertainty):

{to_markdown(spread(metrics, primary), index=True)}

## Congestion features: ablation (report only)
Each family with all features vs without the congestion features (`queue_year` kept and no
recalibration in both). AUC is blank where there are fewer than {MIN_CASES} cases.

{to_markdown(ablation_table(metrics))}

## Bootstrap units
Median width of grouped intervals relative to project intervals (1.0 = same width):

{to_markdown(unit_width_ratio(boot), index=True)}

## Recalibration constants for the primary family
Hazard multipliers fitted at each origin on the reference cohort Y−4 (protocol section 7).
Values below 1 mean the model's hazards were too high for the most recent fully observed cohort.

{to_markdown(pd.DataFrame(recal_rows))}

## Block B notes
Block B (origins 2018–2020) was seen in Phase 4 and is reported only, never used for decisions.
Rows: {len(b_rows):,} metric rows in `reports/evaluation_metrics.csv`. Every point metric with its
counts is in that file. Every bootstrap interval, including paired differences against the
primary configuration (`*_minus_primary`), is in `reports/evaluation_bootstrap.csv`.
"""


def data_gap_section(entries, metrics, primary) -> str:
    """Added after the final run (post-hoc data check; nothing is re-scored)."""
    table = missing_withdrawal_dates_by_year(entries)
    c = metrics[
        (metrics["block"] == "C")
        & (metrics["model"] == primary)
        & (metrics["tau"] == TAU)
        & (metrics["outcome"] == "withdrawn")
    ].iloc[0]
    b = gap_bound(table, C_TEST_YEARS, int(c["n"]), int(c["cases"]), float(c["mean_pred"]))
    final_rows = table.loc[C_TEST_YEARS[0] : C_TEST_YEARS[1]]
    return f"""

## Data check: missing withdrawal dates and the final-test gap (added after the final run)
A post-hoc data check, added after block C was run. It re-scores nothing.

Withdrawn projects with no withdrawal date were excluded from model scope (rule
`missing_event_date`). Each exclusion removes a withdrawal from its entry cohort and lowers the
observed withdrawal rate. Withdrawn projects that pass every other scope rule, by entry year:

{to_markdown(table, index=True)}

For the final-test cohorts ({C_TEST_YEARS[0]}-{C_TEST_YEARS[1]}),
{b["excluded_undated_withdrawals"]} undated withdrawals were excluded
({", ".join(f"{y}: {r['missing_%']}%" for y, r in final_rows.iterrows())} of withdrawn projects).

**Upper bound on the effect.** Suppose every one of them were put back into the test cohort and
had withdrawn within 3 years. The observed 3-year withdrawal rate would rise from
{pct(b["observed"])} to at most {pct(b["observed_upper_bound"])}, against {pct(b["predicted"])}
predicted. That closes at most {100 * b["share_of_gap_explainable"]:.0f}% of the
{b["gap_points"]:.1f}-point gap. **Missing dates can explain only a small part of the
overprediction. Most of it is the model expecting more withdrawals than the 2021–22 entrants
showed.**

One measurement issue can't be checked from a single snapshot: withdrawals that happened but
weren't yet recorded by the data date (status still "active"). That would also lower the observed
rate for recent cohorts. Comparing with the next LBNL edition is the way to test it.
"""


def build_model_card(sel, metrics, boot, results, primary) -> str:
    final = results[("C", C_ORIGIN)]
    m = metrics[
        (metrics["block"] == "C") & (metrics["model"] == primary) & (metrics["tau"] == TAU)
    ].set_index("outcome")
    a = metrics[(metrics["block"] == "A-pooled") & (metrics["model"] == primary)].set_index(
        "outcome"
    )
    sp = spread(metrics, primary)

    def ci(block, outcome, metric):
        return ci_text(boot, block, primary, outcome, metric, ["project", "county-year"])

    wd, op = m.loc["withdrawn"], m.loc["operational"]
    awd, aop = a.loc["withdrawn"], a.loc["operational"]

    def auc_cell(r, block, outcome):
        if np.isnan(r["auc"]):
            return "not estimable (fewer than 20 cases)"
        return f"{r['auc']:.3f} {ci(block, outcome, 'auc')}"

    table = pd.DataFrame(
        [
            [
                "Withdrawn: cases / n",
                f"{int(wd['cases'])} / {int(wd['n']):,}",
                f"{int(awd['cases'])} / {int(awd['n']):,}",
            ],
            [
                "Withdrawn: AUC",
                auc_cell(wd, "C", "withdrawn"),
                auc_cell(awd, "A-pooled", "withdrawn"),
            ],
            [
                "Withdrawn: Brier skill vs null",
                f"{wd['skill']:.3f} {ci('C', 'withdrawn', 'skill')}",
                f"{awd['skill']:.3f} {ci('A-pooled', 'withdrawn', 'skill')}",
            ],
            [
                "Withdrawn: predicted vs observed",
                f"{pct(wd['mean_pred'])} vs {pct(wd['observed'])}",
                f"{pct(awd['mean_pred'])} vs {pct(awd['observed'])}",
            ],
            [
                "Operational: cases / n",
                f"{int(op['cases'])} / {int(op['n']):,}",
                f"{int(aop['cases'])} / {int(aop['n']):,}",
            ],
            [
                "Operational: AUC",
                auc_cell(op, "C", "operational"),
                auc_cell(aop, "A-pooled", "operational"),
            ],
            [
                "Operational: Brier skill vs null",
                f"{op['skill']:.3f} {ci('C', 'operational', 'skill')}",
                f"{aop['skill']:.3f} {ci('A-pooled', 'operational', 'skill')}",
            ],
            [
                "Operational: predicted vs observed",
                f"{pct(op['mean_pred'])} vs {pct(op['observed'])}",
                f"{pct(aop['mean_pred'])} vs {pct(aop['observed'])}",
            ],
        ],
        columns=[
            "3-year horizon",
            "Final test (block C: 2021-22 entrants)",
            "Development (block A pooled: 2014-17 entrants)",
        ],
    )
    return f"""# Model card: QueueIQ interconnection outcome model

Generated by `make evaluate` from the pre-registered evaluation (`{sel["protocol_tag"]}`).
Every number here comes from `reports/evaluation_metrics.csv` and `reports/evaluation_bootstrap.csv`.

## Model
- **Configuration:** `{primary}`. Selected on development block A by the pre-registered score
  ({sel["selection_reason"]}).
- **What it predicts:** for a proposed project at the moment it enters an interconnection queue,
  the cumulative probability of reaching commercial operation and of withdrawing, month by month
  for up to 8 years (competing risks).
- **Inputs:** entry-date information only. Capacity, fuel, hybrid/storage flags, ISO and state,
  plus congestion in the same county and state over the previous 5 years (D-023 to D-026).
- **Training data:** LBNL Queued Up 2026 edition, PJM, MISO, CAISO and SPP, entry years 2010+.

## Intended use
- Comparing the relative outlook of candidate projects or sites at screening stage, alongside
  expert judgement.
- Teaching and portfolio demonstration of leakage-safe survival modelling.

## Not intended for
- Investment, financing or siting decisions on its own.
- Regions outside PJM, MISO, CAISO and SPP, or projects entering after 2022 without re-evaluation.
- Causal statements (for example, "adding storage causes withdrawal").

## Performance
{to_markdown(table)}

Intervals are 95% bootstrap intervals: [projects] [county × entry-year groups].
Across the seven development and extension origins, the primary configuration's 3-year AUC and skill
ranged as follows:

{to_markdown(sp, index=True)}

## Evaluation design
A pre-registered protocol with time-based splits only:
- training outcomes are cut off at each origin date;
- test cohorts have full follow-up to the horizon;
- selection was made on 2014–2017 cohorts only, and frozen in git before the extension (2018–2020)
  and the final test (2021–22) were run once.

The final test has {len(final.ids):,} projects.

## Limitations and failure cases
- **Regime shift:** outcome patterns changed sharply across entry years (Phase 2, Phase 4 D-031).
  A model trained on earlier cohorts ranks withdrawal risk only slightly better than chance, and its
  probability levels can be wrong for new cohorts. See the rolling-origin results.
- **Rare outcome:** very few recent projects reach operation within 3 years, so the ability to
  rank operational outcomes can't be checked on recent cohorts at all.
- **Scope:** 4 of the 9 regions. Others lack usable event dates (D-011). Pre-2010 cohorts are
  excluded (D-012).
- **Data gaps:** about 14% of in-scope CAISO projects have no county, so their county features are
  missing. Withdrawals without a date are never counted as earlier withdrawals (D-024).
- **Features:** nothing about financing, power purchase agreements, upgrade cost allocations or
  study results, which drive most real withdrawals.
- **Not causal:** associations only.
"""


if __name__ == "__main__":
    import sys

    render_only() if "--render-only" in sys.argv else main()
