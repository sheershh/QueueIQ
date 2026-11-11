import numpy as np
import pandas as pd
import pytest

from queueiq.evaluation.backtest import (
    ablation_names,
    candidate_names,
    complexity_key,
    point_metrics,
    run_origin,
)
from queueiq.evaluation.dev import select, selection_scores
from tests.model_data import synthetic


@pytest.fixture(scope="module")
def result():
    entries, features = synthetic(n=900, seed=5)
    return run_origin(entries, features, "A", 2016, (2016, 2016), (36,))


def test_candidates_are_the_protocol_fifteen():
    names = candidate_names()
    assert len(names) == 15 and len(set(names)) == 15
    assert sum("queue_year" in n or "qy=" in n for n in names) == 12


def test_run_origin_produces_every_candidate_with_valid_curves(result):
    expected = set(candidate_names()) | set(ablation_names()) | {"null_aj"}
    assert expected <= set(result.preds)
    for name in expected:
        cif = result.preds[name]
        assert cif["operational"].shape == (len(result.ids), 96)
        assert (cif["operational"] + cif["withdrawn"] <= 1 + 1e-6).all()
    assert (result.entry_year == 2016).all()
    assert all(c["operational"] > 0 for c in result.recal_constants.values())


def test_point_metrics_counts_and_null_skill(result):
    rows = pd.DataFrame(point_metrics(result, ["null_aj", "cox_cause_specific|qy=1|recal=0"], 36))
    null = rows[rows["model"] == "null_aj"]
    assert np.allclose(null["skill"], 0.0)
    assert (rows["cases"] <= rows["n"]).all()
    small = rows[rows["cases"] < 20]
    assert small["auc"].isna().all() and small["cindex"].isna().all()


def test_complexity_order():
    order = sorted(
        [
            "rsf_cause_specific|qy=0|recal=0",
            "cox_cause_specific|qy=1|recal=1",
            "cox_cause_specific|qy=0|recal=1",
            "recent_cohort",
            "baseline_aj_iso_fuel|recal=1",
        ],
        key=complexity_key,
    )
    assert order == [
        "recent_cohort",
        "baseline_aj_iso_fuel|recal=1",
        "cox_cause_specific|qy=0|recal=1",
        "cox_cause_specific|qy=1|recal=1",
        "rsf_cause_specific|qy=0|recal=0",
    ]


def fake_metrics(skill_of: dict[str, float]) -> pd.DataFrame:
    rows = []
    for name in candidate_names():
        for origin in (2014, 2015, 2016, 2017):
            for outcome in ("operational", "withdrawn"):
                rows.append(
                    {
                        "model": name,
                        "origin": origin,
                        "outcome": outcome,
                        "tau": 36,
                        "skill": skill_of.get(name, 0.0),
                    }
                )
    return pd.DataFrame(rows)


def test_selection_picks_the_best_when_clearly_ahead():
    m = fake_metrics(
        {"rsf_cause_specific|qy=1|recal=1": 0.20, "cox_cause_specific|qy=1|recal=0": 0.10}
    )
    assert select(selection_scores(m))[0] == "rsf_cause_specific|qy=1|recal=1"


def test_tie_rule_prefers_the_simpler_candidate_within_margin():
    m = fake_metrics(
        {"rsf_cause_specific|qy=1|recal=1": 0.200, "cox_cause_specific|qy=0|recal=0": 0.196}
    )
    chosen, reason = select(selection_scores(m))
    assert chosen == "cox_cause_specific|qy=0|recal=0" and "tie rule" in reason


def test_selection_refuses_missing_skills():
    m = fake_metrics({})
    m = m[~((m["model"] == "recent_cohort") & (m["origin"] == 2017))]
    with pytest.raises(ValueError, match="every candidate"):
        selection_scores(m)
