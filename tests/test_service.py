import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from queueiq.evaluation.recalibrate import apply_scaling, past_error_band
from queueiq.features.build import build_features
from queueiq.features.congestion import county_key
from queueiq.models.data import MODEL_FEATURES
from queueiq.service.features import project_features
from queueiq.service.inputs import MAX_COMPARE, InputError, ProjectInput
from queueiq.service.predict import ENTRIES_FILE, months_in_queue
from tests.service_data import MULTIPLIERS, synthetic_entries, synthetic_service

CAT = ["fuel_group", "iso", "state"]
NUM = [c for c in MODEL_FEATURES if c not in CAT]


@pytest.fixture(scope="module")
def svc():
    return synthetic_service()


def as_input(row) -> ProjectInput:
    fuel = row.fuel_type if row.fuel_type in ("solar", "wind", "battery", "gas") else "other"
    key = county_key(row.to_frame().T).iloc[0]
    return ProjectInput(
        iso=row.iso,
        state=row.state,
        county=key if pd.notna(key) else None,
        fuel=fuel,
        capacity_mw=row.capacity_mw,
        storage=bool(row.has_storage) and fuel != "battery",
    )


def same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    nums = np.allclose(a[NUM].astype(float), b[NUM].astype(float), equal_nan=True)
    return nums and (a[CAT].astype(str).to_numpy() == b[CAT].astype(str).to_numpy()).all()


def test_features_match_training_pipeline(svc):
    """A real row re-entered through the service gets exactly its training features."""
    entries = synthetic_entries()
    feats = build_features(entries)
    rows = entries[entries["in_model_scope"]].sample(60, random_state=1)
    for _, row in rows.iterrows():
        got = project_features(as_input(row), svc.hist, svc.options, row["queue_date"])
        expected = feats[feats["project_id"] == row["project_id"]][MODEL_FEATURES]
        assert same(got, expected.reset_index(drop=True)), row["project_id"]


def test_county_neighbours_listed_in_another_state_count(svc):
    """Row 0 lists NC but has a VA county FIPS: it counts for that county, as in training."""
    entries = synthetic_entries()
    neighbour = entries.loc[0]
    p = ProjectInput(iso="PJM", state="VA", county="51001", fuel="solar", capacity_mw=50)
    date = neighbour["queue_date"] + pd.Timedelta(days=1)
    with_it = project_features(p, svc.hist, svc.options, date)
    without = project_features(p, svc.hist.drop(index=0), svc.options, date)
    assert with_it["county_prior_requests_5y"][0] == without["county_prior_requests_5y"][0] + 1


def test_features_ignore_anything_after_entry(svc):
    """Moving later withdrawals/entries around cannot change a project's features."""
    p = ProjectInput(iso="PJM", state="VA", county="51003", fuel="wind", capacity_mw=80)
    date = pd.Timestamp("2019-07-01")
    base = project_features(p, svc.hist, svc.options, date)
    mutated = svc.hist.copy()
    later = mutated["withdrawal_date"] >= date
    assert later.any()
    mutated.loc[later, "withdrawal_date"] = pd.NaT
    after = mutated["queue_date"] >= date
    assert after.any()
    mutated.loc[after, "capacity_mw"] = 99999.0
    assert same(project_features(p, mutated, svc.options, date), base)


def test_input_validation_rules():
    with pytest.raises(ValidationError):
        ProjectInput(iso="PJM", state="VA", fuel="solar", capacity_mw=0)
    with pytest.raises(ValidationError):
        ProjectInput(iso="ERCOT", state="TX", fuel="solar", capacity_mw=10)
    with pytest.raises(ValidationError):
        ProjectInput(iso="PJM", state="VA", fuel="coal", capacity_mw=10)
    with pytest.raises(ValidationError):
        ProjectInput(iso="PJM", state="VA", fuel="solar", capacity_mw=10, status="active")


def test_storage_flags_follow_lbnl_combinations():
    def flags(**kw):
        return ProjectInput(iso="PJM", state="VA", capacity_mw=10, **kw).flags()

    assert flags(fuel="battery", storage=False) == (True, False)
    assert flags(fuel="solar", storage=True) == (True, True)
    assert flags(fuel="wind", storage=False) == (False, False)


def test_options_reject_what_the_model_cannot_answer(svc):
    with pytest.raises(InputError) as e:
        svc.features(ProjectInput(iso="PJM", state="IL", fuel="solar", capacity_mw=10))
    assert "IL is not offered for PJM" in e.value.problems[0]
    with pytest.raises(InputError):
        svc.features(
            ProjectInput(iso="PJM", state="VA", county="17001", fuel="solar", capacity_mw=10)
        )
    with pytest.raises(InputError) as e:
        svc.features(ProjectInput(iso="PJM", state="VA", fuel="solar", capacity_mw=1e6))
    assert "at most" in e.value.problems[0]
    assert "NC" not in svc.options.states["PJM"]  # out-of-scope row is history, not an option


def test_prediction_is_coherent(svc):
    out = svc.predict(ProjectInput(iso="MISO", state="IL", fuel="gas", capacity_mw=300))
    op, wd = (np.array(out["curves"][k]["model"]) for k in ("operational", "withdrawn"))
    assert len(op) == len(out["months"]) == 96
    assert (np.diff(op) >= -1e-12).all() and (np.diff(wd) >= -1e-12).all()
    assert (op + wd <= 1 + 1e-9).all()
    for k in ("operational", "withdrawn"):
        c = out["curves"][k]
        assert (np.array(c["low"]) <= np.array(c["high"]) + 1e-12).all()
    for h in out["headline"]:
        assert h["low"] <= h["high"]
        if h["outcome"]:
            assert h["model"] == pytest.approx(
                out["curves"][h["outcome"]]["model"][h["months"] - 1]
            )
            assert h["low"] == pytest.approx(out["curves"][h["outcome"]]["low"][h["months"] - 1])
        else:
            assert 0 <= h["model"] <= 60


def test_past_error_band_is_the_envelope_of_redrawn_curves():
    rng = np.random.default_rng(0)
    inc = rng.uniform(0, 0.004, (3, 96, 2)).cumsum(axis=1)
    cif = {"operational": inc[..., 0], "withdrawn": inc[..., 1]}
    one = past_error_band(cif, [{"operational": 1.0, "withdrawn": 1.0}])
    assert np.allclose(one["withdrawn_low"], cif["withdrawn"], atol=1e-9)
    band = past_error_band(cif, MULTIPLIERS)
    for c in MULTIPLIERS:
        r = apply_scaling(cif, c)
        for k in cif:
            assert (band[f"{k}_low"] <= r[k] + 1e-12).all()
            assert (r[k] <= band[f"{k}_high"] + 1e-12).all()


def test_months_in_queue():
    zero = {"operational": np.zeros((1, 96)), "withdrawn": np.zeros((1, 96))}
    assert months_in_queue(zero)[0] == pytest.approx(60)
    gone = {"operational": np.ones((1, 96)), "withdrawn": np.zeros((1, 96))}
    assert months_in_queue(gone)[0] == pytest.approx(0.5)  # leaves during month 1


def test_compare_limits_and_names_problems(svc):
    p = ProjectInput(iso="PJM", state="VA", fuel="solar", capacity_mw=10)
    with pytest.raises(InputError):
        svc.compare([p] * (MAX_COMPARE + 1))
    bad = ProjectInput(name="Too big", iso="PJM", state="VA", fuel="solar", capacity_mw=1e6)
    with pytest.raises(InputError) as e:
        svc.compare([p, bad])
    assert e.value.problems[0].startswith("Too big:")
    assert len(svc.compare([p, p.model_copy(update={"fuel": "gas"})])) == 2


def test_explanation_adds_up(svc):
    ex = svc.explain(ProjectInput(iso="PJM", state="VA", fuel="battery", capacity_mw=100))
    for t in ex["targets"].values():
        total = sum(c["points"] for c in t["contributions"]) / 100
        assert total == pytest.approx(t["prediction"] - t["baseline"], abs=1e-9)
    assert set(ex["text"]) == set(ex["targets"])


@pytest.mark.skipif(not ENTRIES_FILE.exists(), reason="needs the real processed data")
def test_real_data_features_match_pipeline():
    from queueiq.service.features import history
    from queueiq.service.inputs import Options

    entries = pd.read_parquet(ENTRIES_FILE)
    feats = pd.read_parquet(ENTRIES_FILE.with_name("features.parquet"))
    hist, opts = history(entries), Options.from_entries(entries)
    rows = entries[entries["in_model_scope"] & (entries["capacity_mw"] > 0)].sample(
        150, random_state=3
    )
    checked = 0
    for _, row in rows.iterrows():
        p = as_input(row)
        if row["state"] not in opts.states[row["iso"]] or p.flags() != (
            bool(row["has_storage"]),
            bool(row["is_hybrid"]),
        ):
            continue
        got = project_features(p, hist, opts, row["queue_date"])
        expected = feats[feats["project_id"] == row["project_id"]][MODEL_FEATURES]
        assert same(got, expected.reset_index(drop=True)), row["project_id"]
        checked += 1
    assert checked > 100


def test_origin_multipliers_reproduce_observed_rates():
    """The past-error multipliers map a past cohort's mean prediction onto what happened."""
    from queueiq.evaluation.backtest import OriginResult
    from queueiq.evaluation.error_range import origin_multipliers

    rng = np.random.default_rng(1)
    n = 400
    inc = rng.uniform(0, 0.01, (n, 96, 2)).cumsum(axis=1)
    cif = {"operational": inc[..., 0] * 0.3, "withdrawn": inc[..., 1]}
    time = rng.integers(1, 97, n)
    event = rng.choice([0, 1, 2], n, p=[0.5, 0.1, 0.4])
    res = OriginResult(
        block="A",
        origin_year=2015,
        test_years=(2015, 2015),
        horizons=(36,),
        ids=np.arange(n),
        time=time,
        event=event,
        iso=np.array(["PJM"] * n),
        entry_year=np.full(n, 2015),
        county_year=np.arange(n),
        preds={"m": cif},
    )
    row = origin_multipliers(res, "m")
    redrawn = apply_scaling(cif, row["multiplier"])
    for k, code in (("operational", 1), ("withdrawn", 2)):
        observed = ((event == code) & (time <= 36)).mean()
        assert row["observed"][k] == pytest.approx(observed)
        assert redrawn[k][:, 35].mean() == pytest.approx(observed, abs=1e-6)
