import re
import tomllib
from pathlib import Path

import pytest

from queueiq.app import charts
from queueiq.app.present import (
    band_check_text,
    compare_table,
    curve_table,
    long_date,
    pct,
    summary_html,
    unique_name,
)
from queueiq.service.inputs import ProjectInput
from queueiq.service.predict import Service
from queueiq.style import UI, UI_FONT_SIZES
from tests.service_data import synthetic_service

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "queueiq" / "app" / "main.py"
P = ProjectInput(name="VA solar", iso="PJM", state="VA", fuel="solar", capacity_mw=100)


@pytest.fixture(scope="module")
def svc():
    return synthetic_service()


def test_theme_matches_palette_module():
    theme = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())["theme"]
    assert theme["backgroundColor"] == UI["paper"]
    assert theme["textColor"] == UI["ink"]
    assert theme["primaryColor"] == UI["accent"]
    assert theme["borderColor"] == UI["rule"]
    assert theme["baseFontSize"] == UI_FONT_SIZES["body"]


def test_css_uses_only_the_three_font_sizes():
    css = (APP.parent / "style.css").read_text()
    sizes = {int(s) for s in re.findall(r"font-size:\s*(\d+)px", css)}
    assert sizes <= set(UI_FONT_SIZES.values())


def test_formatting_helpers():
    assert pct(0.5156) == "51.6%"
    assert long_date("2026-01-01") == "1 January 2026"
    assert unique_name("A", ["A", "A (2)"]) == "A (3)"
    assert unique_name("B", ["A"]) == "B"


def test_summary_and_tables(svc):
    pred = svc.predict(P)
    html = summary_html(pred)
    assert html.count('class="qi-stat ') == 4 and "Past-error range" in html
    assert len(curve_table(pred)) == 16  # every 6 months up to 96
    table = compare_table(svc.compare([P, P.model_copy(update={"name": "VA gas", "fuel": "gas"})]))
    assert list(table["Project"]) == ["VA solar", "VA gas"]


def test_charts_are_valid_vega_lite(svc):
    pred = svc.predict(P)
    ctx = svc.context()
    charts.outlook(pred, ctx["observed_reference"], ctx["evaluated_months"]).to_dict()
    charts.compare([pred, pred], "withdrawn_3y", "withdrawn", "Withdraws").to_dict()
    ex = svc.explain(P)
    for t in ex["targets"].values():
        charts.contributions(t).to_dict()


def test_nice_step_covers_extent():
    for extent in (0.1, 0.9, 3.0, 8.3, 19.0, 60.0):
        assert 2 * charts.nice_step(extent) >= extent


@pytest.fixture
def app(svc, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    st.cache_data.clear()
    monkeypatch.setattr(Service, "load", classmethod(lambda cls, *a, **k: svc))
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    return at


def test_app_renders_outlook_without_errors(app):
    assert not app.exception
    assert not app.error
    assert [t.label for t in app.tabs] == ["Outlook", "Compare", "Why this prediction"]
    assert any("Past-error range" in m.value for m in app.markdown)


def test_app_validates_capacity(app):
    app.number_input[0].set_value(0.0).run()
    assert [e.value for e in app.error] == ["Capacity must be greater than 0 MW."]


def test_app_shows_missing_model_message(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()

    def missing(cls, *a, **k):
        raise FileNotFoundError("Missing models/. Run `make data && make train` first.")

    monkeypatch.setattr(Service, "load", classmethod(missing))
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert "make train" in at.error[0].value


def test_band_check_wording_flags_the_miss():
    from tests.service_data import FINAL_TEST_CHECK

    text = band_check_text({"final_test_check": FINAL_TEST_CHECK})
    assert "withdrawing within 3 years, the observed 34.0% fell inside the band" in text
    assert "off by more than the band" in text and "0.03% happened (1 project)" in text


def test_app_says_which_county_was_assumed(app):
    assert any("County assumed:" in m.value for m in app.markdown)
    assert any("off by more than the band" in m.value for m in app.markdown)
    assert any("own accuracy is unknown" in m.value for m in app.markdown)


def test_why_tab_names_the_assumed_county(svc):
    from queueiq.app.present import assumed_county_text

    ex = svc.explain(ProjectInput(iso="PJM", state="VA", fuel="solar", capacity_mw=50))
    assert ex["county"]["assumed"]
    assert ex["county"]["name"] in assumed_county_text(ex, {"state": "VA"})
