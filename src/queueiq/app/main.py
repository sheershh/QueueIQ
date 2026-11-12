"""QueueIQ dashboard (Streamlit). Run with `make app`; hosted on Streamlit Community Cloud.

UI only: every number comes from queueiq.service (which calls models/ and explain/).
Layout (decisions D-057 to D-060), top to bottom:
  header -> where (3D county map, click to choose) -> what (technology, size, storage)
  -> answer (what happened to comparable past projects; how the model ranks this one)
  -> details tabs (model curves, why, shortlist, how to read this) -> footer.
"""

import json
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from queueiq.app import charts, geomap
from queueiq.app.present import (
    FUEL_LABELS,
    STAT_LABELS,
    TECH_LABELS,
    assumed_county_text,
    band_check_text,
    compare_table,
    curve_table,
    describe_project,
    history_sentence,
    long_date,
    outcome_grid_html,
    pct,
    percentile_html,
    plural,
    share_text,
    short_warning,
    summary_html,
    unique_name,
)
from queueiq.service.inputs import MAX_COMPARE, UNKNOWN_COUNTY, InputError, ProjectInput
from queueiq.service.predict import Service
from queueiq.style import CROWDING_RAMP, OUTCOME_COLORS, OUTCOME_LABELS

MODEL_CARD = "https://github.com/sheershh/QueueIQ/blob/main/docs/model_card.md"
REPO = "https://github.com/sheershh/QueueIQ"
CSS = Path(__file__).with_name("style.css")
CAPACITY_PRESETS = {"50 MW": 50.0, "150 MW": 150.0, "500 MW": 500.0}

st.set_page_config(page_title="QueueIQ", layout="wide", initial_sidebar_state="collapsed")
st.markdown(f"<style>{CSS.read_text()}</style>", unsafe_allow_html=True)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# -- data access (cached) ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model ...")
def load_service() -> Service:
    return Service.load()


def setup_key(project: ProjectInput) -> str:
    """Cache key: the project's setup without its name (renaming must not recompute)."""
    return project.model_copy(update={"name": "Project"}).model_dump_json()


@st.cache_data(show_spinner=False, max_entries=256)
def _predict(setup_json: str) -> dict:
    return load_service().predict(ProjectInput.model_validate_json(setup_json))


@st.cache_data(show_spinner=False, max_entries=256)
def _percentiles(setup_json: str) -> dict:
    return load_service().percentiles(ProjectInput.model_validate_json(setup_json))


@st.cache_data(show_spinner=False, max_entries=64)
def _explain(setup_json: str) -> dict:
    return load_service().explain(ProjectInput.model_validate_json(setup_json))


def predict(project: ProjectInput) -> dict:
    return {**_predict(setup_key(project)), "project": project.model_dump()}


def explain(project: ProjectInput) -> dict:
    return {**_explain(setup_key(project)), "project": project.model_dump()}


@st.cache_data(show_spinner=False, max_entries=64)
def compare(projects_json: str) -> list[dict]:
    projects = [ProjectInput.model_validate(p) for p in json.loads(projects_json)]
    return load_service().compare(projects)


# -- location state ------------------------------------------------------------------------------
def init_location(svc: Service) -> None:
    opts = svc.options
    if "loc_state" not in st.session_state:
        st.session_state.loc_iso = "PJM"
        st.session_state.loc_state = opts.largest_state["PJM"]
        st.session_state.loc_county = None


def on_map_select() -> None:
    """A county clicked on the map becomes the selected location (state and ISO follow)."""
    key = geomap.selected_key(st.session_state.get("qi_map"))
    if not key:
        return
    opts = load_service().options
    for state, counties in opts.counties.items():
        if state in opts.state_isos and any(c.key == key for c in counties):
            st.session_state.loc_state = state
            st.session_state.loc_county = key
            st.session_state.loc_iso = opts.state_isos[state][0]
            return


def on_state_change() -> None:
    opts = load_service().options
    st.session_state.loc_county = None
    st.session_state.loc_iso = opts.state_isos[st.session_state.loc_state][0]


# -- sections ------------------------------------------------------------------------------------
def header(ctx: dict) -> None:
    html(
        '<header class="qi-header"><div class="qi-brand">QueueIQ</div>'
        '<div class="qi-tagline">What usually happens to a power project after it asks to connect '
        "to the US grid, and how a new one compares.</div>"
        '<p class="qi-muted">Covers the PJM, MISO, CAISO and SPP queues: '
        f"{ctx['trained_on_rows']:,} projects that entered from 2010 to 2025, with outcomes up to "
        f"{long_date(ctx['outcomes_as_of'])}. Other regions lack the dates needed.</p></header>"
    )


def where_section(svc: Service) -> None:
    opts = svc.options
    html('<div class="qi-section">Where is the project?</div>')
    left, right = st.columns([2.3, 1], gap="large")
    with left:
        st.pydeck_chart(
            geomap.deck(svc.crowding(), st.session_state.loc_county),
            height=460,
            on_select=on_map_select,
            selection_mode="single-object",
            key="qi_map",
        )
        ramp = "".join(f'<i style="background:{c}"></i>' for c in CROWDING_RAMP)
        html(
            '<div class="qi-maplegend">Column height: requests filed in the county in the 5 years '
            "before 2026 (square-root scale). Color: share already withdrawn, low to high "
            f'<span class="qi-ramp">{ramp}</span>Grey: no requests, or a region the model does '
            "not cover. Drag to rotate, scroll to zoom, click a county to choose it.</div>"
        )
    with right:
        state = st.session_state.loc_state
        county = (
            opts.county(state, st.session_state.loc_county) if st.session_state.loc_county else None
        )
        busiest = opts.busiest_county(state)
        crowd = svc.crowding()
        shown = county or busiest
        html(
            f'<div class="qi-pctl-label">Selected</div><div class="qi-pctl-value">'
            f"{shown.name} County, {state}</div>"
            + (
                ""
                if county
                else '<p class="qi-muted">Not chosen yet: this is the busiest county in the state. '
                "Click the map to pick another.</p>"
            )
        )
        if shown.key in crowd.index:
            r = crowd.loc[shown.key]
            html(
                f'<p class="qi-muted">{plural(int(r["requests_5y"]), "request")} filed here in the '
                f"last 5 years, {share_text(r['withdrawn_share'])} of them already withdrawn.</p>"
            )
        isos = opts.state_isos[state]
        if st.session_state.loc_iso not in isos:
            st.session_state.loc_iso = isos[0]
        if len(isos) > 1:
            st.segmented_control(
                f"{state} is served by more than one grid operator",
                isos,
                key="loc_iso",
                help="Ordered by how many past projects each has in this state.",
            )
        else:
            html(f'<p class="qi-muted">Grid operator: {isos[0]}</p>')
        with st.expander("Choose from a list instead"):
            st.selectbox(
                "State",
                sorted(opts.state_isos),
                key="loc_state",
                on_change=on_state_change,
                help="States with at least 20 past projects in a covered grid operator.",
            )
            choices = [c.key for c in opts.counties.get(st.session_state.loc_state, [])]
            st.selectbox(
                "County",
                [*choices, UNKNOWN_COUNTY],
                key="loc_county",
                index=None,
                placeholder="Not chosen: busiest county assumed",
                format_func=lambda k: (
                    "Not recorded"
                    if k == UNKNOWN_COUNTY
                    else opts.county(st.session_state.loc_state, k).name
                ),
            )


def what_section(svc: Service) -> tuple[ProjectInput | None, list[str]]:
    html('<div class="qi-section">What is it?</div>')
    c1, c2, c3 = st.columns([1.6, 1.4, 1], gap="large")
    with c1:
        fuel = (
            st.segmented_control(
                "Technology",
                list(TECH_LABELS),
                format_func=TECH_LABELS.get,
                key="tech",
                default="solar",
            )
            or "solar"
        )
        if fuel == "other":
            html(f'<p class="qi-muted">{FUEL_LABELS["other"]}</p>')
    with c2:
        if "capacity" not in st.session_state:
            st.session_state.capacity = 150.0
        preset = st.pills("Size", list(CAPACITY_PRESETS), key="cap_preset", default=None)
        if preset and st.session_state.get("_last_preset") != preset:
            st.session_state.capacity = CAPACITY_PRESETS[preset]
        st.session_state._last_preset = preset
        capacity = st.number_input(
            "Capacity (MW)",
            min_value=0.0,
            max_value=float(svc.options.capacity_max),
            step=10.0,
            key="capacity",
        )
    with c3:
        storage = st.toggle(
            "Paired with battery storage",
            disabled=fuel == "battery",
            help="A battery project is storage by definition.",
        )
    if capacity <= 0:
        return None, ["Capacity must be greater than 0 MW."]
    try:
        p = ProjectInput(
            name=st.session_state.get("project_name", "Project A").strip() or "Project",
            iso=st.session_state.loc_iso,
            state=st.session_state.loc_state,
            county=st.session_state.loc_county,
            fuel=fuel,
            capacity_mw=capacity,
            storage=storage,
        )
        svc.options.check(p)
    except ValidationError as e:
        return None, [err["msg"] for err in e.errors()]
    except InputError as e:
        return None, e.problems
    return p, []


def answer_section(svc: Service, project: ProjectInput, ctx: dict) -> None:
    html('<div class="qi-section">What to expect</div>')
    left, right = st.columns(2, gap="large")
    with left:
        html(
            '<div class="qi-pctl-label">What happened to comparable past projects '
            "(real outcomes)</div>"
        )
        span = (
            st.segmented_control(
                "Time after entering the queue",
                ["Within 3 years", "Within 5 years"],
                key="horizon",
                default="Within 3 years",
                label_visibility="collapsed",
            )
            or "Within 3 years"
        )
        c = svc.comparable(project)["36" if span == "Within 3 years" else "60"]
        html(f'<div class="qi-pctl-value">{history_sentence(c, project.iso, project.fuel)}</div>')
        html(outcome_grid_html(c))
    with right:
        html(
            '<div class="qi-pctl-label">How the model ranks this project against recent '
            "entrants</div>"
        )
        pc = _percentiles(setup_key(project))
        years = pc["reference_years"]
        html(
            percentile_html(
                "Chance of reaching operation within 3 years",
                pc["targets"]["operational_3y"],
                years,
            )
        )
        html(
            percentile_html(
                "Risk of withdrawing within 3 years", pc["targets"]["withdrawn_3y"], years
            )
        )
        html('<p class="qi-muted">See what drives this under "Why" in the details below.</p>')
    html(f'<p class="qi-caution">{short_warning(ctx["phase5_final_test_calibration"])}</p>')


def legend() -> str:
    wd, op = OUTCOME_COLORS["withdrawn"], OUTCOME_COLORS["operational"]
    return (
        '<div class="qi-legend">'
        f'<span><i class="qi-key" style="background:{wd}"></i>{OUTCOME_LABELS["withdrawn"]}</span>'
        f'<span><i class="qi-key" style="background:{op}"></i>'
        f"{OUTCOME_LABELS['operational']}</span>"
        f'<span><i class="qi-key qi-band" style="background:{wd}"></i>Past-error range</span>'
        '<span><i class="qi-key qi-ring"></i>Observed rate, latest fully followed cohort</span>'
        "</div>"
    )


def curves_tab(project: ProjectInput, ctx: dict) -> None:
    pred = predict(project)
    html(
        '<p class="qi-prose">The model\'s month-by-month chance of each outcome, with the range '
        "its past errors imply. Use these for comparing setups, not as odds.</p>"
    )
    st.altair_chart(
        charts.outlook(pred, ctx["observed_reference"], ctx["evaluated_months"]),
        width="stretch",
        theme=None,
    )
    html(legend())
    if pred["county"]["assumed"]:
        html(f'<p class="qi-muted">{assumed_county_text(pred, project)}</p>')
    html(summary_html(pred))
    caution = f"{ctx['calibration_warning']} {band_check_text(ctx['past_error_range'])}"
    html(f'<p class="qi-caution">{caution}</p>')
    with st.expander("Show the curves as a table"):
        st.dataframe(curve_table(pred), hide_index=True, width="stretch")


def why_tab(project: ProjectInput, ctx: dict) -> None:
    with st.spinner("Working out the explanation ..."):
        ex = explain(project)
    if ex["county"]["assumed"]:
        html(f'<p class="qi-muted">{assumed_county_text(ex, project)}</p>')
    html(
        '<p class="qi-prose">Each bar shows how many percentage points a group of features moves '
        "this project's prediction, compared with the model's average prediction for "
        f"{ctx['background_years'][0]}–{ctx['background_years'][1]} entrants. This is what the "
        "model relies on, not proof of cause, and a group can move predictions without improving "
        "them.</p>"
    )
    cols = st.columns(2, gap="large")
    for col, (key, t) in zip(cols, ex["targets"].items(), strict=True):
        ref = ctx["observed_reference"][key]
        with col:
            st.subheader(f"{STAT_LABELS[key]}: {pct(t['prediction'])}")
            html(
                f'<p class="qi-muted">Model average for recent entrants {pct(t["baseline"])}. '
                f"Observed for {ref['cohort']} entrants {pct(ref['observed'])}.</p>"
            )
            st.altair_chart(charts.contributions(t), width="stretch", theme=None)
            html(f'<p class="qi-prose">{ex["drivers"][key]}</p>')


def shortlist_tab(project: ProjectInput | None) -> None:
    saved: list[dict] = st.session_state.setdefault("compare", [])
    names = [p["name"] for p in saved]
    full = len(saved) >= MAX_COMPARE
    c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
    c1.text_input("Name for this setup", value="Project A", max_chars=40, key="project_name")
    if c2.button(
        "Add to shortlist",
        disabled=project is None or full,
        help=f"Up to {MAX_COMPARE} setups." if full else None,
    ):
        entry = project.model_dump()
        entry["name"] = unique_name(st.session_state.project_name.strip() or "Project", names)
        saved.append(entry)
        st.rerun()
    if not saved:
        html(
            '<p class="qi-muted qi-prose">Nothing shortlisted yet. Add this setup, change one '
            "thing (technology, size, county) and add it again to see them side by side.</p>"
        )
        return
    preds = compare(json.dumps(saved))
    left, right = st.columns(2, gap="large")
    with left:
        st.altair_chart(
            charts.compare(preds, "withdrawn_3y", "withdrawn", STAT_LABELS["withdrawn_3y"]),
            width="stretch",
            theme=None,
        )
    with right:
        st.altair_chart(
            charts.compare(preds, "operational_5y", "operational", STAT_LABELS["operational_5y"]),
            width="stretch",
            theme=None,
        )
    html('<p class="qi-muted">Dot: model. Line: past-error range.</p>')
    st.dataframe(compare_table(preds), hide_index=True, width="stretch")
    for i, (p, pred) in enumerate(zip(saved, preds, strict=True)):
        a, b = st.columns([5, 1], vertical_alignment="center")
        a.markdown(f"**{p['name']}**: {describe_project(p, pred['county'])}")
        if b.button("Remove", key=f"remove-{i}"):
            saved.pop(i)
            st.rerun()


def read_tab(ctx: dict) -> None:
    years = ctx["past_error_range"]["origins"]
    first, last = years[0]["origin"], years[-1]["origin"]
    st.markdown(
        f"""
- **Why the main answer is real history.** The model's percentages ran high on recent projects,
  but it does better at ranking projects against each other. So the page leads with what actually
  happened to comparable past projects, and uses the model only to place this project among
  recent entrants. How far each ranking can be trusted is shown next to it.
- **Comparable projects** share the grid operator and technology, and entered in the latest four
  years that are fully followed up for the chosen horizon. If fewer than 30 match, all of the grid
  operator's projects are used, and the sentence says so.
- **The model** was trained on {ctx["trained_on_rows"]:,} projects in PJM, MISO, CAISO and SPP that
  entered from 2010 on, with outcomes as of {ctx["outcomes_as_of"]}.
- **The shaded band** under "Model curves" is the past-error range. For each of the {len(years)}
  past test years ({first}–{last} entrants), we measured how far the model's average 3-year
  prediction was from what happened, then redrew the curves with that error. It is not a confidence
  interval. {band_check_text(ctx["past_error_range"])}
- **Horizon.** Predictions were checked against real outcomes up to {ctx["evaluated_months"]}
  months only.
- **The deployed model's own accuracy is unknown.** It is refit on every project up to the end of
  2025, so there is no later cohort to check it against yet. That takes 3 to 5 more years.
- **Not causal.** Explanations describe what the model relies on, not what causes withdrawal or
  completion.
- Full details: [model card]({MODEL_CARD}).
"""
    )


def footer(ctx: dict) -> None:
    a = ctx["data_attribution"]
    html(
        f'<footer class="qi-footer">Data: <a href="{a["source"]}">Queued Up</a>, Lawrence Berkeley '
        f'National Laboratory and GridTracker, <a href="{a["license"]}">CC BY 4.0</a>. Modified: '
        "filtered, cleaned and modelled. Not endorsed by LBNL or GridTracker. County shapes: US "
        f'Census Bureau. <a href="{MODEL_CARD}">Model card</a>. <a href="{REPO}">Source code</a>.'
        "</footer>"
    )


def main() -> None:
    try:
        svc = load_service()
    except FileNotFoundError as e:
        st.error(f"The model isn't available. {e}")
        st.stop()
    ctx = svc.context()
    init_location(svc)
    header(ctx)
    where_section(svc)
    project, problems = what_section(svc)
    if problems:
        for msg in problems:
            st.error(msg)
    else:
        answer_section(svc, project, ctx)
    html('<div class="qi-section">Details</div>')
    # on_change="rerun" tracks the open tab, so heavy content runs only when its tab is open.
    tabs = st.tabs(
        ["Model curves", "Why", "Shortlist", "How to read this"], key="details", on_change="rerun"
    )
    with tabs[0]:
        if project and tabs[0].open:
            curves_tab(project, ctx)
    with tabs[1]:
        if project and tabs[1].open:
            why_tab(project, ctx)
    with tabs[2]:
        if tabs[2].open:
            shortlist_tab(project)
    with tabs[3]:
        read_tab(ctx)
    footer(ctx)


main()
