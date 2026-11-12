"""QueueIQ dashboard (Streamlit). Run with `make app` or `docker compose up`.

UI only: every number comes from queueiq.service (which calls models/ and explain/).
Layout: project inputs on the left; on the right, tabs for the outlook, a what-if comparison and
"why this prediction".
"""

import json
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from queueiq.app import charts
from queueiq.app.present import (
    FUEL_LABELS,
    STAT_LABELS,
    assumed_county_text,
    band_check_text,
    compare_table,
    curve_table,
    describe_project,
    long_date,
    pct,
    summary_html,
    unique_name,
)
from queueiq.service.inputs import MAX_COMPARE, UNKNOWN_COUNTY, InputError, ProjectInput
from queueiq.service.predict import Service
from queueiq.style import OUTCOME_COLORS, OUTCOME_LABELS

MODEL_CARD = "https://github.com/sheershh/QueueIQ/blob/main/docs/model_card.md"
CSS = Path(__file__).with_name("style.css")

st.set_page_config(page_title="QueueIQ", layout="wide", initial_sidebar_state="collapsed")
st.markdown(f"<style>{CSS.read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the model ...")
def load_service() -> Service:
    return Service.load()


def setup_key(project: ProjectInput) -> str:
    """Cache key: the project's setup without its name (renaming must not recompute)."""
    return project.model_copy(update={"name": "Project"}).model_dump_json()


@st.cache_data(show_spinner=False, max_entries=256)
def _predict(setup_json: str) -> dict:
    return load_service().predict(ProjectInput.model_validate_json(setup_json))


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


def project_form(svc: Service) -> tuple[ProjectInput | None, list[str]]:
    """Inputs in the left column. Returns the project, or None plus the problems to show."""
    opts = svc.options
    name = st.text_input(
        "Name", value="Project A", max_chars=40, help="Used to tell projects apart in Compare."
    )
    iso = st.selectbox("Grid operator", list(opts.states), index=0)
    states = opts.states[iso]
    state = st.selectbox(
        "State",
        states,
        index=states.index(opts.largest_state[iso]),
        help="States with at least 20 past projects in this grid operator.",
    )
    busiest = opts.busiest_county(state)
    county = st.selectbox(
        "County",
        [*[c.key for c in opts.counties.get(state, [])], UNKNOWN_COUNTY],
        index=None,
        placeholder=f"Not chosen: {busiest.name} is assumed" if busiest else "Not chosen",
        format_func=lambda k: "Not recorded" if k == UNKNOWN_COUNTY else opts.county(state, k).name,
        help="Used to count earlier requests nearby. If you don't choose one, the county with the "
        "most past requests in the state is assumed, and the results say so. 'Not recorded' is "
        "treated like past projects whose county LBNL did not record, which itself shifts the "
        "prediction.",
    )
    fuel = st.selectbox("Technology", list(FUEL_LABELS), format_func=FUEL_LABELS.get)
    storage = st.checkbox(
        "Paired with battery storage",
        value=False,
        disabled=fuel == "battery",
        help="A battery project is storage by definition.",
    )
    capacity = st.number_input(
        "Capacity (MW)", min_value=0.0, max_value=float(opts.capacity_max), value=150.0, step=10.0
    )
    st.markdown(
        '<p class="qi-muted">Assumes the project enters the queue on '
        f"{long_date(svc.meta['entry_date'])}, the day after the data ends.</p>",
        unsafe_allow_html=True,
    )
    if capacity <= 0:
        return None, ["Capacity must be greater than 0 MW."]
    try:
        p = ProjectInput(
            name=name.strip() or "Project",
            iso=iso,
            state=state,
            county=county,
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


def limitations(ctx: dict) -> None:
    years = ctx["past_error_range"]["origins"]
    first, last = years[0]["origin"], years[-1]["origin"]
    with st.expander("How to read this and limitations"):
        st.markdown(
            f"""
- **What the lines are.** The model's chance that a project like this one has withdrawn, or reached
  operation, by each month after entering the queue. It was trained on {ctx["trained_on_rows"]:,}
  projects in PJM, MISO, CAISO and SPP that entered from 2010 on, with outcomes as of
  {ctx["outcomes_as_of"]}.
- **What the shaded band is.** The past-error range. For each of the {len(years)} past test years
  ({first}–{last} entrants), we measured how far the model's average 3-year prediction was from
  what happened, then redrew this project's curves with that error. The band spans those
  redrawn curves. It is not a confidence interval, and it does not cover errors larger than the
  worst past year.
- **Why the band sits below the withdrawal line.** In every past test year, fewer projects
  withdrew than the model predicted. Treat the line as an upper end.
- **Horizon.** Predictions were checked against real outcomes up to {ctx["evaluated_months"]}
  months only. Beyond that (the grey area), the curves are unchecked.
- **The deployed model's own accuracy is unknown.** It is refit on every project up to the end of
  2025, so there is no later cohort to check it against yet. That takes 3 to 5 more years of
  outcomes. The evidence behind it is the backtest of the same fitting procedure on earlier years.
- **Use.** Compare setups with each other. The probabilities are not literal odds, and the
  explanations describe what the model relies on, not what causes withdrawal or completion.
- Full details: [model card]({MODEL_CARD}).
"""
        )


def outlook_tab(svc: Service, project: ProjectInput, ctx: dict) -> None:
    pred = predict(project)
    st.altair_chart(
        charts.outlook(pred, ctx["observed_reference"], ctx["evaluated_months"]),
        width="stretch",
        theme=None,
    )
    st.markdown(legend(), unsafe_allow_html=True)
    if pred["county"]["assumed"]:
        st.markdown(
            f'<p class="qi-muted">{assumed_county_text(pred, project)}</p>', unsafe_allow_html=True
        )
    st.markdown(summary_html(pred), unsafe_allow_html=True)
    caution = f"{ctx['calibration_warning']} {band_check_text(ctx['past_error_range'])}"
    st.markdown(f'<p class="qi-caution">{caution}</p>', unsafe_allow_html=True)
    with st.expander("Show the curves as a table"):
        st.dataframe(curve_table(pred), hide_index=True, width="stretch")
    limitations(ctx)


def compare_tab(project: ProjectInput | None) -> None:
    saved: list[dict] = st.session_state.setdefault("compare", [])
    names = [p["name"] for p in saved]
    full = len(saved) >= MAX_COMPARE
    if st.button(
        "Add current project to comparison",
        disabled=project is None or full,
        help=f"Up to {MAX_COMPARE} projects." if full else None,
    ):
        entry = project.model_dump()
        entry["name"] = unique_name(entry["name"], names)
        saved.append(entry)
        st.rerun()
    if not saved:
        st.markdown(
            '<p class="qi-muted qi-prose">No projects to compare yet. Set up a project on the '
            "left, add it here, then change one thing (technology, size, state) and add it "
            "again.</p>",
            unsafe_allow_html=True,
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
    st.markdown(
        '<p class="qi-muted">Dot: model. Line: past-error range.</p>', unsafe_allow_html=True
    )
    st.dataframe(compare_table(preds), hide_index=True, width="stretch")
    for i, (p, pred) in enumerate(zip(saved, preds, strict=True)):
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.markdown(f"**{p['name']}**: {describe_project(p, pred['county'])}")
        if c2.button("Remove", key=f"remove-{i}"):
            saved.pop(i)
            st.rerun()


def why_tab(project: ProjectInput, ctx: dict) -> None:
    with st.spinner("Working out the explanation ..."):
        ex = explain(project)
    if ex["county"]["assumed"]:
        st.markdown(
            f'<p class="qi-muted">{assumed_county_text(ex, project)}</p>', unsafe_allow_html=True
        )
    st.markdown(
        '<p class="qi-prose">Each bar shows how many percentage points a group of features moves '
        "this project's prediction, compared with the model's average prediction for "
        f"{ctx['background_years'][0]}–{ctx['background_years'][1]} entrants. This is what the "
        "model relies on, not proof of cause, and a group can move predictions without improving "
        "them.</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2, gap="large")
    for col, (key, t) in zip(cols, ex["targets"].items(), strict=True):
        ref = ctx["observed_reference"][key]
        with col:
            st.subheader(f"{STAT_LABELS[key]}: {pct(t['prediction'])}")
            st.markdown(
                f'<p class="qi-muted">Model average for recent entrants {pct(t["baseline"])}. '
                f"Observed for {ref['cohort']} entrants {pct(ref['observed'])}.</p>",
                unsafe_allow_html=True,
            )
            st.altair_chart(charts.contributions(t), width="stretch", theme=None)
            st.markdown(f'<p class="qi-prose">{ex["drivers"][key]}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="qi-caution">{ctx["calibration_warning"]}</p>', unsafe_allow_html=True)


def main() -> None:
    try:
        svc = load_service()
    except FileNotFoundError as e:
        st.error(f"The model isn't available. {e}")
        st.stop()
    ctx = svc.context()
    left, right = st.columns([1, 2.6], gap="large")
    with left:
        st.markdown(
            '<p class="qi-mark">QueueIQ</p><p class="qi-sub">Interconnection queue outlook</p>',
            unsafe_allow_html=True,
        )
        project, problems = project_form(svc)
        a = ctx["data_attribution"]
        st.markdown(
            f'<p class="qi-muted">Data: <a href="{a["source"]}">Queued Up</a>, Lawrence Berkeley '
            f'National Laboratory and GridTracker, <a href="{a["license"]}">CC BY 4.0</a>. '
            "Modified: filtered, cleaned and modelled. Not endorsed by LBNL or GridTracker.</p>",
            unsafe_allow_html=True,
        )
    with right:
        # on_change="rerun" tracks the open tab, so the explanation (about 1 s) is computed only
        # when its tab is open.
        outlook, comparison, why = st.tabs(
            ["Outlook", "Compare", "Why this prediction"], key="view", on_change="rerun"
        )
        with outlook:
            if problems:
                for msg in problems:
                    st.error(msg)
            else:
                outlook_tab(svc, project, ctx)
        with comparison:
            compare_tab(project)
        with why:
            if problems:
                st.markdown(
                    '<p class="qi-muted">Fix the inputs on the left to see an explanation.</p>',
                    unsafe_allow_html=True,
                )
            elif why.open:
                why_tab(project, ctx)


main()
