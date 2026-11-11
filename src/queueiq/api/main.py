"""Small HTTP API over the prediction service (decisions D-048).

    GET  /health   model loaded? which configuration, trained on how many projects
    GET  /options  valid ISOs, states per ISO, counties per state, capacity range
    GET  /context  reference rates, Phase 5 warning, past-error range, horizon limits
    POST /predict  one project -> curves + headline numbers (+ explanation if ?explain=true)
    POST /compare  up to 5 projects -> curves + headline numbers for each

Run: `make api` (or `docker compose up`). Interactive docs at /docs.
Inputs that are well-formed but outside what the model can answer (a state with too little
history, capacity beyond the training range) return 422 with a list of problems.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from queueiq.service.inputs import MAX_COMPARE, InputError, ProjectInput
from queueiq.service.predict import Service

_state: dict = {"service": None, "error": None}


def set_service(service: Service | None, error: str | None = None) -> None:
    """Install the service (tests use this to inject a small synthetic one)."""
    _state["service"], _state["error"] = service, error


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _state["service"] is None:
        try:
            set_service(Service.load())
        except FileNotFoundError as e:  # start anyway, so /health can explain what is missing
            set_service(None, str(e))
    yield


app = FastAPI(
    title="QueueIQ",
    description="Competing-risks predictions for US interconnection-queue projects.",
    version="0.1.0",
    lifespan=lifespan,
)


class CompareRequest(BaseModel):
    projects: list[ProjectInput] = Field(min_length=1, max_length=MAX_COMPARE)


@app.exception_handler(InputError)
async def input_error(_: Request, exc: InputError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": exc.problems})


def service() -> Service:
    if _state["service"] is None:
        raise HTTPException(status_code=503, detail=_state["error"] or "Model not loaded.")
    return _state["service"]


@app.get("/health")
def health():
    if _state["service"] is None:
        return JSONResponse(
            status_code=503, content={"status": "unavailable", "detail": _state["error"]}
        )
    meta = _state["service"].meta
    return {
        "status": "ok",
        "model": meta["primary_configuration"],
        "trained_on_rows": meta["trained_on_rows"],
        "outcomes_as_of": meta["outcomes_as_of"],
        "entry_date": meta["entry_date"],
    }


@app.get("/options")
def options():
    return service().options.as_dict()


@app.get("/context")
def context():
    return service().context()


@app.post("/predict")
def predict(project: ProjectInput, explain: bool = Query(False)):
    svc = service()
    out = svc.predict(project)
    if explain:
        out["explanation"] = svc.explain(project)
    return out


@app.post("/compare")
def compare(req: CompareRequest):
    return {"projects": service().compare(req.projects)}
