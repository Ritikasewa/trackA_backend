"""FastAPI wrapper around the framework-free inference core (app/core/inference.py, unchanged). Endpoints: POST /assess, POST /explain (additive), GET /meta."""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .artifacts import build_meta, load_bundle, verify_manifest
from .core.explain import Explainer
from .core.inference import Assessor, ContractError
from .settings import ARTIFACT_DIR
def _strip(schema): return {k: v for k, v in schema.items() if k not in ("$schema", "$id")}
def create_app(artifact_dir=None):
    d = Path(artifact_dir or ARTIFACT_DIR); verify_manifest(d); bundle = load_bundle(d)
    @asynccontextmanager
    async def lifespan(app):
        app.state.assessor = Assessor(str(d)); app.state.explainer = Explainer(app.state.assessor); app.state.meta = build_meta(bundle["contract"], bundle["card"]); yield
    app = FastAPI(title="Track A - Cardiovascular Risk Assessment API", version=bundle["contract"]["contract_version"], lifespan=lifespan,
                  description="Decision support / educational use only. Not a diagnosis and not a substitute for formal diagnostic imaging.")
    err_schema = _strip(bundle["error_schema"])
    @app.post("/assess", summary="Assess CAD and LAD/LCX/RCA stenosis probabilities for one patient",
              openapi_extra={"requestBody": {"required": True, "content": {"application/json": {"schema": _strip(bundle["request_schema"])}}}},
              responses={200: {"description": "Assessment", "content": {"application/json": {"schema": _strip(bundle["response_schema"])}}},
                         422: {"description": "Contract validation error", "content": {"application/json": {"schema": err_schema}}}})
    async def assess(request: Request):
        try: body = await request.json()
        except ValueError: return JSONResponse(status_code=422, content={"errors": [{"field": "body", "code": "invalid_type", "message": "Request body must be valid JSON."}]})
        try: result = request.app.state.assessor.assess(body)
        except ContractError as e: return JSONResponse(status_code=422, content={"errors": e.errors})
        return JSONResponse(result)
    @app.post("/explain", include_in_schema=False)  # additive; documented in EXPLAIN_API.md (kept out of the frozen OpenAPI document so the /assess contract surface is unchanged)
    async def explain(request: Request):
        try: body = await request.json()
        except ValueError: return JSONResponse(status_code=422, content={"errors": [{"field": "body", "code": "invalid_type", "message": "Request body must be valid JSON."}]})
        try: result = request.app.state.explainer.explain(body)
        except ContractError as e: return JSONResponse(status_code=422, content={"errors": e.errors})
        return JSONResponse(result)
    @app.get("/meta", summary="Frozen model-card performance, limitations, disclaimer")
    async def meta(request: Request): return request.app.state.meta
    return app
app = create_app()
