# Track A backend (FastAPI) - contract v1.0.0
Thin FastAPI layer around the framework-free inference core `app/core/inference.py` (byte-identical to the approved file; hash-checked by tests).
Models, preprocessing, hyper-parameters and validation results are frozen in `artifacts/v1.0.0/` (verified against `manifest.json` at startup; the API refuses to start if any file differs).

    pip install -r requirements-dev.txt
    uvicorn app.main:app --port 8000          # POST /assess, GET /meta (FastAPI also serves /docs, /openapi.json)
    pytest -q                                 # 576 tests

Env: `TRACKA_ARTIFACT_DIR` overrides the artifact directory.
Errors: HTTP 422 `{"errors":[{"field","code","message"}]}` (see artifacts/v1.0.0/schema/error.schema.json). All 50 patient fields are required; nothing is imputed.
DECISION SUPPORT / EDUCATIONAL USE ONLY - not a diagnosis, not a substitute for formal diagnostic imaging.
