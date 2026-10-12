import os
from pathlib import Path
BACKEND_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.environ.get("TRACKA_ARTIFACT_DIR", BACKEND_DIR / "artifacts" / "v1.0.0"))
