"""Global settings for DistillFW backend."""

import os
from pydantic_settings import BaseSettings


def _detect_gcp_project() -> str:
    proj = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", "")).strip()
    if proj:
        return proj
    try:
        import subprocess
        res = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0 and res.stdout.strip() and "(unset)" not in res.stdout:
            return res.stdout.strip()
    except Exception:
        pass
    return "distillfw"


class Settings(BaseSettings):
    PROJECT_NAME: str = "DistillFW API"
    VERSION: str = "1.0.0"
    GCP_PROJECT_ID: str = _detect_gcp_project()
    GCP_REGION: str = os.getenv("GCP_REGION", "us-central1")
    DEFAULT_BUCKET: str = os.getenv("DEFAULT_BUCKET", "distillfw-workspaces")
    
    # Custom training container image in Artifact Registry
    TRAINER_IMAGE_URI: str = os.getenv(
        "TRAINER_IMAGE_URI",
        f"{os.getenv('GCP_REGION', 'us-central1')}-docker.pkg.dev/{_detect_gcp_project()}/distillfw-docker-repo/distillfw-trainer:latest"
    )
    TRAINER_SA: str = os.getenv("TRAINER_SA", "")
    
    # Storage mode: 'gcs' or 'local' (auto-fallback if credentials not found or testing)
    STORAGE_MODE: str = os.getenv("STORAGE_MODE", "auto")
    LOCAL_STORAGE_ROOT: str = os.getenv("LOCAL_STORAGE_ROOT", os.path.abspath(".local_workspace"))

    model_config = {"env_file": ".env", "extra": "allow"}


settings = Settings()

