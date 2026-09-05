"""Main FastAPI Application Entrypoint for DistillFW."""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.core.config import settings
from backend.api.routes import (
    workspaces,
    config,
    dataset,
    teacher,
    cost,
    training,
    evaluation,
    deployment,
    logs
)
from backend.services.storage import storage_service
from backend.services.logger import operations_logger

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    operations_logger.log("DistillFW Backend starting up...", level="INFO", source="STARTUP")
    try:
        storage_service.create_bucket_if_not_exists(settings.DEFAULT_BUCKET)
    except Exception as e:
        operations_logger.log(f"Default bucket creation notice: {e}", level="WARNING", source="STARTUP")
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="DistillFW API: Managed Distillation Framework on Google Cloud Platform",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routers
app.include_router(workspaces.router, prefix="/api")
app.include_router(config.router, prefix="/api")
app.include_router(dataset.router, prefix="/api")
app.include_router(teacher.router, prefix="/api")
app.include_router(cost.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(evaluation.router, prefix="/api")
app.include_router(deployment.router, prefix="/api")
app.include_router(logs.router, prefix="/api")


@app.get("/healthz")
@app.get("/api/healthz")
def health_check():
    return {"status": "HEALTHY", "version": settings.VERSION}


# Mount built static frontend if present
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            return None
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
