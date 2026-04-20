from contextlib import asynccontextmanager
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.model_management import router as model_management_router
from app.core.config import get_settings
from app.db.session import init_database, new_session
from app.services.metrics import observe_http_request
from app.services.model_management import ModelManagementService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.export_dir.mkdir(parents=True, exist_ok=True)
    init_database(settings)
    session = new_session()
    try:
        ModelManagementService(settings, session).bootstrap_defaults()
    finally:
        session.close()
    yield


settings = get_settings()
app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_http_middleware(request: Request, call_next):
    started = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        observe_http_request(
            method=request.method,
            path=request.url.path,
            status=response.status_code if response is not None else 500,
            duration_seconds=max(time.perf_counter() - started, 0.0),
        )

app.include_router(health_router)
app.include_router(chat_router, prefix=settings.api_prefix)
app.include_router(documents_router, prefix=settings.api_prefix)
app.include_router(model_management_router, prefix=settings.api_prefix)
