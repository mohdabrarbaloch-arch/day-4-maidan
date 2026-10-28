"""Maidan — a local LLM playground.

FastAPI app factory. Two things here are deliberate and load-bearing:

1. ``init_db()`` runs at import time, not only in a lifespan handler. Serverless
   platforms import the module and skip lifespan events, so anything that only
   happens on startup silently never happens in production.
2. The static mount is guarded by a real directory check. An unguarded mount on
   a missing directory raises at import and turns every route into a 500, which
   is exactly how a whole app can look "deployed" while being completely dead.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.db import init_db
from app.rate_limit import RateLimitMiddleware
from app.routers import auth, playground

__version__ = "1.0.0"

logger = logging.getLogger("maidan")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Import-time table creation. See the module docstring for why this is not
# inside a lifespan handler.
init_db()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Maidan API",
        description=(
            "A local LLM playground: pick a model, tune the sampler, watch tokens "
            "arrive one at a time. The local engine runs entirely on this machine."
        ),
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        RateLimitMiddleware,
        per_minute=settings.rate_limit_per_minute,
    )

    origins = settings.cors_origin_list()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=origins != ["*"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=()")
        return response

    app.include_router(auth.router)
    app.include_router(playground.router)

    @app.get("/api/health", tags=["meta"])
    def health() -> dict:
        from app.engine.local_model import list_models

        return {
            "status": "ok",
            "app": "Maidan",
            "version": __version__,
            "models": len(list_models()),
            "cloud_providers": sorted(settings.cloud_backends().keys()),
        }

    @app.get("/api/version", tags=["meta"])
    def version() -> dict:
        return {"version": __version__, "environment": settings.environment}

    # --- static frontend -------------------------------------------------
    if STATIC_DIR.is_dir() and (STATIC_DIR / "index.html").is_file():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

        @app.get("/favicon.ico", include_in_schema=False)
        def favicon() -> FileResponse:
            return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")

    else:
        # Fail loudly and usefully instead of raising at import. A missing
        # static bundle is a packaging bug, and the message should say so.
        logger.warning("Static bundle not found at %s", STATIC_DIR)

        @app.get("/", include_in_schema=False)
        def index_missing() -> JSONResponse:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "frontend_not_bundled",
                    "detail": (
                        f"Static assets were not found at {STATIC_DIR}. The API is "
                        "healthy; only the UI bundle is missing. Check that app/static "
                        "is included in the deployment bundle."
                    ),
                },
            )

    # Explicit JSON 404 for unknown paths. Without this, a broken route returns
    # the SPA shell with a 200 and every probe looks green.
    @app.exception_handler(404)
    async def not_found(request: Request, exc) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "path": request.url.path},
        )

    return app


app = create_app()
