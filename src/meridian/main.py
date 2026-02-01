"""FastAPI application factory."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from meridian.api.middleware.correlation import CorrelationMiddleware
from meridian.api.middleware.exception_handler import setup_exception_handlers
from meridian.api.middleware.security_headers import SecurityHeadersMiddleware
from meridian.api.middleware.timing import TimingMiddleware
from meridian.api.routers import health
from meridian.api.routers.v1 import (
    admin,
    customers,
    experiments,
    forecasting,
    monitoring,
    pricing,
    uplift,
)
from meridian.core.config import settings
from meridian.core.logging import setup_logging
from meridian.infrastructure.cache.redis_cache import close_redis, init_redis
from meridian.infrastructure.database.connection import close_db, init_db


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    # Startup
    setup_logging()
    await init_db()
    await init_redis()

    yield

    # Shutdown
    await close_db()
    await close_redis()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Marketing Analytics & Causal Inference Platform",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Middleware (order matters - first added = outermost)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(CorrelationMiddleware)
    app.add_middleware(TimingMiddleware)

    # Exception handlers
    setup_exception_handlers(app)

    # Routers
    app.include_router(health.router)
    app.include_router(uplift.router, prefix="/v1/uplift", tags=["uplift"])
    app.include_router(forecasting.router, prefix="/v1/forecast", tags=["forecasting"])
    app.include_router(pricing.router, prefix="/v1/pricing", tags=["pricing"])
    app.include_router(experiments.router, prefix="/v1/experiments", tags=["experiments"])
    app.include_router(customers.router, prefix="/v1/customers", tags=["customers"])
    app.include_router(admin.router, prefix="/v1/admin", tags=["admin"])
    app.include_router(monitoring.router, prefix="/v1/monitoring", tags=["monitoring"])

    return app


# Application instance
app = create_app()
