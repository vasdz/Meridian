"""Tracing setup for FastAPI using OpenTelemetry."""

from meridian.core.config import settings
from meridian.core.logging import get_logger

logger = get_logger(__name__)


def setup_tracing(app) -> None:
    """Configure OpenTelemetry tracing for the FastAPI app."""
    if not settings.tracing_enabled:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Tracing disabled: OpenTelemetry not available", error=str(exc))
        return

    resource = Resource.create(
        {
            "service.name": settings.tracing_service_name,
            "service.version": settings.app_version,
            "deployment.environment": settings.environment,
        }
    )
    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=settings.tracing_otlp_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)

    logger.info(
        "Tracing initialized",
        endpoint=settings.tracing_otlp_endpoint,
        service=settings.tracing_service_name,
    )
