# Architecture Decision Record: Why FastAPI?
- Pydantic v2 migration required for future updates
- Must use async-compatible libraries (asyncpg, aioredis)
- Team needs to understand async/await patterns
## Consequences

- **Sanic**: Smaller community, less ecosystem support
- **Django**: Too heavy, ORM-centric, overkill for API-only
- **Flask**: More mature but sync-only, less performant
### Alternatives Considered

- Native support for WebSockets (future use)
- Excellent integration with SQLAlchemy, Redis, Celery
- Large community and active development
### Ecosystem

- Minimal boilerplate code
- Excellent IDE support with type hints
- Auto-generated OpenAPI documentation
- Automatic request/response validation via Pydantic
### Developer Experience

- Supports async/await natively
- Built on Starlette and Pydantic for async performance
- FastAPI is one of the fastest Python frameworks, on par with NodeJS and Go
### Performance

## Rationale

We chose **FastAPI** as our web framework.
## Decision

- Production-ready features
- Easy integration with ML models
- Automatic OpenAPI documentation
- Modern Python type hints
- High-performance async operations
We need to choose a web framework for the Meridian API. The framework should support:
## Context

Accepted
## Status


