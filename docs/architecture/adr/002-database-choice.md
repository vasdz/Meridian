# Architecture Decision Record: Database Choice

## Status
Accepted

## Context
We need to select a primary database for Meridian. Requirements:
- ACID compliance for transactional data
- Support for JSON/JSONB for flexible schemas
- Scalability for growing data volumes
- Strong ecosystem and tooling

## Decision
We chose **PostgreSQL 16** as our primary database.

## Rationale

### Features
- JSONB support for semi-structured data (features, results)
- Strong ACID compliance
- Excellent query performance with proper indexing
- Native array and JSON operators

### Operational Excellence
- Proven reliability in production environments
- Excellent tooling (pg_dump, pgAdmin, etc.)
- Strong replication and backup options
- Cloud-managed options (RDS, Cloud SQL, etc.)

### Integration
- Excellent SQLAlchemy support (including async)
- Alembic migrations
- TimescaleDB extension for time-series (future)

### Alternatives Considered
- **MySQL**: Less powerful JSON support, weaker feature set
- **MongoDB**: Not ACID by default, less suitable for transactional data
- **DynamoDB**: Vendor lock-in, complex pricing model

## Consequences
- Need async driver (asyncpg) for FastAPI integration
- Schema migrations required for changes
- Connection pooling needed for high concurrency

