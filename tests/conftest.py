"""Pytest configuration and fixtures."""

import asyncio
import pytest
from typing import AsyncGenerator, Generator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from httpx import AsyncClient

from meridian.main import app
from meridian.infrastructure.database.models.base import Base


# Test database URL (SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client."""
    from httpx import ASGITransport
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_customer_data() -> dict:
    """Sample customer data for testing."""
    return {
        "id": "cust-001",
        "external_id": "EXT-001",
        "segment": "high_value",
        "region": "North",
        "channel": "online",
        "age": 35,
        "tenure_days": 365,
        "total_spend": 1500.0,
        "transaction_count": 25,
        "avg_basket_size": 60.0,
    }


@pytest.fixture
def sample_experiment_data() -> dict:
    """Sample experiment data for testing."""
    return {
        "name": "Test Experiment",
        "hypothesis": "Testing hypothesis for uplift improvement",
        "primary_metric": "conversion_rate",
        "variants": [
            {"name": "control", "weight": 0.5},
            {"name": "treatment", "weight": 0.5},
        ],
    }


@pytest.fixture
def auth_headers() -> dict:
    """Authentication headers for testing.

    Uses dynamically generated test API key to avoid secret detection.
    """
    import secrets
    # Generate a unique test key for each test run
    test_key = f"mk_test_{secrets.token_hex(16)}"
    return {"X-API-Key": test_key}

