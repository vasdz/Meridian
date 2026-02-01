"""Database session dependency."""

from typing import Annotated, AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from meridian.infrastructure.database.connection import get_async_session


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session."""
    async with get_async_session() as session:
        yield session


# Type alias for dependency injection
DBSession = Annotated[AsyncSession, Depends(get_db_session)]

