"""Base repository interface."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional

ModelType = TypeVar("ModelType")


class AbstractRepository(ABC, Generic[ModelType]):
    """Abstract repository pattern interface."""

    @abstractmethod
    async def get(self, id: str) -> Optional[ModelType]:
        """Get entity by ID."""
        pass

    @abstractmethod
    async def get_all(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelType]:
        """Get all entities with pagination."""
        pass

    @abstractmethod
    async def add(self, entity: ModelType) -> ModelType:
        """Add new entity."""
        pass

    @abstractmethod
    async def update(self, entity: ModelType) -> ModelType:
        """Update existing entity."""
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete entity by ID."""
        pass

    @abstractmethod
    async def exists(self, id: str) -> bool:
        """Check if entity exists."""
        pass

