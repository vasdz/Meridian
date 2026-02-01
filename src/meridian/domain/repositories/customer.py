"""Customer repository interface."""

from abc import abstractmethod

from meridian.domain.models.customer import Customer
from meridian.domain.repositories.base import AbstractRepository


class CustomerRepository(AbstractRepository[Customer]):
    """Customer repository interface."""

    @abstractmethod
    async def get_by_external_id(self, external_id: str) -> Customer | None:
        """Get customer by external ID."""
        pass

    @abstractmethod
    async def get_by_segment(
        self,
        segment: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Customer]:
        """Get customers by segment."""
        pass

    @abstractmethod
    async def get_features(
        self,
        customer_ids: list[str],
    ) -> dict[str, dict]:
        """Get features for multiple customers."""
        pass

    @abstractmethod
    async def update_segment(
        self,
        customer_id: str,
        new_segment: str,
    ) -> bool:
        """Update customer segment."""
        pass
