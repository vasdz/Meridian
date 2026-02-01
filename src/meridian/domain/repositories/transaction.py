"""Transaction repository interface."""

from abc import abstractmethod
from datetime import date

from meridian.domain.models.transaction import Transaction
from meridian.domain.repositories.base import AbstractRepository


class TransactionRepository(AbstractRepository[Transaction]):
    """Transaction repository interface."""

    @abstractmethod
    async def get_by_customer(
        self,
        customer_id: str,
        limit: int = 100,
    ) -> list[Transaction]:
        """Get transactions for a customer."""
        pass

    @abstractmethod
    async def get_by_date_range(
        self,
        start_date: date,
        end_date: date,
        limit: int = 1000,
    ) -> list[Transaction]:
        """Get transactions within date range."""
        pass

    @abstractmethod
    async def get_customer_totals(
        self,
        customer_id: str,
    ) -> dict:
        """Get aggregated totals for a customer."""
        pass
