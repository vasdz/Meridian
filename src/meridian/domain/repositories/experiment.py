"""Experiment repository interface."""

from abc import abstractmethod
from typing import Optional

from meridian.domain.models.experiment import Experiment
from meridian.domain.repositories.base import AbstractRepository


class ExperimentRepository(AbstractRepository[Experiment]):
    """Experiment repository interface."""

    @abstractmethod
    async def get_by_status(
        self,
        status: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Experiment]:
        """Get experiments by status."""
        pass

    @abstractmethod
    async def get_running(self) -> list[Experiment]:
        """Get all running experiments."""
        pass

    @abstractmethod
    async def update_status(
        self,
        experiment_id: str,
        new_status: str,
    ) -> bool:
        """Update experiment status."""
        pass

    @abstractmethod
    async def set_results(
        self,
        experiment_id: str,
        results: dict,
    ) -> bool:
        """Set experiment results."""
        pass

