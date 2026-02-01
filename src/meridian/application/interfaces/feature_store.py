"""Feature store interface."""

from abc import ABC, abstractmethod
from typing import Optional


class FeatureStoreInterface(ABC):
    """Interface for feature store."""

    @abstractmethod
    async def get_online_features(
        self,
        entity_type: str,
        entity_ids: list[str],
        feature_names: list[str],
    ) -> dict[str, dict]:
        """Get online features for entities."""
        pass

    @abstractmethod
    async def get_historical_features(
        self,
        entity_type: str,
        entity_ids: list[str],
        feature_names: list[str],
        timestamp: str,
    ) -> dict[str, dict]:
        """Get historical features at a point in time."""
        pass

    @abstractmethod
    async def materialize_features(
        self,
        feature_view: str,
        start_date: str,
        end_date: str,
    ) -> None:
        """Materialize features to online store."""
        pass

