"""Feast feature store adapter."""

from typing import Optional

from meridian.application.interfaces.feature_store import FeatureStoreInterface
from meridian.core.logging import get_logger


logger = get_logger(__name__)


class FeastAdapter(FeatureStoreInterface):
    """Feast feature store adapter."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self._store = None

    def _get_store(self):
        """Get Feast store instance (lazy)."""
        if self._store is None:
            try:
                from feast import FeatureStore
                self._store = FeatureStore(repo_path=self.repo_path)
            except ImportError:
                logger.warning("Feast not installed")
        return self._store

    async def get_online_features(
        self,
        entity_type: str,
        entity_ids: list[str],
        feature_names: list[str],
    ) -> dict[str, dict]:
        """Get online features from Feast."""
        store = self._get_store()

        if store is None:
            # Mock response
            return {eid: {fn: 0.0 for fn in feature_names} for eid in entity_ids}

        try:
            entity_rows = [{entity_type: eid} for eid in entity_ids]

            features = store.get_online_features(
                features=feature_names,
                entity_rows=entity_rows,
            ).to_dict()

            result = {}
            for i, eid in enumerate(entity_ids):
                result[eid] = {
                    fn: features[fn][i]
                    for fn in feature_names
                }

            return result

        except Exception as e:
            logger.error("Failed to get online features", error=str(e))
            return {}

    async def get_historical_features(
        self,
        entity_type: str,
        entity_ids: list[str],
        feature_names: list[str],
        timestamp: str,
    ) -> dict[str, dict]:
        """Get historical features from Feast."""
        # Simplified - would use get_historical_features in real impl
        return await self.get_online_features(entity_type, entity_ids, feature_names)

    async def materialize_features(
        self,
        feature_view: str,
        start_date: str,
        end_date: str,
    ) -> None:
        """Materialize features to online store."""
        store = self._get_store()

        if store is None:
            return

        try:
            from datetime import datetime

            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)

            store.materialize(start_date=start, end_date=end)
            logger.info("Features materialized", feature_view=feature_view)

        except Exception as e:
            logger.error("Failed to materialize features", error=str(e))

