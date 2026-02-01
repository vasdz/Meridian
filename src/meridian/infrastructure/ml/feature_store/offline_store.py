"""Offline feature store using Spark."""

from typing import Optional

from meridian.core.logging import get_logger


logger = get_logger(__name__)


class OfflineStore:
    """Spark-based offline feature store."""

    def __init__(self, spark_session=None):
        self.spark = spark_session

    def compute_features(
        self,
        entity_df,
        feature_definitions: list[dict],
        output_path: str,
    ) -> None:
        """Compute features and save to offline store."""
        if self.spark is None:
            logger.warning("No Spark session available")
            return

        logger.info("Computing features", n_features=len(feature_definitions))

        # Would implement feature computation logic here
        pass

    def get_training_data(
        self,
        feature_view: str,
        entity_ids: list[str],
        start_date: str,
        end_date: str,
    ):
        """Get training data from offline store."""
        if self.spark is None:
            return None

        # Would read from parquet/delta files
        pass

