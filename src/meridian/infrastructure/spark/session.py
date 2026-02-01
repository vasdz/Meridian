"""Spark session builder."""

from typing import Optional

from meridian.core.config import settings
from meridian.core.logging import get_logger


logger = get_logger(__name__)


class SparkSessionBuilder:
    """Builder for PySpark sessions."""

    def __init__(self, app_name: str = "Meridian"):
        self.app_name = app_name
        self._spark = None

    def build(
        self,
        master: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        """Build and return SparkSession."""
        try:
            from pyspark.sql import SparkSession

            master = master or settings.spark_master

            builder = (
                SparkSession.builder
                .appName(self.app_name)
                .master(master)
            )

            # Apply configurations
            default_config = {
                "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
                "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
                "spark.sql.shuffle.partitions": "200",
            }

            if config:
                default_config.update(config)

            for key, value in default_config.items():
                builder = builder.config(key, value)

            self._spark = builder.getOrCreate()
            logger.info("SparkSession created", master=master)

            return self._spark

        except ImportError:
            logger.warning("PySpark not installed")
            return None

    def get_or_create(self):
        """Get existing or create new SparkSession."""
        if self._spark is None:
            return self.build()
        return self._spark

    def stop(self):
        """Stop SparkSession."""
        if self._spark:
            self._spark.stop()
            self._spark = None
            logger.info("SparkSession stopped")


# Global builder instance
spark_builder = SparkSessionBuilder()


def get_spark():
    """Get SparkSession."""
    return spark_builder.get_or_create()

