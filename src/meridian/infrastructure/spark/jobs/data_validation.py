"""Data validation Spark job using Great Expectations concepts."""

from typing import Optional

from meridian.core.logging import get_logger


logger = get_logger(__name__)


class DataValidator:
    """Data validation for Spark DataFrames."""

    def __init__(self, spark):
        self.spark = spark
        self.validation_results = []

    def expect_column_to_exist(self, df, column: str) -> bool:
        """Check if column exists in DataFrame."""
        result = column in df.columns
        self.validation_results.append({
            "check": "column_exists",
            "column": column,
            "passed": result,
        })
        return result

    def expect_column_values_to_not_be_null(
        self,
        df,
        column: str,
        threshold: float = 0.0,
    ) -> bool:
        """Check null rate is below threshold."""
        from pyspark.sql import functions as F

        total = df.count()
        null_count = df.filter(F.col(column).isNull()).count()
        null_rate = null_count / total if total > 0 else 0

        result = null_rate <= threshold
        self.validation_results.append({
            "check": "null_rate",
            "column": column,
            "null_rate": null_rate,
            "threshold": threshold,
            "passed": result,
        })
        return result

    def expect_column_values_to_be_between(
        self,
        df,
        column: str,
        min_value: float,
        max_value: float,
    ) -> bool:
        """Check if values are within range."""
        from pyspark.sql import functions as F

        out_of_range = df.filter(
            (F.col(column) < min_value) | (F.col(column) > max_value)
        ).count()

        result = out_of_range == 0
        self.validation_results.append({
            "check": "value_range",
            "column": column,
            "min": min_value,
            "max": max_value,
            "out_of_range_count": out_of_range,
            "passed": result,
        })
        return result

    def get_validation_report(self) -> dict:
        """Get validation report."""
        passed = sum(1 for r in self.validation_results if r["passed"])
        total = len(self.validation_results)

        return {
            "passed": passed,
            "failed": total - passed,
            "total": total,
            "success_rate": passed / total if total > 0 else 1.0,
            "results": self.validation_results,
        }


def validate_transactions_df(spark, df) -> dict:
    """Validate transactions DataFrame."""
    validator = DataValidator(spark)

    # Required columns
    for col in ["customer_id", "amount", "created_at"]:
        validator.expect_column_to_exist(df, col)

    # No nulls in key columns
    validator.expect_column_values_to_not_be_null(df, "customer_id")
    validator.expect_column_values_to_not_be_null(df, "amount")

    # Valid ranges
    validator.expect_column_values_to_be_between(df, "amount", 0, 100000)

    report = validator.get_validation_report()
    logger.info("Validation complete", **report)

    return report

