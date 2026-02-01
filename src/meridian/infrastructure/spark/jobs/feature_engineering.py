"""Feature engineering Spark job."""

from meridian.core.logging import get_logger

logger = get_logger(__name__)


def compute_customer_features(spark, transactions_df, output_path: str):
    """Compute customer-level features from transactions."""
    from pyspark.sql import functions as F

    logger.info("Computing customer features")

    # Aggregate features
    customer_features = transactions_df.groupBy("customer_id").agg(
        F.count("*").alias("transaction_count"),
        F.sum("amount").alias("total_spend"),
        F.avg("amount").alias("avg_transaction_amount"),
        F.max("amount").alias("max_transaction_amount"),
        F.countDistinct("product_id").alias("unique_products"),
        F.countDistinct("category").alias("unique_categories"),
        F.max("created_at").alias("last_transaction_date"),
        F.min("created_at").alias("first_transaction_date"),
    )

    # Calculate derived features
    customer_features = customer_features.withColumn(
        "tenure_days",
        F.datediff(F.current_date(), F.col("first_transaction_date")),
    ).withColumn(
        "days_since_last_purchase",
        F.datediff(F.current_date(), F.col("last_transaction_date")),
    )

    # Write features
    customer_features.write.format("delta").mode("overwrite").save(output_path)

    logger.info(
        "Customer features computed",
        n_customers=customer_features.count(),
    )

    return customer_features


def compute_product_features(spark, transactions_df, output_path: str):
    """Compute product-level features."""
    from pyspark.sql import functions as F

    product_features = transactions_df.groupBy("product_id").agg(
        F.count("*").alias("purchase_count"),
        F.sum("quantity").alias("total_quantity_sold"),
        F.avg("unit_price").alias("avg_price"),
        F.countDistinct("customer_id").alias("unique_buyers"),
    )

    product_features.write.format("delta").mode("overwrite").save(output_path)

    return product_features
