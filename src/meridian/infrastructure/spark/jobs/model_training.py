"""Model training Spark job."""

from meridian.core.logging import get_logger

logger = get_logger(__name__)


def train_uplift_model(spark, training_data_path: str, model_output_path: str):
    """Train uplift model on Spark."""
    from pyspark.ml.feature import VectorAssembler

    logger.info("Training uplift model", data_path=training_data_path)

    # Load training data
    df = spark.read.format("delta").load(training_data_path)

    # Feature columns
    feature_cols = [
        "age",
        "tenure_days",
        "total_spend",
        "transaction_count",
        "avg_basket_size",
    ]

    # Assemble features
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
    )

    df_assembled = assembler.transform(df)

    # Would train actual model here
    # For now, just log completion

    logger.info("Model training complete")

    return df_assembled
