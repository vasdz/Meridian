"""Spark utilities."""

from typing import Optional


def optimize_for_join(df, column: str, num_partitions: int = 200):
    """Repartition DataFrame for efficient joins."""
    return df.repartition(num_partitions, column)


def write_delta(
    df,
    path: str,
    mode: str = "overwrite",
    partition_by: Optional[list] = None,
):
    """Write DataFrame as Delta table."""
    writer = df.write.format("delta").mode(mode)

    if partition_by:
        writer = writer.partitionBy(*partition_by)

    writer.save(path)


def read_delta(spark, path: str):
    """Read Delta table."""
    return spark.read.format("delta").load(path)

