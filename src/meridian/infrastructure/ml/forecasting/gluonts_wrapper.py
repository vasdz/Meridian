"""GluonTS wrapper utilities."""

from typing import List, Optional

import pandas as pd

from meridian.core.logging import get_logger


logger = get_logger(__name__)


def create_dataset(
    df: pd.DataFrame,
    target_column: str,
    timestamp_column: str,
    item_id_column: Optional[str] = None,
    freq: str = "D",
):
    """Create GluonTS dataset from pandas DataFrame."""
    try:
        from gluonts.dataset.pandas import PandasDataset

        if item_id_column:
            return PandasDataset.from_long_dataframe(
                df,
                target=target_column,
                timestamp=timestamp_column,
                item_id=item_id_column,
                freq=freq,
            )
        else:
            return PandasDataset(
                df.set_index(timestamp_column)[target_column],
                freq=freq,
            )
    except ImportError:
        logger.warning("GluonTS not installed")
        return None


def forecast_to_dataframe(
    forecasts,
    item_ids: List[str],
    quantiles: List[float] = None,
) -> pd.DataFrame:
    """Convert GluonTS forecasts to pandas DataFrame."""
    quantiles = quantiles or [0.1, 0.5, 0.9]

    records = []
    for item_id, forecast in zip(item_ids, forecasts):
        for i, date in enumerate(forecast.index):
            record = {
                "item_id": item_id,
                "date": date,
                "mean": forecast.mean[i],
            }
            for q in quantiles:
                record[f"q{int(q*100):02d}"] = forecast.quantile(q)[i]
            records.append(record)

    return pd.DataFrame(records)

