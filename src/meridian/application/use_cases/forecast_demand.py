"""Forecast demand use case."""

from datetime import date, timedelta
from typing import Optional

from meridian.core.logging import get_logger


logger = get_logger(__name__)


class ForecastDemandUseCase:
    """Use case: Forecast demand for items."""

    def __init__(
        self,
        model_registry=None,
        feature_store=None,
        cache=None,
    ):
        self.model_registry = model_registry
        self.feature_store = feature_store
        self.cache = cache

    async def execute(
        self,
        items: list[dict],
        horizon_days: int = 14,
        model_id: str = "deepar_v1",
        quantiles: list[float] = None,
    ) -> list[dict]:
        """
        Execute demand forecasting.

        Returns forecasts for each item for each day in the horizon.
        """
        quantiles = quantiles or [0.1, 0.5, 0.9]

        logger.info(
            "Executing demand forecast",
            item_count=len(items),
            horizon_days=horizon_days,
            model_id=model_id,
        )

        forecasts = []
        today = date.today()

        import random
        for item in items:
            for day in range(horizon_days):
                forecast_date = today + timedelta(days=day + 1)

                # Mock forecast
                base = random.uniform(50, 150)
                variance = base * 0.2

                forecast = {
                    "item_id": item.get("item_id"),
                    "store_id": item.get("store_id"),
                    "date": forecast_date.isoformat(),
                    "point_forecast": base,
                    "quantiles": {
                        str(q): base + (q - 0.5) * variance * 2
                        for q in quantiles
                    },
                }
                forecasts.append(forecast)

        return forecasts

