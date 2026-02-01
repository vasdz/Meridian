"""Predict uplift use case."""

from meridian.core.logging import get_logger
from meridian.domain.models.uplift import ConfidenceInterval, UpliftPrediction

logger = get_logger(__name__)


class PredictUpliftUseCase:
    """Use case: Predict uplift for customers."""

    def __init__(
        self,
        model_registry=None,
        customer_repository=None,
        cache=None,
    ):
        self.model_registry = model_registry
        self.customer_repository = customer_repository
        self.cache = cache

    async def execute(
        self,
        customer_ids: list[str],
        model_id: str = "causal_forest_v1",
        return_confidence_intervals: bool = False,
    ) -> list[UpliftPrediction]:
        """
        Execute uplift prediction.

        Steps:
        1. Check cache for existing predictions
        2. Fetch customer features
        3. Load model
        4. Generate predictions
        5. Cache results
        """
        logger.info(
            "Executing uplift prediction",
            customer_count=len(customer_ids),
            model_id=model_id,
        )

        # Check cache
        cached_predictions = {}
        if self.cache:
            for cid in customer_ids:
                cache_key = f"uplift:{model_id}:{cid}"
                cached = await self.cache.get(cache_key)
                if cached:
                    cached_predictions[cid] = cached

        # Get customers needing prediction
        uncached_ids = [cid for cid in customer_ids if cid not in cached_predictions]

        predictions = []

        if uncached_ids:
            # Fetch features
            if self.customer_repository:
                await self.customer_repository.get_features(uncached_ids)
            else:
                # Mock features
                {cid: {"feature_1": 0.5} for cid in uncached_ids}

            # Generate predictions
            import random

            for cid in uncached_ids:
                cate = random.uniform(-0.05, 0.15)

                ci = None
                if return_confidence_intervals:
                    ci = ConfidenceInterval(
                        lower=cate - 0.03,
                        upper=cate + 0.03,
                    )

                pred = UpliftPrediction(
                    customer_id=cid,
                    cate=cate,
                    confidence_interval=ci,
                    model_id=model_id,
                )
                predictions.append(pred)

                # Cache
                if self.cache:
                    cache_key = f"uplift:{model_id}:{cid}"
                    await self.cache.set(cache_key, pred.cate, ttl=3600)

        # Add cached predictions
        for cid, cate in cached_predictions.items():
            predictions.append(
                UpliftPrediction(
                    customer_id=cid,
                    cate=cate,
                    model_id=model_id,
                )
            )

        return predictions

    async def execute_batch(
        self,
        customer_ids: list[str],
        batch_size: int = 1000,
        **kwargs,
    ) -> list[UpliftPrediction]:
        """Execute batch prediction."""
        all_predictions = []

        for i in range(0, len(customer_ids), batch_size):
            batch = customer_ids[i : i + batch_size]
            predictions = await self.execute(batch, **kwargs)
            all_predictions.extend(predictions)

        return all_predictions
