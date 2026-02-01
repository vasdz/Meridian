"""Unit tests for predict uplift use case."""

from unittest.mock import AsyncMock

import pytest

from meridian.application.use_cases.predict_uplift import PredictUpliftUseCase
from meridian.domain.models.uplift import UpliftPrediction


class TestPredictUpliftUseCase:
    """Tests for PredictUpliftUseCase."""

    @pytest.mark.asyncio
    async def test_execute_returns_predictions(self):
        """Test that execute returns predictions for all customers."""
        use_case = PredictUpliftUseCase()

        customer_ids = ["c1", "c2", "c3"]
        predictions = await use_case.execute(customer_ids)

        assert len(predictions) == 3
        assert all(isinstance(p, UpliftPrediction) for p in predictions)

    @pytest.mark.asyncio
    async def test_execute_with_confidence_intervals(self):
        """Test predictions with confidence intervals."""
        use_case = PredictUpliftUseCase()

        predictions = await use_case.execute(
            ["c1"],
            return_confidence_intervals=True,
        )

        assert len(predictions) == 1
        assert predictions[0].confidence_interval is not None

    @pytest.mark.asyncio
    async def test_execute_uses_cache(self):
        """Test that predictions are cached."""
        mock_cache = AsyncMock()
        mock_cache.get = AsyncMock(return_value=0.05)

        use_case = PredictUpliftUseCase(cache=mock_cache)

        predictions = await use_case.execute(["c1"])

        assert len(predictions) == 1
        assert predictions[0].cate == 0.05
        mock_cache.get.assert_called()

    @pytest.mark.asyncio
    async def test_execute_batch(self):
        """Test batch prediction."""
        use_case = PredictUpliftUseCase()

        customer_ids = [f"c{i}" for i in range(100)]
        predictions = await use_case.execute_batch(
            customer_ids,
            batch_size=30,
        )

        assert len(predictions) == 100
