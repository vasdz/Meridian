"""Optimize price use case."""

from meridian.core.logging import get_logger
from meridian.domain.models.pricing import PriceOptimizationResult, PricePoint
from meridian.domain.services.pricing_optimizer import PricingOptimizer

logger = get_logger(__name__)


class OptimizePriceUseCase:
    """Use case: Optimize prices for products."""

    def __init__(
        self,
        pricing_optimizer: PricingOptimizer | None = None,
        elasticity_model=None,
    ):
        self.optimizer = pricing_optimizer or PricingOptimizer()
        self.elasticity_model = elasticity_model

    async def execute(
        self,
        products: list[dict],
        objective: str = "maximize_profit",
        constraints: dict | None = None,
    ) -> PriceOptimizationResult:
        """
        Execute price optimization.

        Steps:
        1. Estimate price elasticity (if not provided)
        2. Calculate optimal prices
        3. Apply constraints
        4. Return recommendations
        """
        logger.info(
            "Executing price optimization",
            product_count=len(products),
            objective=objective,
        )

        constraints = constraints or {}

        price_points = []
        for product in products:
            elasticity = product.get("elasticity", -1.5)

            pp = PricePoint(
                product_id=product["product_id"],
                current_price=product["current_price"],
                cost=product.get("cost"),
                price_elasticity=elasticity,
            )
            price_points.append(pp)

        result = self.optimizer.optimize_prices(
            price_points=price_points,
            objective=objective,
            constraints=constraints,
        )

        logger.info(
            "Price optimization complete",
            product_count=len(result.recommendations),
            expected_revenue_change=result.expected_revenue_change,
        )

        return result
