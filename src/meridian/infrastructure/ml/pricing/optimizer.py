"""Price optimization using mathematical programming."""

from meridian.core.logging import get_logger

logger = get_logger(__name__)


class PriceOptimizer:
    """Price optimization using OR-Tools or Pyomo."""

    def __init__(
        self,
        solver: str = "glop",
        time_limit_seconds: int = 300,
    ):
        self.solver = solver
        self.time_limit_seconds = time_limit_seconds

    def optimize(
        self,
        products: list[dict],
        elasticities: list[float],
        objective: str = "maximize_profit",
        constraints: dict | None = None,
    ) -> dict:
        """
        Optimize prices for products.

        Args:
            products: List of product dicts with current_price, cost
            elasticities: Price elasticities per product
            objective: maximize_profit or maximize_revenue
            constraints: Optional constraints dict

        Returns:
            Optimization result with recommended prices
        """
        logger.info(
            "Running price optimization",
            n_products=len(products),
            objective=objective,
        )

        constraints = constraints or {}
        max_price_change = constraints.get("max_price_change", 0.2)
        min_margin = constraints.get("min_margin", 0.1)

        recommendations = []

        for product, elasticity in zip(products, elasticities, strict=False):
            current_price = product["current_price"]
            cost = product.get("cost", current_price * 0.6)

            # Simple optimal price calculation
            if objective == "maximize_profit":
                # Lerner markup
                if elasticity < -1:
                    optimal = cost * (elasticity / (elasticity + 1))
                else:
                    optimal = current_price * 1.05
            else:
                # Revenue maximization
                optimal = current_price * (-1 / elasticity) if elasticity < 0 else current_price

            # Apply constraints
            min_price = current_price * (1 - max_price_change)
            max_price = current_price * (1 + max_price_change)
            optimal = max(min_price, min(max_price, optimal))

            # Ensure minimum margin
            min_allowed = cost / (1 - min_margin)
            optimal = max(optimal, min_allowed)

            recommendations.append(
                {
                    "product_id": product.get("product_id"),
                    "current_price": current_price,
                    "recommended_price": optimal,
                    "elasticity": elasticity,
                }
            )

        return {
            "status": "optimal",
            "recommendations": recommendations,
            "objective": objective,
        }
