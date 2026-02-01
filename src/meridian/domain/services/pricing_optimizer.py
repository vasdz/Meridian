"""Pricing optimizer domain service."""

from meridian.core.logging import get_logger
from meridian.domain.models.pricing import PriceOptimizationResult, PricePoint

logger = get_logger(__name__)


class PricingOptimizer:
    """Domain service for price optimization."""

    def estimate_elasticity(
        self,
        prices: list[float],
        quantities: list[float],
    ) -> float:
        """
        Estimate price elasticity using log-linear regression.

        Elasticity = % change in quantity / % change in price
        """
        import numpy as np

        log_prices = np.log(prices)
        log_quantities = np.log(quantities)

        # Simple OLS
        x_mean = np.mean(log_prices)
        y_mean = np.mean(log_quantities)

        numerator = np.sum((log_prices - x_mean) * (log_quantities - y_mean))
        denominator = np.sum((log_prices - x_mean) ** 2)

        elasticity = numerator / denominator if denominator != 0 else -1.0

        return float(elasticity)

    def calculate_optimal_price(
        self,
        current_price: float,
        elasticity: float,
        cost: float | None = None,
        objective: str = "maximize_profit",
    ) -> float:
        """
        Calculate optimal price using elasticity.

        For profit maximization with constant elasticity:
        P* = (e / (e + 1)) * MC
        where e is elasticity (negative) and MC is marginal cost.
        """
        if cost is None:
            cost = current_price * 0.6  # Assume 40% margin

        if elasticity >= -1:
            # Inelastic demand - increase price
            return current_price * 1.1

        if objective == "maximize_profit":
            # Lerner index approach
            optimal = cost * (elasticity / (elasticity + 1))
            return max(optimal, cost * 1.1)  # Ensure positive margin

        elif objective == "maximize_revenue":
            # Revenue maximization
            # For log-linear demand: P* = -1/e * P_current
            return current_price * (-1 / elasticity)

        return current_price

    def optimize_prices(
        self,
        price_points: list[PricePoint],
        objective: str = "maximize_profit",
        constraints: dict | None = None,
    ) -> PriceOptimizationResult:
        """Optimize prices for multiple products."""
        constraints = constraints or {}

        recommendations = []
        total_revenue_change = 0.0
        total_profit_change = 0.0

        for pp in price_points:
            optimal = self.calculate_optimal_price(
                current_price=pp.current_price,
                elasticity=pp.price_elasticity,
                cost=pp.cost,
                objective=objective,
            )

            # Apply constraints
            max_change = constraints.get("max_price_change", 0.2)
            min_price = pp.current_price * (1 - max_change)
            max_price = pp.current_price * (1 + max_change)

            optimal = max(min_price, min(max_price, optimal))

            # Apply min margin constraint
            if pp.cost:
                min_margin = constraints.get("min_margin", 0.1)
                min_allowed_price = pp.cost / (1 - min_margin)
                optimal = max(optimal, min_allowed_price)

            pp.recommended_price = optimal

            # Estimate impact
            demand_change = pp.predict_demand_change(optimal)
            revenue_change = (optimal / pp.current_price - 1) + demand_change

            total_revenue_change += revenue_change / len(price_points)

            recommendations.append(pp)

        return PriceOptimizationResult(
            recommendations=recommendations,
            objective=objective,
            solver_status="optimal",
            expected_revenue_change=total_revenue_change,
            expected_profit_change=total_profit_change,
        )
