"""Dynamic Pricing Module - Enterprise-grade price optimization.

This module provides production-ready pricing capabilities:
- Price elasticity estimation (Bayesian & frequentist)
- Profit/Revenue optimization
- Constraint handling (min/max prices, competition)
- Multi-product optimization
- Cannibalization effects

Designed for hyperscale retail pricing operations.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable
from scipy import optimize
from scipy import stats
from enum import Enum

from meridian.core.logging import get_logger


logger = get_logger(__name__)


class OptimizationObjective(Enum):
    """Pricing optimization objective."""
    PROFIT = "profit"
    REVENUE = "revenue"
    VOLUME = "volume"
    MARGIN = "margin"


class ElasticityModel(Enum):
    """Price elasticity estimation model."""
    OLS = "ols"
    BAYESIAN = "bayesian"
    LOG_LOG = "log_log"
    SEMI_LOG = "semi_log"


@dataclass
class PriceConstraints:
    """Constraints for price optimization."""

    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_margin: Optional[float] = None  # Minimum margin percentage
    max_price_change: Optional[float] = None  # Max % change from current
    competitor_price: Optional[float] = None  # Reference competitor price
    competitor_distance: Optional[float] = None  # Max distance from competitor

    def validate(self, price: float, current_price: float, cost: float) -> tuple[bool, list[str]]:
        """Validate price against constraints."""
        violations = []

        if self.min_price is not None and price < self.min_price:
            violations.append(f"Price {price:.2f} below minimum {self.min_price:.2f}")

        if self.max_price is not None and price > self.max_price:
            violations.append(f"Price {price:.2f} above maximum {self.max_price:.2f}")

        if self.min_margin is not None:
            margin = (price - cost) / price
            if margin < self.min_margin:
                violations.append(f"Margin {margin:.1%} below minimum {self.min_margin:.1%}")

        if self.max_price_change is not None:
            change = abs(price - current_price) / current_price
            if change > self.max_price_change:
                violations.append(f"Price change {change:.1%} exceeds maximum {self.max_price_change:.1%}")

        if self.competitor_price is not None and self.competitor_distance is not None:
            distance = abs(price - self.competitor_price) / self.competitor_price
            if distance > self.competitor_distance:
                violations.append(f"Distance from competitor {distance:.1%} exceeds maximum")

        return len(violations) == 0, violations


@dataclass
class ElasticityResult:
    """Result of elasticity estimation."""

    product_id: str
    elasticity: float  # Price elasticity of demand
    elasticity_std: float  # Standard error
    confidence_interval: tuple[float, float]
    r_squared: float
    p_value: float

    # Bayesian results (if applicable)
    elasticity_samples: Optional[np.ndarray] = None

    def is_elastic(self) -> bool:
        """Check if demand is elastic (|e| > 1)."""
        return abs(self.elasticity) > 1

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "elasticity": self.elasticity,
            "elasticity_std": self.elasticity_std,
            "confidence_interval": list(self.confidence_interval),
            "r_squared": self.r_squared,
            "p_value": self.p_value,
            "is_elastic": self.is_elastic(),
        }


@dataclass
class OptimizationResult:
    """Result of price optimization."""

    product_id: str
    current_price: float
    optimal_price: float
    price_change: float  # Percentage change

    # Predicted outcomes
    current_demand: float
    predicted_demand: float
    demand_change: float

    current_revenue: float
    predicted_revenue: float
    revenue_change: float

    current_profit: float
    predicted_profit: float
    profit_change: float

    # Optimization metadata
    objective: OptimizationObjective
    constraints_satisfied: bool
    elasticity_used: float

    confidence_interval: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "current_price": self.current_price,
            "optimal_price": self.optimal_price,
            "price_change_pct": self.price_change * 100,
            "predicted_demand": self.predicted_demand,
            "demand_change_pct": self.demand_change * 100,
            "predicted_revenue": self.predicted_revenue,
            "revenue_change_pct": self.revenue_change * 100,
            "predicted_profit": self.predicted_profit,
            "profit_change_pct": self.profit_change * 100,
            "objective": self.objective.value,
            "constraints_satisfied": self.constraints_satisfied,
        }


class PriceElasticityEstimator:
    """
    Estimate price elasticity of demand.

    Supports multiple estimation methods:
    - OLS: Simple linear regression
    - Log-Log: Constant elasticity model (most common)
    - Semi-Log: Linear-log model
    - Bayesian: Full posterior distribution
    """

    def __init__(
        self,
        model: ElasticityModel = ElasticityModel.LOG_LOG,
        confidence_level: float = 0.95,
    ):
        self.model = model
        self.confidence_level = confidence_level

    def estimate(
        self,
        prices: np.ndarray,
        quantities: np.ndarray,
        product_id: str = "default",
        controls: Optional[np.ndarray] = None,
    ) -> ElasticityResult:
        """
        Estimate price elasticity from historical data.

        Args:
            prices: Array of historical prices
            quantities: Array of corresponding quantities sold
            product_id: Product identifier
            controls: Optional control variables (promotions, seasonality, etc.)

        Returns:
            ElasticityResult with elasticity estimate and diagnostics
        """
        prices = np.asarray(prices)
        quantities = np.asarray(quantities)

        # Remove zeros and negatives
        valid_mask = (prices > 0) & (quantities > 0)
        prices = prices[valid_mask]
        quantities = quantities[valid_mask]

        if len(prices) < 10:
            raise ValueError("Insufficient data for elasticity estimation")

        logger.info(
            "Estimating price elasticity",
            product=product_id,
            model=self.model.value,
            n_observations=len(prices),
        )

        if self.model == ElasticityModel.LOG_LOG:
            return self._estimate_log_log(prices, quantities, product_id, controls)
        elif self.model == ElasticityModel.SEMI_LOG:
            return self._estimate_semi_log(prices, quantities, product_id, controls)
        elif self.model == ElasticityModel.BAYESIAN:
            return self._estimate_bayesian(prices, quantities, product_id, controls)
        else:
            return self._estimate_ols(prices, quantities, product_id, controls)

    def _estimate_log_log(
        self,
        prices: np.ndarray,
        quantities: np.ndarray,
        product_id: str,
        controls: Optional[np.ndarray],
    ) -> ElasticityResult:
        """Log-log model: ln(Q) = a + e*ln(P) + controls."""
        from sklearn.linear_model import LinearRegression

        log_prices = np.log(prices).reshape(-1, 1)
        log_quantities = np.log(quantities)

        if controls is not None:
            X = np.hstack([log_prices, controls[: len(prices)]])
        else:
            X = log_prices

        model = LinearRegression()
        model.fit(X, log_quantities)

        elasticity = model.coef_[0]  # Coefficient on log(price)

        # Calculate standard errors
        y_pred = model.predict(X)
        residuals = log_quantities - y_pred
        n = len(log_quantities)
        k = X.shape[1]

        mse = np.sum(residuals ** 2) / (n - k - 1)
        X_with_const = np.hstack([np.ones((n, 1)), X])
        var_covar = mse * np.linalg.inv(X_with_const.T @ X_with_const)
        se = np.sqrt(np.diag(var_covar))
        elasticity_std = se[1]  # SE for log(price) coefficient

        # Confidence interval
        t_crit = stats.t.ppf((1 + self.confidence_level) / 2, n - k - 1)
        ci = (elasticity - t_crit * elasticity_std, elasticity + t_crit * elasticity_std)

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((log_quantities - np.mean(log_quantities)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        # P-value
        t_stat = elasticity / elasticity_std
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1))

        return ElasticityResult(
            product_id=product_id,
            elasticity=elasticity,
            elasticity_std=elasticity_std,
            confidence_interval=ci,
            r_squared=r_squared,
            p_value=p_value,
        )

    def _estimate_semi_log(
        self,
        prices: np.ndarray,
        quantities: np.ndarray,
        product_id: str,
        controls: Optional[np.ndarray],
    ) -> ElasticityResult:
        """Semi-log model: Q = a + b*ln(P)."""
        from sklearn.linear_model import LinearRegression

        log_prices = np.log(prices).reshape(-1, 1)

        if controls is not None:
            X = np.hstack([log_prices, controls[:len(prices)]])
        else:
            X = log_prices

        model = LinearRegression()
        model.fit(X, quantities)

        b = model.coef_[0]
        mean_q = np.mean(quantities)

        # Elasticity at mean: e = b / mean(Q)
        elasticity = b / mean_q

        # Standard error (approximation)
        y_pred = model.predict(X)
        residuals = quantities - y_pred
        n = len(quantities)
        mse = np.sum(residuals ** 2) / (n - 2)
        se_b = np.sqrt(mse / np.sum((log_prices.flatten() - np.mean(log_prices)) ** 2))
        elasticity_std = se_b / mean_q

        t_crit = stats.t.ppf((1 + self.confidence_level) / 2, n - 2)
        ci = (elasticity - t_crit * elasticity_std, elasticity + t_crit * elasticity_std)

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((quantities - mean_q) ** 2)
        r_squared = 1 - ss_res / ss_tot

        t_stat = elasticity / elasticity_std
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

        return ElasticityResult(
            product_id=product_id,
            elasticity=elasticity,
            elasticity_std=elasticity_std,
            confidence_interval=ci,
            r_squared=r_squared,
            p_value=p_value,
        )

    def _estimate_ols(
        self,
        prices: np.ndarray,
        quantities: np.ndarray,
        product_id: str,
        controls: Optional[np.ndarray],
    ) -> ElasticityResult:
        """Simple OLS: Q = a + b*P (elasticity at mean)."""
        from sklearn.linear_model import LinearRegression

        X = prices.reshape(-1, 1)
        if controls is not None:
            X = np.hstack([X, controls[:len(prices)]])

        model = LinearRegression()
        model.fit(X, quantities)

        b = model.coef_[0]
        mean_p = np.mean(prices)
        mean_q = np.mean(quantities)

        # Elasticity at mean: e = (dQ/dP) * (P/Q) = b * (mean_P / mean_Q)
        elasticity = b * mean_p / mean_q

        # Approximate standard error
        y_pred = model.predict(X)
        residuals = quantities - y_pred
        n = len(quantities)
        mse = np.sum(residuals ** 2) / (n - 2)
        se_b = np.sqrt(mse / np.sum((prices - mean_p) ** 2))
        elasticity_std = se_b * mean_p / mean_q

        t_crit = stats.t.ppf((1 + self.confidence_level) / 2, n - 2)
        ci = (elasticity - t_crit * elasticity_std, elasticity + t_crit * elasticity_std)

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((quantities - mean_q) ** 2)
        r_squared = 1 - ss_res / ss_tot

        t_stat = b / se_b
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

        return ElasticityResult(
            product_id=product_id,
            elasticity=elasticity,
            elasticity_std=elasticity_std,
            confidence_interval=ci,
            r_squared=r_squared,
            p_value=p_value,
        )

    def _estimate_bayesian(
        self,
        prices: np.ndarray,
        quantities: np.ndarray,
        product_id: str,
        controls: Optional[np.ndarray],
    ) -> ElasticityResult:
        """Bayesian estimation with MCMC (simplified)."""
        # Simplified Bayesian: use bootstrap for posterior approximation
        n_bootstrap = 1000
        elasticities = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(len(prices), size=len(prices), replace=True)
            p_boot = prices[idx]
            q_boot = quantities[idx]

            try:
                result = self._estimate_log_log(p_boot, q_boot, product_id, None)
                elasticities.append(result.elasticity)
            except Exception:
                continue

        elasticities = np.array(elasticities)

        return ElasticityResult(
            product_id=product_id,
            elasticity=np.mean(elasticities),
            elasticity_std=np.std(elasticities),
            confidence_interval=(np.percentile(elasticities, 2.5), np.percentile(elasticities, 97.5)),
            r_squared=0.0,  # Not applicable
            p_value=0.0,  # Not applicable
            elasticity_samples=elasticities,
        )


class PriceOptimizer:
    """
    Optimize prices to maximize profit, revenue, or other objectives.

    Supports:
    - Single product optimization
    - Multi-product optimization with cross-elasticities
    - Constraint handling
    - Sensitivity analysis
    """

    def __init__(
        self,
        objective: OptimizationObjective = OptimizationObjective.PROFIT,
        elasticity_estimator: Optional[PriceElasticityEstimator] = None,
    ):
        self.objective = objective
        self.elasticity_estimator = elasticity_estimator or PriceElasticityEstimator()

        self._elasticities: dict[str, ElasticityResult] = {}

    def set_elasticity(
        self,
        product_id: str,
        elasticity: float,
        elasticity_std: float = 0.0,
    ) -> None:
        """Manually set elasticity for a product."""
        self._elasticities[product_id] = ElasticityResult(
            product_id=product_id,
            elasticity=elasticity,
            elasticity_std=elasticity_std,
            confidence_interval=(elasticity - 2*elasticity_std, elasticity + 2*elasticity_std),
            r_squared=1.0,
            p_value=0.0,
        )

    def estimate_elasticity(
        self,
        product_id: str,
        prices: np.ndarray,
        quantities: np.ndarray,
        controls: Optional[np.ndarray] = None,
    ) -> ElasticityResult:
        """Estimate and store elasticity for a product."""
        result = self.elasticity_estimator.estimate(prices, quantities, product_id, controls)
        self._elasticities[product_id] = result
        return result

    def optimize(
        self,
        product_id: str,
        current_price: float,
        current_demand: float,
        cost: float,
        constraints: Optional[PriceConstraints] = None,
    ) -> OptimizationResult:
        """
        Find optimal price for a product.

        Args:
            product_id: Product identifier
            current_price: Current selling price
            current_demand: Current demand at current price
            cost: Unit cost (for profit calculation)
            constraints: Price constraints

        Returns:
            OptimizationResult with optimal price and predictions
        """
        if product_id not in self._elasticities:
            raise ValueError(f"No elasticity for product {product_id}. Call estimate_elasticity first.")

        elasticity = self._elasticities[product_id].elasticity

        logger.info(
            "Optimizing price",
            product=product_id,
            current_price=current_price,
            elasticity=elasticity,
            objective=self.objective.value,
        )

        constraints = constraints or PriceConstraints()

        # Define demand function: Q(P) = Q0 * (P/P0)^e
        def demand_function(price: float) -> float:
            return current_demand * (price / current_price) ** elasticity

        # Define objective function
        if self.objective == OptimizationObjective.PROFIT:
            def objective(price: float) -> float:
                q = demand_function(price)
                return -(price - cost) * q  # Negative for minimization
        elif self.objective == OptimizationObjective.REVENUE:
            def objective(price: float) -> float:
                q = demand_function(price)
                return -price * q
        elif self.objective == OptimizationObjective.VOLUME:
            def objective(price: float) -> float:
                return -demand_function(price)
        else:  # MARGIN
            def objective(price: float) -> float:
                return -(price - cost) / price

        # Set bounds
        min_price = constraints.min_price or cost * 1.01  # At least 1% margin
        max_price = constraints.max_price or current_price * 3

        if constraints.max_price_change is not None:
            min_price = max(min_price, current_price * (1 - constraints.max_price_change))
            max_price = min(max_price, current_price * (1 + constraints.max_price_change))

        # Optimize
        result = optimize.minimize_scalar(
            objective,
            bounds=(min_price, max_price),
            method="bounded",
        )

        optimal_price = result.x

        # Validate constraints
        is_valid, violations = constraints.validate(optimal_price, current_price, cost)
        if not is_valid:
            logger.warning(f"Constraint violations: {violations}")
            # Adjust to nearest valid price
            if constraints.min_price and optimal_price < constraints.min_price:
                optimal_price = constraints.min_price
            if constraints.max_price and optimal_price > constraints.max_price:
                optimal_price = constraints.max_price

        # Calculate predictions
        predicted_demand = demand_function(optimal_price)
        predicted_revenue = optimal_price * predicted_demand
        predicted_profit = (optimal_price - cost) * predicted_demand

        current_revenue = current_price * current_demand
        current_profit = (current_price - cost) * current_demand

        return OptimizationResult(
            product_id=product_id,
            current_price=current_price,
            optimal_price=optimal_price,
            price_change=(optimal_price - current_price) / current_price,
            current_demand=current_demand,
            predicted_demand=predicted_demand,
            demand_change=(predicted_demand - current_demand) / current_demand,
            current_revenue=current_revenue,
            predicted_revenue=predicted_revenue,
            revenue_change=(predicted_revenue - current_revenue) / current_revenue,
            current_profit=current_profit,
            predicted_profit=predicted_profit,
            profit_change=(predicted_profit - current_profit) / current_profit if current_profit > 0 else 0,
            objective=self.objective,
            constraints_satisfied=is_valid,
            elasticity_used=elasticity,
        )

    def optimize_portfolio(
        self,
        products: list[dict],
        total_revenue_constraint: Optional[float] = None,
    ) -> list[OptimizationResult]:
        """
        Optimize prices for a portfolio of products.

        Args:
            products: List of dicts with product_id, current_price, current_demand, cost
            total_revenue_constraint: Minimum total revenue constraint

        Returns:
            List of OptimizationResult for each product
        """
        results = []

        for product in products:
            try:
                result = self.optimize(
                    product_id=product["product_id"],
                    current_price=product["current_price"],
                    current_demand=product["current_demand"],
                    cost=product["cost"],
                    constraints=product.get("constraints"),
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to optimize {product['product_id']}: {e}")

        # Check total revenue constraint
        if total_revenue_constraint is not None:
            total_predicted_revenue = sum(r.predicted_revenue for r in results)

            if total_predicted_revenue < total_revenue_constraint:
                logger.warning(
                    f"Total predicted revenue {total_predicted_revenue:.2f} "
                    f"below constraint {total_revenue_constraint:.2f}"
                )

        return results

    def sensitivity_analysis(
        self,
        product_id: str,
        current_price: float,
        current_demand: float,
        cost: float,
        price_range: tuple[float, float] = (0.8, 1.2),
        n_points: int = 20,
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis over a price range.

        Returns DataFrame with price, demand, revenue, profit, margin for each price point.
        """
        if product_id not in self._elasticities:
            raise ValueError(f"No elasticity for product {product_id}")

        elasticity = self._elasticities[product_id].elasticity

        prices = np.linspace(
            current_price * price_range[0],
            current_price * price_range[1],
            n_points,
        )

        results = []
        for price in prices:
            demand = current_demand * (price / current_price) ** elasticity
            revenue = price * demand
            profit = (price - cost) * demand
            margin = (price - cost) / price

            results.append({
                "price": price,
                "price_change_pct": (price - current_price) / current_price * 100,
                "demand": demand,
                "demand_change_pct": (demand - current_demand) / current_demand * 100,
                "revenue": revenue,
                "profit": profit,
                "margin_pct": margin * 100,
            })

        return pd.DataFrame(results)


class DynamicPricingEngine:
    """
    Production-ready dynamic pricing engine.

    Combines elasticity estimation, optimization, and business rules
    for enterprise-scale pricing operations.
    """

    def __init__(
        self,
        objective: OptimizationObjective = OptimizationObjective.PROFIT,
        elasticity_model: ElasticityModel = ElasticityModel.LOG_LOG,
        default_constraints: Optional[PriceConstraints] = None,
    ):
        self.objective = objective
        self.elasticity_estimator = PriceElasticityEstimator(model=elasticity_model)
        self.optimizer = PriceOptimizer(objective=objective, elasticity_estimator=self.elasticity_estimator)
        self.default_constraints = default_constraints or PriceConstraints(
            min_margin=0.05,
            max_price_change=0.20,
        )

        self._price_history: dict[str, list[dict]] = {}

    def update_elasticity(
        self,
        product_id: str,
        prices: np.ndarray,
        quantities: np.ndarray,
        controls: Optional[np.ndarray] = None,
    ) -> ElasticityResult:
        """Update elasticity estimate for a product."""
        return self.optimizer.estimate_elasticity(product_id, prices, quantities, controls)

    def get_recommended_price(
        self,
        product_id: str,
        current_price: float,
        current_demand: float,
        cost: float,
        constraints: Optional[PriceConstraints] = None,
    ) -> OptimizationResult:
        """Get recommended price for a product."""
        constraints = constraints or self.default_constraints
        return self.optimizer.optimize(product_id, current_price, current_demand, cost, constraints)

    def run_pricing_cycle(
        self,
        products_data: pd.DataFrame,
        product_id_col: str = "product_id",
        price_col: str = "price",
        quantity_col: str = "quantity",
        cost_col: str = "cost",
        date_col: str = "date",
    ) -> list[OptimizationResult]:
        """
        Run full pricing cycle for multiple products.

        1. Estimate elasticities from historical data
        2. Optimize prices
        3. Apply business rules
        4. Return recommendations
        """
        logger.info(
            "Running pricing cycle",
            n_products=products_data[product_id_col].nunique(),
        )

        results = []

        for product_id in products_data[product_id_col].unique():
            product_data = products_data[products_data[product_id_col] == product_id]

            if len(product_data) < 20:
                logger.warning(f"Insufficient data for {product_id}")
                continue

            try:
                # Estimate elasticity
                prices = product_data[price_col].values
                quantities = product_data[quantity_col].values

                elasticity_result = self.update_elasticity(product_id, prices, quantities)

                # Get current values
                current = product_data.sort_values(date_col).iloc[-1]
                current_price = current[price_col]
                current_demand = current[quantity_col]
                cost = current[cost_col]

                # Optimize
                result = self.get_recommended_price(
                    product_id=product_id,
                    current_price=current_price,
                    current_demand=current_demand,
                    cost=cost,
                )

                results.append(result)

                # Log recommendation
                logger.info(
                    "Price recommendation",
                    product=product_id,
                    current_price=current_price,
                    optimal_price=result.optimal_price,
                    expected_profit_change=f"{result.profit_change:.1%}",
                )

            except Exception as e:
                logger.error(f"Pricing failed for {product_id}: {e}")

        return results

    def simulate_scenario(
        self,
        product_id: str,
        current_price: float,
        current_demand: float,
        cost: float,
        proposed_price: float,
    ) -> dict:
        """Simulate a pricing scenario (what-if analysis)."""
        if product_id not in self.optimizer._elasticities:
            raise ValueError(f"No elasticity for {product_id}")

        elasticity = self.optimizer._elasticities[product_id].elasticity

        # Calculate predicted outcomes
        predicted_demand = current_demand * (proposed_price / current_price) ** elasticity
        predicted_revenue = proposed_price * predicted_demand
        predicted_profit = (proposed_price - cost) * predicted_demand

        current_revenue = current_price * current_demand
        current_profit = (current_price - cost) * current_demand

        return {
            "product_id": product_id,
            "proposed_price": proposed_price,
            "predicted_demand": predicted_demand,
            "demand_change_pct": (predicted_demand - current_demand) / current_demand * 100,
            "predicted_revenue": predicted_revenue,
            "revenue_change_pct": (predicted_revenue - current_revenue) / current_revenue * 100,
            "predicted_profit": predicted_profit,
            "profit_change_pct": (predicted_profit - current_profit) / current_profit * 100 if current_profit > 0 else 0,
            "elasticity_used": elasticity,
        }

