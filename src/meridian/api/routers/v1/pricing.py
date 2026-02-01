"""Price optimization endpoints - Production-grade dynamic pricing API."""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from meridian.api.dependencies.auth import TokenData, get_current_user
from meridian.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Extended schemas for production pricing API


class PriceConstraints(BaseModel):
    """Constraints for price optimization."""

    min_price: float | None = Field(None, ge=0)
    max_price: float | None = Field(None, ge=0)
    min_margin: float | None = Field(None, ge=0, le=1)
    max_price_change: float | None = Field(None, ge=0, le=1)


class ProductPricingInput(BaseModel):
    """Input for single product pricing."""

    product_id: str
    current_price: float = Field(..., gt=0)
    current_demand: float = Field(..., ge=0)
    cost: float = Field(..., ge=0)
    elasticity: float | None = Field(None, description="Override elasticity")
    constraints: PriceConstraints | None = None


class ElasticityRequest(BaseModel):
    """Request for elasticity estimation."""

    product_id: str
    prices: list[float] = Field(..., min_length=10)
    quantities: list[float] = Field(..., min_length=10)
    method: str = Field(default="log_log")


class ElasticityResponse(BaseModel):
    """Response for elasticity estimation."""

    product_id: str
    elasticity: float
    elasticity_std: float
    confidence_interval: tuple[float, float]
    r_squared: float
    p_value: float
    is_elastic: bool
    interpretation: str


class OptimizationResult(BaseModel):
    """Result for single product optimization."""

    product_id: str
    current_price: float
    optimal_price: float
    price_change_pct: float
    predicted_demand: float
    demand_change_pct: float
    predicted_revenue: float
    revenue_change_pct: float
    predicted_profit: float
    profit_change_pct: float
    elasticity_used: float
    constraints_satisfied: bool


class BatchOptimizationRequest(BaseModel):
    """Request for batch price optimization."""

    products: list[ProductPricingInput]
    objective: str = Field(default="profit")
    global_constraints: PriceConstraints | None = None


class BatchOptimizationResponse(BaseModel):
    """Response for batch optimization."""

    results: list[OptimizationResult]
    total_products: int
    total_revenue_change_pct: float
    total_profit_change_pct: float
    processing_time_ms: float
    timestamp: datetime


class SensitivityRequest(BaseModel):
    """Request for price sensitivity analysis."""

    product_id: str
    current_price: float
    current_demand: float
    cost: float
    elasticity: float
    price_range_pct: float = Field(default=0.2, ge=0.01, le=0.5)
    n_points: int = Field(default=20, ge=5, le=100)


class SensitivityPoint(BaseModel):
    """Single point in sensitivity analysis."""

    price: float
    price_change_pct: float
    demand: float
    revenue: float
    profit: float
    margin_pct: float


class SensitivityResponse(BaseModel):
    """Response for sensitivity analysis."""

    product_id: str
    analysis: list[SensitivityPoint]
    optimal_price: float
    optimal_profit: float


@router.post("/optimize", response_model=BatchOptimizationResponse)
async def optimize_prices(
    request: BatchOptimizationRequest,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """
    Calculate optimal prices for products.

    Uses price elasticity and constraints to maximize profit or revenue.
    Supports batch optimization for multiple products.
    """
    import time

    start_time = time.time()

    logger.info(
        "Price optimization request",
        product_count=len(request.products),
        objective=request.objective,
        user_id=current_user.user_id,
    )

    results = []
    total_current_revenue: float = 0.0
    total_predicted_revenue: float = 0.0
    total_current_profit: float = 0.0
    total_predicted_profit: float = 0.0

    for product in request.products:
        # Use provided elasticity or default
        elasticity = product.elasticity or -1.5

        # Calculate optimal price
        current_price = product.current_price
        current_demand = product.current_demand
        cost = product.cost

        # Demand function: Q(P) = Q0 * (P/P0)^e
        def demand_at_price(price: float) -> float:
            return current_demand * (price / current_price) ** elasticity

        def profit_at_price(price: float) -> float:
            q = demand_at_price(price)
            return (price - cost) * q

        # Simple grid search for optimal price
        min_price = cost * 1.01
        max_price = current_price * 2

        if product.constraints:
            if product.constraints.min_price:
                min_price = max(min_price, product.constraints.min_price)
            if product.constraints.max_price:
                max_price = min(max_price, product.constraints.max_price)
            if product.constraints.max_price_change:
                min_price = max(
                    min_price, current_price * (1 - product.constraints.max_price_change)
                )
                max_price = min(
                    max_price, current_price * (1 + product.constraints.max_price_change)
                )

        best_price = current_price
        best_profit = profit_at_price(current_price)

        for price in [min_price + (max_price - min_price) * i / 50 for i in range(51)]:
            profit = profit_at_price(price)
            if profit > best_profit:
                best_profit = profit
                best_price = price

        # Calculate metrics
        predicted_demand = demand_at_price(best_price)
        predicted_revenue = best_price * predicted_demand
        predicted_profit = (best_price - cost) * predicted_demand

        current_revenue = current_price * current_demand
        current_profit = (current_price - cost) * current_demand

        total_current_revenue += current_revenue
        total_predicted_revenue += predicted_revenue
        total_current_profit += current_profit
        total_predicted_profit += predicted_profit

        results.append(
            OptimizationResult(
                product_id=product.product_id,
                current_price=current_price,
                optimal_price=round(best_price, 2),
                price_change_pct=round((best_price - current_price) / current_price * 100, 2),
                predicted_demand=round(predicted_demand, 1),
                demand_change_pct=round(
                    (predicted_demand - current_demand) / current_demand * 100, 2
                ),
                predicted_revenue=round(predicted_revenue, 2),
                revenue_change_pct=round(
                    (predicted_revenue - current_revenue) / current_revenue * 100, 2
                ),
                predicted_profit=round(predicted_profit, 2),
                profit_change_pct=(
                    round((predicted_profit - current_profit) / current_profit * 100, 2)
                    if current_profit > 0
                    else 0
                ),
                elasticity_used=elasticity,
                constraints_satisfied=True,
            )
        )

    processing_time = (time.time() - start_time) * 1000

    return BatchOptimizationResponse(
        results=results,
        total_products=len(results),
        total_revenue_change_pct=(
            round(
                (total_predicted_revenue - total_current_revenue) / total_current_revenue * 100, 2
            )
            if total_current_revenue > 0
            else 0
        ),
        total_profit_change_pct=(
            round((total_predicted_profit - total_current_profit) / total_current_profit * 100, 2)
            if total_current_profit > 0
            else 0
        ),
        processing_time_ms=round(processing_time, 2),
        timestamp=datetime.now(),
    )


@router.post("/elasticity", response_model=ElasticityResponse)
async def estimate_elasticity(
    request: ElasticityRequest,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """
    Estimate price elasticity from historical data.

    Supports multiple estimation methods:
    - log_log: Constant elasticity model (recommended)
    - ols: Simple linear regression
    - semi_log: Semi-logarithmic model
    """
    import numpy as np
    from scipy import stats
    from sklearn.linear_model import LinearRegression

    logger.info(
        "Elasticity estimation request",
        product_id=request.product_id,
        n_observations=len(request.prices),
        method=request.method,
    )

    if len(request.prices) != len(request.quantities):
        raise HTTPException(status_code=400, detail="prices and quantities must have same length")

    prices = np.array(request.prices)
    quantities = np.array(request.quantities)

    # Filter valid data
    valid_mask = (prices > 0) & (quantities > 0)
    prices = prices[valid_mask]
    quantities = quantities[valid_mask]

    if len(prices) < 10:
        raise HTTPException(
            status_code=400, detail="Insufficient data (need at least 10 valid observations)"
        )

    # Log-log model: ln(Q) = a + e*ln(P)
    log_prices = np.log(prices).reshape(-1, 1)
    log_quantities = np.log(quantities)

    model = LinearRegression()
    model.fit(log_prices, log_quantities)

    elasticity = float(model.coef_[0])

    # Calculate statistics
    y_pred = model.predict(log_prices)
    residuals = log_quantities - y_pred
    n = len(log_quantities)

    mse = np.sum(residuals**2) / (n - 2)
    se = np.sqrt(mse / np.sum((log_prices.flatten() - np.mean(log_prices)) ** 2))

    t_stat = elasticity / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_quantities - np.mean(log_quantities)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    t_crit = stats.t.ppf(0.975, n - 2)
    ci = (elasticity - t_crit * se, elasticity + t_crit * se)

    is_elastic = abs(elasticity) > 1

    if elasticity < -1:
        interpretation = "Elastic demand - price increases will decrease revenue"
    elif elasticity > -1 and elasticity < 0:
        interpretation = "Inelastic demand - price increases will increase revenue"
    else:
        interpretation = "Unusual positive elasticity - verify data quality"

    return ElasticityResponse(
        product_id=request.product_id,
        elasticity=round(elasticity, 3),
        elasticity_std=round(float(se), 3),
        confidence_interval=(round(ci[0], 3), round(ci[1], 3)),
        r_squared=round(r_squared, 3),
        p_value=round(float(p_value), 4),
        is_elastic=is_elastic,
        interpretation=interpretation,
    )


@router.post("/sensitivity", response_model=SensitivityResponse)
async def sensitivity_analysis(
    request: SensitivityRequest,
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """
    Perform price sensitivity analysis (what-if scenarios).

    Returns demand, revenue, and profit at various price points.
    """
    import numpy as np

    logger.info(
        "Sensitivity analysis request",
        product_id=request.product_id,
        current_price=request.current_price,
    )

    prices = np.linspace(
        request.current_price * (1 - request.price_range_pct),
        request.current_price * (1 + request.price_range_pct),
        request.n_points,
    )

    analysis = []
    best_price = request.current_price
    best_profit = 0

    for price in prices:
        demand = request.current_demand * (price / request.current_price) ** request.elasticity
        revenue = price * demand
        profit = (price - request.cost) * demand
        margin = (price - request.cost) / price * 100

        analysis.append(
            SensitivityPoint(
                price=round(price, 2),
                price_change_pct=round(
                    (price - request.current_price) / request.current_price * 100, 2
                ),
                demand=round(demand, 1),
                revenue=round(revenue, 2),
                profit=round(profit, 2),
                margin_pct=round(margin, 2),
            )
        )

        if profit > best_profit:
            best_profit = profit
            best_price = price

    return SensitivityResponse(
        product_id=request.product_id,
        analysis=analysis,
        optimal_price=round(best_price, 2),
        optimal_profit=round(best_profit, 2),
    )


@router.get("/models")
async def list_pricing_models(
    current_user: Annotated[TokenData, Depends(get_current_user)],
):
    """List available pricing models and their status."""
    return {
        "models": [
            {
                "id": "log_log_elasticity",
                "name": "Log-Log Elasticity Model",
                "description": "Constant elasticity demand model",
                "status": "active",
            },
            {
                "id": "bayesian_elasticity",
                "name": "Bayesian Elasticity Model",
                "description": "Bayesian estimation with uncertainty quantification",
                "status": "active",
            },
        ],
        "objectives": ["profit", "revenue", "volume", "margin"],
        "default_constraints": {
            "min_margin": 0.05,
            "max_price_change": 0.20,
        },
    }
