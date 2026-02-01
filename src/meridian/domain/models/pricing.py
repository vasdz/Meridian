"""Pricing domain model."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PricePoint:
    """A product price point."""

    product_id: str
    current_price: float
    recommended_price: Optional[float] = None
    cost: Optional[float] = None

    # Elasticity
    price_elasticity: float = -1.5

    # Constraints
    min_price: Optional[float] = None
    max_price: Optional[float] = None

    @property
    def margin(self) -> Optional[float]:
        """Calculate margin."""
        if self.cost is None:
            return None
        return (self.current_price - self.cost) / self.current_price

    @property
    def recommended_margin(self) -> Optional[float]:
        """Calculate recommended margin."""
        if self.cost is None or self.recommended_price is None:
            return None
        return (self.recommended_price - self.cost) / self.recommended_price

    def predict_demand_change(self, new_price: float) -> float:
        """Predict demand change for price change."""
        price_change_pct = (new_price - self.current_price) / self.current_price
        demand_change_pct = self.price_elasticity * price_change_pct
        return demand_change_pct

    def is_price_valid(self, price: float) -> bool:
        """Check if price is within constraints."""
        if self.min_price and price < self.min_price:
            return False
        if self.max_price and price > self.max_price:
            return False
        return True


@dataclass
class PriceOptimizationResult:
    """Result of price optimization."""

    recommendations: list[PricePoint] = field(default_factory=list)
    objective: str = "maximize_profit"
    solver_status: str = "optimal"

    # Expected impact
    expected_revenue_change: float = 0.0
    expected_profit_change: float = 0.0

