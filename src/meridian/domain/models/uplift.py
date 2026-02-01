"""Uplift prediction domain model."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConfidenceInterval:
    """Confidence interval for predictions."""

    lower: float
    upper: float
    level: float = 0.95

    @property
    def width(self) -> float:
        """Calculate interval width."""
        return self.upper - self.lower


@dataclass
class UpliftPrediction:
    """Uplift prediction result."""

    customer_id: str
    cate: float  # Conditional Average Treatment Effect

    # Optional confidence interval
    confidence_interval: ConfidenceInterval | None = None

    # Model info
    model_id: str | None = None
    model_version: str | None = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)

    def should_treat(self, threshold: float = 0.0) -> bool:
        """Determine if customer should receive treatment."""
        return self.cate > threshold

    def get_expected_value(self, treatment_cost: float = 0.0) -> float:
        """Calculate expected value of treatment."""
        return self.cate - treatment_cost

    def is_confident(self, min_width: float = 0.1) -> bool:
        """Check if prediction is confident (narrow interval)."""
        if self.confidence_interval is None:
            return True
        return self.confidence_interval.width < min_width

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "customer_id": self.customer_id,
            "cate": self.cate,
            "should_treat": self.should_treat(),
        }

        if self.confidence_interval:
            result["confidence_interval"] = {
                "lower": self.confidence_interval.lower,
                "upper": self.confidence_interval.upper,
            }

        return result
