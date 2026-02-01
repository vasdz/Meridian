"""Customer domain model."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from meridian.core.constants import CustomerSegment


@dataclass
class Customer:
    """Customer aggregate root."""

    id: str
    external_id: Optional[str] = None
    segment: CustomerSegment = CustomerSegment.NEW
    region: Optional[str] = None
    channel: Optional[str] = None

    # Demographics
    age: Optional[int] = None

    # Behavioral metrics
    tenure_days: int = 0
    total_spend: float = 0.0
    transaction_count: int = 0
    avg_basket_size: float = 0.0

    # Features for ML
    features: dict = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_feature_dict(self) -> dict:
        """Convert to feature dictionary for ML models."""
        return {
            "age": self.age,
            "tenure_days": self.tenure_days,
            "total_spend": self.total_spend,
            "transaction_count": self.transaction_count,
            "avg_basket_size": self.avg_basket_size,
            "segment": self.segment.value,
            "region": self.region,
            "channel": self.channel,
            **self.features,
        }

    def update_segment(self, new_segment: CustomerSegment) -> None:
        """Update customer segment."""
        self.segment = new_segment
        self.updated_at = datetime.utcnow()

    def calculate_rfm_score(self) -> dict:
        """Calculate RFM (Recency, Frequency, Monetary) scores."""
        # Simplified scoring
        recency_score = min(5, max(1, 6 - self.tenure_days // 100))
        frequency_score = min(5, 1 + self.transaction_count // 10)
        monetary_score = min(5, 1 + int(self.total_spend / 500))

        return {
            "recency": recency_score,
            "frequency": frequency_score,
            "monetary": monetary_score,
            "total": recency_score + frequency_score + monetary_score,
        }

