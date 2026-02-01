"""Transaction domain model."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Transaction:
    """Transaction entity."""

    id: str
    customer_id: str
    amount: float
    quantity: int = 1

    # Product info
    product_id: str | None = None
    category: str | None = None

    # Channel info
    channel: str | None = None
    store_id: str | None = None

    # Pricing
    unit_price: float | None = None
    discount_amount: float = 0.0

    # Flags
    is_return: bool = False

    # Promotion
    promotion_id: str | None = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def net_amount(self) -> float:
        """Calculate net amount after discount."""
        amount = self.amount - self.discount_amount
        return -amount if self.is_return else amount

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "amount": self.amount,
            "quantity": self.quantity,
            "product_id": self.product_id,
            "category": self.category,
            "channel": self.channel,
            "created_at": self.created_at.isoformat(),
        }
