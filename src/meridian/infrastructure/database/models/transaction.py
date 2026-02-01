"""Transaction ORM model."""

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from meridian.infrastructure.database.models.base import Base, TimestampMixin


class TransactionModel(Base, TimestampMixin):
    """Transaction database model."""

    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("customers.id"),
        index=True,
    )

    # Amount
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, default=1)

    # Product
    product_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Channel
    channel: Mapped[str | None] = mapped_column(String(32), nullable=True)
    store_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Pricing
    unit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    discount_amount: Mapped[float] = mapped_column(Float, default=0.0)

    # Flags
    is_return: Mapped[bool] = mapped_column(Boolean, default=False)

    # Promotion
    promotion_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    def __repr__(self) -> str:
        return f"<Transaction(id={self.id}, amount={self.amount})>"
