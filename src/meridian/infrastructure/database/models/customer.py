"""Customer ORM model."""

from sqlalchemy import JSON, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from meridian.infrastructure.database.models.base import Base, TimestampMixin


class CustomerModel(Base, TimestampMixin):
    """Customer database model."""

    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    external_id: Mapped[str | None] = mapped_column(String(128), unique=True, nullable=True)

    # Segmentation
    segment: Mapped[str] = mapped_column(String(32), default="new", index=True)
    region: Mapped[str | None] = mapped_column(String(64), nullable=True)
    channel: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Demographics
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Behavioral metrics
    tenure_days: Mapped[int] = mapped_column(Integer, default=0)
    total_spend: Mapped[float] = mapped_column(Float, default=0.0)
    transaction_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_basket_size: Mapped[float] = mapped_column(Float, default=0.0)

    # Flexible features storage
    features: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=dict)

    def __repr__(self) -> str:
        return f"<Customer(id={self.id}, segment={self.segment})>"
