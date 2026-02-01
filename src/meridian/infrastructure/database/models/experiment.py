"""Experiment ORM model."""

from datetime import date

from sqlalchemy import JSON, Date, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from meridian.infrastructure.database.models.base import Base, TimestampMixin


class ExperimentModel(Base, TimestampMixin):
    """Experiment database model."""

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    hypothesis: Mapped[str] = mapped_column(String(1000), nullable=False)

    # Status
    status: Mapped[str] = mapped_column(String(32), default="draft", index=True)

    # Variants (stored as JSON)
    variants: Mapped[list] = mapped_column(JSON, default=list)

    # Metrics
    primary_metric: Mapped[str] = mapped_column(String(64), default="conversion_rate")
    secondary_metrics: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Design
    target_sample_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_mde: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_level: Mapped[float] = mapped_column(Float, default=0.95)

    # Duration
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Results (stored as JSON)
    results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    winner_variant: Mapped[str | None] = mapped_column(String(64), nullable=True)

    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, name={self.name}, status={self.status})>"
