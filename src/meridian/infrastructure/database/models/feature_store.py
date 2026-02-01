"""Feature store ORM model."""

from typing import Optional
from datetime import datetime

from sqlalchemy import String, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column

from meridian.infrastructure.database.models.base import Base


class FeatureRegistryModel(Base):
    """Feature registry database model."""

    __tablename__ = "feature_registry"

    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(64), index=True)

    # Feature definition
    data_type: Mapped[str] = mapped_column(String(32))
    description: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # Source
    source_table: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    transform_sql: Mapped[Optional[str]] = mapped_column(String(2000), nullable=True)

    # Metadata
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, onupdate=datetime.utcnow)

