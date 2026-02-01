"""Audit log ORM model."""

from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, DateTime, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column

from meridian.infrastructure.database.models.base import Base


class AuditLogModel(Base):
    """Audit log database model for security tracking."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Request identification
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    correlation_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # User/client
    user_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    api_key_prefix: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    client_ip: Mapped[str] = mapped_column(String(45))  # IPv6 compatible
    user_agent: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # Request details
    method: Mapped[str] = mapped_column(String(10))
    path: Mapped[str] = mapped_column(String(256))
    query_params: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    # Response
    status_code: Mapped[int] = mapped_column(Integer)
    response_time_ms: Mapped[float] = mapped_column(Integer)

    # Additional context
    action: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    resource_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    resource_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Error details (if any)
    error_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        index=True,
    )

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, path={self.path}, status={self.status_code})>"

