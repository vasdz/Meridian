"""Security audit logger."""

from meridian.core.logging import get_logger

logger = get_logger(__name__)


class AuditLogger:
    """Security audit trail logger."""

    @staticmethod
    async def log_request(audit_data: dict) -> None:
        """Log API request to audit trail."""
        # In production, would insert into audit_logs table
        logger.info(
            "Audit log",
            event_type="api_request",
            **audit_data,
        )

    @staticmethod
    async def log_authentication(
        user_id: str | None,
        success: bool,
        method: str,
        client_ip: str,
        details: dict | None = None,
    ) -> None:
        """Log authentication attempt."""
        logger.info(
            "Audit log",
            event_type="authentication",
            user_id=user_id,
            success=success,
            method=method,
            client_ip=client_ip,
            details=details,
        )

    @staticmethod
    async def log_authorization(
        user_id: str,
        resource: str,
        action: str,
        allowed: bool,
    ) -> None:
        """Log authorization decision."""
        logger.info(
            "Audit log",
            event_type="authorization",
            user_id=user_id,
            resource=resource,
            action=action,
            allowed=allowed,
        )

    @staticmethod
    async def log_data_access(
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
    ) -> None:
        """Log sensitive data access."""
        logger.info(
            "Audit log",
            event_type="data_access",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
        )

    @staticmethod
    async def log_security_event(
        event_type: str,
        severity: str,
        description: str,
        details: dict | None = None,
    ) -> None:
        """Log security event (suspicious activity, etc.)."""
        log_func = logger.warning if severity == "high" else logger.info

        log_func(
            "Security event",
            event_type=event_type,
            severity=severity,
            description=description,
            details=details,
        )
