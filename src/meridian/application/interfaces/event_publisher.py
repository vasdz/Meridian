"""Event publisher interface."""

from abc import ABC, abstractmethod

from meridian.core.events import DomainEvent


class EventPublisherInterface(ABC):
    """Interface for publishing domain events."""

    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish a domain event."""
        pass

    @abstractmethod
    async def publish_batch(self, events: list[DomainEvent]) -> None:
        """Publish multiple domain events."""
        pass
