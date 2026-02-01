"""Event bus implementation."""

from meridian.core.events import DomainEvent, event_bus
from meridian.application.interfaces.event_publisher import EventPublisherInterface
from meridian.infrastructure.messaging.kafka_producer import KafkaProducer
from meridian.core.config import settings
from meridian.core.logging import get_logger


logger = get_logger(__name__)


class EventBusPublisher(EventPublisherInterface):
    """Event bus publisher that uses both in-memory and Kafka."""

    def __init__(self):
        self._kafka = KafkaProducer() if settings.kafka_enabled else None

    async def publish(self, event: DomainEvent) -> None:
        """Publish event to both local bus and Kafka."""
        # Local in-memory bus
        event_bus.publish(event)

        # Kafka
        if self._kafka:
            self._kafka.send(
                topic=f"meridian.events.{event.event_type.lower()}",
                key=event.event_id,
                value={
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp.isoformat(),
                    "correlation_id": event.correlation_id,
                    "data": event.__dict__,
                },
            )

        logger.debug("Event published", event_type=event.event_type)

    async def publish_batch(self, events: list[DomainEvent]) -> None:
        """Publish multiple events."""
        for event in events:
            await self.publish(event)

