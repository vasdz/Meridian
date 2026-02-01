"""Domain events system."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass
class DomainEvent:
    """Base class for domain events."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: str | None = None

    @property
    def event_type(self) -> str:
        """Get event type name."""
        return self.__class__.__name__


@dataclass
class UpliftPredictionRequested(DomainEvent):
    """Event: Uplift prediction was requested."""

    customer_ids: list[str] = field(default_factory=list)
    model_id: str = ""
    user_id: str = ""


@dataclass
class UpliftPredictionCompleted(DomainEvent):
    """Event: Uplift prediction completed."""

    customer_ids: list[str] = field(default_factory=list)
    model_id: str = ""
    prediction_count: int = 0
    duration_ms: float = 0.0


@dataclass
class ExperimentCreated(DomainEvent):
    """Event: New experiment was created."""

    experiment_id: str = ""
    name: str = ""
    created_by: str = ""


@dataclass
class ExperimentStarted(DomainEvent):
    """Event: Experiment was started."""

    experiment_id: str = ""


@dataclass
class ExperimentCompleted(DomainEvent):
    """Event: Experiment was completed."""

    experiment_id: str = ""
    winner_variant: str | None = None
    is_significant: bool = False


@dataclass
class ModelRetrainTriggered(DomainEvent):
    """Event: Model retrain was triggered."""

    model_id: str = ""
    triggered_by: str = ""


class EventBus:
    """Simple in-memory event bus."""

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {}

    def subscribe(
        self,
        event_type: type[DomainEvent],
        handler: Callable[[DomainEvent], None],
    ) -> None:
        """Subscribe to an event type."""
        event_name = event_type.__name__
        if event_name not in self._handlers:
            self._handlers[event_name] = []
        self._handlers[event_name].append(handler)

    def publish(self, event: DomainEvent) -> None:
        """Publish an event to all subscribers."""
        event_name = event.event_type
        handlers = self._handlers.get(event_name, [])

        for handler in handlers:
            try:
                handler(event)
            except Exception:
                # Log but don't fail
                pass

    def clear(self) -> None:
        """Clear all handlers."""
        self._handlers.clear()


# Global event bus instance
event_bus = EventBus()
