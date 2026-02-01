"""Experiment domain model."""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

from meridian.core.constants import ExperimentStatus


@dataclass
class ExperimentVariant:
    """Experiment variant (treatment group)."""

    name: str
    weight: float = 0.5
    description: Optional[str] = None


@dataclass
class ExperimentResults:
    """Experiment analysis results."""

    lift: float  # Treatment effect
    lift_confidence_interval: tuple[float, float] = (0.0, 0.0)
    p_value: float = 1.0
    is_significant: bool = False

    # Sample sizes
    control_size: int = 0
    treatment_size: int = 0

    # Metrics
    control_metric: float = 0.0
    treatment_metric: float = 0.0


@dataclass
class Experiment:
    """A/B experiment aggregate root."""

    id: str
    name: str
    hypothesis: str
    status: ExperimentStatus = ExperimentStatus.DRAFT

    # Variants
    variants: list[ExperimentVariant] = field(default_factory=list)

    # Metrics
    primary_metric: str = "conversion_rate"
    secondary_metrics: list[str] = field(default_factory=list)

    # Design
    target_sample_size: Optional[int] = None
    target_mde: Optional[float] = None  # Minimum Detectable Effect
    confidence_level: float = 0.95

    # Duration
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    # Results
    results: Optional[ExperimentResults] = None
    winner_variant: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def can_start(self) -> bool:
        """Check if experiment can be started."""
        return (
            self.status == ExperimentStatus.DRAFT
            and len(self.variants) >= 2
        )

    def start(self) -> None:
        """Start the experiment."""
        if not self.can_start():
            raise ValueError("Experiment cannot be started")

        self.status = ExperimentStatus.RUNNING
        self.start_date = date.today()
        self.updated_at = datetime.utcnow()

    def stop(self, results: Optional[ExperimentResults] = None) -> None:
        """Stop the experiment."""
        if self.status != ExperimentStatus.RUNNING:
            raise ValueError("Experiment is not running")

        self.status = ExperimentStatus.COMPLETED
        self.end_date = date.today()

        if results:
            self.results = results
            if results.is_significant:
                # Determine winner based on lift direction
                if results.lift > 0:
                    # Find treatment variant
                    for v in self.variants:
                        if v.name != "control":
                            self.winner_variant = v.name
                            break

        self.updated_at = datetime.utcnow()

    def is_significant(self) -> bool:
        """Check if results are statistically significant."""
        if self.results is None:
            return False
        return self.results.is_significant

    @property
    def duration_days(self) -> Optional[int]:
        """Get experiment duration in days."""
        if self.start_date is None:
            return None
        end = self.end_date or date.today()
        return (end - self.start_date).days

