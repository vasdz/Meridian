"""Run A/B test use case."""

from meridian.core.logging import get_logger
from meridian.domain.models.experiment import Experiment, ExperimentResults, ExperimentVariant
from meridian.domain.services.experiment_design import ExperimentDesignService

logger = get_logger(__name__)


class RunABTestUseCase:
    """Use case: Run and analyze A/B test."""

    def __init__(
        self,
        experiment_repository=None,
        experiment_design_service: ExperimentDesignService | None = None,
    ):
        self.experiment_repository = experiment_repository
        self.design_service = experiment_design_service or ExperimentDesignService()

    async def create_experiment(
        self,
        name: str,
        hypothesis: str,
        primary_metric: str = "conversion_rate",
        mde: float = 0.05,
        power: float = 0.8,
    ) -> Experiment:
        """Create a new experiment."""
        import uuid

        # Calculate required sample size
        sample_calc = self.design_service.calculate_sample_size(
            baseline_rate=0.05,  # Would come from historical data
            mde=mde,
            power=power,
        )

        experiment = Experiment(
            id=str(uuid.uuid4()),
            name=name,
            hypothesis=hypothesis,
            primary_metric=primary_metric,
            target_sample_size=sample_calc["total_sample_size"],
            target_mde=mde,
            variants=[
                ExperimentVariant(name="control", weight=0.5),
                ExperimentVariant(name="treatment", weight=0.5),
            ],
        )

        if self.experiment_repository:
            experiment = await self.experiment_repository.add(experiment)

        logger.info("Experiment created", experiment_id=experiment.id)

        return experiment

    async def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment."""
        if self.experiment_repository:
            experiment = await self.experiment_repository.get(experiment_id)
        else:
            # Mock
            experiment = Experiment(
                id=experiment_id,
                name="Test",
                hypothesis="Test hypothesis",
            )

        experiment.start()

        if self.experiment_repository:
            await self.experiment_repository.update(experiment)

        logger.info("Experiment started", experiment_id=experiment_id)

        return experiment

    async def analyze_experiment(
        self,
        experiment_id: str,
        control_conversions: int,
        control_size: int,
        treatment_conversions: int,
        treatment_size: int,
    ) -> ExperimentResults:
        """Analyze experiment results."""
        analysis = self.design_service.analyze_results(
            control_conversions=control_conversions,
            control_size=control_size,
            treatment_conversions=treatment_conversions,
            treatment_size=treatment_size,
        )

        results = ExperimentResults(
            lift=analysis["lift"],
            lift_confidence_interval=analysis["confidence_interval"],
            p_value=analysis["p_value"],
            is_significant=analysis["is_significant"],
            control_size=control_size,
            treatment_size=treatment_size,
            control_metric=analysis["control_rate"],
            treatment_metric=analysis["treatment_rate"],
        )

        logger.info(
            "Experiment analyzed",
            experiment_id=experiment_id,
            is_significant=results.is_significant,
            lift=results.lift,
        )

        return results

    async def stop_experiment(
        self,
        experiment_id: str,
        results: ExperimentResults | None = None,
    ) -> Experiment:
        """Stop an experiment."""
        if self.experiment_repository:
            experiment = await self.experiment_repository.get(experiment_id)
        else:
            experiment = Experiment(
                id=experiment_id,
                name="Test",
                hypothesis="Test hypothesis",
            )
            experiment.status = "running"

        experiment.stop(results)

        if self.experiment_repository:
            await self.experiment_repository.update(experiment)

        logger.info("Experiment stopped", experiment_id=experiment_id)

        return experiment
