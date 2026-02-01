"""ML Pipeline Module - Production-grade ML workflow orchestration.

This module provides enterprise-level ML pipeline capabilities:
- Feature engineering pipelines
- Model training pipelines
- Batch & real-time inference
- Model versioning and registry
- Pipeline monitoring and alerting

Designed for hyperscale ML operations.
"""

import hashlib
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from meridian.core.logging import get_logger

logger = get_logger(__name__)


T = TypeVar("T")


class PipelineStatus(Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepType(Enum):
    """Type of pipeline step."""

    EXTRACT = "extract"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    TRAIN = "train"
    EVALUATE = "evaluate"
    DEPLOY = "deploy"
    INFERENCE = "inference"


@dataclass
class StepResult:
    """Result of a pipeline step execution."""

    step_name: str
    step_type: StepType
    status: PipelineStatus
    start_time: datetime
    end_time: datetime
    duration_seconds: float

    # Data
    output: Any | None = None
    output_shape: tuple | None = None

    # Metrics
    metrics: dict[str, float] = field(default_factory=dict)

    # Error info
    error_message: str | None = None
    error_traceback: str | None = None

    def to_dict(self) -> dict:
        return {
            "step_name": self.step_name,
            "step_type": self.step_type.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "output_shape": self.output_shape,
            "metrics": self.metrics,
            "error_message": self.error_message,
        }


@dataclass
class PipelineResult:
    """Result of a complete pipeline execution."""

    pipeline_name: str
    run_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: datetime
    total_duration_seconds: float

    step_results: list[StepResult] = field(default_factory=list)
    final_output: Any | None = None
    aggregate_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pipeline_name": self.pipeline_name,
            "run_id": self.run_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_duration_seconds": self.total_duration_seconds,
            "steps": [s.to_dict() for s in self.step_results],
            "aggregate_metrics": self.aggregate_metrics,
        }


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""

    def __init__(
        self,
        name: str,
        step_type: StepType,
        retry_count: int = 0,
        timeout_seconds: int | None = None,
    ):
        self.name = name
        self.step_type = step_type
        self.retry_count = retry_count
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    def execute(self, input_data: Any, context: dict) -> Any:
        """Execute the pipeline step."""
        pass

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data before execution."""
        return True

    def validate_output(self, output_data: Any) -> bool:
        """Validate output data after execution."""
        return True


class FunctionStep(PipelineStep):
    """Pipeline step that wraps a function."""

    def __init__(
        self,
        name: str,
        func: Callable,
        step_type: StepType = StepType.TRANSFORM,
        **kwargs,
    ):
        super().__init__(name, step_type, **kwargs)
        self.func = func

    def execute(self, input_data: Any, context: dict) -> Any:
        return self.func(input_data, **context)


class DataValidationStep(PipelineStep):
    """Step for validating data quality."""

    def __init__(
        self,
        name: str = "data_validation",
        checks: list[Callable] | None = None,
        fail_on_error: bool = True,
    ):
        super().__init__(name, StepType.VALIDATE)
        self.checks = checks or []
        self.fail_on_error = fail_on_error

    def add_check(self, check: Callable[[pd.DataFrame], tuple[bool, str]]) -> None:
        """Add a validation check."""
        self.checks.append(check)

    def execute(self, input_data: pd.DataFrame, context: dict) -> pd.DataFrame:
        """Run all validation checks."""
        validation_results = []

        for check in self.checks:
            try:
                passed, message = check(input_data)
                validation_results.append(
                    {
                        "check": check.__name__,
                        "passed": passed,
                        "message": message,
                    }
                )

                if not passed and self.fail_on_error:
                    raise ValueError(f"Validation failed: {message}")

            except Exception as e:
                if self.fail_on_error:
                    raise
                validation_results.append(
                    {
                        "check": check.__name__,
                        "passed": False,
                        "message": str(e),
                    }
                )

        context["validation_results"] = validation_results
        return input_data


class FeatureEngineeringStep(PipelineStep):
    """Step for feature engineering transformations."""

    def __init__(
        self,
        name: str = "feature_engineering",
        transformations: list[Callable] | None = None,
    ):
        super().__init__(name, StepType.TRANSFORM)
        self.transformations = transformations or []

    def add_transformation(self, transform: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        """Add a transformation."""
        self.transformations.append(transform)

    def execute(self, input_data: pd.DataFrame, context: dict) -> pd.DataFrame:
        """Apply all transformations sequentially."""
        df = input_data.copy()

        for transform in self.transformations:
            df = transform(df)
            logger.debug(f"Applied transformation: {transform.__name__}")

        context["feature_columns"] = list(df.columns)
        return df


class ModelTrainingStep(PipelineStep):
    """Step for model training."""

    def __init__(
        self,
        name: str = "model_training",
        model_factory: Callable | None = None,
        target_column: str = "target",
        feature_columns: list[str] | None = None,
    ):
        super().__init__(name, StepType.TRAIN)
        self.model_factory = model_factory
        self.target_column = target_column
        self.feature_columns = feature_columns

    def execute(self, input_data: pd.DataFrame, context: dict) -> dict:
        """Train the model."""
        from sklearn.model_selection import train_test_split

        feature_cols = self.feature_columns or context.get("feature_columns", [])
        feature_cols = [c for c in feature_cols if c != self.target_column]

        X = input_data[feature_cols]
        y = input_data[self.target_column]

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train model
        model = self.model_factory() if self.model_factory else self._default_model()
        model.fit(X_train, y_train)

        # Evaluate on validation set
        val_predictions = model.predict(X_val)

        from sklearn.metrics import mean_squared_error, r2_score

        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_val, val_predictions)),
            "r2": r2_score(y_val, val_predictions),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }

        context["metrics"] = metrics
        context["feature_columns"] = feature_cols

        return {
            "model": model,
            "metrics": metrics,
            "feature_columns": feature_cols,
        }

    def _default_model(self):
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(n_estimators=100, max_depth=5)


class InferenceStep(PipelineStep):
    """Step for model inference."""

    def __init__(
        self,
        name: str = "inference",
        model: Any | None = None,
        feature_columns: list[str] | None = None,
        batch_size: int = 1000,
    ):
        super().__init__(name, StepType.INFERENCE)
        self.model = model
        self.feature_columns = feature_columns
        self.batch_size = batch_size

    def set_model(self, model: Any, feature_columns: list[str]) -> None:
        """Set model for inference."""
        self.model = model
        self.feature_columns = feature_columns

    def execute(self, input_data: pd.DataFrame, context: dict) -> pd.DataFrame:
        """Run inference on input data."""
        if self.model is None:
            self.model = context.get("model")

        if self.model is None:
            raise ValueError("No model set for inference")

        feature_cols = self.feature_columns or context.get("feature_columns", [])
        X = input_data[feature_cols]

        # Batch prediction for memory efficiency
        predictions = []
        for i in range(0, len(X), self.batch_size):
            batch = X.iloc[i : i + self.batch_size]
            batch_pred = self.model.predict(batch)
            predictions.extend(batch_pred)

        result = input_data.copy()
        result["prediction"] = predictions

        return result


class Pipeline:
    """
    Production-grade ML pipeline.

    Features:
    - Sequential step execution
    - Error handling and retries
    - Logging and monitoring
    - Context propagation
    - Result caching
    """

    def __init__(
        self,
        name: str,
        steps: list[PipelineStep] | None = None,
        enable_caching: bool = False,
    ):
        self.name = name
        self.steps = steps or []
        self.enable_caching = enable_caching
        self._cache: dict[str, Any] = {}

    def add_step(self, step: PipelineStep) -> "Pipeline":
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{self.name}_{timestamp}"
        return hashlib.sha256(hash_input.encode(), usedforsecurity=False).hexdigest()[:12]

    def run(
        self,
        input_data: Any,
        context: dict | None = None,
    ) -> PipelineResult:
        """
        Execute the pipeline.

        Args:
            input_data: Initial input data
            context: Additional context for steps

        Returns:
            PipelineResult with execution details
        """
        run_id = self._generate_run_id()
        context = context or {}
        context["run_id"] = run_id
        context["pipeline_name"] = self.name

        start_time = datetime.now()
        step_results = []
        current_data = input_data
        overall_status = PipelineStatus.RUNNING

        logger.info(
            "Starting pipeline",
            pipeline=self.name,
            run_id=run_id,
            n_steps=len(self.steps),
        )

        try:
            for step in self.steps:
                step_start = datetime.now()
                step_status = PipelineStatus.RUNNING
                step_error = None
                step_traceback = None

                logger.info(
                    "Executing step",
                    step=step.name,
                    type=step.step_type.value,
                )

                try:
                    # Validate input
                    if not step.validate_input(current_data):
                        raise ValueError(f"Input validation failed for {step.name}")

                    # Execute with retries
                    for attempt in range(step.retry_count + 1):
                        try:
                            current_data = step.execute(current_data, context)
                            break
                        except Exception as e:
                            if attempt == step.retry_count:
                                raise
                            logger.warning(
                                f"Step {step.name} failed, retrying",
                                attempt=attempt + 1,
                                error=str(e),
                            )
                            time.sleep(2**attempt)  # Exponential backoff

                    # Validate output
                    if not step.validate_output(current_data):
                        raise ValueError(f"Output validation failed for {step.name}")

                    step_status = PipelineStatus.SUCCESS

                except Exception as e:
                    import traceback

                    step_status = PipelineStatus.FAILED
                    step_error = str(e)
                    step_traceback = traceback.format_exc()

                    logger.error(
                        f"Step {step.name} failed",
                        error=step_error,
                    )

                    raise

                finally:
                    step_end = datetime.now()

                    # Get output shape if possible
                    output_shape = None
                    if hasattr(current_data, "shape"):
                        output_shape = current_data.shape
                    elif isinstance(current_data, dict):
                        output_shape = (len(current_data),)

                    step_result = StepResult(
                        step_name=step.name,
                        step_type=step.step_type,
                        status=step_status,
                        start_time=step_start,
                        end_time=step_end,
                        duration_seconds=(step_end - step_start).total_seconds(),
                        output_shape=output_shape,
                        metrics=context.get("metrics", {}),
                        error_message=step_error,
                        error_traceback=step_traceback,
                    )
                    step_results.append(step_result)

            overall_status = PipelineStatus.SUCCESS

        except Exception as e:
            overall_status = PipelineStatus.FAILED
            logger.error(f"Pipeline {self.name} failed", error=str(e))

        finally:
            end_time = datetime.now()

        # Aggregate metrics
        aggregate_metrics = {}
        for result in step_results:
            for key, value in result.metrics.items():
                aggregate_metrics[f"{result.step_name}_{key}"] = value

        pipeline_result = PipelineResult(
            pipeline_name=self.name,
            run_id=run_id,
            status=overall_status,
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=(end_time - start_time).total_seconds(),
            step_results=step_results,
            final_output=current_data if overall_status == PipelineStatus.SUCCESS else None,
            aggregate_metrics=aggregate_metrics,
        )

        logger.info(
            "Pipeline completed",
            pipeline=self.name,
            run_id=run_id,
            status=overall_status.value,
            duration=pipeline_result.total_duration_seconds,
        )

        return pipeline_result


class PipelineRegistry:
    """Registry for managing ML pipelines."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pipelines = {}
            cls._instance._runs = {}
        return cls._instance

    def register(self, pipeline: Pipeline) -> None:
        """Register a pipeline."""
        self._pipelines[pipeline.name] = pipeline
        logger.info(f"Registered pipeline: {pipeline.name}")

    def get(self, name: str) -> Pipeline | None:
        """Get pipeline by name."""
        return self._pipelines.get(name)

    def list_pipelines(self) -> list[str]:
        """List all registered pipelines."""
        return list(self._pipelines.keys())

    def record_run(self, result: PipelineResult) -> None:
        """Record a pipeline run."""
        if result.pipeline_name not in self._runs:
            self._runs[result.pipeline_name] = []
        self._runs[result.pipeline_name].append(result)

    def get_run_history(self, pipeline_name: str, limit: int = 10) -> list[PipelineResult]:
        """Get run history for a pipeline."""
        runs = self._runs.get(pipeline_name, [])
        return sorted(runs, key=lambda r: r.start_time, reverse=True)[:limit]


# Factory functions for common pipelines


def create_training_pipeline(
    name: str,
    model_factory: Callable,
    target_column: str,
    validation_checks: list[Callable] | None = None,
    transformations: list[Callable] | None = None,
) -> Pipeline:
    """Create a standard training pipeline."""
    pipeline = Pipeline(name)

    # Add validation step
    if validation_checks:
        validation = DataValidationStep(checks=validation_checks)
        pipeline.add_step(validation)

    # Add feature engineering
    if transformations:
        fe_step = FeatureEngineeringStep(transformations=transformations)
        pipeline.add_step(fe_step)

    # Add training step
    training = ModelTrainingStep(
        model_factory=model_factory,
        target_column=target_column,
    )
    pipeline.add_step(training)

    return pipeline


def create_inference_pipeline(
    name: str,
    model: Any,
    feature_columns: list[str],
    transformations: list[Callable] | None = None,
) -> Pipeline:
    """Create a standard inference pipeline."""
    pipeline = Pipeline(name)

    # Add feature engineering
    if transformations:
        fe_step = FeatureEngineeringStep(transformations=transformations)
        pipeline.add_step(fe_step)

    # Add inference step
    inference = InferenceStep(
        model=model,
        feature_columns=feature_columns,
    )
    pipeline.add_step(inference)

    return pipeline
