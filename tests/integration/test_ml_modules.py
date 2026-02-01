"""Integration tests for new ML modules."""

import numpy as np
import pandas as pd
import pytest


class TestDemandForecaster:
    """Tests for demand forecasting module."""

    def test_lightgbm_forecaster(self):
        """Test LightGBM-based forecaster."""
        from meridian.infrastructure.ml.forecasting.demand_forecaster import (
            LightGBMForecaster,
            ForecastConfig,
        )

        # Generate sample time series
        np.random.seed(42)
        n = 100
        timestamps = pd.date_range("2024-01-01", periods=n, freq="D")
        y = 100 + np.arange(n) * 0.5 + np.sin(np.arange(n) / 7 * 2 * np.pi) * 20 + np.random.randn(n) * 5

        config = ForecastConfig(horizon=14, quantiles=[0.1, 0.5, 0.9])
        model = LightGBMForecaster(config)

        model.fit(y, timestamps.values)
        point_forecast, quantile_forecasts = model.predict(14)

        assert len(point_forecast) == 14
        assert 0.1 in quantile_forecasts
        assert 0.5 in quantile_forecasts
        assert 0.9 in quantile_forecasts
        assert all(point_forecast >= 0)

    def test_ensemble_forecaster(self):
        """Test ensemble forecaster."""
        from meridian.infrastructure.ml.forecasting.demand_forecaster import (
            EnsembleForecaster,
            ForecastConfig,
        )

        np.random.seed(42)
        n = 100
        timestamps = pd.date_range("2024-01-01", periods=n, freq="D")
        y = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 5

        config = ForecastConfig(horizon=7)
        model = EnsembleForecaster(config)

        model.fit(y, timestamps.values)
        point_forecast, _ = model.predict(7)

        assert len(point_forecast) == 7
        assert model.get_model_name() == "ensemble"


class TestPricingOptimizer:
    """Tests for pricing optimization module."""

    def test_elasticity_estimation(self):
        """Test price elasticity estimation."""
        from meridian.infrastructure.ml.pricing.price_optimizer import (
            PriceElasticityEstimator,
            ElasticityModel,
        )

        np.random.seed(42)

        # Generate data with known elasticity
        true_elasticity = -1.5
        prices = np.random.uniform(5, 15, 100)
        quantities = 1000 * (prices / 10) ** true_elasticity + np.random.randn(100) * 10
        quantities = np.maximum(quantities, 1)

        estimator = PriceElasticityEstimator(model=ElasticityModel.LOG_LOG)
        result = estimator.estimate(prices, quantities, "test_product")

        # Should be close to true elasticity
        assert -2.5 < result.elasticity < -0.5
        assert result.elasticity_std > 0
        assert result.r_squared > 0.5

    def test_price_optimization(self):
        """Test price optimization."""
        from meridian.infrastructure.ml.pricing.price_optimizer import (
            PriceOptimizer,
            OptimizationObjective,
        )

        optimizer = PriceOptimizer(objective=OptimizationObjective.PROFIT)

        # Set elasticity manually
        optimizer.set_elasticity("product_1", elasticity=-1.5)

        result = optimizer.optimize(
            product_id="product_1",
            current_price=100,
            current_demand=1000,
            cost=60,
        )

        assert result.optimal_price > 0
        assert result.predicted_demand > 0
        assert result.predicted_profit > 0

    def test_sensitivity_analysis(self):
        """Test price sensitivity analysis."""
        from meridian.infrastructure.ml.pricing.price_optimizer import PriceOptimizer

        optimizer = PriceOptimizer()
        optimizer.set_elasticity("product_1", elasticity=-1.5)

        df = optimizer.sensitivity_analysis(
            product_id="product_1",
            current_price=100,
            current_demand=1000,
            cost=60,
        )

        assert len(df) == 20
        assert "price" in df.columns
        assert "profit" in df.columns
        assert "margin_pct" in df.columns


class TestMLPipeline:
    """Tests for ML pipeline module."""

    def test_pipeline_execution(self):
        """Test basic pipeline execution."""
        from meridian.infrastructure.ml.pipeline import (
            Pipeline,
            FunctionStep,
            StepType,
            PipelineStatus,
        )

        # Create simple pipeline
        def transform_data(data, **kwargs):
            data["processed"] = data["value"] * 2
            return data

        pipeline = Pipeline("test_pipeline")
        pipeline.add_step(FunctionStep("transform", transform_data, StepType.TRANSFORM))

        input_data = pd.DataFrame({"value": [1, 2, 3]})
        result = pipeline.run(input_data)

        assert result.status == PipelineStatus.SUCCESS
        assert len(result.step_results) == 1
        assert result.final_output is not None

    def test_pipeline_with_validation(self):
        """Test pipeline with validation step."""
        from meridian.infrastructure.ml.pipeline import (
            Pipeline,
            DataValidationStep,
        )

        def check_not_empty(df):
            return len(df) > 0, f"DataFrame has {len(df)} rows"

        validation = DataValidationStep()
        validation.add_check(check_not_empty)

        pipeline = Pipeline("validation_test")
        pipeline.add_step(validation)

        result = pipeline.run(pd.DataFrame({"a": [1, 2, 3]}))

        assert result.status.value == "success"


class TestMonitoring:
    """Tests for ML monitoring module."""

    def test_drift_detector(self):
        """Test drift detection."""
        from meridian.infrastructure.monitoring import DriftDetector

        np.random.seed(42)

        # Reference data
        reference = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, 1000),
            "feature_2": np.random.normal(10, 2, 1000),
        })

        # Current data with drift
        current = pd.DataFrame({
            "feature_1": np.random.normal(0.5, 1, 500),  # Shifted mean
            "feature_2": np.random.normal(10, 2, 500),   # No drift
        })

        detector = DriftDetector(drift_threshold=0.1, method="psi")
        detector.set_reference(reference)

        results = detector.detect_drift(current)

        assert len(results) == 2
        # Feature 1 should show drift
        feature_1_result = next(r for r in results if r.feature_name == "feature_1")
        assert feature_1_result.drift_score > 0

    def test_prediction_monitor(self):
        """Test prediction monitoring."""
        from meridian.infrastructure.monitoring import PredictionMonitor

        monitor = PredictionMonitor("test_model")
        monitor.set_reference_stats(mean=0.5, std=0.1)

        # Record normal predictions
        for _ in range(10):
            alert = monitor.record_prediction(0.48, latency_ms=20)
            assert alert is None

        # Record anomalous prediction
        alert = monitor.record_prediction(10.0, latency_ms=20)  # Very anomalous
        assert alert is not None
        assert alert.severity.value == "warning"

    def test_sla_monitor(self):
        """Test SLA monitoring."""
        from meridian.infrastructure.monitoring import SLAMonitor

        monitor = SLAMonitor("test_model", latency_sla_ms=100, error_rate_sla=0.01)

        # Record normal requests
        for _ in range(100):
            monitor.record_request(latency_ms=50, is_error=False)

        report = monitor.get_sla_report()

        assert report["total_requests"] == 100
        assert report["error_count"] == 0
        assert report["error_sla_compliance"] is True


class TestCausalDiscovery:
    """Tests for causal discovery module."""

    def test_pc_algorithm(self):
        """Test PC algorithm for causal discovery."""
        from meridian.domain.services.causal_discovery import PCAlgorithm

        np.random.seed(42)
        n = 500

        # Generate causal data: X -> Y -> Z
        X = np.random.randn(n)
        Y = 0.8 * X + np.random.randn(n) * 0.3
        Z = 0.6 * Y + np.random.randn(n) * 0.4

        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        pc = PCAlgorithm(alpha=0.05)
        graph = pc.fit(data)

        assert len(graph.nodes) == 3
        assert len(graph.edges) > 0  # Should find some edges


class TestAttribution:
    """Tests for attribution module."""

    def test_shapley_attribution(self):
        """Test Shapley value attribution."""
        from meridian.domain.services.attribution import ShapleyAttribution

        conversion_paths = [
            ["email", "push"],
            ["push", "email", "push"],
            ["email"],
            ["push", "sms"],
            ["sms", "email"],
        ]

        shapley = ShapleyAttribution()
        report = shapley.calculate(conversion_paths)

        assert report.total_conversions == 5
        assert len(report.channel_attributions) == 3

        # All attributions should sum to 1
        total_attribution = sum(a.attribution_value for a in report.channel_attributions)
        assert abs(total_attribution - 1.0) < 0.01

    def test_simple_attribution(self):
        """Test simple attribution models."""
        from meridian.domain.services.attribution import SimpleAttribution

        paths = [
            ["email", "push", "sms"],
            ["push", "email"],
        ]

        last_touch = SimpleAttribution.last_touch(paths)
        first_touch = SimpleAttribution.first_touch(paths)
        linear = SimpleAttribution.linear(paths)

        assert last_touch.model_name == "last_touch"
        assert first_touch.model_name == "first_touch"
        assert linear.model_name == "linear"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

