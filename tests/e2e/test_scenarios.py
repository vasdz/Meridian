"""End-to-end test scenarios."""

import pytest
from httpx import AsyncClient


class TestUpliftScenario:
    """E2E test for complete uplift prediction scenario."""

    @pytest.mark.asyncio
    async def test_full_uplift_workflow(self, client: AsyncClient, auth_headers):
        """Test complete uplift prediction workflow."""
        # 1. Request uplift predictions
        response = await client.post(
            "/v1/uplift/predict",
            json={
                "customer_ids": ["customer-001", "customer-002", "customer-003"],
                "model_id": "causal_forest_v1",
                "return_confidence_intervals": True,
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        predictions = response.json()["predictions"]

        # 2. Verify predictions have required fields
        for pred in predictions:
            assert "customer_id" in pred
            assert "cate" in pred

        # 3. Get list of models
        response = await client.get(
            "/v1/uplift/models",
            headers=auth_headers,
        )

        assert response.status_code == 200


class TestExperimentScenario:
    """E2E test for complete experiment workflow."""

    @pytest.mark.asyncio
    async def test_full_experiment_workflow(
        self, client: AsyncClient, auth_headers, sample_experiment_data
    ):
        """Test complete experiment lifecycle."""
        # 1. Create experiment
        response = await client.post(
            "/v1/experiments",
            json=sample_experiment_data,
            headers=auth_headers,
        )

        assert response.status_code == 201
        experiment = response.json()
        experiment_id = experiment["id"]

        # 2. Get experiment
        response = await client.get(
            f"/v1/experiments/{experiment_id}",
            headers=auth_headers,
        )

        assert response.status_code == 200

        # 3. List experiments
        response = await client.get(
            "/v1/experiments",
            headers=auth_headers,
        )

        assert response.status_code == 200


class TestHealthScenario:
    """E2E test for health checks."""

    @pytest.mark.asyncio
    async def test_health_endpoints(self, client: AsyncClient):
        """Test health check endpoints."""
        # Health check (no auth required)
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        # Readiness check
        response = await client.get("/ready")
        assert response.status_code == 200

