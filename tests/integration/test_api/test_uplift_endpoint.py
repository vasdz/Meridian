"""Integration tests for uplift API endpoint."""

import pytest
from httpx import AsyncClient


class TestUpliftEndpoint:
    """Integration tests for /v1/uplift endpoints."""

    @pytest.mark.asyncio
    async def test_predict_uplift_success(self, client: AsyncClient, auth_headers):
        """Test successful uplift prediction."""
        response = await client.post(
            "/v1/uplift/predict",
            json={
                "customer_ids": ["c1", "c2", "c3"],
                "model_id": "causal_forest_v1",
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3

    @pytest.mark.asyncio
    async def test_predict_uplift_with_confidence_intervals(
        self, client: AsyncClient, auth_headers
    ):
        """Test uplift prediction with confidence intervals."""
        response = await client.post(
            "/v1/uplift/predict",
            json={
                "customer_ids": ["c1"],
                "return_confidence_intervals": True,
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["predictions"][0].get("confidence_interval") is not None

    @pytest.mark.asyncio
    async def test_predict_uplift_unauthorized(self, client: AsyncClient):
        """Test that unauthorized request is rejected."""
        response = await client.post(
            "/v1/uplift/predict",
            json={"customer_ids": ["c1"]},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_predict_uplift_empty_customers(self, client: AsyncClient, auth_headers):
        """Test validation of empty customer list."""
        response = await client.post(
            "/v1/uplift/predict",
            json={"customer_ids": []},
            headers=auth_headers,
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_uplift_invalid_customer_id(self, client: AsyncClient, auth_headers):
        """Test validation of invalid customer ID format."""
        response = await client.post(
            "/v1/uplift/predict",
            json={"customer_ids": ["<script>alert('xss')</script>"]},
            headers=auth_headers,
        )

        assert response.status_code == 422
