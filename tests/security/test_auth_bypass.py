"""Authentication bypass tests."""

import pytest
from httpx import AsyncClient


class TestAuthBypass:
    """Authentication bypass vulnerability tests."""

    @pytest.mark.asyncio
    async def test_missing_auth_header(self, client: AsyncClient):
        """Test request without authentication."""
        response = await client.post(
            "/v1/uplift/predict",
            json={"customer_ids": ["c1"]},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_api_key(self, client: AsyncClient):
        """Test request with invalid API key."""
        response = await client.post(
            "/v1/uplift/predict",
            json={"customer_ids": ["c1"]},
            headers={"X-API-Key": "invalid_key"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_bearer_token(self, client: AsyncClient):
        """Test request with empty bearer token."""
        response = await client.post(
            "/v1/uplift/predict",
            json={"customer_ids": ["c1"]},
            headers={"Authorization": "Bearer "},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_malformed_jwt(self, client: AsyncClient):
        """Test request with malformed JWT."""
        response = await client.post(
            "/v1/uplift/predict",
            json={"customer_ids": ["c1"]},
            headers={"Authorization": "Bearer not.a.valid.jwt"},
        )

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_expired_jwt(self, client: AsyncClient):
        """Test request with expired JWT."""
        # Would need to generate expired token
        pass

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, client: AsyncClient):
        """Test that health endpoint works without auth."""
        response = await client.get("/health")

        assert response.status_code == 200

