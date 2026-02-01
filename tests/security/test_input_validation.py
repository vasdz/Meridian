"""Input validation tests."""

import pytest
from httpx import AsyncClient


class TestInputValidation:
    """Input validation security tests."""

    XSS_PAYLOADS = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
        "<body onload=alert('XSS')>",
        "<iframe src='javascript:alert(1)'>",
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    async def test_xss_in_customer_id(self, client: AsyncClient, auth_headers, payload):
        """Test XSS payloads in customer_id."""
        response = await client.post(
            "/v1/uplift/predict",
            json={"customer_ids": [payload]},
            headers=auth_headers,
        )

        assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    async def test_xss_in_experiment_name(self, client: AsyncClient, auth_headers, payload):
        """Test XSS payloads in experiment name."""
        response = await client.post(
            "/v1/experiments",
            json={
                "name": payload,
                "hypothesis": "Test hypothesis for security",
            },
            headers=auth_headers,
        )

        assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_oversized_payload(self, client: AsyncClient, auth_headers):
        """Test rejection of oversized payloads."""
        large_list = [f"customer_{i}" for i in range(100001)]

        response = await client.post(
            "/v1/uplift/predict",
            json={"customer_ids": large_list},
            headers=auth_headers,
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_json(self, client: AsyncClient, auth_headers):
        """Test handling of invalid JSON."""
        response = await client.post(
            "/v1/uplift/predict",
            content="{invalid json",
            headers={**auth_headers, "Content-Type": "application/json"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_null_bytes(self, client: AsyncClient, auth_headers):
        """Test handling of null bytes."""
        response = await client.post(
            "/v1/uplift/predict",
            json={"customer_ids": ["customer\x00id"]},
            headers=auth_headers,
        )

        assert response.status_code in [400, 422]
