"""SQL injection tests."""

import pytest
from httpx import AsyncClient


class TestSQLInjection:
    """SQL injection vulnerability tests."""

    SQL_INJECTION_PAYLOADS = [
        "'; DROP TABLE customers; --",
        "1' OR '1'='1",
        "1; DELETE FROM users",
        "' UNION SELECT * FROM users --",
        "1' AND '1'='1",
        "admin'--",
        "'; EXEC xp_cmdshell('dir'); --",
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    async def test_uplift_customer_id_injection(self, client: AsyncClient, auth_headers, payload):
        """Test SQL injection via customer_ids."""
        response = await client.post(
            "/v1/uplift/predict",
            json={"customer_ids": [payload]},
            headers=auth_headers,
        )

        # Should be rejected by validation, not cause SQL error
        assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    async def test_experiment_name_injection(self, client: AsyncClient, auth_headers, payload):
        """Test SQL injection via experiment name."""
        response = await client.post(
            "/v1/experiments",
            json={
                "name": payload,
                "hypothesis": "Test hypothesis",
            },
            headers=auth_headers,
        )

        assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_query_parameter_injection(self, client: AsyncClient, auth_headers):
        """Test SQL injection via query parameters."""
        response = await client.get(
            "/v1/experiments",
            params={"status": "'; DROP TABLE experiments; --"},
            headers=auth_headers,
        )

        assert response.status_code in [400, 422]
