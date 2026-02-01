"""API fuzzing tests."""

import pytest
import random
import string
from httpx import AsyncClient


def random_string(length: int) -> str:
    """Generate random string."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def random_unicode() -> str:
    """Generate random unicode string."""
    return "".join(chr(random.randint(0, 0xFFFF)) for _ in range(10))


class TestAPIFuzzing:
    """API fuzzing tests."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("_", range(20))
    async def test_random_customer_ids(
        self, client: AsyncClient, auth_headers, _
    ):
        """Fuzz customer_ids with random data."""
        customer_ids = [random_string(random.randint(1, 100)) for _ in range(5)]

        response = await client.post(
            "/v1/uplift/predict",
            json={"customer_ids": customer_ids},
            headers=auth_headers,
        )

        # Should not crash (200 or 4xx expected)
        assert response.status_code in [200, 400, 422]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("_", range(10))
    async def test_random_unicode_input(
        self, client: AsyncClient, auth_headers, _
    ):
        """Fuzz with random unicode."""
        response = await client.post(
            "/v1/experiments",
            json={
                "name": random_unicode()[:50],
                "hypothesis": random_unicode()[:200],
            },
            headers=auth_headers,
        )

        # Should handle gracefully
        assert response.status_code in [200, 201, 400, 422]

    @pytest.mark.asyncio
    async def test_deeply_nested_json(self, client: AsyncClient, auth_headers):
        """Test deeply nested JSON."""
        nested = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}

        response = await client.post(
            "/v1/uplift/predict",
            json={"customer_ids": ["c1"], "extra": nested},
            headers=auth_headers,
        )

        # Should not crash
        assert response.status_code in [200, 400, 422]

    @pytest.mark.asyncio
    async def test_large_numbers(self, client: AsyncClient, auth_headers):
        """Test very large numbers."""
        response = await client.post(
            "/v1/pricing/optimize",
            json={
                "products": [
                    {
                        "product_id": "p1",
                        "current_price": 10**100,
                    }
                ],
            },
            headers=auth_headers,
        )

        # Should handle gracefully
        assert response.status_code in [200, 400, 422]

    @pytest.mark.asyncio
    async def test_negative_values(self, client: AsyncClient, auth_headers):
        """Test negative values where positive expected."""
        response = await client.post(
            "/v1/pricing/optimize",
            json={
                "products": [
                    {
                        "product_id": "p1",
                        "current_price": -100,
                    }
                ],
            },
            headers=auth_headers,
        )

        assert response.status_code in [400, 422]

