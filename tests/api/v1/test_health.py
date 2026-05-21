import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_returns_ok(client: AsyncClient):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "app" in data
    assert "version" in data
    assert "uptime_seconds" in data


@pytest.mark.asyncio
async def test_health_response_headers(client: AsyncClient):
    response = await client.get("/api/v1/health")
    assert "x-request-id" in response.headers
    assert "x-process-time" in response.headers
