import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_rejects_weak_password(client: AsyncClient):
    response = await client.post(
        "/api/v1/auth/register",
        json={"username": "testuser", "email": "test@example.com", "password": "weak"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"] == "Validation failed."


@pytest.mark.asyncio
async def test_register_rejects_invalid_email(client: AsyncClient):
    response = await client.post(
        "/api/v1/auth/register",
        json={"username": "testuser", "email": "not-an-email", "password": "SecurePass@1"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_register_rejects_short_username(client: AsyncClient):
    response = await client.post(
        "/api/v1/auth/register",
        json={"username": "ab", "email": "test@example.com", "password": "SecurePass@1"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_me_requires_auth(client: AsyncClient):
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_me_rejects_invalid_token(client: AsyncClient):
    response = await client.get(
        "/api/v1/auth/me", headers={"Authorization": "Bearer not.a.real.token"}
    )
    assert response.status_code == 401
