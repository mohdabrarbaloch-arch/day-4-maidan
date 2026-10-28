"""API tests: auth, generation, history, error paths."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.db import Base, engine
from app.main import app


@pytest.fixture()
def client():
    """Fresh schema per test so ordering never matters."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth(client):
    res = client.post(
        "/api/auth/register",
        json={"email": "test@maidan.dev", "display_name": "Tester", "password": "password123"},
    )
    assert res.status_code == 201
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["models"] == 3


def test_models_catalogue(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    assert len(res.json()) == 3


def test_backends_reports_no_cloud_without_keys(client):
    body = client.get("/api/backends").json()
    assert body["local"] is True
    assert body["cloud"] is False
    assert "OPENAI_API_KEY" in body["message"]


def test_root_serves_frontend(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Maidan" in res.text


def test_unknown_path_returns_json_404(client):
    """A 200 SPA shell on a dead route hides broken deployments."""
    res = client.get("/definitely-not-a-route")
    assert res.status_code == 404
    assert res.json()["error"] == "not_found"


def test_register_returns_token(client):
    res = client.post(
        "/api/auth/register",
        json={"email": "new@maidan.dev", "display_name": "New", "password": "password123"},
    )
    assert res.status_code == 201
    assert res.json()["access_token"]
    assert res.json()["user"]["display_name"] == "New"


def test_duplicate_email_conflicts(client, auth):
    res = client.post(
        "/api/auth/register",
        json={"email": "test@maidan.dev", "display_name": "Dup", "password": "password123"},
    )
    assert res.status_code == 409


def test_short_password_rejected(client):
    res = client.post(
        "/api/auth/register",
        json={"email": "x@maidan.dev", "display_name": "X", "password": "short"},
    )
    assert res.status_code == 422


def test_login_wrong_password(client, auth):
    res = client.post(
        "/api/auth/login", json={"email": "test@maidan.dev", "password": "wrongpassword"}
    )
    assert res.status_code == 401


def test_login_succeeds(client, auth):
    res = client.post(
        "/api/auth/login", json={"email": "test@maidan.dev", "password": "password123"}
    )
    assert res.status_code == 200


def test_me_requires_auth(client):
    assert client.get("/api/auth/me").status_code == 401


def test_me_returns_user(client, auth):
    res = client.get("/api/auth/me", headers=auth)
    assert res.status_code == 200
    assert res.json()["email"] == "test@maidan.dev"


def test_generate_requires_auth(client):
    res = client.post("/api/generate", json={"prompt": "hello", "model_id": "maidan-tiny"})
    assert res.status_code == 401


def test_generate_returns_completion(client, auth):
    res = client.post(
        "/api/generate",
        headers=auth,
        json={
            "prompt": "The context window is",
            "model_id": "maidan-tiny",
            "max_tokens": 20,
            "seed": 5,
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["completion"]
    assert body["tokens_generated"] > 0
    assert body["seed"] == 5
    assert body["run"]["id"]


def test_generate_unknown_model_404(client, auth):
    res = client.post("/api/generate", headers=auth, json={"prompt": "hi", "model_id": "nope"})
    assert res.status_code == 404


def test_cloud_without_key_returns_503(client, auth):
    """The cloud toggle must fail honestly, never fabricate an answer."""
    res = client.post(
        "/api/generate",
        headers=auth,
        json={"prompt": "hi", "model_id": "cloud-gpt-4o-mini", "backend": "cloud"},
    )
    assert res.status_code == 503
    assert "API_KEY" in res.json()["detail"]


def test_blank_prompt_rejected(client, auth):
    res = client.post(
        "/api/generate", headers=auth, json={"prompt": "   ", "model_id": "maidan-tiny"}
    )
    assert res.status_code == 422


def test_temperature_out_of_range_rejected(client, auth):
    res = client.post(
        "/api/generate",
        headers=auth,
        json={"prompt": "hi", "model_id": "maidan-tiny", "temperature": 5.0},
    )
    assert res.status_code == 422


def test_runs_are_listed_and_deleted(client, auth):
    client.post(
        "/api/generate",
        headers=auth,
        json={"prompt": "The model reads", "model_id": "maidan-tiny", "max_tokens": 10, "seed": 1},
    )
    runs = client.get("/api/runs", headers=auth).json()
    assert len(runs) == 1

    run_id = runs[0]["id"]
    assert client.delete(f"/api/runs/{run_id}", headers=auth).status_code == 200
    assert client.get("/api/runs", headers=auth).json() == []


def test_runs_are_scoped_to_the_owner(client, auth):
    client.post(
        "/api/generate",
        headers=auth,
        json={"prompt": "The model reads", "model_id": "maidan-tiny", "max_tokens": 10, "seed": 1},
    )
    other = client.post(
        "/api/auth/register",
        json={"email": "other@maidan.dev", "display_name": "Other", "password": "password123"},
    ).json()
    other_headers = {"Authorization": f"Bearer {other['access_token']}"}
    assert client.get("/api/runs", headers=other_headers).json() == []


def test_stream_endpoint_emits_sse(client, auth):
    with client.stream(
        "POST",
        "/api/generate/stream",
        headers=auth,
        json={"prompt": "The model reads", "model_id": "maidan-tiny", "max_tokens": 8, "seed": 2},
    ) as res:
        assert res.status_code == 200
        body = "".join(res.iter_text())
    assert "data: " in body
    assert '"type": "token"' in body
    assert '"type": "done"' in body
