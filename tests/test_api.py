"""API contract tests: auth, generation, history, ownership isolation."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
 def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-not-for-production")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    with TestClient(app) as c:
        yield c


def register(client, email="a@b.com", password="password123"):
    res = client.post(
        "/api/auth/register",
        json={"email": email, "display_name": "Tester", "password": password},
    )
    assert res.status_code == 201, res.text
    return res.json()["access_token"]


def auth(token):
    return {"Authorization": f"Bearer {token}"}


# ── health & catalogue ──────────────────────────────────────────────


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["models"] >= 1


def test_models_returns_bare_array(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    models = res.json()
    assert isinstance(models, list)
    assert any(m["id"] == "maidan-tiny" for m in models)


def test_models_report_a_literal_context_window(client):
    """A trigram model conditions on two tokens. An earlier draft claimed 512."""
    models = client.get("/api/models").json()
    tiny = next(m for m in models if m["id"] == "maidan-tiny")
    assert tiny["context_window"] == 2


def test_backends_reports_cloud_unavailable_without_keys(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    body = client.get("/api/backends").json()
    assert body["local"] is True
    assert body["cloud"] is False
    assert "OPENAI_API_KEY" in body["message"]


# ── auth ────────────────────────────────────────────────────────────


def test_register_returns_token_and_user(client):
    res = client.post(
        "/api/auth/register",
        json={"email": "new@b.com", "display_name": "New", "password": "password123"},
    )
    assert res.status_code == 201
    body = res.json()
    assert body["access_token"]
    assert body["user"]["email"] == "new@b.com"
    assert "password" not in body["user"]
    assert "password_hash" not in body["user"]


def test_duplicate_email_is_409(client):
    register(client, "dup@b.com")
    res = client.post(
        "/api/auth/register",
        json={"email": "dup@b.com", "display_name": "Dup", "password": "password123"},
    )
    assert res.status_code == 409


def test_email_is_case_insensitive(client):
    register(client, "Case@B.com")
    res = client.post(
        "/api/auth/login", json={"email": "case@b.com", "password": "password123"}
    )
    assert res.status_code == 200


def test_short_password_is_422(client):
    res = client.post(
        "/api/auth/register",
        json={"email": "x@b.com", "display_name": "X", "password": "short"},
    )
    assert res.status_code == 422


def test_login_wrong_password_is_401(client):
    register(client, "pw@b.com")
    res = client.post(
        "/api/auth/login", json={"email": "pw@b.com", "password": "wrongpassword"}
    )
    assert res.status_code == 401


def test_login_unknown_email_gives_the_same_message(client):
    """Distinguishing the two would let an attacker enumerate accounts."""
    register(client, "known@b.com")
    wrong_pw = client.post(
        "/api/auth/login", json={"email": "known@b.com", "password": "wrongpassword"}
    )
    unknown = client.post(
        "/api/auth/login", json={"email": "nobody@b.com", "password": "wrongpassword"}
    )
    assert wrong_pw.status_code == unknown.status_code == 401
    assert wrong_pw.json()["detail"] == unknown.json()["detail"]


def test_me_requires_a_token(client):
    assert client.get("/api/auth/me").status_code == 401


def test_me_returns_the_current_user(client):
    token = register(client, "me@b.com")
    body = client.get("/api/auth/me", headers=auth(token)).json()
    assert body["email"] == "me@b.com"


def test_garbage_token_is_401(client):
    res = client.get("/api/auth/me", headers=auth("not-a-jwt"))
    assert res.status_code == 401


# ── generation ──────────────────────────────────────────────────────


def test_generate_requires_auth(client):
    res = client.post("/api/generate", json={"prompt": "hello"})
    assert res.status_code == 401


def test_generate_returns_a_completion_and_a_nested_run(client):
    token = register(client)
    res = client.post(
        "/api/generate",
        json={"prompt": "The context window is", "max_tokens": 20, "seed": 42},
        headers=auth(token),
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["completion"]
    assert body["tokens_generated"] > 0
    assert body["stop_reason"] in ("eos", "max_tokens")
    # The saved row is nested, not flattened — there is no top-level run_id.
    assert "run_id" not in body
    assert body["run"]["id"] > 0
    assert body["run"]["seed"] == 42


def test_generate_is_reproducible_with_a_seed(client):
    token = register(client)
    payload = {"prompt": "A good test", "max_tokens": 25, "seed": 1234}
    first = client.post("/api/generate", json=payload, headers=auth(token)).json()
    second = client.post("/api/generate", json=payload, headers=auth(token)).json()
    assert first["completion"] == second["completion"]


def test_different_seeds_diverge(client):
    token = register(client)
    base = {"prompt": "A good test", "max_tokens": 40, "temperature": 1.2}
    a = client.post("/api/generate", json={**base, "seed": 1}, headers=auth(token)).json()
    b = client.post("/api/generate", json={**base, "seed": 2}, headers=auth(token)).json()
    assert a["completion"] != b["completion"]


def test_unknown_model_is_404(client):
    token = register(client)
    res = client.post(
        "/api/generate", json={"prompt": "hi", "model_id": "nope"}, headers=auth(token)
    )
    assert res.status_code == 404


def test_empty_prompt_is_422(client):
    token = register(client)
    res = client.post("/api/generate", json={"prompt": "   "}, headers=auth(token))
    assert res.status_code == 422


def test_out_of_range_temperature_is_422(client):
    token = register(client)
    res = client.post(
        "/api/generate", json={"prompt": "hi", "temperature": 9.0}, headers=auth(token)
    )
    assert res.status_code == 422


def test_cloud_without_a_key_is_503_not_a_fake_reply(client, monkeypatch):
    """A fabricated answer would be indistinguishable from a real one in history."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    token = register(client)
    res = client.post(
        "/api/generate",
        json={"prompt": "hi", "backend": "cloud", "model_id": "cloud-gpt-4o-mini"},
        headers=auth(token),
    )
    assert res.status_code == 503
    assert "OPENAI_API_KEY" in res.json()["detail"]


# ── streaming ───────────────────────────────────────────────────────


def test_stream_emits_typed_frames(client):
    token = register(client)
    with client.stream(
        "POST",
        "/api/generate/stream",
        json={"prompt": "The model reads", "max_tokens": 8, "seed": 3},
        headers=auth(token),
    ) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        body = "".join(res.iter_text())

    assert '"type": "start"' in body
    assert '"type": "token"' in body
    assert '"type": "done"' in body


def test_stream_done_frame_carries_the_run_id(client):
    token = register(client)
    with client.stream(
        "POST",
        "/api/generate/stream",
        json={"prompt": "The model reads", "max_tokens": 6, "seed": 3},
        headers=auth(token),
    ) as res:
        body = "".join(res.iter_text())

    done = [line for line in body.split("\n") if '"type": "done"' in line]
    assert len(done) == 1
    assert '"run_id":' in done[0]


# ── history & ownership ─────────────────────────────────────────────


def test_runs_returns_a_bare_array(client):
    token = register(client)
    client.post(
        "/api/generate", json={"prompt": "hi", "max_tokens": 5}, headers=auth(token)
    )
    runs = client.get("/api/runs", headers=auth(token)).json()
    assert isinstance(runs, list)
    assert len(runs) == 1


def test_runs_are_newest_first(client):
    token = register(client)
    for prompt in ("first", "second", "third"):
        client.post(
            "/api/generate", json={"prompt": prompt, "max_tokens": 4}, headers=auth(token)
        )
    runs = client.get("/api/runs", headers=auth(token)).json()
    assert [r["prompt"] for r in runs] == ["third", "second", "first"]


def test_runs_are_scoped_to_the_owner(client):
    alice = register(client, "alice@b.com")
    bob = register(client, "bob@b.com")
    client.post(
        "/api/generate", json={"prompt": "alice only", "max_tokens": 4}, headers=auth(alice)
    )
    assert client.get("/api/runs", headers=auth(bob)).json() == []


def test_foreign_run_id_is_404_not_403(client):
    """403 would confirm the row exists — an enumeration oracle."""
    alice = register(client, "a2@b.com")
    bob = register(client, "b2@b.com")
    run = client.post(
        "/api/generate", json={"prompt": "secret", "max_tokens": 4}, headers=auth(alice)
    ).json()["run"]

    assert client.get(f"/api/runs/{run['id']}", headers=auth(bob)).status_code == 404
    assert client.delete(f"/api/runs/{run['id']}", headers=auth(bob)).status_code == 404


def test_delete_returns_the_id(client):
    token = register(client)
    run = client.post(
        "/api/generate", json={"prompt": "bye", "max_tokens": 4}, headers=auth(token)
    ).json()["run"]
    res = client.delete(f"/api/runs/{run['id']}", headers=auth(token))
    assert res.status_code == 200
    assert res.json() == {"deleted": run["id"]}
    assert client.get(f"/api/runs/{run['id']}", headers=auth(token)).status_code == 404


def test_limit_is_clamped(client):
    token = register(client)
    res = client.get("/api/runs?limit=9999", headers=auth(token))
    assert res.status_code == 200
