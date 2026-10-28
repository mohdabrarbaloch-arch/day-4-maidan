"""Packaging assertions.

These exist because this account has shipped a broken Vercel deploy before: the
legacy ``builds`` block resolved ``includeFiles`` relative to ``api/``, so the
``app`` package never reached the function bundle and every request died on
``ModuleNotFoundError``. The site still served a friendly page at ``/``, which is
what made it invisible for so long.

Each test below pins one of the three things that made that failure possible.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_vercel_config_uses_modern_functions_not_legacy_builds() -> None:
    config = json.loads((ROOT / "vercel.json").read_text())

    assert "builds" not in config, (
        "Legacy 'builds' resolves includeFiles relative to api/, so app/** "
        "becomes api/app/** and the package never ships."
    )
    assert "functions" in config
    assert config["functions"]["api/index.py"]["includeFiles"] == "app/**"


def test_vercel_rewrites_everything_to_the_entrypoint() -> None:
    config = json.loads((ROOT / "vercel.json").read_text())
    rewrites = config["rewrites"]

    assert any(r["destination"] == "/api/index.py" for r in rewrites)


def test_serverless_entrypoint_initialises_the_database_at_import() -> None:
    source = (ROOT / "api" / "index.py").read_text()

    # Vercel never fires lifespan events, so table creation must not live in a
    # startup handler.
    assert "init_db" in source or "from app.main import app" in source
    assert "sqlite:////tmp" in source, "SQLite must point at /tmp on Vercel"


def test_static_mount_is_guarded() -> None:
    source = (ROOT / "app" / "main.py").read_text()

    # An unguarded StaticFiles(directory=...) raises at import time when the
    # directory is absent from the bundle, which takes the whole app down.
    assert "is_dir()" in source or "exists()" in source


def test_no_hardcoded_secrets_in_source() -> None:
    offenders = []
    for path in (ROOT / "app").rglob("*.py"):
        text = path.read_text()
        for marker in ("sk-", "sk_live_", "AKIA"):
            if marker in text:
                offenders.append(f"{path.relative_to(ROOT)}: {marker}")

    assert not offenders, f"Possible hardcoded secret: {offenders}"


def test_env_example_documents_the_required_variables() -> None:
    text = (ROOT / ".env.example").read_text()

    for key in ("SECRET_KEY", "DATABASE_URL", "CORS_ORIGINS", "RATE_LIMIT_PER_MINUTE"):
        assert key in text, f"{key} missing from .env.example"


def test_dockerfile_runs_as_non_root() -> None:
    text = (ROOT / "Dockerfile").read_text()

    assert "USER " in text
    assert "USER root" not in text
