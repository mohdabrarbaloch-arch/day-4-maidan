"""Application settings, loaded from the environment only.

Nothing here has a hardcoded secret. If SECRET_KEY is missing outside of a
local dev run, the app refuses to sign tokens with a guessable default.
"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def running_on_vercel() -> bool:
    """Serverless detection — used to relocate SQLite to /tmp and to relax
    a couple of filesystem assumptions that do not hold there."""
    return bool(os.getenv("VERCEL")) or bool(os.getenv("VERCEL_ENV"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_name: str = "Maidan"
    environment: str = Field(default="development")

    secret_key: str = Field(default="")
    access_token_expire_minutes: int = 60 * 24 * 7
    algorithm: str = "HS256"

    database_url: str = "sqlite:///./maidan.db"

    cors_origins: str = "*"

    # Rate limiting. Off by default in tests so a suite can hammer the API
    # without tripping its own guard.
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60

    # Optional cloud backend. Absent keys mean the cloud toggle reports itself
    # as unavailable instead of pretending to answer.
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"

    max_prompt_chars: int = 8000
    max_new_tokens: int = 400

    def resolved_secret_key(self) -> str:
        if self.secret_key:
            return self.secret_key
        if self.environment in {"development", "test"}:
            # Ephemeral per-process key: tokens do not survive a restart, which
            # is the correct behaviour when no key is configured.
            #
            # It must be generated ONCE and reused. Returning a fresh random
            # string on every call means the key that signed a token is not the
            # key that verifies it, so every token is instantly invalid — which
            # looks exactly like a broken auth system.
            cached = getattr(self, "_ephemeral_key", None)
            if cached is None:
                cached = secrets.token_urlsafe(48)
                object.__setattr__(self, "_ephemeral_key", cached)
            return cached
        raise RuntimeError(
            "SECRET_KEY is not set. Refusing to start in a non-development "
            "environment with an auto-generated signing key."
        )

    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def cloud_backends(self) -> dict[str, dict[str, str]]:
        """Which cloud providers are actually configured right now."""
        found: dict[str, dict[str, str]] = {}
        if self.openai_api_key:
            found["openai"] = {
                "base_url": self.openai_base_url,
                "api_key": self.openai_api_key,
            }
        if self.groq_api_key:
            found["groq"] = {"base_url": self.groq_base_url, "api_key": self.groq_api_key}
        return found


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
