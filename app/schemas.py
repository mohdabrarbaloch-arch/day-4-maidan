"""Request/response models. Validation lives here, not in the routers."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

Backend = Literal["local", "cloud"]


class RegisterRequest(BaseModel):
    email: EmailStr
    display_name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("display_name")
    @classmethod
    def _trim_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("display_name cannot be blank")
        return cleaned


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    display_name: str
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ModelInfo(BaseModel):
    id: str
    name: str
    family: str
    params: str
    context_window: int
    vocab_size: int | None = None
    trigram_count: int | None = None
    description: str
    strengths: list[str]
    backend: Backend = "local"
    available: bool = True


class GenerateRequest(BaseModel):
    """A single generation request.

    The defaults are deliberately not extremes — temperature 0 is deterministic
    and temperature 2 is noise, so neither makes a good first impression.
    """

    prompt: str = Field(min_length=1, max_length=8000)
    model_id: str = Field(default="maidan-tiny", max_length=80)
    backend: Backend = "local"

    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=500)
    max_tokens: int = Field(default=80, ge=1, le=400)
    repetition_penalty: float = Field(default=1.15, ge=1.0, le=2.0)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt cannot be blank")
        return value


class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_id: str
    backend: str
    prompt: str
    completion: str
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    repetition_penalty: float
    seed: int | None
    tokens_generated: int
    latency_ms: int
    stop_reason: str
    created_at: datetime


class GenerateResponse(BaseModel):
    completion: str
    tokens_generated: int
    latency_ms: int
    stop_reason: str
    seed: int
    model_id: str
    backend: str
    run: RunOut


class BackendStatus(BaseModel):
    local: bool
    cloud: bool
    providers: list[str]
    message: str


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    models: int
    cloud_providers: list[str]
