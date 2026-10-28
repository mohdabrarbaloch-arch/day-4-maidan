"""Optional cloud passthrough.

The playground's whole pitch is the local/cloud toggle, so the cloud side has to
either genuinely work or genuinely say it can't. It never fabricates a response.
"""

from __future__ import annotations

import time

import httpx

from app.config import get_settings
from app.engine.generate import GenerationParams, resolve_seed

CLOUD_MODELS: dict[str, str] = {
    "cloud-gpt-4o-mini": "gpt-4o-mini",
    "cloud-llama-3.1-8b": "llama-3.1-8b-instant",
}

MODEL_TO_PROVIDER: dict[str, str] = {
    "cloud-gpt-4o-mini": "openai",
    "cloud-llama-3.1-8b": "groq",
}


class CloudUnavailable(RuntimeError):
    """Raised when no provider key is configured for the requested model."""


def available_cloud_models() -> list[str]:
    settings = get_settings()
    configured = settings.cloud_backends()
    return [model_id for model_id, provider in MODEL_TO_PROVIDER.items() if provider in configured]


def generate_cloud(prompt: str, model_id: str, params: GenerationParams) -> dict:
    """Call the configured provider and normalise the reply.

    Raises CloudUnavailable rather than returning a placeholder — a fake answer
    here would be indistinguishable from a real one in the run history, which is
    exactly the kind of thing this project refuses to do.
    """
    settings = get_settings()
    provider = MODEL_TO_PROVIDER.get(model_id)
    if provider is None:
        raise CloudUnavailable(f"Unknown cloud model: {model_id}")

    backends = settings.cloud_backends()
    if provider not in backends:
        raise CloudUnavailable(
            f"No {provider.upper()}_API_KEY configured. Set it in .env to enable "
            f"the cloud backend, or switch the toggle back to local."
        )

    backend = backends[provider]
    upstream_model = CLOUD_MODELS[model_id]
    seed = resolve_seed(params.seed)

    payload = {
        "model": upstream_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_tokens": params.max_tokens,
    }

    started = time.perf_counter()
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{backend['base_url'].rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {backend['api_key']}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
    except httpx.HTTPError as exc:
        raise CloudUnavailable(f"Could not reach {provider}: {exc}") from exc

    if response.status_code >= 400:
        # Surface the provider's own message instead of a generic 500 — the
        # usual cause is an expired key or a model that moved.
        detail = response.text[:400]
        raise CloudUnavailable(f"{provider} returned HTTP {response.status_code}: {detail}")

    data = response.json()
    try:
        completion = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise CloudUnavailable(f"Unexpected {provider} response shape") from exc

    usage = data.get("usage") or {}
    elapsed = int((time.perf_counter() - started) * 1000)

    return {
        "completion": completion.strip(),
        "tokens_generated": int(usage.get("completion_tokens") or 0),
        "latency_ms": elapsed,
        "stop_reason": "eos",
        "seed": seed,
        "model_id": model_id,
        "backend": "cloud",
    }
