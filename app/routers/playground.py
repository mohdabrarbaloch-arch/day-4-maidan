"""Playground routes: model catalogue, backend status, generation, history."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal, get_db
from app.engine.cloud import CloudUnavailable, available_cloud_models, generate_cloud
from app.engine.generate import GenerationParams, generate, resolve_seed
from app.engine.local_model import get_model, list_models
from app.models import Run, User
from app.schemas import (
    BackendStatus,
    GenerateRequest,
    GenerateResponse,
    ModelInfo,
    RunOut,
)
from app.security import get_current_user

router = APIRouter(prefix="/api", tags=["playground"])

# Cloud models are only advertised when a key exists, so the UI never shows a
# toggle position that cannot actually answer.
_CLOUD_CATALOGUE = [
    {
        "id": "cloud-gpt-4o-mini",
        "name": "GPT-4o mini",
        "family": "OpenAI",
        "params": "hosted",
        "context_window": 128000,
        "description": (
            "Fast hosted model. Useful as a side-by-side reference: run the same "
            "prompt locally and here, and the gap tells you what scale buys."
        ),
        "strengths": ["Reasoning", "Instructions", "Long context"],
    },
    {
        "id": "cloud-llama-3.1-8b",
        "name": "Llama 3.1 8B (Groq)",
        "family": "Meta",
        "params": "8B",
        "context_window": 131072,
        "description": (
            "The same model family the tiny local one imitates, at full scale. "
            "The comparison is the interesting part."
        ),
        "strengths": ["Open weights", "Fast inference", "General chat"],
    },
]


@router.get("/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    local = [ModelInfo(**entry) for entry in list_models()]
    available = set(available_cloud_models())
    cloud = [
        ModelInfo(**{**entry, "backend": "cloud"})
        for entry in _CLOUD_CATALOGUE
        if entry["id"] in available
    ]
    return local + cloud


@router.get("/backends", response_model=BackendStatus)
def backends() -> BackendStatus:
    settings = get_settings()
    providers = sorted(settings.cloud_backends().keys())
    if providers:
        message = f"Local engine ready. Cloud available via {', '.join(providers)}."
    else:
        message = (
            "Local engine ready. No cloud provider key is configured, so the cloud "
            "toggle has nothing to call — set OPENAI_API_KEY or GROQ_API_KEY to enable it."
        )
    return BackendStatus(
        local=True,
        cloud=bool(providers),
        providers=providers,
        message=message,
    )


def _persist(db: Session, user: User, payload: GenerateRequest, result: dict) -> Run:
    run = Run(
        user_id=user.id,
        model_id=result["model_id"],
        backend=result["backend"],
        prompt=payload.prompt,
        completion=result["completion"],
        temperature=payload.temperature,
        top_p=payload.top_p,
        top_k=payload.top_k,
        max_tokens=payload.max_tokens,
        repetition_penalty=payload.repetition_penalty,
        seed=result["seed"],
        tokens_generated=result["tokens_generated"],
        latency_ms=result["latency_ms"],
        stop_reason=result["stop_reason"],
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


@router.post("/generate", response_model=GenerateResponse)
def create_generation(
    payload: GenerateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> GenerateResponse:
    settings = get_settings()
    if len(payload.prompt) > settings.max_prompt_chars:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Prompt exceeds {settings.max_prompt_chars} characters.",
        )

    params = GenerationParams(
        temperature=payload.temperature,
        top_p=payload.top_p,
        top_k=payload.top_k,
        max_tokens=payload.max_tokens,
        repetition_penalty=payload.repetition_penalty,
        seed=payload.seed,
    )

    if payload.backend == "cloud":
        try:
            result = generate_cloud(payload.prompt, payload.model_id, params)
        except CloudUnavailable as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc
    else:
        local_model = get_model(payload.model_id)
        if local_model is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown model: {payload.model_id}",
            )
        generated = generate(payload.prompt, payload.model_id, params)
        if generated is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown model: {payload.model_id}",
            )
        result = {
            "completion": generated.completion,
            "tokens_generated": generated.tokens_generated,
            "latency_ms": generated.latency_ms,
            "stop_reason": generated.stop_reason,
            "seed": generated.seed,
            "model_id": payload.model_id,
            "backend": "local",
        }

    run = _persist(db, user, payload, result)
    return GenerateResponse(**result, run=RunOut.model_validate(run))


@router.post("/generate/stream")
def create_generation_stream(
    payload: GenerateRequest,
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Server-sent events, one token per frame.

    Streaming is what makes a small model feel alive: you watch it commit to
    each token instead of staring at a spinner. Cloud models are not streamed
    here — the whole reply arrives and is emitted as a single final frame, which
    the client renders identically.
    """
    settings = get_settings()
    if len(payload.prompt) > settings.max_prompt_chars:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Prompt exceeds {settings.max_prompt_chars} characters.",
        )

    params = GenerationParams(
        temperature=payload.temperature,
        top_p=payload.top_p,
        top_k=payload.top_k,
        max_tokens=payload.max_tokens,
        repetition_penalty=payload.repetition_penalty,
        seed=payload.seed,
    )

    def event_stream() -> Iterator[str]:
        # A generator runs after the response has been sent, so it cannot use
        # the request-scoped dependency. Open its own session instead.
        session = SessionLocal()
        started = time.perf_counter()
        try:
            yield f"data: {json.dumps({'type': 'start', 'model': payload.model_id})}\n\n"

            if payload.backend == "cloud":
                try:
                    result = generate_cloud(payload.prompt, payload.model_id, params)
                except CloudUnavailable as exc:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                    return
                yield f"data: {json.dumps({'type': 'token', 'token': result['completion']})}\n\n"
            else:
                local_model = get_model(payload.model_id)
                if local_model is None:
                    yield (
                        "data: "
                        + json.dumps({"type": "error", "message": "Unknown model"})
                        + "\n\n"
                    )
                    return

                from app.engine.generate import StopReason, stream_generate
                from app.engine.tokenizer import NO_SPACE_AFTER, NO_SPACE_BEFORE

                text = ""
                token_count = 0
                seed = resolve_seed(params.seed)
                stop = StopReason()
                for token in stream_generate(
                    payload.prompt,
                    local_model,
                    GenerationParams(**{**params.__dict__, "seed": seed}),
                    stop,
                ):
                    if not text:
                        text = token
                    elif token in NO_SPACE_BEFORE or text[-1] in NO_SPACE_AFTER:
                        text += token
                    else:
                        text += " " + token
                    token_count += 1
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

                result = {
                    "completion": text.strip(),
                    "tokens_generated": token_count,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "stop_reason": stop.value,
                    "seed": seed,
                    "model_id": payload.model_id,
                    "backend": "local",
                }

            db_user = session.scalar(select(User).where(User.email == user.email))
            run_id = None
            if db_user is not None:
                run = _persist(session, db_user, payload, result)
                run_id = run.id

            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "done",
                        "run_id": run_id,
                        "tokens_generated": result["tokens_generated"],
                        "latency_ms": result["latency_ms"],
                        "seed": result["seed"],
                        "stop_reason": result["stop_reason"],
                    }
                )
                + "\n\n"
            )
        finally:
            session.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/runs", response_model=list[RunOut])
def list_runs(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> list[RunOut]:
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    rows = db.scalars(
        select(Run)
        .where(Run.user_id == user.id)
        .order_by(desc(Run.created_at), desc(Run.id))
        .limit(limit)
        .offset(offset)
    ).all()
    return [RunOut.model_validate(row) for row in rows]


@router.get("/runs/{run_id}", response_model=RunOut)
def get_run(
    run_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> RunOut:
    run = db.scalar(select(Run).where(Run.id == run_id, Run.user_id == user.id))
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return RunOut.model_validate(run)


@router.delete("/runs/{run_id}")
def delete_run(
    run_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    run = db.scalar(select(Run).where(Run.id == run_id, Run.user_id == user.id))
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    db.delete(run)
    db.commit()
    # Deliberately 200 with a small body rather than 204: FastAPI asserts that a
    # 204 route declares no response model, and a tiny confirmation payload is
    # friendlier to the client than an empty response anyway.
    return {"deleted": run_id}
