"""The generation loop: prompt in, tokens out, with a real stop condition."""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.engine.local_model import NgramModel, get_model
from app.engine.sampler import build_distribution, sample_token


@dataclass
class GenerationParams:
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int = 80
    repetition_penalty: float = 1.15
    seed: int | None = None


@dataclass
class GenerationResult:
    completion: str
    tokens: list[str] = field(default_factory=list)
    stop_reason: str = "max_tokens"
    seed: int = 0
    latency_ms: int = 0

    @property
    def tokens_generated(self) -> int:
        return len(self.tokens)


def resolve_seed(seed: int | None) -> int:
    """Absent a seed, draw one and *return it* so the UI can replay the run."""
    if seed is not None:
        return seed
    return random.SystemRandom().randrange(0, 2**31 - 1)


def _join_tokens(tokens: list[str]) -> str:
    """Assemble streamed tokens into readable text."""
    from app.engine.tokenizer import NO_SPACE_AFTER, NO_SPACE_BEFORE

    out = ""
    for token in tokens:
        if not out:
            out = token
            continue
        if token in NO_SPACE_BEFORE or (out and out[-1] in NO_SPACE_AFTER):
            out += token
        else:
            out += " " + token
    return out.strip()


def model_word_limit(model: NgramModel) -> int:
    """Context window in tokens.

    A trigram conditions on exactly two tokens, so this is 2 — not 512. Keeping
    a longer window would change nothing about the output and would make the
    advertised context window a lie.
    """
    return 2


def stream_generate(
    prompt: str,
    model: NgramModel,
    params: GenerationParams,
) -> Iterator[str]:
    """Yield tokens one at a time.

    Stops on EOS (the corpus learned where sentences end) or when the token
    budget runs out.
    """
    seed = resolve_seed(params.seed)
    rng = random.Random(seed)

    tokenizer = model.tokenizer
    context = tokenizer.encode(prompt)[-model_word_limit(model) :]
    if not context:
        context = [tokenizer.bos_id]

    generated: list[int] = []
    for _ in range(params.max_tokens):
        logits = model.next_token_logits(context)
        distribution = build_distribution(
            logits,
            generated=generated,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            repetition_penalty=params.repetition_penalty,
        )
        token_id = sample_token(distribution, rng)

        if token_id == tokenizer.eos_id:
            return

        rendered = tokenizer.render_token(token_id)
        if rendered:
            generated.append(token_id)
            context = (context + [token_id])[-model_word_limit(model) :]
            yield rendered


def generate(
    prompt: str,
    model_id: str,
    params: GenerationParams,
) -> GenerationResult | None:
    """Blocking generation. Returns None when the model id is unknown."""
    model = get_model(model_id)
    if model is None:
        return None

    seed = resolve_seed(params.seed)
    started = time.perf_counter()

    tokens: list[str] = []
    stop_reason = "max_tokens"
    for token in stream_generate(
        prompt, model, GenerationParams(**{**params.__dict__, "seed": seed})
    ):
        tokens.append(token)
    else:
        # The generator was exhausted rather than returning early on EOS.
        stop_reason = "max_tokens"

    elapsed = int((time.perf_counter() - started) * 1000)

    return GenerationResult(
        completion=_join_tokens(tokens),
        tokens=tokens,
        stop_reason=stop_reason,
        seed=seed,
        latency_ms=elapsed,
    )
