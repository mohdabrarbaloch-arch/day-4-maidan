"""Sampling primitives for the local engine.

These are the actual maths used to pick the next token: temperature scaling,
top-k truncation, nucleus (top-p) truncation, repetition penalty, and a
categorical draw. Kept pure and side-effect free so every one of them can be
unit-tested against a hand-computed expectation.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence

# A logit of -1e9 behaves as "impossible" for our purposes without risking
# exp() underflow to exactly zero in the softmax.
NEG_INF = -1e9


def apply_repetition_penalty(
    logits: Mapping[int, float],
    generated: Iterable[int],
    penalty: float,
) -> dict[int, float]:
    """CTRL-style repetition penalty.

    A penalty of 1.0 is a no-op. Above 1.0, logits for tokens already seen are
    pulled toward zero: positives are divided, negatives are multiplied. This
    is asymmetric on purpose — dividing a negative logit would make it *more*
    likely, which is the opposite of what we want.
    """
    if penalty <= 1.0:
        return dict(logits)

    seen = set(generated)
    adjusted: dict[int, float] = {}
    for token, logit in logits.items():
        if token in seen and logit != NEG_INF:
            adjusted[token] = logit / penalty if logit > 0 else logit * penalty
        else:
            adjusted[token] = logit
    return adjusted


def top_k_filter(logits: Mapping[int, float], k: int) -> dict[int, float]:
    """Keep only the k highest logits.

    k <= 0 disables the filter entirely, which is the conventional meaning of
    "top_k = 0" in most inference servers.
    """
    if k <= 0 or k >= len(logits):
        return dict(logits)

    ranked = sorted(logits.items(), key=lambda item: item[1], reverse=True)
    return dict(ranked[:k])


def softmax(logits: Mapping[int, float], temperature: float) -> dict[int, float]:
    """Temperature-scaled softmax.

    temperature == 0 is handled as greedy rather than as a division by zero:
    every probability mass goes to the single highest logit. In practice the
    caller short-circuits to argmax, but returning a valid distribution here
    keeps the function total.
    """
    if not logits:
        return {}

    if temperature <= 0.0:
        best = max(logits.items(), key=lambda item: item[1])[0]
        return {token: (1.0 if token == best else 0.0) for token in logits}

    scaled = {token: logit / temperature for token, logit in logits.items()}
    peak = max(scaled.values())
    exps = {token: math.exp(value - peak) for token, value in scaled.items()}
    total = sum(exps.values())
    if total <= 0.0:
        # Every logit was NEG_INF. Fall back to uniform so we never divide by 0.
        share = 1.0 / len(exps)
        return {token: share for token in exps}
    return {token: value / total for token, value in exps.items()}


def top_p_filter(probs: Mapping[int, float], p: float) -> dict[int, float]:
    """Nucleus sampling: smallest set of tokens whose mass reaches p.

    Ties are broken by token id so the result is deterministic. At least one
    token always survives — a zero-token distribution would make sampling
    impossible.
    """
    if not probs:
        return {}
    if p >= 1.0:
        return dict(probs)

    ranked = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
    kept: dict[int, float] = {}
    cumulative = 0.0
    # Float addition is not exact: 0.7 + 0.2 lands on 0.8999999999999999, which
    # is less than 0.9, so a strict comparison keeps one token too many. The
    # tolerance is far below any meaningful probability difference.
    tolerance = 1e-9
    for token, prob in ranked:
        kept[token] = prob
        cumulative += prob
        if cumulative >= p - tolerance:
            break

    if not kept:
        kept[ranked[0][0]] = ranked[0][1]

    # Renormalise — truncation removes mass from the tail.
    total = sum(kept.values())
    return {token: prob / total for token, prob in kept.items()} if total > 0 else kept


def sample_token(probs: Mapping[int, float], rng: random.Random) -> int:
    """Draw one token from a categorical distribution."""
    if not probs:
        raise ValueError("cannot sample from an empty distribution")

    # Sort for reproducibility: dict order is insertion order, which depends on
    # how the model built the logits, not on the seed.
    items = sorted(probs.items(), key=lambda item: item[0])
    threshold = rng.random()
    cumulative = 0.0
    for token, prob in items:
        cumulative += prob
        if threshold <= cumulative:
            return token
    return items[-1][0]


def build_distribution(
    logits: Mapping[int, float],
    *,
    generated: Sequence[int] = (),
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> dict[int, float]:
    """Full pipeline: penalty -> top-k -> temperature softmax -> top-p.

    Order matters and follows the usual implementation: penalties and truncation
    happen in logit space, and nucleus truncation happens after the softmax
    because it needs normalised mass.
    """
    penalised = apply_repetition_penalty(logits, generated, repetition_penalty)
    truncated = top_k_filter(penalised, top_k)
    probs = softmax(truncated, temperature)
    return top_p_filter(probs, top_p)
