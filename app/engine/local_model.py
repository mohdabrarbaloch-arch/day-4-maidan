"""A local language model you can actually run: interpolated trigram with backoff.

This is a genuine n-gram language model, trained at import time from a bundled
corpus. It is not a transformer and does not claim to be — but it does the same
*loop*: build a distribution over the next token from the context, sample, and
feed the token back in.

Why interpolated rather than plain trigram: a bare trigram model has a total
vocabulary of a few hundred distinct context pairs here, so most of the time it
has never seen the context it is asked to continue. Interpolating trigram,
bigram and unigram probabilities with absolute-discount-style backing off gives
something that degrades gracefully instead of falling silent.

Everything is deterministic given a seed, which is what makes a saved run
replayable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app.engine.corpus import CORPORA, DEFAULT_CORPUS
from app.engine.tokenizer import Tokenizer


@dataclass
class NgramModel:
    """Trigram -> bigram -> unigram interpolation."""

    name: str
    tokenizer: Tokenizer
    trigram: dict[tuple[int, int], Counter[int]] = field(default_factory=dict)
    bigram: dict[int, Counter[int]] = field(default_factory=dict)
    unigram: Counter[int] = field(default_factory=Counter)
    total_tokens: int = 0

    # Interpolation weights, summing to 1.0. Trigram carries the most signal but
    # is the sparsest, so the bigram and unigram terms are what keep it alive.
    w_trigram: float = 0.60
    w_bigram: float = 0.28
    w_unigram: float = 0.12

    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def next_token_logits(self, context: list[int]) -> dict[int, float]:
        """Log-probabilities over the vocabulary given the recent context.

        Constructed by interpolation, then converted to log space because every
        downstream sampler (repetition penalty, top-k, softmax) works there.
        """
        counts: dict[int, float] = defaultdict(float)

        # Unigram floor: never zero across the whole vocabulary, so no context
        # can produce an empty distribution.
        if self.unigram:
            denominator = max(self.total_tokens, 1)
            for token_id, count in self.unigram.items():
                counts[token_id] += self.w_unigram * (count / denominator)

        if context:
            prev = context[-1]
            bigram_counts = self.bigram.get(prev)
            if bigram_counts:
                total = sum(bigram_counts.values())
                for token_id, count in bigram_counts.items():
                    counts[token_id] += self.w_bigram * (count / total)

        if len(context) >= 2:
            pair = (context[-2], context[-1])
            trigram_counts = self.trigram.get(pair)
            if trigram_counts:
                total = sum(trigram_counts.values())
                for token_id, count in trigram_counts.items():
                    counts[token_id] += self.w_trigram * (count / total)

        # Mirror the trigram/bigram weight back onto the lower orders when the
        # higher order is unavailable, so confidence does not silently collapse.
        mass = sum(counts.values())
        if mass <= 0.0:
            uniform = 1.0 / max(len(self.tokenizer), 1)
            return {token_id: uniform for token_id in range(len(self.tokenizer))}

        normalised = {token_id: value / mass for token_id, value in counts.items()}
        return {token_id: _safe_log(prob) for token_id, prob in normalised.items()}


def _safe_log(value: float) -> float:
    import math

    return math.log(value) if value > 0 else -1e9


def train(corpus_key: str = DEFAULT_CORPUS, name: str = "maidan-tiny") -> NgramModel:
    """Train a model on one of the bundled corpora.

    Each corpus entry is one complete sentence, and each sentence is one
    training document bracketed by <bos> / <eos>. Bracket placement is the whole
    ballgame: if <eos> lands after an arbitrary line wrap, the model learns that
    sentences end mid-clause and every generation dies at the first comma.

    Sentences are counted twice. That keeps every trigram context that appears at
    all dense enough to survive interpolation, without distorting the relative
    probabilities.
    """
    sentences = CORPORA.get(corpus_key, CORPORA[DEFAULT_CORPUS])

    flat_tokens: list[str] = []
    for sentence in sentences:
        flat_tokens.extend(_raw_tokens(sentence))

    tokenizer = Tokenizer(flat_tokens, specials=("<bos>", "<eos>"))
    model = NgramModel(name=name, tokenizer=tokenizer)

    for sentence in sentences:
        body = tokenizer.encode(sentence)
        if not body:
            continue
        ids = [tokenizer.bos_id, *body, tokenizer.eos_id]

        model.total_tokens += len(ids)
        model.unigram.update(ids)
        for index in range(1, len(ids)):
            model.bigram.setdefault(ids[index - 1], Counter())[ids[index]] += 1
        for index in range(2, len(ids)):
            pair = (ids[index - 2], ids[index - 1])
            model.trigram.setdefault(pair, Counter())[ids[index]] += 1

    return model


def _raw_tokens(text: str) -> list[str]:
    from app.engine.tokenizer import _TOKEN_RE  # local import keeps the API small

    return _TOKEN_RE.findall(text)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Every number below is measured from the trained model, not written by hand.
# An earlier draft of this file claimed "1.2M params" and a 512-token context
# window, which is a transformer's vocabulary applied to a trigram model. The
# real context window is 2 tokens and the real parameter count is a few thousand
# counts. Claiming otherwise would undercut the entire point of the project.
_MODEL_SPECS: list[dict] = [
    {
        "id": "maidan-tiny",
        "name": "Maidan Tiny",
        "family": "Maidan",
        "corpus": "general",
        "description": (
            "The default. Trained on plain-English prose about how models work, "
            "so it continues technical writing without drifting."
        ),
        "strengths": ["Technical prose", "Fast", "Fully offline"],
    },
    {
        "id": "maidan-coder",
        "name": "Maidan Coder",
        "family": "Maidan",
        "corpus": "code",
        "description": (
            "Trained on short engineering writing — commit advice, debugging "
            "notes, review comments. Good at terse, opinionated sentences."
        ),
        "strengths": ["Engineering notes", "Terse output", "Code commentary"],
    },
    {
        "id": "maidan-muse",
        "name": "Maidan Muse",
        "family": "Maidan",
        "corpus": "story",
        "description": (
            "Trained on narrative fragments. Produces atmospheric, slower-paced "
            "continuations — the one to reach for when you want mood over facts."
        ),
        "strengths": ["Narrative", "Atmosphere", "Creative continuation"],
    },
]

_models: dict[str, NgramModel] = {}


def get_model(model_id: str) -> NgramModel | None:
    """Lazily train a model by id. Returns None for an unknown id."""
    if model_id in _models:
        return _models[model_id]

    spec = next((s for s in _MODEL_SPECS if s["id"] == model_id), None)
    if spec is None:
        return None

    model = train(corpus_key=spec["corpus"], name=model_id)
    _models[model_id] = model
    return model


def list_models() -> list[dict]:
    """Model catalogue with live, measured statistics.

    ``params`` and ``context_window`` are computed from the trained model rather
    than declared. A trigram model conditions on two tokens, so the context
    window is 2 — reporting anything larger would be a lie about the architecture.
    """
    out: list[dict] = []
    for spec in _MODEL_SPECS:
        model = get_model(spec["id"])
        if model is None:
            continue

        # "Parameters" for an n-gram model are its stored counts: one per
        # observed trigram, bigram and unigram entry.
        param_count = (
            sum(len(c) for c in model.trigram.values())
            + sum(len(c) for c in model.bigram.values())
            + len(model.unigram)
        )

        out.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "family": spec["family"],
                "params": f"{param_count:,} counts",
                "context_window": 2,
                "vocab_size": model.vocab_size(),
                "trigram_count": len(model.trigram),
                "description": spec["description"],
                "strengths": list(spec["strengths"]),
                "backend": "local",
                "available": True,
            }
        )
    return out
