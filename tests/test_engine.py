"""Engine tests: tokenizer, sampler, model, generation loop.

These target the parts where a silent bug is expensive — a wrong special-token
id, a sampler that ignores its parameters, or a seed that does not actually
reproduce. Every assertion is against a hand-computed expectation, not a
snapshot of whatever the code happened to output.
"""

from __future__ import annotations

import math
import random

import pytest

from app.engine.corpus import CORPORA
from app.engine.generate import GenerationParams, generate, resolve_seed, stream_generate
from app.engine.local_model import get_model, list_models, train
from app.engine.sampler import (
    NEG_INF,
    apply_repetition_penalty,
    build_distribution,
    sample_token,
    softmax,
    top_k_filter,
    top_p_filter,
)
from app.engine.tokenizer import Tokenizer

# ── tokenizer ────────────────────────────────────────────────────────────────


def test_tokenizer_splits_words_and_punctuation():
    tk = Tokenizer(["The", "model", "reads", "a", "prompt", "."])
    assert tk.split("The model reads a prompt.") == [
        "The",
        "model",
        "reads",
        "a",
        "prompt",
        ".",
    ]


def test_tokenizer_keeps_numbers_together():
    tk = Tokenizer(["3.14", "and", "42"])
    assert tk.split("3.14 and 42") == ["3.14", "and", "42"]


def test_special_token_ids_are_not_zero():
    """Regression: eos_id defaulting to 0 made the first comma look like EOS.

    Index 0 is a real vocabulary token, so a special id of 0 silently truncates
    every generation after one token.
    """
    tk = Tokenizer(["a", "b", "c"], specials=("<bos>", "<eos>"))
    assert tk.eos_id != 0
    assert tk.bos_id != 0
    assert tk.itos[tk.eos_id] == "<eos>"
    assert tk.itos[tk.bos_id] == "<bos>"


def test_tokenizer_roundtrip_is_readable():
    tk = Tokenizer(["The", "model", "reads", "a", "prompt", "."])
    ids = tk.encode("The model reads a prompt.")
    assert tk.decode(ids) == "The model reads a prompt."


def test_unknown_token_maps_to_unk():
    tk = Tokenizer(["known"], specials=("<unk>",))
    assert tk.encode("known unknownword")[1] == tk.unk_id


# ── sampler ──────────────────────────────────────────────────────────────────


def test_softmax_sums_to_one():
    probs = softmax({1: 1.0, 2: 2.0, 3: 3.0}, temperature=1.0)
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9)


def test_softmax_zero_temperature_is_greedy():
    probs = softmax({1: 1.0, 2: 5.0, 3: 2.0}, temperature=0.0)
    assert probs[2] == 1.0
    assert probs[1] == 0.0


def test_low_temperature_sharpens_distribution():
    logits = {1: 1.0, 2: 2.0, 3: 3.0}
    cold = softmax(logits, temperature=0.2)
    warm = softmax(logits, temperature=2.0)
    # The top token should hold more mass when cold than when warm.
    assert cold[3] > warm[3]


def test_top_k_keeps_exactly_k():
    logits = {1: 1.0, 2: 5.0, 3: 3.0, 4: 0.5}
    kept = top_k_filter(logits, 2)
    assert set(kept) == {2, 3}


def test_top_k_zero_disables_filter():
    logits = {1: 1.0, 2: 5.0, 3: 3.0}
    assert top_k_filter(logits, 0) == logits


def test_top_p_keeps_smallest_nucleus():
    # 0.7 + 0.2 = 0.9 >= 0.9, so two tokens survive and one is dropped.
    probs = {1: 0.7, 2: 0.2, 3: 0.1}
    kept = top_p_filter(probs, 0.9)
    assert set(kept) == {1, 2}
    assert math.isclose(sum(kept.values()), 1.0, rel_tol=1e-9)


def test_top_p_always_keeps_at_least_one_token():
    probs = {1: 0.4, 2: 0.35, 3: 0.25}
    kept = top_p_filter(probs, 0.01)
    assert len(kept) == 1


def test_repetition_penalty_pushes_seen_tokens_down():
    logits = {1: 4.0, 2: 2.0}
    adjusted = apply_repetition_penalty(logits, generated=[1], penalty=2.0)
    assert adjusted[1] == pytest.approx(2.0)
    assert adjusted[2] == 2.0


def test_repetition_penalty_handles_negative_logits():
    """A negative logit must get *less* likely, not more.

    Dividing a negative number by a penalty > 1 moves it toward zero, which
    would increase its probability — the opposite of the intent.
    """
    logits = {1: -4.0, 2: 1.0}
    adjusted = apply_repetition_penalty(logits, generated=[1], penalty=2.0)
    assert adjusted[1] == pytest.approx(-8.0)


def test_repetition_penalty_of_one_is_noop():
    logits = {1: 4.0, 2: 2.0}
    assert apply_repetition_penalty(logits, generated=[1], penalty=1.0) == logits


def test_sample_token_is_deterministic_for_a_seed():
    probs = {1: 0.5, 2: 0.3, 3: 0.2}
    a = sample_token(probs, random.Random(1234))
    b = sample_token(probs, random.Random(1234))
    assert a == b


def test_sample_token_never_returns_impossible_token():
    probs = {1: 1.0, 2: 0.0}
    for seed in range(50):
        assert sample_token(probs, random.Random(seed)) == 1


def test_build_distribution_respects_top_k():
    logits = {1: 1.0, 2: 5.0, 3: 3.0, 4: 0.5}
    dist = build_distribution(logits, top_k=2, temperature=1.0)
    assert set(dist) <= {2, 3}


def test_build_distribution_ignores_neg_inf_tokens():
    logits = {1: 5.0, 2: NEG_INF}
    dist = build_distribution(logits, temperature=1.0)
    assert dist.get(2, 0.0) == pytest.approx(0.0, abs=1e-12)


# ── model ────────────────────────────────────────────────────────────────────


def test_train_builds_all_three_orders():
    model = train("general", "test-general")
    assert model.vocab_size() > 50
    assert len(model.unigram) > 50
    assert len(model.bigram) > 50
    assert len(model.trigram) > 50


def test_model_returns_logits_for_every_vocab_token():
    model = train("general", "test-logits")
    context = model.tokenizer.encode("The model reads")
    logits = model.next_token_logits(context)
    assert len(logits) == model.vocab_size()
    assert all(value <= 0 for value in logits.values())


def test_model_handles_empty_context():
    model = train("general", "test-empty")
    logits = model.next_token_logits([])
    assert len(logits) == model.vocab_size()


def test_unknown_model_id_returns_none():
    assert get_model("does-not-exist") is None


def test_list_models_reports_three_presets():
    models = list_models()
    assert len(models) == 3
    assert {m["id"] for m in models} == {"maidan-tiny", "maidan-coder", "maidan-muse"}
    assert all(m["available"] for m in models)


def test_every_corpus_is_sentence_shaped():
    """Each entry must be a complete sentence.

    Wrapped paragraphs taught the model that sentences end mid-clause, which
    truncated generation at the first comma.
    """
    for key, sentences in CORPORA.items():
        assert sentences, f"{key} is empty"
        for sentence in sentences:
            assert sentence[0].isupper(), f"{key}: {sentence[:40]!r} does not start a sentence"
            assert sentence[-1] in ".!?", f"{key}: {sentence[:40]!r} does not end a sentence"


# ── generation ───────────────────────────────────────────────────────────────


def test_generate_produces_output():
    result = generate(
        "The context window is",
        "maidan-tiny",
        GenerationParams(max_tokens=20, seed=1),
    )
    assert result is not None
    assert result.completion
    assert result.tokens_generated > 0


def test_same_seed_reproduces_exactly():
    params = GenerationParams(max_tokens=20, seed=4242, temperature=0.9)
    first = generate("The context window is", "maidan-tiny", params)
    second = generate("The context window is", "maidan-tiny", params)
    assert first.completion == second.completion


def test_different_seeds_diverge():
    prompt = "The context window is"
    a = generate(prompt, "maidan-tiny", GenerationParams(max_tokens=30, seed=1, temperature=1.3))
    b = generate(prompt, "maidan-tiny", GenerationParams(max_tokens=30, seed=2, temperature=1.3))
    assert a.completion != b.completion


def test_max_tokens_is_respected():
    result = generate(
        "The model reads",
        "maidan-tiny",
        GenerationParams(max_tokens=5, seed=3, repetition_penalty=1.0),
    )
    assert result.tokens_generated <= 5


def test_resolve_seed_returns_given_seed():
    assert resolve_seed(99) == 99


def test_resolve_seed_generates_when_absent():
    seed = resolve_seed(None)
    assert isinstance(seed, int)
    assert 0 <= seed < 2**31


def test_stream_generate_yields_tokens():
    model = get_model("maidan-tiny")
    assert model is not None
    tokens = list(stream_generate("The model reads", model, GenerationParams(max_tokens=8, seed=7)))
    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


def test_stop_reason_reports_eos_when_the_model_ends_a_sentence():
    """Regression: every run used to report "max_tokens".

    A generator's early ``return`` and its normal exhaustion are
    indistinguishable from the caller's ``for`` loop, so the old
    ``for ... else`` always took the ``else`` branch. The holder is what
    makes an EOS stop visible.
    """
    from app.engine.generate import StopReason

    model = get_model("maidan-tiny")
    assert model is not None

    # A budget large enough that hitting it would mean something is wrong, so
    # whatever comes out almost certainly stops on <eos>.
    stop = StopReason()
    tokens = list(
        stream_generate(
            "The model reads",
            model,
            GenerationParams(max_tokens=400, seed=11, temperature=0.7),
            stop,
        )
    )
    assert len(tokens) < 400
    assert stop.value == "eos"


def test_stop_reason_reports_max_tokens_when_the_budget_runs_out():
    """The other half of the same regression: the label must still be honest
    when the model is cut off mid-sentence rather than choosing to stop."""
    from app.engine.generate import StopReason

    model = get_model("maidan-tiny")
    assert model is not None

    stop = StopReason()
    list(
        stream_generate(
            "The model reads",
            model,
            GenerationParams(max_tokens=3, seed=11, temperature=0.7),
            stop,
        )
    )
    assert stop.value == "max_tokens"


def test_blocking_generate_reports_the_same_stop_reason_as_the_stream():
    """The route reads ``result.stop_reason``; if the two paths disagree the UI
    shows one thing in the stream and saves another to history."""
    params = GenerationParams(max_tokens=400, seed=5, temperature=0.7)
    model = get_model("maidan-tiny")
    assert model is not None

    from app.engine.generate import StopReason

    stop = StopReason()
    streamed = list(stream_generate("The model reads", model, params, stop))
    blocking = generate("The model reads", "maidan-tiny", params)

    assert blocking is not None
    assert blocking.tokens_generated == len(streamed)
    assert blocking.stop_reason == stop.value


def test_generate_unknown_model_returns_none():
    assert generate("hello", "nope", GenerationParams()) is None
