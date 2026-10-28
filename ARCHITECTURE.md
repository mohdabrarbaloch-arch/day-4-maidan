# Architecture

## The one-sentence version

A FastAPI app that trains a small n-gram language model at import time and streams
its tokens to a browser over SSE, with the sampler parameters exposed as real
controls rather than decoration.

---

## System diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  Browser (no framework, no build step)                               │
│                                                                      │
│   index.html ── styles.css ── app.js                                 │
│                                  │                                   │
│                                  ├── fetch /api/auth/*   (JWT)       │
│                                  ├── fetch /api/models               │
│                                  └── fetch /api/generate/stream (SSE)│
└──────────────────────────────────┬───────────────────────────────────┘
                                   │  HTTPS
┌──────────────────────────────────▼───────────────────────────────────┐
│  FastAPI (app/main.py)                                               │
│                                                                      │
│   SecurityHeaders ──► RateLimit ──► CORS ──► Router                  │
│                                                                      │
│   ┌────────────┐  ┌──────────────┐  ┌──────────────────────────┐    │
│   │ auth.py    │  │ playground.py│  │ static mount (guarded)    │    │
│   │ register   │  │ /models      │  │ only if the dir exists    │    │
│   │ login      │  │ /generate    │  └──────────────────────────┘    │
│   │ me         │  │ /generate/   │                                   │
│   └─────┬──────┘  │   stream     │                                   │
│         │         │ /runs        │                                   │
│         │         └──────┬───────┘                                   │
│         │                │                                           │
│   ┌─────▼────────────────▼──────────────────────────────────────┐    │
│   │  SQLAlchemy 2.0  ──►  SQLite (dev) / PostgreSQL (prod)      │    │
│   │  User, Run                                                  │    │
│   └─────────────────────────────────────────────────────────────┘    │
│                          │                                           │
│   ┌──────────────────────▼──────────────────────────────────────┐    │
│   │  Engine (app/engine/)                                       │    │
│   │                                                             │    │
│   │  corpus.py ──► tokenizer.py ──► local_model.py              │    │
│   │   (sentences)    (ids)           (trigram + backoff)        │    │
│   │                                        │                    │    │
│   │                                        ▼                    │    │
│   │                                  generate.py                │    │
│   │                                        │                    │    │
│   │                                  sampler.py                 │    │
│   │                          (penalty → top-k → temp → top-p)   │    │
│   └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Web | FastAPI 0.115 | Async, typed, generates OpenAPI for free |
| Server | Uvicorn | ASGI, supports SSE without hacks |
| ORM | SQLAlchemy 2.0 | Typed `Mapped[]` models, one codebase for SQLite and Postgres |
| Validation | Pydantic v2 | Constraints live in the schema, not in `if` statements |
| Auth | python-jose + passlib/bcrypt | HS256 JWT, 12-round bcrypt |
| Engine | Pure Python | No numpy, no torch — the point is that it's readable |
| Frontend | Vanilla JS | No build step means no build to break |
| Tests | pytest | 64 tests, no fixtures beyond a temp DB |

**Deliberately absent:** numpy, torch, transformers, LangChain. A dependency that
hides the maths defeats the purpose.

---

## Data flow: one generation

```
1. POST /api/generate/stream
   { prompt, model_id, temperature, top_k, top_p, max_tokens, seed }

2. Auth dependency resolves the JWT → User

3. Pydantic validates:
   - prompt: 1..8000 chars, stripped, non-empty
   - temperature: 0.0..2.0
   - top_k: 0..200   (0 = disabled)
   - top_p: 0.0..1.0
   - max_tokens: 1..400

4. Model registry lookup → NgramModel (trained once, cached in a module dict)

5. tokenizer.encode(prompt) → [ids]

6. Loop, up to max_tokens times:
     a. model.next_logits(context)      → {token_id: logit}
     b. apply_repetition_penalty(...)   → penalise what's already been said
     c. top_k_filter(...)               → keep the k best
     d. softmax(logits / temperature)   → probabilities
     e. top_p_filter(...)               → smallest nucleus reaching p
     f. sample(probs, rng)              → one token
     g. if token == eos: stop
     h. yield SSE frame: {"type": "token", "token": "..."}

7. On completion: persist Run (prompt, params, seed, output, latency, stop_reason)
   and emit a final frame: {"type": "done", "run_id": ..., "stop_reason": ...}
```

**Why the stop reason needs a holder.** A generator's early `return` and its
normal exhaustion are indistinguishable from the caller's `for` loop, so
`for ... else` always took the `else` branch and every run reported
`max_tokens`. A small mutable `StopReason` is passed in and written by the loop.

**Why tokens are joined client-side.** The server emits raw tokens without the
separator space, so a client that appends them verbatim renders
`thecontextwindow`. The browser does the same joining `_join_tokens` does.

**Why the seed is stored:** a run is only reproducible if you keep the seed *and*
the parameters. Storing the output alone makes history a log; storing the seed
makes it a lab notebook.

---

## The model, precisely

**Training.** Each corpus is a list of complete sentences. For every sentence:

```
<s> <s> w1 w2 w3 ... wn </s>
```

Counts are collected for trigrams `(w[i-2], w[i-1], w[i])`, bigrams, and unigrams.

**Inference.** For a context `(a, b)`:

```
p(w | a, b) = 0.60 · p_tri(w | a, b)
            + 0.28 · p_bi (w | b)
            + 0.12 · p_uni(w)
```

Each term falls back when its count is zero, so an unseen trigram degrades to a
bigram rather than to nothing. The weights are constants, not learned — this is
interpolation, not a neural net, and pretending otherwise would be dishonest.

**Why sentence-bracketed training data matters.** The first version of the corpus
was wrapped paragraphs. The model learned that sentences end at the wrap point, so
`eos` fired after a random comma and every generation was one token long. Splitting
the corpus into sentences fixed it. There is a regression test for this
(`test_every_corpus_is_sentence_shaped`).

---

## Security

| Concern | Handling |
|---|---|
| Password storage | bcrypt, 12 rounds, per-password salt |
| Session | HS256 JWT, 7-day expiry, `sub` = user id |
| Secret key | Env only. Blank in dev → ephemeral per-process key (cached, so tokens survive within a process). **Production refuses to boot without it.** |
| Rate limiting | Sliding window, 60/min, keyed on `X-Forwarded-For` then peer IP. Applied to auth and generate routes only. |
| Headers | `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`, HSTS in production |
| CORS | `*` in dev, explicit allow-list in production |
| Input | Pydantic constraints on every field; prompt length capped; no SQL string building anywhere |
| Secrets in code | None. `grep -r "sk-" app/` returns nothing. |
| Run ownership | Every `/runs/{id}` query filters on `user_id`; a foreign id returns 404, not 403 — no existence leak |

**One deliberate choice:** a foreign run id returns **404**, not **403**. Returning
403 confirms the row exists, which is an enumeration oracle.

---

## Scaling notes

**Where it breaks first: rate limiting.** The limiter is an in-process dict. Two
instances behind a load balancer each allow 60/min, so the effective limit doubles.
Fix: move the counter to Redis (`INCR` + `EXPIRE`), keyed the same way.

**Second: model training.** Models are trained lazily and cached in a module-level
dict. That is correct for one process and wasteful for N — every worker trains its
own copy (~40ms each, so it's tolerable, but it's N× the memory). Fix: train once
at build time, serialise the count tables to JSON, load them at startup.

**Third: SSE and serverless.** Streaming works on a long-lived process. On Vercel,
a function has a wall-clock limit and the connection is not guaranteed to stay
open for a long generation. The blocking `/api/generate` endpoint is the safe path
there; the stream endpoint is for self-hosted deployments.

**Fourth: SQLite.** Fine for one writer. The moment two users generate
concurrently, writes serialise. Postgres is already wired in `docker-compose.yml`.

**What would actually scale:** the engine is CPU-bound and single-threaded per
request. Real concurrency needs either a process pool or a move to a batched
inference server. Neither is worth building for a playground whose model fits in
300 tokens of vocabulary.

---

## Deployment

**Vercel.** `vercel.json` uses the modern `functions` + `rewrites` form:

```json
{
  "functions": { "api/index.py": { "includeFiles": "app/**" } },
  "rewrites": [{ "source": "/(.*)", "destination": "/api/index.py" }]
}
```

The legacy `builds` block resolved `includeFiles` relative to `api/`, so `app/**`
became `api/app/**` — a path that does not exist. The package never shipped and
every request died on `ModuleNotFoundError: No module named 'app'`. That is the
exact failure this config avoids, and `tests/test_packaging.py` asserts it.

`api/index.py` also calls `init_db()` at import, because Vercel never fires
lifespan events.

**Docker.** Multi-stage-free but lean: `python:3.11-slim`, non-root user (uid
10001), healthcheck on `/api/health`, Postgres 16 via compose with a
`pg_isready` gate so the app never races the database.
