# Maidan — میدان

**A local LLM playground that actually runs a model.**

Pick a model, tune the sampler, watch tokens arrive one at a time. No API key, no
network call, no GPU. The engine is a real interpolated trigram language model
trained at startup from a bundled corpus — small, honest about what it is, and
genuinely running on your machine.

[![Python](https://img.shields.io/badge/python-3.11-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-64%20passing-4c9?logo=pytest&logoColor=white)](#testing)
[![Ruff](https://img.shields.io/badge/lint-ruff%20clean-261230)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## Why this exists

Every "LLM playground" is a thin form over someone else's API. You learn what the
sliders are called, not what they do.

Maidan flips that. The model is small enough to read in one sitting, so when you
drag temperature to 1.8 and the output falls apart, you can open `sampler.py` and
see exactly which line did it. The sliders are wired to real maths, not to a
request body.

**What it is:** a working inference loop — tokenizer, n-gram model, sampler,
streaming — with a UI that makes the parameters legible.

**What it is not:** a transformer. There is no attention, no embeddings, no
gradient descent. It is a trigram model with interpolation and backoff, and it
says so everywhere it matters.

---

## Features

**A real engine, not a mock**
- Interpolated trigram → bigram → unigram model with graceful backoff
- Trained at import from three bundled corpora — no download, no network
- Three presets: `maidan-tiny` (technical prose), `maidan-coder` (engineering
  notes), `maidan-muse` (narrative)
- Deterministic: same prompt + parameters + seed = byte-identical output

**Sampler controls that do something**
- Temperature, top-k, top-p, repetition penalty, max tokens, seed
- Every control maps to a tested function in `app/engine/sampler.py`
- Repetition penalty is asymmetric on purpose — negatives get *less* likely, not more

**Streaming**
- Server-sent events, one token per frame
- You watch the model commit to each token instead of waiting for a finished string
- The cursor blinks while it thinks

**Local / cloud toggle**
- Local needs no keys at all
- Cloud is optional and honest: with no key configured, the API returns **503 with
  an explanation** rather than inventing a reply

**Everything else**
- JWT auth with bcrypt (12 rounds), per-user run history
- Every run saved with its full parameter set and seed, so it can be replayed
- Rate limiting, security headers, input validation, JSON 404s
- Mobile-first dark UI, no build step, no framework

---

## Screenshots

| Playground | Streaming output |
|---|---|
| ![Playground](docs/screenshots/playground.png) | ![Streaming](docs/screenshots/streaming.png) |

| Sampler panel | Run history |
|---|---|
| ![Sampler](docs/screenshots/sampler.png) | ![History](docs/screenshots/history.png) |

| Mobile (375px) |
|---|
| ![Mobile](docs/screenshots/mobile.png) |

> Screenshots are captured from a local run. See [docs/setup.md](docs/setup.md) to
> reproduce them.

---

## Live demo

**Not deployed yet.** There is no `VERCEL_TOKEN` in the build environment, so no
live URL exists and none is claimed here. The repo ships `vercel.json`, a
serverless entrypoint and a Dockerfile — it is deploy-ready.

To deploy yourself:

```bash
vercel --prod
```

Set `SECRET_KEY` and `DATABASE_URL=sqlite:////tmp/maidan.db` in the Vercel
dashboard first. The `/tmp` path is not optional: Vercel's filesystem is read-only
everywhere else.

---

## Quick start

```bash
git clone https://github.com/mohdabrarbaloch-arch/day-4-maidan.git
cd day-4-maidan

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env

uvicorn app.main:app --reload
```

Open **http://localhost:8000**. Register an account, write a prompt, press
Generate.

No API key needed. The model trains itself on first request — about 40ms.

### With Docker

```bash
export SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
docker compose up --build
```

That brings up the app on `:8000` and PostgreSQL 16 behind it.

---

## How it works

```
prompt
  │
  ▼
tokenizer ──► [ids]
  │
  ▼
trigram model ──► logits over the vocabulary
  │                 (interpolated: 0.60 trigram + 0.28 bigram + 0.12 unigram)
  ▼
repetition penalty ──► top-k ──► temperature softmax ──► top-p
  │
  ▼
sample one token ──► append ──► feed back in
  │
  ▼
SSE frame ──► browser
```

The loop stops on `<eos>` or when the token budget runs out. Both are reported
back to the client as `stop_reason`.

Full detail — data flow, scaling notes, and the trade-offs — is in
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `/api/health` | — | Liveness + model count |
| `GET` | `/api/models` | — | Model catalogue |
| `GET` | `/api/backends` | — | Which backends are configured |
| `POST` | `/api/auth/register` | — | Create account → JWT |
| `POST` | `/api/auth/login` | — | Sign in → JWT |
| `GET` | `/api/auth/me` | ✅ | Current user |
| `POST` | `/api/generate` | ✅ | Blocking generation |
| `POST` | `/api/generate/stream` | ✅ | SSE token stream |
| `GET` | `/api/runs` | ✅ | Run history (paginated) |
| `GET` | `/api/runs/{id}` | ✅ | One run |
| `DELETE` | `/api/runs/{id}` | ✅ | Delete a run |

Interactive docs at **`/api/docs`**. Full reference in
[docs/api.md](docs/api.md).

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The context window is","model_id":"maidan-tiny","max_tokens":30,"seed":42}'
```

---

## Testing

```bash
pytest -q          # 64 passed
ruff check .       # All checks passed
```

The suite covers the parts where a silent bug is expensive:

- **Special-token ids** — a regression test for the bug that made `eos_id` point
  at index 0, so the first comma ended every generation after one token
- **Sampler maths** — softmax sums to 1, top-k keeps exactly k, top-p keeps the
  smallest nucleus, negative logits get *less* likely under repetition penalty
- **Reproducibility** — same seed produces identical output, different seeds diverge
- **Stop reasons** — a regression test for the bug where every run reported
  `max_tokens`: a generator's early `return` and its normal exhaustion look
  identical to the caller's `for` loop, so `for ... else` always took the `else`
  branch. The blocking and streaming paths are now asserted to agree.
- **Corpus shape** — every training entry is a complete sentence, because wrapped
  paragraphs taught the model that sentences end mid-clause
- **API contracts** — 401/404/409/422/503 paths, run ownership isolation, SSE frames

---

## Project layout

```
day-4-maidan/
├── app/
│   ├── engine/
│   │   ├── sampler.py       # temperature, top-k, top-p, repetition penalty
│   │   ├── tokenizer.py     # word/punctuation tokenizer with special tokens
│   │   ├── corpus.py        # bundled training data (sentence-per-line)
│   │   ├── local_model.py   # interpolated trigram model + registry
│   │   ├── generate.py      # the generation loop
│   │   └── cloud.py         # optional provider passthrough
│   ├── routers/             # auth, playground
│   ├── static/              # index.html, styles.css, app.js
│   ├── config.py            # settings from env only
│   ├── db.py                # engine + session
│   ├── models.py            # User, Run
│   ├── schemas.py           # request/response validation
│   ├── security.py          # bcrypt, JWT, auth dependency
│   ├── rate_limit.py        # sliding-window middleware
│   └── main.py              # app factory
├── api/index.py             # Vercel serverless entrypoint
├── tests/                   # 64 tests
├── docs/                    # setup, usage, API reference
├── ARCHITECTURE.md
├── Dockerfile
└── docker-compose.yml
```

---

## Configuration

Every variable is documented in [`.env.example`](.env.example). The ones that
matter:

| Variable | Default | Notes |
|---|---|---|
| `SECRET_KEY` | *(empty)* | **Required in production.** Blank in dev = ephemeral key, tokens die on restart |
| `DATABASE_URL` | `sqlite:///./maidan.db` | Postgres URL works as-is |
| `CORS_ORIGINS` | `*` | Comma-separated list in production |
| `RATE_LIMIT_PER_MINUTE` | `60` | Applies to `/api/auth/*` and `/api/generate*` |
| `OPENAI_API_KEY` | *(empty)* | Only enables the cloud toggle |
| `GROQ_API_KEY` | *(empty)* | Same |

---

## Known limits

Being straight about this, because the whole project is about being straight about
what a model is:

- **It is a trigram model.** It has no long-range coherence. It will produce a
  fluent clause and then wander. That is the honest behaviour of the architecture,
  not a bug.
- **The vocabulary is ~300 tokens.** Anything outside the corpus becomes `<unk>`.
- **Rate limiting is in-process.** Correct for one instance, wrong behind a load
  balancer. See ARCHITECTURE.md → Scaling.
- **SQLite on Vercel is ephemeral.** `/tmp` survives a warm invocation, not a cold
  start. Use Postgres for anything real.

---

## License

MIT — see [LICENSE](LICENSE).

---

Built as Day 4 of a 30-day autonomous engineering challenge. Day 1 was a
conversational assistant, Day 2 a multimodal workspace, Day 3 a long-context
reasoning tool. This one is the opposite: instead of hiding the model behind an
API, it puts a small one on the table and hands you the sliders.
