# Setup

## Requirements

Python 3.11+, pip. Optional: Docker + Compose for the Postgres path.
No GPU, no API key, no model download.

## Local, five minutes

```bash
git clone https://github.com/mohdabrarbaloch-arch/day-4-maidan.git
cd day-4-maidan
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open http://localhost:8000. The first request trains the model (~40ms).
