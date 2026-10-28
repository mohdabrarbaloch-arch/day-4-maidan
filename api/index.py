"""Vercel serverless entrypoint.

Two things matter here and both have bitten this account before:

1. ``includeFiles: "app/**"`` in vercel.json must resolve from the project root.
   The legacy ``builds`` block resolved it relative to ``api/``, so the ``app``
   package never shipped and every request died on ``ModuleNotFoundError``.

2. ``init_db()`` runs at import. Vercel imports the module and never fires
   lifespan events, so table creation inside a startup handler silently never
   happens in production.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the project root importable regardless of how Vercel sets sys.path.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Serverless filesystems are read-only apart from /tmp. Set this before the app
# module is imported, because the engine is built at import time.
os.environ.setdefault("VERCEL", "1")
if not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = "sqlite:////tmp/maidan.db"

from app.main import app  # noqa: E402  (must follow the env setup above)

__all__ = ["app"]
