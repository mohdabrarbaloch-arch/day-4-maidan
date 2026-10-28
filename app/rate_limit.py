"""Rate limiting as middleware.

Deliberately not a decorator. slowapi's ``@limiter.limit`` wraps the endpoint
function, and with ``from __future__ import annotations`` in play the wrapper's
resolved signature loses the body parameter — FastAPI then reads ``payload`` as
a query string and every POST returns 422. A middleware sidesteps the whole
class of problem: it sees the request before routing and never touches the
handler's signature.

Storage is an in-process sliding window. That is correct for one instance and
explicitly insufficient behind a load balancer — see ARCHITECTURE.md.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window counter keyed by client IP and route path."""

    def __init__(self, app, per_minute: int = 60, window_seconds: int = 60) -> None:
        super().__init__(app)
        self.per_minute = per_minute
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        # X-Forwarded-For is set by every proxy in front of us; the first entry
        # is the original client. Fall back to the socket address locally.
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _is_limited_route(self, path: str) -> bool:
        # Only the expensive or abusable routes. Static assets and health checks
        # must never be throttled — a rate-limited health check reads as an
        # outage to every monitor watching it.
        return path.startswith("/api/auth") or path.startswith("/api/generate")

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        if not settings.rate_limit_enabled or not self._is_limited_route(request.url.path):
            return await call_next(request)

        key = f"{self._client_key(request)}:{request.url.path}"
        now = time.monotonic()
        bucket = self._hits[key]

        while bucket and now - bucket[0] > self.window:
            bucket.popleft()

        if len(bucket) >= self.per_minute:
            retry_after = int(self.window - (now - bucket[0])) + 1
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limited",
                    "detail": (
                        f"Too many requests. Limit is {self.per_minute} per minute "
                        f"for this endpoint."
                    ),
                },
                headers={"Retry-After": str(max(retry_after, 1))},
            )

        bucket.append(now)
        return await call_next(request)
