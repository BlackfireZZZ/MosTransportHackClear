from __future__ import annotations

import logging
import re
from time import perf_counter
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

REQUEST_ID_HEADER = "X-Request-ID"
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, logs the outcome, and turns a crash into JSON.

    The JSON-on-crash part lives here rather than in an `Exception` handler on the
    app. Such a handler is invoked by Starlette's ServerErrorMiddleware, which sits
    outside every middleware this app adds -- including CORS -- so its response
    carries no Access-Control-Allow-Origin. A browser then rejects the 500 as a CORS
    failure and never reads the body, which defeats the point of answering in JSON.
    Catching here keeps the response inside the chain, so it travels back out
    through CORS like any other.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._logger = logging.getLogger("tramflow.http")

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        supplied_id = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = supplied_id if _SAFE_REQUEST_ID.fullmatch(supplied_id) else uuid4().hex
        started_at = perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
        except Exception:
            self._logger.exception(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": getattr(request.scope.get("route"), "path", "<unmatched>"),
                },
            )
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal Server Error", "request_id": request_id},
            )
        else:
            status_code = response.status_code
        finally:
            self._logger.info(
                "request_completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": getattr(request.scope.get("route"), "path", "<unmatched>"),
                    "status_code": status_code,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 3),
                },
            )

        response.headers[REQUEST_ID_HEADER] = request_id
        return response
