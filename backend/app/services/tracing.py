from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from starlette.requests import Request


TRACE_HEADER = "X-Trace-Id"
_logger = logging.getLogger("app.tracing")


def ensure_trace_id(request: Request) -> str:
    incoming = request.headers.get(TRACE_HEADER, "").strip()
    trace_id = incoming or str(uuid4())
    request.state.trace_id = trace_id
    return trace_id


def get_trace_id(request: Request) -> str | None:
    return getattr(request.state, "trace_id", None)


def emit_trace_event(event: str, **fields: object) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    _logger.info(json.dumps(payload, default=str, ensure_ascii=True))
