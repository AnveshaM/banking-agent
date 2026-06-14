"""
tools/wrapper.py — Layer 1: reliable_tool decorator.

Every tool goes through this wrapper:
  - Pydantic validation before the function runs
  - Uniform error envelope on any failure
  - Exponential-backoff retry on retryable errors
  - Stack-trace sanitisation (raw exceptions never reach the model)
"""
import logging
import time
from functools import wraps
from typing import Callable, Type

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)


class ToolError(Exception):
    """Raise inside a tool for expected failures. The wrapper catches it."""
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


def reliable_tool(
    input_model: Type[BaseModel],
    *,
    max_retries: int = 2,
    backoff_s: float = 0.5,
):
    """
    Decorator that adds:
      1. Pydantic input validation
      2. Uniform { status, error_code, error_message, retryable } envelope
      3. Retry with exponential backoff for retryable failures

    Error codes the model learns once, uses everywhere:
      INVALID_INPUT   — fix args and retry
      NOT_FOUND       — tell the user, don't retry
      INVALID_STATE   — wrong state (already refunded, unpaid, etc)
      UPSTREAM_5XX    — wrapper already retried; tell user there's an outage
      CIRCUIT_OPEN    — guardrail tripped; stop completely
      INTERNAL_ERROR  — stop and apologise; never retry
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(**kwargs) -> dict:
            # ── 1. Validate ───────────────────────────────────────────────────
            try:
                validated = input_model(**kwargs)
            except ValidationError as exc:
                log.warning("[%s] INVALID_INPUT: %s", fn.__name__, _first_error(exc))
                return {
                    "status": "error",
                    "error_code": "INVALID_INPUT",
                    "error_message": _first_error(exc),
                    "retryable": False,
                }

            # ── 2. Run with retry ─────────────────────────────────────────────
            attempt = 0
            while True:
                try:
                    result = fn(**validated.model_dump())
                    return {"status": "success", "data": result}

                except ToolError as exc:
                    if exc.retryable and attempt < max_retries:
                        wait = backoff_s * (2 ** attempt)
                        log.info(
                            "[%s] %s — attempt %d/%d, retrying in %.1fs",
                            fn.__name__, exc.code, attempt + 1, max_retries, wait,
                        )
                        time.sleep(wait)
                        attempt += 1
                        continue
                    log.warning("[%s] %s: %s", fn.__name__, exc.code, exc.message)
                    return {
                        "status": "error",
                        "error_code": exc.code,
                        "error_message": exc.message,
                        "retryable": exc.retryable,
                    }

                except Exception:
                    log.exception("[%s] unexpected error", fn.__name__)
                    return {
                        "status": "error",
                        "error_code": "INTERNAL_ERROR",
                        "error_message": "An internal error occurred. The team has been notified.",
                        "retryable": False,
                    }

        wrapper.__doc__ = fn.__doc__
        return wrapper
    return decorator


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors(include_url=False)
    if not errors:
        return str(exc)
    first = errors[0]
    field = " → ".join(str(loc) for loc in first.get("loc", []))
    msg = first.get("msg", "validation error").replace("Value error, ", "")
    return f"Field '{field}': {msg}" if field else msg
