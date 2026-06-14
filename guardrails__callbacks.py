"""
guardrails/callbacks.py — Layer 4: ADK tool callbacks.

before_tool_callback runs before every tool call in the agent loop.
Returning a dict overrides the tool's response — the tool does NOT run.
Returning None lets the tool run normally.

The circuit breaker check happens here. The model cannot bypass it.
"""
import logging
from typing import Optional

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from guardrails.circuit_breaker import get_breaker
from observability.telemetry import log_tool_call

log = logging.getLogger(__name__)


def make_before_tool_callback(guarded_tools: set[str]):
    """
    Factory that returns a before_tool_callback guarding the specified tools.

    Args:
        guarded_tools: Names of tools that should go through the circuit breaker.
    """
    def before_tool_callback(
        tool: BaseTool,
        args: dict,
        tool_context: ToolContext,
    ) -> Optional[dict]:
        session_id = tool_context.state.get("session_id", "default")

        # ── Log the tool call for observability ───────────────────────────────
        log_tool_call(
            session_id=session_id,
            tool_name=tool.name,
            args=args,
        )

        # ── Circuit breaker check for guarded tools ───────────────────────────
        if tool.name in guarded_tools:
            breaker = get_breaker()
            allowed, reason = breaker.check(session_id)
            if not allowed:
                log.warning("[circuit_breaker] DENIED %s in session %s: %s",
                            tool.name, session_id, reason)
                # Return a structured error — same shape as the error envelope
                # so the agent's error-handling instruction applies.
                return {
                    "status": "error",
                    "error_code": "CIRCUIT_OPEN",
                    "error_message": reason,
                    "retryable": False,
                }

            # Record the call AFTER letting it through
            # (we record success/failure in after_tool_callback or in the tool itself)
            breaker.record_call(session_id, success=True)
            log.info("[circuit_breaker] ALLOWED %s in session %s", tool.name, session_id)

        return None  # proceed normally

    return before_tool_callback
