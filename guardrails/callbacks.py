"""
guardrails/callbacks.py — Layer 4: Pre-tool guardrail checks.

Called before any guarded tool executes. Returns an error envelope dict if
the circuit breaker denies the call, otherwise returns None to proceed.
"""
import logging

from guardrails.circuit_breaker import get_breaker
from observability.telemetry import log_tool_call

log = logging.getLogger(__name__)


def check_before_tool(
    tool_name: str,
    args: dict,
    session_id: str,
    guarded_tools: set[str],
) -> dict | None:
    """
    Run before a tool call. Returns an error envelope if denied, else None.

    Args:
        tool_name:     Name of the tool about to run.
        args:          Arguments being passed to the tool.
        session_id:    Current session ID for circuit breaker scoping.
        guarded_tools: Set of tool names subject to circuit breaker checks.
    """
    log_tool_call(session_id=session_id, tool_name=tool_name, args=args)

    if tool_name in guarded_tools:
        breaker = get_breaker()
        allowed, reason = breaker.check(session_id)
        if not allowed:
            log.warning("[circuit_breaker] DENIED %s in session %s: %s",
                        tool_name, session_id, reason)
            return {
                "status": "error",
                "error_code": "CIRCUIT_OPEN",
                "error_message": reason,
                "retryable": False,
            }
        breaker.record_call(session_id, success=True)
        log.info("[circuit_breaker] ALLOWED %s in session %s", tool_name, session_id)

    return None
