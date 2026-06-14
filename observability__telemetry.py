"""
observability/telemetry.py — Layer 5: Structured logging and cost tracking.

Two instruments:
  1. Structured JSON logs — every turn and tool call becomes a log line
     you can ship to Cloud Logging, Datadog, or any log aggregator.
  2. Per-session cost tracking — input/output tokens and estimated cost per turn.

In production:
  - Ship logs to your observability stack (Datadog, Grafana, Cloud Logging).
  - Add Cloud Trace spans (ADK's adk deploy gives you this for free).
  - Wire cost_tracker.session_summary() into your billing dashboard.
"""
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

# Configure root logger for structured output
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
)
_log = logging.getLogger("bank_support.telemetry")


def _emit(record: dict) -> None:
    """Write a structured JSON log line."""
    _log.info(json.dumps(record, default=str))


# ── Turn logging ──────────────────────────────────────────────────────────────

def log_turn_start(session_id: str, user_input: str) -> float:
    """Log a new turn and return the start timestamp."""
    start = time.monotonic()
    _emit({
        "event": "turn_start",
        "session_id": session_id,
        "user_input_preview": user_input[:120],
        "ts": time.time(),
    })
    return start


def log_turn_end(
    session_id: str,
    response_preview: str,
    start_time: float,
    agent_name: str = "unknown",
) -> None:
    latency_ms = int((time.monotonic() - start_time) * 1000)
    _emit({
        "event": "turn_end",
        "session_id": session_id,
        "agent": agent_name,
        "response_preview": response_preview[:120],
        "latency_ms": latency_ms,
        "ts": time.time(),
    })


# ── Tool call logging ─────────────────────────────────────────────────────────

def log_tool_call(session_id: str, tool_name: str, args: dict) -> None:
    # Mask any sensitive fields before logging
    safe_args = {k: ("***" if k in {"idempotency_key"} else v) for k, v in args.items()}
    _emit({
        "event": "tool_call",
        "session_id": session_id,
        "tool": tool_name,
        "args": safe_args,
        "ts": time.time(),
    })


def log_tool_result(session_id: str, tool_name: str, result: dict) -> None:
    _emit({
        "event": "tool_result",
        "session_id": session_id,
        "tool": tool_name,
        "status": result.get("status"),
        "error_code": result.get("error_code"),
        "ts": time.time(),
    })


# ── Cost tracking ─────────────────────────────────────────────────────────────
# Gemini Flash pricing (approximate, May 2026)
_PRICE_INPUT_PER_1K  = 0.000075   # $0.075 per 1M input tokens
_PRICE_OUTPUT_PER_1K = 0.0003     # $0.30 per 1M output tokens


@dataclass
class _SessionCost:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    turn_count: int = 0


class CostTracker:
    def __init__(self):
        self._sessions: dict[str, _SessionCost] = defaultdict(_SessionCost)

    def record_turn(
        self,
        session_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> dict:
        cost = (
            input_tokens  * _PRICE_INPUT_PER_1K  / 1000 +
            output_tokens * _PRICE_OUTPUT_PER_1K / 1000
        )
        s = self._sessions[session_id]
        s.total_input_tokens  += input_tokens
        s.total_output_tokens += output_tokens
        s.total_cost_usd      += cost
        s.turn_count          += 1
        turn_summary = {
            "event": "cost_turn",
            "session_id": session_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "session_total_usd": round(s.total_cost_usd, 6),
        }
        _emit(turn_summary)
        return turn_summary

    def session_summary(self, session_id: str) -> dict:
        s = self._sessions[session_id]
        return {
            "session_id": session_id,
            "turns": s.turn_count,
            "total_input_tokens": s.total_input_tokens,
            "total_output_tokens": s.total_output_tokens,
            "total_cost_usd": round(s.total_cost_usd, 6),
        }


cost_tracker = CostTracker()
