"""
guardrails/circuit_breaker.py — Layer 4: Circuit breaker.

Enforces hard limits on state-mutating tool calls.
Lives OUTSIDE the agent loop — the model cannot prompt-inject past code.

Three limits per session:
  1. max_calls_per_session — absolute cap on refund proposals
  2. max_calls_per_minute  — rate limit
  3. max_consecutive_failures — trip on repeated errors (prevents loops)
"""
import time
from collections import defaultdict
from dataclasses import dataclass, field

from config import REFUND_PROPOSALS_PER_SESSION, TOOL_CALLS_PER_MINUTE


@dataclass
class CircuitBreaker:
    max_calls_per_session: int = REFUND_PROPOSALS_PER_SESSION
    max_calls_per_minute: int = TOOL_CALLS_PER_MINUTE
    max_consecutive_failures: int = 3

    # Per-session counters
    _call_count: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _recent_timestamps: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _consecutive_failures: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def check(self, session_id: str) -> tuple[bool, str | None]:
        """
        Returns (allowed, reason_if_denied).
        Call BEFORE executing any guarded tool.
        """
        # Absolute per-session cap
        if self._call_count[session_id] >= self.max_calls_per_session:
            return False, (
                f"Session limit of {self.max_calls_per_session} refund proposals reached. "
                "Please contact support@bank.example.com to continue."
            )

        # Per-minute rate limit
        now = time.monotonic()
        recent = [t for t in self._recent_timestamps[session_id] if now - t < 60]
        self._recent_timestamps[session_id] = recent
        if len(recent) >= self.max_calls_per_minute:
            return False, "Too many requests per minute. Please wait a moment and try again."

        # Consecutive failure trip
        if self._consecutive_failures[session_id] >= self.max_consecutive_failures:
            return False, (
                f"Circuit breaker tripped after {self.max_consecutive_failures} consecutive "
                "failures. Contact support@bank.example.com."
            )

        return True, None

    def record_call(
        self,
        session_id: str,
        *,
        success: bool,
    ) -> None:
        """Call AFTER executing a guarded tool to update counters."""
        self._call_count[session_id] += 1
        self._recent_timestamps[session_id].append(time.monotonic())
        if success:
            self._consecutive_failures[session_id] = 0
        else:
            self._consecutive_failures[session_id] += 1

    def reset_session(self, session_id: str) -> None:
        """Clear state for a session (testing / admin)."""
        self._call_count.pop(session_id, None)
        self._recent_timestamps.pop(session_id, None)
        self._consecutive_failures.pop(session_id, None)


# Module-level singleton
_breaker = CircuitBreaker()


def get_breaker() -> CircuitBreaker:
    return _breaker
