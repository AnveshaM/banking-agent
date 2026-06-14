"""
tools/idempotency.py — Layer 1: Idempotency store.

Prevents duplicate state mutations when the agent retries.
Swap InMemoryIdempotencyStore for a Redis/Postgres-backed one in production.
"""
import threading
from datetime import datetime, timedelta


class InMemoryIdempotencyStore:
    """Thread-safe, TTL-expiring in-memory idempotency store."""

    def __init__(self, ttl_hours: int = 24):
        self._store: dict[str, tuple[dict, datetime]] = {}
        self._ttl = timedelta(hours=ttl_hours)
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            result, stored_at = entry
            if datetime.utcnow() - stored_at > self._ttl:
                del self._store[key]
                return None
            return result

    def set(self, key: str, result: dict) -> None:
        with self._lock:
            self._store[key] = (result, datetime.utcnow())

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# Singleton — swap for dependency injection in production
_store = InMemoryIdempotencyStore()


def get_store() -> InMemoryIdempotencyStore:
    return _store


# ── Proposal queue (for HITL) ─────────────────────────────────────────────────

class ProposalQueue:
    """
    Pending refund proposals waiting for human approval.
    In production: back this with a database and a dashboard UI.
    """
    def __init__(self):
        self._proposals: list[dict] = []
        self._lock = threading.Lock()

    def enqueue(self, proposal: dict) -> str:
        import uuid
        proposal_id = f"PROP-{uuid.uuid4().hex[:8].upper()}"
        with self._lock:
            self._proposals.append({**proposal, "proposal_id": proposal_id, "state": "pending"})
        return proposal_id

    def pending(self) -> list[dict]:
        with self._lock:
            return [p for p in self._proposals if p["state"] == "pending"]

    def approve(self, proposal_id: str) -> dict | None:
        with self._lock:
            for p in self._proposals:
                if p["proposal_id"] == proposal_id:
                    p["state"] = "approved"
                    return p
        return None

    def reject(self, proposal_id: str) -> dict | None:
        with self._lock:
            for p in self._proposals:
                if p["proposal_id"] == proposal_id:
                    p["state"] = "rejected"
                    return p
        return None

    def clear(self) -> None:
        with self._lock:
            self._proposals.clear()


_proposal_queue = ProposalQueue()


def get_proposal_queue() -> ProposalQueue:
    return _proposal_queue
