"""config.py — Central configuration. All env reads live here."""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Model selection ──────────────────────────────────────────────────────────
# Swap ORCHESTRATOR_MODEL to a frontier model (e.g. gemini-2.0-pro) for
# more reliable routing. WORKER_MODEL can be a smaller/cheaper model.
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "gemini-2.0-flash")
WORKER_MODEL       = os.getenv("WORKER_MODEL",       "gemini-2.0-flash")

# ── App identity ─────────────────────────────────────────────────────────────
APP_NAME = "bank_support"

# ── Circuit breaker limits ────────────────────────────────────────────────────
# How many refund proposals the agent can make per session.
REFUND_PROPOSALS_PER_SESSION = 3
# How many tool calls per minute before we rate-limit.
TOOL_CALLS_PER_MINUTE = 20
