"""config.py — Central configuration. All env reads live here."""
import os
from dotenv import load_dotenv

load_dotenv()

# ── HuggingFace config ───────────────────────────────────────────────────────
# Get a free token at https://huggingface.co/settings/tokens
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ── Model selection ──────────────────────────────────────────────────────────
# Free HuggingFace Inference API models. Qwen2.5-72B is the most capable free option.
# Alternatives: "mistralai/Mistral-7B-Instruct-v0.3", "HuggingFaceH4/zephyr-7b-beta"
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "Qwen/Qwen2.5-72B-Instruct")
WORKER_MODEL       = os.getenv("WORKER_MODEL",       "Qwen/Qwen2.5-72B-Instruct")

# ── App identity ─────────────────────────────────────────────────────────────
APP_NAME = "bank_support"

# ── Circuit breaker limits ────────────────────────────────────────────────────
# How many refund proposals the agent can make per session.
REFUND_PROPOSALS_PER_SESSION = 3
# How many tool calls per minute before we rate-limit.
TOOL_CALLS_PER_MINUTE = 20
