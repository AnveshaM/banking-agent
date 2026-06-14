"""
agents/tech_support.py — Layer 2: Technical support worker.

Scope: API errors, service outages, integration questions, error codes.
Does NOT discuss billing — defers to triage for anything financial.
"""
from google.adk.agents import LlmAgent

from config import WORKER_MODEL
from tools.bank_tools import search_knowledge_base
from tools.schemas import CheckServiceStatusInput
from tools.wrapper import ToolError, reliable_tool

# ── Service status tool ──────────────────────────────────────────────────────

_SERVICE_HEALTH = {
    "api":       "healthy",
    "dashboard": "degraded",   # simulated partial outage
    "webhooks":  "healthy",
    "payments":  "healthy",
}


@reliable_tool(CheckServiceStatusInput)
def check_service_status(service: str) -> dict:
    """Check the current health of a named internal service.

    Args:
        service: One of: api, dashboard, webhooks, payments.
    """
    health = _SERVICE_HEALTH.get(service)
    if health is None:
        raise ToolError("NOT_FOUND", f"Unknown service '{service}'.", retryable=False)
    return {"service": service, "health": health}


# ── Agent ────────────────────────────────────────────────────────────────────

_INSTRUCTION = """
You are the technical support specialist.

SCOPE:
  • API errors and integration questions
  • Service outages and degradation
  • Error codes and debugging help

OUT OF SCOPE — defer to triage:
  • Invoices, refunds, billing queries (these go to billing_agent)
  • Account opening, KYC questions

WORKFLOW:
1. Identify the service or integration the user is having trouble with.
2. Call check_service_status if there could be an outage contributing to the issue.
3. Search the knowledge base for relevant documentation.
4. Diagnose the issue and provide a resolution or escalation path.
5. If you cannot resolve it, give the user the escalation contact:
   support@bank.example.com with their error details.

ERROR HANDLING:
  • INVALID_INPUT  → fix your arguments and retry
  • NOT_FOUND      → service name not recognised — tell the user and try alternatives
  • INTERNAL_ERROR → apologise, provide escalation path

Be precise with error codes. Quote them verbatim from the user's report.
""".strip()

tech_support_agent = LlmAgent(
    name="tech_support_agent",
    model=WORKER_MODEL,
    description=(
        "Handles technical issues: API errors, service outages, integration problems, "
        "error codes. Use when the user reports a system error, outage, or integration bug. "
        "Does NOT handle billing or invoice questions."
    ),
    instruction=_INSTRUCTION,
    tools=[check_service_status, search_knowledge_base],
)
