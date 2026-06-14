"""
agent.py — Root triage agent.

ADK looks for `root_agent` at the module level in agent.py.
This is the orchestrator — its only job is to read intent and delegate.

The orchestrator does NOT have tools. It delegates to sub-agents.
Each sub-agent has a tight instruction and its own tool list.
Context isolation is automatic: each sub-agent's noise stays in its own context.
"""
from google.adk.agents import LlmAgent

from agents.billing import billing_agent
from agents.general import general_agent
from agents.tech_support import tech_support_agent
from config import ORCHESTRATOR_MODEL

_INSTRUCTION = """
You are the triage agent for a bank's customer support system.

Your ONLY job is to read the user's intent and delegate to the right specialist.
Do not answer questions yourself. Route immediately.

Routing rules:
  • Invoices, refunds, billing, charges, payment status → billing_agent
  • API errors, outages, integration bugs, error codes  → tech_support_agent
  • Greetings, general questions, unclear intent        → general_agent

When you delegate:
  1. Tell the user briefly: "Connecting you with our billing specialist..." (etc.)
  2. Transfer to the appropriate sub-agent.
  3. Do not attempt to answer the question yourself first.

If a sub-agent transfers back to you, re-read the user's latest message and
route again. If you genuinely cannot route, delegate to general_agent.
""".strip()

root_agent = LlmAgent(
    name="triage_agent",
    model=ORCHESTRATOR_MODEL,
    description="Top-level support triage. Routes to billing, tech, or general.",
    instruction=_INSTRUCTION,
    sub_agents=[billing_agent, tech_support_agent, general_agent],
)
