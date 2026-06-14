"""
agents/general.py — Fallback general agent.

Handles greetings, FAQs, and anything that doesn't fit billing or tech.
Always routes substantive requests back to triage.
"""
from google.adk.agents import LlmAgent

from config import WORKER_MODEL
from tools.bank_tools import search_knowledge_base

_INSTRUCTION = """
You are the general support agent — the friendly fallback.

Your job is to handle:
  • Greetings, small talk, simple questions
  • Policy questions not covered by billing or tech (use search_knowledge_base)
  • Requests where the intent isn't yet clear

For anything clearly billing or technical, say:
  "Let me connect you with the right specialist." and stop.

Keep your replies concise and friendly. If you're not sure, search first.
""".strip()

general_agent = LlmAgent(
    name="general_agent",
    model=WORKER_MODEL,
    description=(
        "Handles greetings, FAQ questions, small talk, and requests that don't clearly "
        "fit billing or technical support. Always the last-resort fallback."
    ),
    instruction=_INSTRUCTION,
    tools=[search_knowledge_base],
)
