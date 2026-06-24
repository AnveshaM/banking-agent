"""
voice/agents_patch.py — Patched agent factories that accept voice_mode.

Replace your existing agents/billing.py, agents/general.py,
and agents/tech_support.py with these versions.

The only change from the originals: the run(user_input, history) closure
now accepts run(user_input, history, voice_mode=False).
When voice_mode=True, the voice brevity instruction is appended to the
system message before the LLM call.
"""
import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from typing import Literal

from guardrails.callbacks import check_before_tool
from tools.bank_tools import lookup_invoice, propose_refund, search_knowledge_base
from tools.schemas import CheckServiceStatusInput
from tools.state_tools import get_session_state, remember_customer
from tools.wrapper import ToolError, reliable_tool

# ── Voice brevity instruction ─────────────────────────────────────────────────
_VOICE_BREVITY = """

VOICE MODE — you are speaking aloud on a phone call:
- Maximum 2-3 short sentences per response.
- No bullet points, no lists, no markdown, no headers.
- Speak naturally. Don't say "Here is what I found:" — just say it.
- If you need to quote an invoice ID, say it digit by digit.
"""

# ─── BILLING AGENT ────────────────────────────────────────────────────────────

_BILLING_SYSTEM = """You are the billing specialist for a bank customer support system.

Your job: handle invoice lookups, refund proposals, and payment status questions.

Rules:
- Always use tools — never make up invoice data.
- If the customer hasn't given their customer ID (format CUST-NNNN), ask for it first, then call remember_customer.
- For invoice status: call lookup_invoice.
- For refunds: call lookup_invoice first to confirm status is 'paid', then call propose_refund.
  Use idempotency_key format: "propose-{invoice_id}-{session_id}"
- Tell the customer a refund is PENDING REVIEW, not issued.
- If the question is about API errors or technical issues (not billing), say you'll transfer them to tech support."""


def make_billing_agent(llm):
    def _make_tools(session_id: str):
        guarded = {"propose_refund"}

        @tool
        def lookup_invoice_tool(invoice_id: str) -> str:
            """Look up an invoice by its ID (format INV-NNNNN). Returns status, amount, and customer."""
            denied = check_before_tool("lookup_invoice", {"invoice_id": invoice_id}, session_id, guarded)
            if denied:
                return json.dumps(denied)
            return json.dumps(lookup_invoice(invoice_id=invoice_id))

        @tool
        def remember_customer_tool(customer_id: str) -> str:
            """Pin a customer ID (format CUST-NNNN) for this session."""
            denied = check_before_tool("remember_customer", {"customer_id": customer_id}, session_id, guarded)
            if denied:
                return json.dumps(denied)
            return json.dumps(remember_customer(customer_id=customer_id))

        @tool
        def propose_refund_tool(
            invoice_id: str,
            reason: str,
            idempotency_key: str,
            customer_message: str,
        ) -> str:
            """Propose a refund for a paid invoice. A human reviewer will approve or reject it."""
            args = {
                "invoice_id": invoice_id,
                "reason": reason,
                "idempotency_key": idempotency_key,
                "customer_message": customer_message,
            }
            denied = check_before_tool("propose_refund", args, session_id, guarded)
            if denied:
                return json.dumps(denied)
            return json.dumps(propose_refund(**args))

        return [lookup_invoice_tool, remember_customer_tool, propose_refund_tool]

    def run(
        user_input: str,
        history: list[tuple[str, str]],
        voice_mode: bool = False,
    ) -> str:
        session_id = get_session_state().get("session_id", "default")
        tools = _make_tools(session_id)
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        system_content = _BILLING_SYSTEM
        if voice_mode:
            system_content += _VOICE_BREVITY

        messages = [SystemMessage(content=system_content)]
        for role, text in history[-8:]:
            messages.append(
                HumanMessage(content=text) if role == "user" else AIMessage(content=text)
            )
        messages.append(HumanMessage(content=user_input))

        for _ in range(8):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return response.content.strip()

            for tc in response.tool_calls:
                fn = tool_map.get(tc["name"])
                if fn:
                    result = fn.invoke(tc["args"])
                else:
                    result = json.dumps({
                        "status": "error",
                        "error_message": f"Unknown tool: {tc['name']}",
                    })
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        return "I'm sorry, I couldn't complete your request. Please try again."

    return run


# ─── TECH SUPPORT AGENT ───────────────────────────────────────────────────────

_SERVICE_HEALTH = {
    "api":       "healthy",
    "dashboard": "degraded",
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


_TECH_SYSTEM = """You are the technical support specialist for a bank.

Handle: API errors, service outages, integration issues, error codes.
For billing questions (invoices, refunds), tell the user you'll connect them with billing.

Rules:
- Use check_service_status to check if a service is down before diagnosing.
- Use search_knowledge_base to find documentation.
- If you cannot resolve the issue, provide the escalation email: support@bank.example.com
- Quote error codes verbatim from the user's report."""


def make_tech_support_agent(llm):
    @tool
    def check_status_tool(service: str) -> str:
        """Check health of a service. Options: api, dashboard, webhooks, payments."""
        return json.dumps(check_service_status(service=service))

    @tool
    def search_kb_tool(query: str, top_k: int = 3) -> str:
        """Search the bank's knowledge base for technical documentation."""
        return json.dumps(search_knowledge_base(query=query, top_k=top_k))

    tools = [check_status_tool, search_kb_tool]
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    def run(
        user_input: str,
        history: list[tuple[str, str]],
        voice_mode: bool = False,
    ) -> str:
        system_content = _TECH_SYSTEM
        if voice_mode:
            system_content += _VOICE_BREVITY

        messages = [SystemMessage(content=system_content)]
        for role, text in history[-6:]:
            messages.append(
                HumanMessage(content=text) if role == "user" else AIMessage(content=text)
            )
        messages.append(HumanMessage(content=user_input))

        for _ in range(6):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return response.content.strip()

            for tc in response.tool_calls:
                fn = tool_map.get(tc["name"])
                result = fn.invoke(tc["args"]) if fn else json.dumps({"error": "Unknown tool"})
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        return "I'm sorry, I couldn't complete your request. Please try again."

    return run


# ─── GENERAL AGENT ────────────────────────────────────────────────────────────

_GENERAL_SYSTEM = """You are the general support agent for a bank — friendly and concise.

Handle greetings, policy questions, and unclear requests.
For billing questions (invoices, refunds), tell the user you'll connect them with billing.
For technical issues (API errors, outages), tell the user you'll connect them with tech support.

Use the search_knowledge_base tool to answer policy questions."""


def make_general_agent(llm):
    @tool
    def search_kb_tool(query: str, top_k: int = 3) -> str:
        """Search the bank's knowledge base for policy and procedure information."""
        return json.dumps(search_knowledge_base(query=query, top_k=top_k))

    tools = [search_kb_tool]
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    def run(
        user_input: str,
        history: list[tuple[str, str]],
        voice_mode: bool = False,
    ) -> str:
        system_content = _GENERAL_SYSTEM
        if voice_mode:
            system_content += _VOICE_BREVITY

        messages = [SystemMessage(content=system_content)]
        for role, text in history[-6:]:
            messages.append(
                HumanMessage(content=text) if role == "user" else AIMessage(content=text)
            )
        messages.append(HumanMessage(content=user_input))

        for _ in range(4):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return response.content.strip()

            for tc in response.tool_calls:
                fn = tool_map.get(tc["name"])
                result = fn.invoke(tc["args"]) if fn else json.dumps({"error": "Unknown tool"})
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        return "I'm sorry, I couldn't complete your request. Please try again."

    return run
