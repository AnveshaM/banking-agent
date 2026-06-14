"""
agents/billing.py — Layer 2: Billing worker agent.

Uses native tool calling via the OpenAI-compatible HF endpoint.
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from guardrails.callbacks import check_before_tool
from tools.bank_tools import lookup_invoice, propose_refund
from tools.state_tools import get_session_state, remember_customer
from tools.schemas import LookupInvoiceInput, ProposeRefundInput, RememberCustomerInput

_SYSTEM = """You are the billing specialist for a bank customer support system.

Your job: handle invoice lookups, refund proposals, and payment status questions.

Rules:
- Always use tools — never make up invoice data.
- If the customer hasn't given their customer ID (format CUST-NNNN), ask for it first, then call remember_customer.
- For invoice status: call lookup_invoice.
- For refunds: call lookup_invoice first to confirm status is 'paid', then call propose_refund.
  Use idempotency_key format: "propose-{invoice_id}-{session_id}"
- Tell the customer a refund is PENDING REVIEW, not issued.
- If the question is about API errors or technical issues (not billing), say you'll transfer them to tech support."""


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
    def propose_refund_tool(invoice_id: str, reason: str, idempotency_key: str, customer_message: str) -> str:
        """Propose a refund for a paid invoice. A human reviewer will approve or reject it."""
        args = {"invoice_id": invoice_id, "reason": reason, "idempotency_key": idempotency_key, "customer_message": customer_message}
        denied = check_before_tool("propose_refund", args, session_id, guarded)
        if denied:
            return json.dumps(denied)
        return json.dumps(propose_refund(**args))

    return [lookup_invoice_tool, remember_customer_tool, propose_refund_tool]


def make_billing_agent(llm):
    def run(user_input: str, history: list[tuple[str, str]]) -> str:
        session_id = get_session_state().get("session_id", "default")
        tools = _make_tools(session_id)
        llm_with_tools = llm.bind_tools(tools)
        tool_map = {t.name: t for t in tools}

        messages = [SystemMessage(content=_SYSTEM)]
        for role, text in history[-8:]:
            messages.append(HumanMessage(content=text) if role == "user" else AIMessage(content=text))
        messages.append(HumanMessage(content=user_input))

        for _ in range(8):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return response.content.strip()

            # Execute each tool call and feed results back
            for tc in response.tool_calls:
                fn = tool_map.get(tc["name"])
                if fn:
                    result = fn.invoke(tc["args"])
                else:
                    result = json.dumps({"status": "error", "error_message": f"Unknown tool: {tc['name']}"})
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        return "I'm sorry, I couldn't complete your request. Please try again."

    return run
