"""
agents/tech_support.py — Layer 2: Technical support worker with native tool calling.
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from typing import Literal

from tools.bank_tools import search_knowledge_base
from tools.schemas import CheckServiceStatusInput
from tools.wrapper import ToolError, reliable_tool

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


_SYSTEM = """You are the technical support specialist for a bank.

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

    def run(user_input: str, history: list[tuple[str, str]]) -> str:
        messages = [SystemMessage(content=_SYSTEM)]
        for role, text in history[-6:]:
            messages.append(HumanMessage(content=text) if role == "user" else AIMessage(content=text))
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
