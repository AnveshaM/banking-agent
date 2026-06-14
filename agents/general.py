"""
agents/general.py — Fallback general agent with native tool calling.
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from tools.bank_tools import search_knowledge_base

_SYSTEM = """You are the general support agent for a bank — friendly and concise.

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

    def run(user_input: str, history: list[tuple[str, str]]) -> str:
        messages = [SystemMessage(content=_SYSTEM)]
        for role, text in history[-6:]:
            messages.append(HumanMessage(content=text) if role == "user" else AIMessage(content=text))
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
