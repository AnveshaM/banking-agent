"""
tools/state_tools.py — Layer 3: Session state tools.

These tools write to ADK's session state — the structured scratchpad
that persists across turns without inflating the context window.

Key principle: don't make the model re-derive things it already found.
Pin customer IDs, auth status, and anything else needed across multiple turns.
"""
from google.adk.tools.tool_context import ToolContext

from tools.schemas import RememberCustomerInput
from tools.wrapper import reliable_tool


@reliable_tool(RememberCustomerInput)
def remember_customer(customer_id: str, tool_context: ToolContext) -> dict:
    """Pin a customer ID for the rest of this session.

    Call this as soon as the customer identifies themselves so you don't
    have to ask again. The ID is stored in session state and available
    to all agents in the session.

    Args:
        customer_id: Customer ID to pin. Format: CUST-NNNN.
    """
    tool_context.state["customer_id"] = customer_id
    tool_context.state["customer_verified"] = True
    return {
        "pinned": customer_id,
        "message": f"Customer {customer_id} pinned for this session.",
    }


def get_customer_from_state(tool_context: ToolContext) -> str | None:
    """Helper used by other tools to read the pinned customer ID."""
    return tool_context.state.get("customer_id")
