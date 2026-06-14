"""
tools/state_tools.py — Layer 3: Session state tools.

Session state is a plain dict set at the start of each session.
Tools read/write it directly. No framework dependency.
"""
from tools.schemas import RememberCustomerInput
from tools.wrapper import reliable_tool

# Module-level session state — reset each session via set_session_state()
_session_state: dict = {}


def set_session_state(state: dict) -> None:
    global _session_state
    _session_state = state


def get_session_state() -> dict:
    return _session_state


@reliable_tool(RememberCustomerInput)
def remember_customer(customer_id: str) -> dict:
    """Pin a customer ID for the rest of this session.

    Call this as soon as the customer identifies themselves so you don't
    have to ask again. The ID is stored in session state and available
    to all agents in the session.

    Args:
        customer_id: Customer ID to pin. Format: CUST-NNNN.
    """
    _session_state["customer_id"] = customer_id
    _session_state["customer_verified"] = True
    return {
        "pinned": customer_id,
        "message": f"Customer {customer_id} pinned for this session.",
    }


def get_customer_from_state() -> str | None:
    return _session_state.get("customer_id")
