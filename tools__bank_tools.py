"""
tools/bank_tools.py — Production bank tools.

Three tools for the billing agent:
  - lookup_invoice    : read-only · Pydantic + error envelope
  - propose_refund    : write · all three contracts
  - search_kb         : read-only · Pydantic + error envelope

Layer 4 note: the agent gets propose_refund, NOT issue_refund.
The actual execution happens outside the agent loop, after human approval.
"""
import random
import uuid

from tools.idempotency import get_proposal_queue, get_store
from tools.schemas import LookupInvoiceInput, ProposeRefundInput, SearchKBInput
from tools.wrapper import ToolError, reliable_tool

# ── Fake databases ────────────────────────────────────────────────────────────

INVOICE_DB: dict[str, dict] = {
    "INV-00123": {"amount_gbp": 49.00,  "status": "paid",     "customer": "Acme Corp",  "customer_id": "CUST-0001"},
    "INV-00456": {"amount_gbp": 120.00, "status": "paid",     "customer": "Globex Ltd", "customer_id": "CUST-0002"},
    "INV-00789": {"amount_gbp": 75.00,  "status": "unpaid",   "customer": "Initech",    "customer_id": "CUST-0003"},
    "INV-00999": {"amount_gbp": 200.00, "status": "refunded", "customer": "Umbrella",   "customer_id": "CUST-0004"},
}

KNOWLEDGE_BASE = [
    {"id": "kb-001", "title": "Refund policy",
     "content": "Refunds are processed within 5–7 business days. Only invoices with status 'paid' are eligible. Refunds must be requested within 30 days of the invoice date."},
    {"id": "kb-002", "title": "Invoice numbering",
     "content": "Invoice IDs follow the format INV-NNNNN (five digits). Example: INV-00123. If you can't find an invoice, check the format — three or four digits is a common mistake."},
    {"id": "kb-003", "title": "Contact and escalation",
     "content": "For urgent issues or disputes, email support@bank.example.com. For refunds over £500, a manager review is required and may take up to 10 business days."},
    {"id": "kb-004", "title": "Payment methods and timing",
     "content": "We accept BACS, CHAPS, and card payments. Refunds go back to the original payment method. BACS refunds take 3–5 days, CHAPS same day."},
    {"id": "kb-005", "title": "Data and privacy",
     "content": "We process financial data under GDPR. You can request a copy of your data or a deletion under our data subject rights process. Contact privacy@bank.example.com."},
]


# ── Tool 1: lookup_invoice ────────────────────────────────────────────────────

@reliable_tool(LookupInvoiceInput)
def lookup_invoice(invoice_id: str) -> dict:
    """Look up an invoice by its ID.

    Returns the invoice amount, status, and customer name.
    Fails with NOT_FOUND if the invoice does not exist.

    Args:
        invoice_id: Invoice identifier. Format: INV-NNNNN. Example: INV-00123.
    """
    invoice = INVOICE_DB.get(invoice_id)
    if invoice is None:
        raise ToolError(
            "NOT_FOUND",
            f"Invoice {invoice_id} does not exist. Please verify the ID with the customer.",
            retryable=False,
        )
    return {"invoice_id": invoice_id, **invoice}


# ── Tool 2: propose_refund — all three contracts ───────────────────────────────
# NOTE: The agent gets THIS tool, not issue_refund.
# A human reviews proposals from the queue and approves/rejects.
# The actual refund execution happens in tools/execute.py, outside the agent.

_upstream_fail_count = 0


@reliable_tool(ProposeRefundInput)
def propose_refund(
    invoice_id: str,
    reason: str,
    idempotency_key: str,
    customer_message: str,
) -> dict:
    """Propose a refund for a paid invoice. A human will review and execute.

    IMPORTANT: Use idempotency_key = f"propose-{invoice_id}-{session_id}" to
    prevent duplicate proposals on retry. Replays return the original proposal.

    Args:
        invoice_id:        Invoice to refund. Must exist and have status 'paid'.
        reason:            Human-readable reason. Min 10 characters.
        idempotency_key:   Format: propose-{invoice_id}-{session_id}. Replay-safe.
        customer_message:  What you will tell the customer about this refund.
    """
    store = get_store()
    queue = get_proposal_queue()

    # ── Idempotency check ────────────────────────────────────────────────────
    cached = store.get(idempotency_key)
    if cached is not None:
        return {**cached, "_idempotency": "replayed"}

    # ── Business validation ──────────────────────────────────────────────────
    invoice = INVOICE_DB.get(invoice_id)
    if invoice is None:
        raise ToolError("NOT_FOUND", f"Invoice {invoice_id} not found.", retryable=False)

    if invoice["status"] == "refunded":
        raise ToolError("INVALID_STATE", f"Invoice {invoice_id} is already refunded.", retryable=False)

    if invoice["status"] != "paid":
        raise ToolError(
            "INVALID_STATE",
            f"Invoice {invoice_id} cannot be refunded — status is '{invoice['status']}'.",
            retryable=False,
        )

    # ── Simulate occasional upstream write failure (shows retry logic) ────────
    global _upstream_fail_count
    if _upstream_fail_count < 1 and random.random() < 0.35:
        _upstream_fail_count += 1
        raise ToolError("UPSTREAM_5XX", "Proposal service returned 503. Retrying.", retryable=True)
    _upstream_fail_count = 0

    # ── Queue the proposal for human review ───────────────────────────────────
    proposal = {
        "invoice_id": invoice_id,
        "amount_gbp": invoice["amount_gbp"],
        "customer": invoice["customer"],
        "reason": reason,
        "customer_message": customer_message,
        "_idempotency": "executed",
    }
    proposal_id = queue.enqueue(proposal)
    result = {**proposal, "proposal_id": proposal_id, "state": "pending_review"}

    # Store for replay
    store.set(idempotency_key, result)
    return result


# ── Tool 3: search_kb ─────────────────────────────────────────────────────────

@reliable_tool(SearchKBInput)
def search_knowledge_base(query: str, top_k: int = 3) -> dict:
    """Search the internal knowledge base for policy and procedure information.

    Use this for refund policy, invoice formatting, payment timelines, and
    any question the user has about how the bank's systems work.

    Args:
        query:  The user's question in their own words.
        top_k:  Number of results (default 3, max 5).
    """
    q = query.lower()
    scored = []
    for article in KNOWLEDGE_BASE:
        score = sum(
            1 for word in q.split()
            if word in article["title"].lower() or word in article["content"].lower()
        )
        if score > 0:
            scored.append((score, article))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [a for _, a in scored[:top_k]] or KNOWLEDGE_BASE[:top_k]
    return {"query": query, "results": results, "count": len(results)}


# ── Execute approved refund (called by runtime, NOT the agent) ────────────────

def execute_approved_refund(proposal_id: str) -> dict:
    """
    Called by the HITL approval flow — never by the agent.
    This is what actually moves the money.
    """
    queue = get_proposal_queue()
    proposal = queue.approve(proposal_id)
    if proposal is None:
        return {"status": "error", "error": f"Proposal {proposal_id} not found or not pending."}

    invoice_id = proposal["invoice_id"]
    if invoice_id in INVOICE_DB:
        INVOICE_DB[invoice_id]["status"] = "refunded"

    ref = f"REF-{uuid.uuid4().hex[:8].upper()}"
    return {
        "status": "success",
        "refund_reference": ref,
        "invoice_id": invoice_id,
        "amount_gbp": proposal["amount_gbp"],
        "customer": proposal["customer"],
    }
