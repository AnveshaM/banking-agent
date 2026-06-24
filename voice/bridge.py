"""
voice/bridge.py — Bridge between LiveKit Agents and BankSupportOrchestrator.

Handles:
  - One BankSupportOrchestrator instance per voice session
  - Async wrapper around the synchronous agent.run() call
  - HITL proposal detection — raises an event after each turn
    so voice_agent.py can prompt the terminal and speak the approval
  - Session state initialisation for voice sessions
"""
import asyncio
import logging
import sys
import uuid
from pathlib import Path

# Make sure the parent repo is importable from voice/
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import BankSupportOrchestrator
from tools.bank_tools import execute_approved_refund
from tools.idempotency import get_proposal_queue
from tools.state_tools import set_session_state

log = logging.getLogger(__name__)


class VoiceBridge:
    """
    One instance per voice call session.

    Usage:
        bridge = VoiceBridge()
        response_text, agent_name = await bridge.run("I need a refund")

        # After the response is spoken, check for pending proposals
        approval_text = await bridge.handle_pending_proposals()
        if approval_text:
            # speak approval_text to the caller
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.orchestrator = BankSupportOrchestrator()

        # Initialise session state — session_id is used for idempotency keys
        set_session_state({
            "session_id": self.session_id,
            "user_id": f"voice-{self.session_id}",
        })
        log.info("[bridge] session %s started", self.session_id)

    async def run(self, user_input: str) -> tuple[str, str]:
        """
        Run one turn. Returns (response_text, agent_name).
        Runs the synchronous orchestrator in a thread pool to avoid
        blocking LiveKit's async event loop.
        """
        loop = asyncio.get_event_loop()
        response, agent_name = await loop.run_in_executor(
            None,
            lambda: self.orchestrator.run(user_input, voice_mode=True),
        )
        log.info("[bridge] %s → %s: %s", agent_name, self.session_id, response[:80])
        return response, agent_name

    async def handle_pending_proposals(self) -> str | None:
        """
        Check the HITL queue for pending refund proposals.
        If found, print to terminal and wait for approval input,
        then execute the approved refund.

        Returns the spoken confirmation text to say to the caller,
        or None if nothing was pending.
        """
        queue = get_proposal_queue()
        pending = queue.pending()
        if not pending:
            return None

        # There should be at most one proposal per turn (the agent's loop
        # runs one propose_refund per session call). Process the first one.
        proposal = pending[0]

        # Print to terminal for the human reviewer
        print(f"\n\033[1m\033[33m")
        print(f"  ┌─── HITL: REFUND APPROVAL REQUIRED ───────────────────────────┐")
        print(f"  │  Proposal:  {proposal['proposal_id']}")
        print(f"  │  Invoice:   {proposal['invoice_id']}  (£{proposal['amount_gbp']:.2f})")
        print(f"  │  Customer:  {proposal['customer']}")
        print(f"  │  Reason:    {proposal['reason']}")
        print(f"  └────────────────────────────────────────────────────────────────┘")
        print(f"\033[0m", end="", flush=True)

        # Non-blocking terminal input — runs in executor so LiveKit doesn't block
        loop = asyncio.get_event_loop()

        def _prompt() -> str:
            while True:
                choice = input("  Approve refund? [y/n]: ").strip().lower()
                if choice in ("y", "yes", "n", "no"):
                    return choice

        choice = await loop.run_in_executor(None, _prompt)

        if choice in ("y", "yes"):
            result = execute_approved_refund(proposal["proposal_id"])
            ref = result.get("refund_reference", "unknown")
            amount = proposal["amount_gbp"]
            print(f"  \033[32m✓ Refund executed — {ref}\033[0m\n")
            return (
                f"Great news — your refund of £{amount:.2f} has just been approved. "
                f"Your reference number is {ref}. "
                f"You should see the funds within 5 business days."
            )
        else:
            queue.reject(proposal["proposal_id"])
            print(f"  \033[31m✗ Rejected\033[0m\n")
            return (
                "I'm sorry — your refund request has been reviewed and "
                "we're unable to process it at this time. "
                "Please contact us at support@bank.example.com for more information."
            )
