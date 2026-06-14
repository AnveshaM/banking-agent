"""
main.py — Interactive CLI runner for the bank support agent.

Runs the full 5-layer production agent in your terminal:

  python main.py

After each turn, the runner checks for pending refund proposals.
If found, you're prompted to approve or reject — this is the HITL layer.
Approved proposals execute the refund. Rejected proposals notify the agent.

Structured JSON logs stream to stdout alongside the conversation.
A cost summary prints at session end.
"""
import sys
import uuid
import logging

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent import root_agent
from config import APP_NAME
from observability.telemetry import cost_tracker, log_turn_end, log_turn_start
from tools.bank_tools import execute_approved_refund
from tools.idempotency import get_proposal_queue

# ── Logging setup ─────────────────────────────────────────────────────────────
# Structured logs go to stderr so they don't pollute the conversation output.
# In production, ship these to your log aggregator.
log = logging.getLogger("bank_support.main")
logging.getLogger("bank_support.telemetry").setLevel(logging.INFO)
for noisy in ["google", "httpx", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Terminal colours ──────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
WHITE  = "\033[37m"


def _print_header():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════╗
║          Bank Support Agent — Production Grade · All 5 Layers         ║
╚══════════════════════════════════════════════════════════════════════╝{RESET}

{DIM}  Layer 1: Tooling   — Pydantic contracts + error envelopes + idempotency
  Layer 2: Agents    — Triage → billing / tech / general workers
  Layer 3: State     — Session state (customer ID pinning)
  Layer 4: Guardrails— Circuit breaker + HITL propose pattern
  Layer 5: Observ.   — Structured JSON logs + cost tracking

  Try: "What's the status of invoice INV-00123?"
  Or:  "I'd like a refund on INV-00456"
  Or:  "The dashboard is showing an error"
  Or:  "What's your refund policy?"

  Type 'quit' or 'exit' to end the session.{RESET}
""")


def _handle_pending_proposals() -> None:
    """
    HITL approval loop. Runs after every agent turn.

    In production this is a dashboard or a Slack message to a human reviewer.
    Here it's a simple CLI prompt.
    """
    queue = get_proposal_queue()
    pending = queue.pending()
    if not pending:
        return

    print(f"\n{BOLD}{YELLOW}  ─── PENDING REFUND PROPOSALS ──────────────────────────────────{RESET}")
    for p in pending:
        print(f"""
  {YELLOW}Proposal ID: {p['proposal_id']}{RESET}
    Invoice:    {p['invoice_id']}  (£{p['amount_gbp']:.2f})
    Customer:   {p['customer']}
    Reason:     {p['reason']}
    Message:    {p['customer_message']}
""")
        while True:
            choice = input(f"  {BOLD}Approve this refund? [y/n]: {RESET}").strip().lower()
            if choice in ("y", "yes"):
                result = execute_approved_refund(p["proposal_id"])
                print(f"  {GREEN}✓ Refund executed. Reference: {result.get('refund_reference')}{RESET}\n")
                break
            elif choice in ("n", "no"):
                queue.reject(p["proposal_id"])
                print(f"  {RED}✗ Proposal rejected.{RESET}\n")
                break
            else:
                print("  Please enter 'y' or 'n'.")

    print(f"  {DIM}───────────────────────────────────────────────────────────────{RESET}\n")


def run_session():
    _print_header()

    session_service = InMemorySessionService()
    session_id = str(uuid.uuid4())[:8]
    user_id = "user_demo"

    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={"session_id": session_id},   # Layer 3: pin session_id in state
    )

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    print(f"  {DIM}Session: {session_id} · type 'quit' to end{RESET}\n")

    while True:
        # ── Get user input ────────────────────────────────────────────────────
        try:
            user_input = input(f"{BOLD}{WHITE}You: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ("quit", "exit", "q", "bye"):
            break
        if not user_input:
            continue

        # ── Layer 5: log turn start ───────────────────────────────────────────
        turn_start = log_turn_start(session_id, user_input)

        # ── Run the agent ─────────────────────────────────────────────────────
        content = types.Content(
            role="user",
            parts=[types.Part(text=user_input)],
        )

        final_response = ""
        responding_agent = "unknown"

        try:
            events = runner.run(
                user_id=user_id,
                session_id=session.id,
                new_message=content,
            )
            for event in events:
                if event.is_final_response():
                    final_response = event.content.parts[0].text
                    # ADK surfaces the active agent name on the event
                    responding_agent = getattr(event, "author", "unknown")
                    break

        except Exception as exc:
            log.exception("Agent run failed")
            print(f"\n  {RED}Agent error: {exc}{RESET}\n")
            continue

        # ── Print the response ────────────────────────────────────────────────
        print(f"\n{BOLD}{CYAN}Agent ({responding_agent}): {RESET}{final_response}\n")

        # ── Layer 5: log turn end + cost ──────────────────────────────────────
        log_turn_end(session_id, final_response, turn_start, responding_agent)
        # Approximate token counts (ADK doesn't expose token counts directly yet)
        # In production, read from the event's usage metadata
        approx_input  = len(user_input.split()) * 2
        approx_output = len(final_response.split()) * 2
        cost_tracker.record_turn(session_id, approx_input, approx_output)

        # ── Layer 4: HITL approval check ─────────────────────────────────────
        _handle_pending_proposals()

    # ── Session summary ───────────────────────────────────────────────────────
    summary = cost_tracker.session_summary(session_id)
    print(f"""
{DIM}  ─── Session Summary ────────────────────────────────────────────────
  Session ID:    {summary['session_id']}
  Turns:         {summary['turns']}
  Est. tokens:   {summary['total_input_tokens']} in / {summary['total_output_tokens']} out
  Est. cost:     ${summary['total_cost_usd']:.6f} USD
  ─────────────────────────────────────────────────────────────────{RESET}
""")


if __name__ == "__main__":
    run_session()
