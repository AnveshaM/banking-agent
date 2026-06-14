# Banking Agent — Architecture

## Diagram

```
                              USER
                           (main.py)
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Orchestrator     │
                    │      agent.py        │
                    │  classifies intent   │
                    │  routes to sub-agent │
                    │  maintains history   │
                    └──────┬──────┬───────┘
                           │      │      │
              ┌────────────┘      │      └────────────┐
              ▼                   ▼                    ▼
   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
   │  billing_agent   │ │  general_agent   │ │ tech_support_agent│
   │ agents/billing   │ │ agents/general   │ │ agents/tech_support│
   │ invoices·refunds │ │ greetings·policy │ │ API errors·outages│
   └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
            │                    │                     │
            └────────────────────┼─────────────────────┘
                                 ▼
              ┌──────────────────────────────────────────┐
              │              tools/                       │
              │  bank_tools.py  │  wrapper.py             │
              │  schemas.py     │  idempotency.py         │
              └──────────────────────────────────────────┘
              ┌──────────────────────────────────────────┐
              │  Layer 3 · tools/state_tools.py           │
              │  session dict · customer_id · verified    │
              └──────────────────────────────────────────┘
              ┌──────────────────────────────────────────┐
              │  Layer 4 · guardrails/                    │
              │  circuit breaker · HITL proposal queue    │
              └──────────────────────────────────────────┘
              ┌──────────────────────────────────────────┐
              │  Layer 5 · observability/telemetry.py     │
              │  structured JSON logs · cost tracking     │
              └──────────────────────────────────────────┘
```

## Request Flow

1. User types a message in `main.py`
2. The **Orchestrator** (`agent.py`) classifies intent — `billing`, `tech`, or `general` — and dispatches to the right sub-agent, passing along conversation history
3. The **sub-agent** calls tools as needed using native HuggingFace function calling, gets results back, and forms a response
4. If a refund was proposed, `main.py` prompts a human to approve or reject it before any money moves

## The 5 Layers

| Layer | Files | What it does |
|---|---|---|
| 1 · Tooling | `tools/` | All tool functions with Pydantic validation, error envelopes, retry logic, and idempotency |
| 2 · Agents | `agent.py`, `agents/` | Orchestrator + 3 specialist sub-agents (billing, tech, general) |
| 3 · State | `tools/state_tools.py` | Session dict that pins customer ID across turns so the agent doesn't ask twice |
| 4 · Guardrails | `guardrails/` | Circuit breaker (rate limits, per-session caps) + HITL refund approval queue |
| 5 · Observability | `observability/telemetry.py` | Structured JSON logs + token/cost tracking per turn |

## Key Design Decisions

**Multi-agent routing** — A lightweight orchestrator classifies intent and hands off to a specialist. Each sub-agent has a tight scope and its own tool list, keeping context clean and reducing hallucination.

**Native function calling** — Agents use HuggingFace's OpenAI-compatible endpoint with `bind_tools`, so the model calls tools the way it was trained to — structured JSON, not fragile text parsing.

**HITL (Human-in-the-Loop)** — The agent can only *propose* a refund via `propose_refund`. The actual execution (`execute_approved_refund`) happens outside the agent loop, after a human approves it in the CLI. The model cannot bypass this.

**Idempotency** — Every write operation uses an idempotency key. Retrying the same request returns the cached result instead of creating a duplicate proposal.

**Circuit breaker** — Guards state-mutating tools (e.g. `propose_refund`) with per-session call caps, per-minute rate limits, and consecutive failure trips. Lives outside the agent loop so the model cannot prompt-inject past it.

## Model

| Role | Model | Provider |
|---|---|---|
| Orchestrator + workers | `Qwen/Qwen2.5-72B-Instruct` | HuggingFace Inference API (free tier) |

The model is configured in `config.py` and can be swapped via `.env`:
```
ORCHESTRATOR_MODEL=Qwen/Qwen2.5-72B-Instruct
WORKER_MODEL=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=your_token_here
```

## File Structure

```
banking-agent/
├── main.py                  # CLI runner, HITL approval loop
├── agent.py                 # Orchestrator, conversation history, routing
├── config.py                # Model names, env vars, circuit breaker limits
│
├── agents/
│   ├── billing.py           # Invoice lookups, refund proposals
│   ├── general.py           # Greetings, policy questions, fallback
│   └── tech_support.py      # API errors, outages, service status
│
├── tools/
│   ├── bank_tools.py        # lookup_invoice, propose_refund, search_knowledge_base
│   ├── wrapper.py           # @reliable_tool: validation + retry + error envelope
│   ├── schemas.py           # Pydantic input models for every tool
│   ├── state_tools.py       # Session state (customer ID pinning)
│   └── idempotency.py       # Idempotency store + HITL proposal queue
│
├── guardrails/
│   ├── callbacks.py         # check_before_tool() — runs before guarded tools
│   └── circuit_breaker.py   # Rate limits, session caps, failure tripping
│
└── observability/
    └── telemetry.py         # Structured JSON logs, cost tracker
```
