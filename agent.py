"""
agent.py — Root triage orchestrator.

Uses HuggingFace's OpenAI-compatible endpoint so models like Qwen2.5
can use native function/tool calling instead of fragile text protocols.
"""
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from config import HF_TOKEN, ORCHESTRATOR_MODEL, WORKER_MODEL
from agents.billing import make_billing_agent
from agents.general import make_general_agent
from agents.tech_support import make_tech_support_agent

HF_BASE_URL = "https://router.huggingface.co/v1"

_ROUTE_SYSTEM = (
    "You are a routing classifier. "
    "Reply with exactly one word — billing, tech, or general — nothing else."
)

_ROUTE_PROMPT = """Classify this bank support message into one category:
  billing  — invoices, refunds, charges, payment status
  tech     — API errors, outages, integration bugs, error codes
  general  — greetings, policy questions, unclear intent

Previous conversation:
{history}

Latest message: {message}

One word answer:"""


def _make_llm(model_id: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_id,
        openai_api_key=HF_TOKEN,
        openai_api_base=HF_BASE_URL,
        temperature=0.1,
        max_tokens=1024,
    )


class BankSupportOrchestrator:
    def __init__(self):
        self._orchestrator = _make_llm(ORCHESTRATOR_MODEL)
        worker_llm = _make_llm(WORKER_MODEL)

        self._agents = {
            "billing": make_billing_agent(worker_llm),
            "tech":    make_tech_support_agent(worker_llm),
            "general": make_general_agent(worker_llm),
        }
        self._agent_names = {
            "billing": "billing_agent",
            "tech":    "tech_support_agent",
            "general": "general_agent",
        }
        self._history: list[tuple[str, str]] = []

    def run(self, user_input: str) -> tuple[str, str]:
        route = self._classify(user_input)
        agent_fn = self._agents[route]
        response = agent_fn(user_input, self._history)
        self._history.append(("user", user_input))
        self._history.append(("assistant", response))
        return response, self._agent_names[route]

    def _classify(self, message: str) -> str:
        history_text = "\n".join(f"{r}: {t}" for r, t in self._history[-6:]) or "(none)"
        resp = self._orchestrator.invoke([
            SystemMessage(content=_ROUTE_SYSTEM),
            HumanMessage(content=_ROUTE_PROMPT.format(history=history_text, message=message)),
        ])
        raw = resp.content.strip().lower()
        for key in ("billing", "tech", "general"):
            if key in raw:
                return key
        return "general"


root_agent = BankSupportOrchestrator()
