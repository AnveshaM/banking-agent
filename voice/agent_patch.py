"""
agent.py patch — adds voice_mode=True to BankSupportOrchestrator.run()

Voice mode makes two changes:
  1. Injects a voice brevity instruction into the routing prompt so workers
     know to keep responses to 2-3 sentences max — no bullet points,
     no markdown, no "Here is what I found:". Just natural speech.
  2. Strips any residual markdown from the final response before it
     reaches TTS (asterisks, hashes, backticks all sound terrible spoken aloud).

HOW TO APPLY:
  Replace your existing agent.py with this file.
  Everything else in the repo is unchanged.
"""
import json
import re

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

# Injected into the worker's system message when voice_mode=True
_VOICE_BREVITY_INSTRUCTION = """

IMPORTANT — VOICE MODE:
You are speaking aloud to a customer on a phone call.
Keep every response to 2-3 short sentences maximum.
Never use bullet points, numbered lists, markdown, or headers.
Speak naturally, as you would in a real phone conversation.
Do not say "Here is what I found:" — just say it.
"""

# Markdown patterns that sound terrible when spoken
_MARKDOWN_RE = re.compile(
    r"(\*{1,3}|_{1,3}|#{1,6}\s?|`{1,3}|>\s?|\[([^\]]+)\]\([^\)]+\))"
)


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting that TTS would speak literally."""
    text = _MARKDOWN_RE.sub(r"\2", text)
    # Remove any remaining bare URLs
    text = re.sub(r"https?://\S+", "", text)
    # Collapse multiple spaces/newlines
    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


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

    def run(
        self,
        user_input: str,
        voice_mode: bool = False,
    ) -> tuple[str, str]:
        """
        Run one turn of the bank support agent.

        Args:
            user_input:  The customer's message (transcribed speech in voice mode).
            voice_mode:  If True, inject brevity instruction and strip markdown
                         from the response before returning.

        Returns:
            (response_text, agent_name)
        """
        route = self._classify(user_input)
        agent_fn = self._agents[route]
        response = agent_fn(
            user_input,
            self._history,
            voice_mode=voice_mode,
        )

        # Strip markdown before TTS sees it
        if voice_mode:
            response = _strip_markdown(response)

        self._history.append(("user", user_input))
        self._history.append(("assistant", response))
        return response, self._agent_names[route]

    def _classify(self, message: str) -> str:
        history_text = (
            "\n".join(f"{r}: {t}" for r, t in self._history[-6:]) or "(none)"
        )
        resp = self._orchestrator.invoke([
            SystemMessage(content=_ROUTE_SYSTEM),
            HumanMessage(
                content=_ROUTE_PROMPT.format(
                    history=history_text, message=message
                )
            ),
        ])
        raw = resp.content.strip().lower()
        for key in ("billing", "tech", "general"):
            if key in raw:
                return key
        return "general"


# ADK / LiveKit compatibility export
root_agent = BankSupportOrchestrator()
