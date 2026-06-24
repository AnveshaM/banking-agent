"""
voice/voice_agent.py — LiveKit Agents entry point.

Pipeline:
  Browser WebRTC → Silero VAD → Deepgram Nova-3 STT
    → BankSupportOrchestrator (all 5 layers)
    → ElevenLabs Flash TTS → Browser WebRTC

Run with:
  cd bank-support-agent
  python -m livekit.agents.cli start voice/voice_agent.py dev

Then open your LiveKit sandbox URL in the browser to start a call.

HITL:
  When the agent proposes a refund, a prompt appears in THIS terminal.
  Type y or n to approve or reject.
  The agent then speaks the approval/rejection to the caller.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

# Make the repo root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()  # loads .env from the repo root
load_dotenv(Path(__file__).parent / ".env.local", override=True)  # voice-specific overrides

from livekit import agents
from livekit.agents import AgentSession, JobContext, cli
from livekit.plugins import deepgram, elevenlabs, silero

from voice.bridge import VoiceBridge

log = logging.getLogger("bank_support.voice")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")


# ── Greeting spoken at the start of every call ───────────────────────────────

_GREETING = (
    "Hello, thank you for calling Bank Support. "
    "I'm your AI assistant. "
    "How can I help you today?"
)


# ── LiveKit Agent definition ─────────────────────────────────────────────────

class BankVoiceAgent(agents.Agent):
    """
    A LiveKit Agent that wraps BankSupportOrchestrator.

    LiveKit handles:
      - WebRTC audio transport
      - VAD (voice activity detection) for turn boundaries
      - Barge-in (if the caller speaks while agent is talking, agent stops)
      - STT / TTS plugin wiring

    This class handles:
      - Forwarding transcribed text to the orchestrator
      - HITL proposal detection and spoken approval/rejection
    """

    def __init__(self):
        super().__init__(
            instructions=(
                "You are a bank customer support voice agent. "
                "You handle invoice lookups, refund proposals, technical issues, "
                "and general policy questions."
            ),
        )
        self.bridge: VoiceBridge | None = None

    async def on_enter(self):
        """Called when a caller connects. Speak the greeting."""
        self.bridge = VoiceBridge()
        log.info("[agent] new call — session %s", self.bridge.session_id)
        await self.session.say(_GREETING, allow_interruptions=True)

    async def on_user_turn_completed(
        self,
        turn_ctx,
        new_message,
    ):
        """
        Called when the caller finishes speaking (VAD + turn detector).
        Transcribed text is in new_message.content.
        """
        user_text = new_message.content
        if not user_text or not user_text.strip():
            return

        log.info("[agent] caller said: %s", user_text[:120])

        # Run the bank agent
        response_text, agent_name = await self.bridge.run(user_text)
        log.info("[agent] %s responding: %s", agent_name, response_text[:120])

        # Speak the response
        await self.session.say(response_text, allow_interruptions=True)

        # Check for pending HITL proposals
        # This awaits terminal input — the caller hears silence while you
        # type y/n, which is intentional and honest about what's happening.
        approval_text = await self.bridge.handle_pending_proposals()
        if approval_text:
            await self.session.say(approval_text, allow_interruptions=False)


# ── LiveKit entry point ──────────────────────────────────────────────────────

async def entrypoint(ctx: JobContext):
    """Called by LiveKit when a new caller connects to the room."""
    await ctx.connect()

    session = AgentSession(
        # STT: Deepgram Nova-3 — 200-400ms latency, no card required
        stt=deepgram.STT(
            model="nova-3",
            language="en",
            smart_format=True,
            # Keyterm prompting — helps Nova-3 catch financial vocabulary
            keywords=[
                ("invoice", 1.5),
                ("refund", 1.5),
                ("INV", 2.0),
                ("CUST", 1.5),
            ],
        ),

        # LLM: not used directly by LiveKit — our bridge calls the orchestrator
        # We use a pass-through placeholder here
        llm=None,

        # TTS: ElevenLabs Flash — best quality, 0.5 credits/char (doubles free allowance)
        tts=elevenlabs.TTS(
            model="eleven_flash_v2_5",
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            # Streaming reduces time-to-first-audio
            streaming=True,
            # Optimise settings for clear phone-call quality
            voice_settings=elevenlabs.VoiceSettings(
                stability=0.5,
                similarity_boost=0.8,
                style=0.0,
                use_speaker_boost=True,
            ),
        ),

        # VAD: Silero — required for barge-in even when Deepgram handles turns
        vad=silero.VAD.load(),
    )

    await session.start(
        room=ctx.room,
        agent=BankVoiceAgent(),
    )


# ── Worker options ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Prewarm: load Silero VAD model once at startup, not per call
            prewarm_fnc=lambda proc: setattr(
                proc, "vad", silero.VAD.load()
            ),
        )
    )
