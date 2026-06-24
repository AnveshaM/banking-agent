# Voice Layer — Bank Support Agent

Adds a voice interface to the existing bank support agent.
Five layers unchanged. Two new surfaces: a LiveKit voice pipeline and
a browser "click to call" UI.

## Stack

| Component | Tool | Cost |
|---|---|---|
| Voice framework | LiveKit Agents | Free |
| STT | Deepgram Nova-3 | $200 free credit (no card) |
| LLM | Qwen2.5-72B via HuggingFace | Free |
| TTS | ElevenLabs Flash | 10k credits/month free |
| VAD + barge-in | Silero | Free |
| Browser transport | WebRTC | Free |

## Setup

### 1. Get your keys (all free, no card required)

| Key | Where to get it |
|---|---|
| `DEEPGRAM_API_KEY` | console.deepgram.com → Create API Key |
| `ELEVENLABS_API_KEY` | elevenlabs.io → Profile → API Keys |
| `LIVEKIT_URL` | cloud.livekit.io → your project → Settings |
| `LIVEKIT_API_KEY` | same |
| `LIVEKIT_API_SECRET` | same |

### 2. Add keys to your .env

```bash
# Add to your existing .env in the repo root:
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

### 3. Apply the agent patches

```bash
# Replace agents with voice_mode-aware versions
cp voice/agents_patch.py agents/billing.py
# (copy the three make_X_agent functions into their respective files)

# Replace agent.py with the voice_mode-aware version
cp voice/agent_patch.py agent.py
```

### 4. Install voice dependencies

```bash
pip install -r voice/requirements.txt
```

### 5. Run (three terminals)

**Terminal 1 — the voice agent:**
```bash
cd bank-support-agent
python voice/voice_agent.py dev
```

**Terminal 2 — the token server (serves browser credentials):**
```bash
python voice/token_server.py
```

**Terminal 3 — optional: watch logs**
```bash
tail -f bank_support.log  # if you redirect output
```

### 6. Open the browser UI

Open `voice/frontend.html` directly in your browser (double-click the file).
Click **Start Call**. You should hear the greeting within 2 seconds.

---

## The demo flow

```
You say:    "What's the status of invoice INV-00123?"
Agent hears: Deepgram transcribes in ~200-400ms
Agent thinks: BankSupportOrchestrator routes to billing_agent
Agent says:  "Invoice INV-00123 is paid — £49 from Acme Corp."
                                                (ElevenLabs speaks this)

You say:    "I'd like a refund on that. There was an overcharge."
Agent says:  "I've confirmed the invoice. Proposing your refund now.
              It needs manager approval — usually takes 1 business day."

--- Terminal 1 shows the HITL prompt ---
Approve refund? [y/n]: y

Agent says:  "Great news — your refund of £49 has been approved.
              Reference REF-9F4E2B1A. You'll see it within 5 days."
```

## Expected latency (Intel Mac, 16GB)

| Step | Time |
|---|---|
| STT (Deepgram) | 200-400ms |
| LLM (Qwen HF free tier) | 2-4s (variable) |
| TTS (ElevenLabs Flash) | 300-500ms |
| **Total per turn** | **~2.5-5s** |

This is real-world latency for a fully local free-tier stack.
It's slower than commercial production (which hits ~800ms)
but fast enough to hold a real conversation.

## ElevenLabs free tier budget

- 10,000 credits/month
- Flash model: ~0.5 credits/character
- Effective budget: ~20,000 characters
- Typical agent response: ~150 chars
- **~130 full agent responses per month**

This is enough to record the demo video multiple times.
To extend: upgrade to Starter ($5/month) for 30k credits + commercial rights.

## The HITL moment (what to show on video)

The most compelling demo moment:

1. Browser left half: transcript streaming in
2. Terminal right half: JSON logs flying, then the HITL prompt appearing
3. You type `y`
4. The agent's voice says "your refund has been approved"

This is the propose-not-execute pattern audible for the first time.
The agent literally says "I'm not allowed to do this myself."
That's the video.

## Changing the voice

Browse voices at elevenlabs.io/voice-library.
Copy the voice ID and set it in `.env`:

```
ELEVENLABS_VOICE_ID=<voice_id_here>
```

Good options for a bank support agent:
- Rachel (default) — clear, professional American English
- Callum — calm British male
- Charlotte — warm British female
